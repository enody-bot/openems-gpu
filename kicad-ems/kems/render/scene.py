"""Render step 2: the exploded field animation in Blender 4.1 (Cycles), from textures.py's SCENE_DIR and a shot.

    blender -b --factory-startup --python scene.py -- SCENE_DIR SHOT.json OUT.blend [--render FRAMES_DIR | --mp4 OUT.mp4]

Without --render / --mp4 it only builds and saves the .blend (cheap: run it anywhere to check a shot; open the .blend
to frame the camera). Rendering is the expensive part: run it on a GPU (make_render_job.py packages it for gpu-run).

The scene, in Blender's frame (x = KiCad x, y = -KiCad y, z up; metres, the board's bottom face at z = 0):
- every copper layer as a see-through sheet: its texture as a faint unlit glow over transparency (lit, a stack of
  layers turns into an opaque beige slab and hides the fields inside it), opacity per layer from the shot;
- every field plane as a sheet playing its frame sequence at the keyed simulation time; the 8-bit log code goes
  through the shot's palette into emission (gain per plane, gamma, a floor below which it is transparent), added on
  top of what is behind it, so the glow of a lane adds to the reference plane's return current under it;
- explode: each copper layer moves to z + (i - anchor) * spacing * explode(t) and each field sheet keeps its place
  between the layers it sits between (or on the face it shows), baked on every frame;
- the camera: keys (position, target, lens, explode, sim_ns) eased between, baked on every frame; a dark void and a
  compositor fog glow.
SHOT.json (times in s; positions in KiCad mm, z above the board's bottom face):
  {"fps": 30, "duration_s": 10, "width": 1920, "height": 1080, "samples": 64, "spacing_mm": 6.0, "anchor": "In4.Cu",
   "keys": [{"t": 0, "sim_ns": 0, "position": [138, 101.5, 300], "target": [138, 101, 0.8], "lens_mm": 85, "explode": 0},
            {"t": 4, "sim_ns": 2.5, ...}, {"t": 10, "sim_ns": 6, "position": [105, 215, 75], ..., "explode": 1}],
   "opacity": {"F.Cu": 0.35, "In3.Cu": 0.25}, "gain": {"E_above_In4": 0.6}, "gamma": 0.8, "floor": 0.08,
   "palette": ["#1b3cff", "#18c8d8", "#7ce35a", "#f9e64f", "#ffffff"], "background": "#07090c", "bloom": 0.5}
A top-down camera exactly above its target has no defined "up": keep the position a hair off the target in y."""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
SCENE_DIR, SHOT_PATH, OUT = argv[0], argv[1], argv[2]
RENDER = argv[argv.index('--render') + 1] if '--render' in argv else None
MP4 = argv[argv.index('--mp4') + 1] if '--mp4' in argv else None
S = json.load(open(os.path.join(SCENE_DIR, 'scene.json')))
SHOT = {**dict(fps=30, duration_s=10.0, width=1920, height=1080, samples=64, spacing_mm=6.0, anchor=None, opacity={},
               gain={}, gamma=0.8, floor=0.08, palette=['#1b3cff', '#18c8d8', '#7ce35a', '#f9e64f', '#ffffff'],
               background='#07090c', bloom=0.5), **json.load(open(SHOT_PATH))}
MM = 1e-3
FIELD_EMIT = 8.0
GHOST_EMIT = 0.5
FPS, FRAMES = int(SHOT['fps']), max(1, int(round(SHOT['duration_s'] * SHOT['fps'])))
X0, Y0, X1, Y1 = S['box']
CU = [L['name'] for L in S['copper']]
ANCHOR = CU.index(SHOT['anchor']) if SHOT['anchor'] in CU else len(CU) // 2


def B(x, y, z):
    """KiCad mm -> Blender metres."""
    return Vector((x * MM, -y * MM, z * MM))


def srgb(h):
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    h = h.lstrip('#')
    return tuple(lin(int(h[i:i + 2], 16)) for i in (0, 2, 4)) + (1.0,)


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def at(t, field):
    """A key field at time t: eased between the keys either side (positions as vectors)."""
    K = SHOT['keys']
    if t <= K[0]['t']:
        return K[0][field]
    for a, b in zip(K[:-1], K[1:]):
        if t <= b['t']:
            u = ease((t - a['t']) / max(b['t'] - a['t'], 1e-9))
            va, vb = a.get(field, b.get(field)), b.get(field, a.get(field))
            if isinstance(va, list):
                return [x + (y - x) * u for x, y in zip(va, vb)]
            return va + (vb - va) * u
    return K[-1][field]


def sheet(name, z_mm):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(X0 * MM, -Y0 * MM, 0), (X1 * MM, -Y0 * MM, 0), (X1 * MM, -Y1 * MM, 0), (X0 * MM, -Y1 * MM, 0)],
                   [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new()
    for li, (u, v) in zip(range(4), [(0, 1), (1, 1), (1, 0), (0, 0)]):      # image rows run KiCad y downward
        uv.data[li].uv = (u, v)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.location.z = z_mm * MM
    return ob


def node_tree(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    mat.blend_method = 'BLEND'
    return nt, out


def layer_material(L, opacity):
    mat = bpy.data.materials.new('cu_' + L['name'])
    nt, out = node_tree(mat)
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(os.path.join(SCENE_DIR, L['png']))
    tex.interpolation = 'Closest'
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = GHOST_EMIT * opacity
    nt.links.new(tex.outputs['Color'], em.inputs['Color'])
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    mix = nt.nodes.new('ShaderNodeMixShader')
    mul = nt.nodes.new('ShaderNodeMath')
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = opacity
    nt.links.new(tex.outputs['Alpha'], mul.inputs[0])
    nt.links.new(mul.outputs[0], mix.inputs['Fac'])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
    return mat


def field_material(P, gain):
    mat = bpy.data.materials.new('field_' + P['name'])
    nt, out = node_tree(mat)
    tex = nt.nodes.new('ShaderNodeTexImage')
    first = os.path.join(SCENE_DIR, P['frames'].replace('####', '0001'))
    img = bpy.data.images.load(first)
    img.source = 'SEQUENCE'
    img.colorspace_settings.name = 'Non-Color'
    tex.image = img
    # Blender clamps the scene frame to frame_duration before adding frame_offset: a duration of
    # P['n'] shows min(f, n) + offset, which runs off the sequence once the shot outlasts it.
    tex.image_user.frame_duration = max(P['n'], FRAMES)
    tex.image_user.frame_start = 1
    tex.image_user.use_auto_refresh = True
    tex.interpolation = 'Linear'
    gam = nt.nodes.new('ShaderNodeMath')
    gam.operation = 'POWER'
    gam.inputs[1].default_value = SHOT['gamma']
    nt.links.new(tex.outputs['Color'], gam.inputs[0])       # Color.r of a grey image = the code / 255
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    pal = SHOT['palette']
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, srgb(pal[0])
    els[1].position, els[1].color = 1.0, srgb(pal[-1])
    for k, c in enumerate(pal[1:-1], 1):
        e = els.new(k / (len(pal) - 1))
        e.color = srgb(c)
    nt.links.new(gam.outputs[0], ramp.inputs['Fac'])
    floor = nt.nodes.new('ShaderNodeMath')
    floor.operation = 'GREATER_THAN'
    floor.inputs[1].default_value = SHOT['floor']
    nt.links.new(tex.outputs['Color'], floor.inputs[0])
    strength = nt.nodes.new('ShaderNodeMath')
    strength.operation = 'MULTIPLY'
    strength.inputs[1].default_value = FIELD_EMIT * gain
    nt.links.new(gam.outputs[0], strength.inputs[0])
    gate = nt.nodes.new('ShaderNodeMath')
    gate.operation = 'MULTIPLY'
    nt.links.new(strength.outputs[0], gate.inputs[0])
    nt.links.new(floor.outputs[0], gate.inputs[1])
    em = nt.nodes.new('ShaderNodeEmission')
    nt.links.new(ramp.outputs['Color'], em.inputs['Color'])
    nt.links.new(gate.outputs[0], em.inputs['Strength'])
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    add = nt.nodes.new('ShaderNodeAddShader')               # glow over whatever is behind it
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs['Surface'])
    return mat, tex


bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
sc.render.fps = FPS
sc.frame_start, sc.frame_end = 1, FRAMES
sc.render.resolution_x, sc.render.resolution_y = int(SHOT['width']), int(SHOT['height'])
sc.render.resolution_percentage = 100
sc.cycles.samples = int(SHOT['samples'])
sc.cycles.use_denoising = True
sc.cycles.denoiser = 'OPENIMAGEDENOISE'                  # runs anywhere; OptiX's needs OptiX (not under WSL2)
sc.cycles.transparent_max_bounces = 64                      # every sheet is a transparent bounce
sc.cycles.max_bounces = 64
sc.view_settings.view_transform = 'AgX'
world = bpy.data.worlds.new('void')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = srgb(SHOT['background'])
sc.world = world

layer_obs = {}
for L in S['copper']:
    ob = sheet(L['name'], L['z_mm'])
    ob.data.materials.append(layer_material(L, float(SHOT['opacity'].get(L['name'], 0.15))))
    layer_obs[L['name']] = (ob, L['z_mm'])
field_obs = []
for P in S['planes']:
    ob = sheet(P['name'], P['z_mm'])
    mat, tex = field_material(P, float(SHOT['gain'].get(P['name'], 0.5)))
    ob.data.materials.append(mat)
    field_obs.append((ob, P, tex))

cam_data = bpy.data.cameras.new('cam')
cam_data.sensor_width = 36.0
cam_data.clip_start, cam_data.clip_end = 1e-3, 50.0
cam = bpy.data.objects.new('cam', cam_data)
sc.collection.objects.link(cam)
sc.camera = cam
target = bpy.data.objects.new('target', None)
sc.collection.objects.link(target)
con = cam.constraints.new('TRACK_TO')
con.target, con.track_axis, con.up_axis = target, 'TRACK_NEGATIVE_Z', 'UP_Y'

spacing = float(SHOT['spacing_mm'])
z_of = {L['name']: L['z_mm'] for L in S['copper']}
t0, dt = S['t_ns'] if S['t_ns'] else (0.0, 1.0)


def layer_z(name, ex):
    return z_of[name] + (ANCHOR - CU.index(name)) * spacing * ex      # CU runs top (F.Cu) to bottom: upper ones rise


def plane_z(P, ex):
    """A field sheet between two layers keeps its fraction of the gap; one on a face or above a layer moves with it."""
    btw = [b for b in P.get('between', []) if b in z_of]
    if len(btw) == 2:
        za, zb = z_of[btw[0]], z_of[btw[1]]
        u = (P['z_mm'] - za) / (zb - za) if zb != za else 0.5
        return layer_z(btw[0], ex) + u * (layer_z(btw[1], ex) - layer_z(btw[0], ex))
    if btw:
        return P['z_mm'] + layer_z(btw[0], ex) - z_of[btw[0]]
    return P['z_mm']


for f in range(1, FRAMES + 1):
    t = (f - 1) / FPS
    ex = float(at(t, 'explode'))
    for name, (ob, z) in layer_obs.items():
        ob.location.z = layer_z(name, ex) * MM
        ob.keyframe_insert('location', index=2, frame=f)
    sim = float(at(t, 'sim_ns'))
    for ob, P, tex in field_obs:
        ob.location.z = plane_z(P, ex) * MM
        ob.keyframe_insert('location', index=2, frame=f)
        idx = int(round((sim - t0) / dt)) + 1
        idx = min(max(idx, 1), P['n'])
        tex.image_user.frame_offset = idx - f               # shown frame = scene frame + offset
        tex.image_user.keyframe_insert('frame_offset', frame=f)
    cam.location = B(*at(t, 'position'))
    cam.keyframe_insert('location', frame=f)
    target.location = B(*at(t, 'target'))
    target.keyframe_insert('location', frame=f)
    cam_data.lens = float(at(t, 'lens_mm'))
    cam_data.keyframe_insert('lens', frame=f)
for ad in [o.animation_data for o in bpy.data.objects if o.animation_data] + \
        [m.node_tree.animation_data for m in bpy.data.materials if m.node_tree and m.node_tree.animation_data] + \
        ([cam_data.animation_data] if cam_data.animation_data else []):
    if ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT'

sc.use_nodes = True
ct = sc.node_tree
for n in list(ct.nodes):
    ct.nodes.remove(n)
rl = ct.nodes.new('CompositorNodeRLayers')
glare = ct.nodes.new('CompositorNodeGlare')
glare.glare_type, glare.quality, glare.mix, glare.size = 'FOG_GLOW', 'HIGH', SHOT['bloom'] - 1.0, 8
comp = ct.nodes.new('CompositorNodeComposite')
ct.links.new(rl.outputs['Image'], glare.inputs['Image'])
ct.links.new(glare.outputs['Image'], comp.inputs['Image'])

bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(OUT))
print('kems scene: %d layers, %d field sheets, %d frames -> %s' % (len(layer_obs), len(field_obs), FRAMES, OUT))

if RENDER or MP4:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for kind in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            if any(d.type == kind for d in prefs.devices):
                break
        except TypeError:
            continue
    for d in prefs.devices:
        d.use = d.type != 'CPU'
    sc.cycles.device = 'GPU'
    print('kems render: %s on %s' % (prefs.compute_device_type, ', '.join(d.name for d in prefs.devices if d.use)))
    if MP4:
        sc.render.image_settings.file_format = 'FFMPEG'
        sc.render.ffmpeg.format, sc.render.ffmpeg.codec = 'MPEG4', 'H264'
        sc.render.ffmpeg.constant_rate_factor = 'HIGH'
        sc.render.filepath = os.path.abspath(MP4)
    else:
        sc.render.image_settings.file_format = 'PNG'
        sc.render.filepath = os.path.join(os.path.abspath(RENDER), 'f_')
    bpy.ops.render.render(animation=True)
