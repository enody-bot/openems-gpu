"""Render step 3: package a scene and a shot as a gpu-run job that renders the animation on a rented GPU.

    python -m kems.render.make_render_job SCENE_DIR SHOT.json --dest jobs/render_NAME [--preview]
    ../gpu-run jobs/render_NAME --gpu "NVIDIA RTX A5000" --max-min 60

The job downloads the official Blender 4.1.1 Linux build from download.blender.org, checks it against the SHA-256
Blender publishes beside it, installs the X / GL libraries Blender loads even headless, builds the scene (scene.py)
and renders it with Cycles on the GPU (OptiX when the driver exposes it, else CUDA) straight to an H.264 MP4.
--preview renders at a quarter of the pixels and 16 samples: check the framing for cents before the full render.

STATUS: untested on gpu-run. The openems-gpu image is a CUDA *base* image and Blender is not in it; the apt step
needs root (RunPod and Modal give it). Blender 4.1 carries CUDA kernels up to sm_89 (Ada): pick an A5000 / A6000 /
L4 / L40S / RTX 4090 class card, not a Blackwell one (RTX 5090, B200), whose sm_100 / sm_120 it cannot run. The
render of a 10 s, 1080p, 64-sample shot of a few sheets is minutes on an A5000; cap it with --max-min anyway."""
import argparse
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER = 'https://download.blender.org/release/Blender4.1/blender-4.1.1-linux-x64.tar.xz'
SUMS = 'https://download.blender.org/release/Blender4.1/blender-4.1.1.sha256'

RUN_SH = """#!/usr/bin/env bash
# kems render job: scene.py over scene/ through shot.json -> out/{name}.mp4 (Blender 4.1.1, Cycles on the GPU).
set -euo pipefail
sha256sum -c SHA256SUMS
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
mkdir -p out /scratch/blender
if [ "$(id -u)" = 0 ]; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends xz-utils libx11-6 libxi6 libxxf86vm1 \\
      libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6 libgl1 libegl1 libglib2.0-0 > /dev/null
fi
cd /scratch/blender
curl -fsSLO {blender}
curl -fsSL {sums} | grep ' blender-4.1.1-linux-x64.tar.xz$' > blender.sha256
sha256sum -c blender.sha256
tar xf blender-4.1.1-linux-x64.tar.xz
cd - > /dev/null
/scratch/blender/blender-4.1.1-linux-x64/blender -b --factory-startup --python scene.py -- scene shot.json \\
    out/{name}.blend --mp4 out/{name}.mp4 2>&1 | grep -v '^Fra:' | tee out/{name}.log
test -s out/{name}.mp4 || {{ echo "FAIL: no video"; exit 6; }}
grep -q 'kems render: \\(OPTIX\\|CUDA\\)' out/{name}.log || {{ echo "FAIL: Cycles did not get the GPU"; exit 3; }}
rm -rf /scratch/blender
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scene')
    ap.add_argument('shot')
    ap.add_argument('--dest', required=True)
    ap.add_argument('--preview', action='store_true')
    a = ap.parse_args()
    name = os.path.basename(a.dest.rstrip('/'))
    if os.path.exists(a.dest):
        shutil.rmtree(a.dest)
    shutil.copytree(a.scene, os.path.join(a.dest, 'scene'))
    shot = json.load(open(a.shot))
    if a.preview:
        shot.update(width=int(shot.get('width', 1920)) // 2, height=int(shot.get('height', 1080)) // 2, samples=16)
    json.dump(shot, open(os.path.join(a.dest, 'shot.json'), 'w'), indent=1)
    shutil.copy(os.path.join(HERE, 'scene.py'), a.dest)
    open(os.path.join(a.dest, 'run.sh'), 'w').write(RUN_SH.format(name=name, blender=BLENDER, sums=SUMS))
    os.chmod(os.path.join(a.dest, 'run.sh'), 0o755)
    sums = []
    for f in ('scene.py', 'shot.json', 'scene/scene.json'):
        sums.append('%s  %s' % (hashlib.sha256(open(os.path.join(a.dest, f), 'rb').read()).hexdigest(), f))
    open(os.path.join(a.dest, 'SHA256SUMS'), 'w').write('\n'.join(sums) + '\n')
    n = sum(len(fs) for _, _, fs in os.walk(os.path.join(a.dest, 'scene')))
    mb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(a.dest) for f in fs) / 1e6
    print('render job %s -> %s: %d scene files, %.1f MB%s' % (name, a.dest, n, mb, ' (preview)' if a.preview else ''))


if __name__ == '__main__':
    main()
