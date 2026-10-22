# Tailnet policy for the GPU pods

GPU pods join the tailnet as ephemeral nodes tagged `tag:gpu-pod`. They can be reached by tailnet
admins and by machines tagged `tag:gpu-controller`, on SSH (Tailscale SSH, as root) and on
tcp 8000-8100 (for a dashboard or a notebook while a job runs). No rule has `tag:gpu-pod` as its
source, so a pod cannot open a connection to anything on the tailnet. A pod could open one only through
a SOCKS or HTTP proxy, which its userspace-mode tailscaled doesn't run.

You apply this yourself in the admin console (**Access controls**, JSON editor).

## 1. Check the existing policy first

Look for any rule with `"*"` on either side, for example the default
`{"action": "accept", "src": ["*"], "dst": ["*:*"]}`. `"*"` includes tagged devices: as a source
it would let the pods reach everything, and as a destination it would open the pods to everyone
on every port. Tailscale has no deny rules, so spell such a rule out instead: `autogroup:member`
(the devices of tailnet members, no tagged devices), the tags that rely on it, your subnet routes,
and `autogroup:internet` if anyone uses an exit node. For example, with `tag:k8s`,
`tag:k8s-operator` and a `10.0.0.0/16` subnet route:

```jsonc
{
  "action": "accept",
  "src": ["autogroup:member", "tag:k8s", "tag:k8s-operator", "10.0.0.0/16"],
  "dst": ["autogroup:member:*", "tag:k8s:*", "tag:k8s-operator:*", "10.0.0.0/16:*", "autogroup:internet:*"]
}
```

The `tests` below fail on save if a pod could still reach your workstation, which catches a rule
you missed. Put your workstation's address (`tailscale ip -4` on it) in place of
`WORKSTATION_TS_IP` first.

## 2. Merge these sections into the policy

Add the keys to the sections the policy already has; don't replace the file. `grants` can sit
next to an existing `acls` section.

```jsonc
{
  "tagOwners": {
    "tag:gpu-pod":        ["autogroup:admin"],
    // the source tag for a future headless launcher; must be defined to be used below
    "tag:gpu-controller": ["autogroup:admin"]
  },

  "grants": [
    {
      "src": ["autogroup:admin", "tag:gpu-controller"],
      "dst": ["tag:gpu-pod"],
      "ip":  ["tcp:22", "tcp:8000-8100"]
    }
  ],

  "ssh": [
    {
      "action": "accept",
      "src":    ["autogroup:admin", "tag:gpu-controller"],
      "dst":    ["tag:gpu-pod"],
      "users":  ["root"]
    }
  ],

  "tests": [
    {
      "src":    "tag:gpu-controller",
      "accept": ["tag:gpu-pod:22", "tag:gpu-pod:8000", "tag:gpu-pod:8100"],
      "deny":   ["tag:gpu-pod:443"]
    },
    {
      // pods reach nothing: not each other, not a controller, not your workstation
      "src":  "tag:gpu-pod",
      "deny": ["tag:gpu-pod:22", "tag:gpu-controller:22", "WORKSTATION_TS_IP:22"]
    }
  ],

  "sshTests": [
    {
      "src":    "tag:gpu-controller",
      "dst":    ["tag:gpu-pod"],
      "accept": ["root"]
    }
  ]
}
```

Notes:

- Tailscale SSH needs both rules: the grant opens port 22, and the `ssh` rule lets the connection
  log in as root. `accept` (not `check`) because gpu-run connects without a browser.
- `autogroup:admin` covers the devices of users with the Admin role. If the editor refuses it in
  the `ssh` rule, or your account's role isn't Admin, put your login email in that
  `src` too.
- If you drop the workstation address from the `tests`, keep the other deny entries.

## 3. OAuth client for the pods

Pods authenticate with an OAuth client secret used as an auth key
(`tailscale up --auth-key="<secret>?ephemeral=true&preauthorized=true" --advertise-tags=tag:gpu-pod`).
The policy must define `tag:gpu-pod` before you create the client.

1. Admin console, **Settings > Trust credentials** (`/admin/settings/trust-credentials`),
   **+ Credential**, **OAuth**.
2. Scope: **Keys > Auth Keys: Write** (`auth_keys`). Tags: **`tag:gpu-pod`** only.
3. **Generate credential**. Copy the client secret (`tskey-client-...`). The console shows it once.
   Don't paste it into a terminal or a file.
4. RunPod console, **Settings > Secrets**, **Create Secret**: name `TS_OAUTH`, value: the client
   secret. Pods get it as `{{ RUNPOD_SECRET_TS_OAUTH }}`; gpu-run only sends that placeholder.
   If RunPod insists on a lowercase name, use `ts_oauth` and run gpu-run with
   `GPU_RUN_TS_SECRET=ts_oauth`.
5. Close the "Credential created" page.

The secret doesn't expire. It can only mint keys for `tag:gpu-pod`, and those nodes can reach
nothing. Revoke it from the same page if it leaks, then create a new one and update the RunPod
secret.
