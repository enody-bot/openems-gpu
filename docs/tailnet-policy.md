# Tailnet policy for the GPU pods

GPU pods join the tailnet as ephemeral nodes tagged `tag:gpu-pod`. They can be reached by tailnet
admins and by machines tagged `tag:gpu-controller`, on SSH (Tailscale SSH, as root) and on
tcp 8000-8100 (for a dashboard or a notebook while a job runs). No rule has `tag:gpu-pod` as its
source, so a pod cannot open a connection to anything on the tailnet. A pod could open one only through
a SOCKS or HTTP proxy, which its userspace-mode tailscaled doesn't run.

You apply this yourself in the admin console (**Access controls**, JSON editor).

## 1. Check the existing policy first

Look for any rule whose source is `"*"`, for example the default
`{"action": "accept", "src": ["*"], "dst": ["*:*"]}`. `"*"` includes tagged devices, so the
pods would get everything that rule grants. Narrow such rules before you add the pods: replace
`"*"` with `"autogroup:member"` (the devices of tailnet members, no tagged devices) plus any tags
that still need the access. This tailnet's machine list shows `tag:k8s` (enody-staging) and
`tag:k8s-operator`, for example:

```jsonc
{"action": "accept", "src": ["autogroup:member", "tag:k8s", "tag:k8s-operator"], "dst": ["*:*"]}
```

The `tests` below fail on save if a pod could still reach this Mac, which catches a rule you missed.

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
      // pods reach nothing: not each other, not a controller, not this Mac (claptrap)
      "src":  "tag:gpu-pod",
      "deny": ["tag:gpu-pod:22", "tag:gpu-controller:22", "100.68.235.75:22"]
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
  the `ssh` rule, or your account's role isn't Admin, put your login (for example
  `carter@enody.lighting`) in that `src` too.
- If you drop the `tests` entry with this Mac's address, keep the other deny test.

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
