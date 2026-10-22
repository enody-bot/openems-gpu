#!/usr/bin/env bash
# Pod entrypoint (under tini): arm the max-runtime watchdog, join the tailnet as an ephemeral
# tag:gpu-pod node with Tailscale SSH, then wait. gpu-run does the rest over SSH (gpu-job).
#
# Env:
#   TS_OAUTH         Tailscale OAuth client secret, from the RunPod secret TS_OAUTH. Never printed.
#   RUNPOD_POD_ID    set by RunPod; the node is named gpu-$RUNPOD_POD_ID.
#   RUNPOD_API_KEY   set by RunPod: a pod-scoped key, not the account key. Used only to stop this pod.
#   MAX_RUNTIME_MIN  the watchdog's limit (default 240). gpu-run passes its --max-min plus 10.
set -uo pipefail

SOCK=/tmp/ts.sock
MAX_RUNTIME_MIN="${MAX_RUNTIME_MIN:-240}"
POD="${RUNPOD_POD_ID:-local-$(hostname)}"

ts() { tailscale --socket="$SOCK" "$@"; }

# api METHOD URL [JSON]: call the RunPod API with the pod-scoped key, which reaches curl on stdin
# (-K -), so it is on no command line and in no log. Sets CODE and BODY.
api() {
  local args=(-sS -K - -X "$1" -o /tmp/runpod-api.out -w '%{http_code}' "$2")
  [ $# -ge 3 ] && args+=(-H 'Content-Type: application/json' --data "$3")
  CODE=$(printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" | curl "${args[@]}" 2>/dev/null)
  BODY=$(cat /tmp/runpod-api.out 2>/dev/null)
}

# Terminate this pod (stop it if termination is refused). RunPod documents the pod-scoped key but
# not which calls it may make, so try the endpoints in turn and log each answer's status code.
self_stop() {
  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "watchdog: no pod-scoped RunPod key; gpu-run (or gpu-run kill-all) must terminate this pod"
    return 1
  fi
  local id=$RUNPOD_POD_ID
  api DELETE "https://rest.runpod.io/v1/pods/$id"; echo "watchdog: terminate (REST v1) -> HTTP $CODE"
  case "$CODE" in 2??) return 0 ;; esac
  api DELETE "https://api.runpod.io/v2/pods/$id"; echo "watchdog: terminate (REST v2) -> HTTP $CODE"
  case "$CODE" in 2??) return 0 ;; esac
  api POST https://api.runpod.io/graphql "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$id\\\"}) }\"}"
  echo "watchdog: terminate (GraphQL) -> HTTP $CODE"
  [ "$CODE" = 200 ] && ! grep -q '"errors"' <<<"$BODY" && return 0
  api POST "https://rest.runpod.io/v1/pods/$id/stop"; echo "watchdog: stop (REST v1) -> HTTP $CODE"
  case "$CODE" in 2??) return 0 ;; esac
  echo "watchdog: the pod-scoped key could not stop this pod; run gpu-run kill-all"
  return 1
}

watchdog() {
  sleep $((MAX_RUNTIME_MIN * 60))
  echo "watchdog: MAX_RUNTIME_MIN=$MAX_RUNTIME_MIN reached"
  self_stop
}

# A broken pod stops itself instead of exiting: RunPod restarts an exited container, still billing.
fail() {
  echo "entrypoint: $*" >&2
  self_stop
  sleep infinity & wait $!
}

trap 'ts logout >/dev/null 2>&1' EXIT
trap 'exit 143' TERM INT

mkdir -p /scratch
watchdog &
echo "entrypoint: watchdog stops this pod after ${MAX_RUNTIME_MIN} min"

[ -n "${TS_OAUTH:-}" ] || fail "TS_OAUTH is not set (RunPod secret TS_OAUTH missing from the pod env)"

env -u TS_OAUTH -u RUNPOD_API_KEY \
  tailscaled --tun=userspace-networking --state=mem: --socket="$SOCK" >/var/log/tailscaled.log 2>&1 &
TAILSCALED=$!
for _ in $(seq 100); do [ -S "$SOCK" ] && break; sleep 0.1; done

if ! ts up --auth-key="${TS_OAUTH}?ephemeral=true&preauthorized=true" \
     --advertise-tags=tag:gpu-pod --hostname="gpu-${POD}" --ssh; then
  unset TS_OAUTH
  tail -n 20 /var/log/tailscaled.log
  fail "tailscale up failed"
fi
unset TS_OAUTH
echo "entrypoint: on the tailnet as gpu-${POD}"

wait "$TAILSCALED"
fail "tailscaled exited; log tail: $(tail -n 5 /var/log/tailscaled.log)"
