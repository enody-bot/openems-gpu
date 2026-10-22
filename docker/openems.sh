# /etc/profile.d/openems.sh: the openEMS build on PATH in every shell, including Tailscale SSH
# sessions, which start from a login environment rather than the container's ENV.
export VIRTUAL_ENV=/opt/openEMS/venv
case ":$PATH:" in
  *":/opt/openEMS/bin:"*) ;;
  *) export PATH="/opt/openEMS/venv/bin:/opt/openEMS/bin:$PATH" ;;
esac
