#!/usr/bin/env bash
# HIP-on-NVIDIA headers + hipconfig, no ROCm runtime:
#   bash hip-nvidia.sh [rocm-version] [prefix]
#
# The 3080 box's setup/20-hip-nvidia.sh (openems-bench), unchanged but for the default prefix.
#
# Why not apt: in AMD's ROCm repo, hip-runtime-nvidia is an empty metapackage
# whose only effect is "Depends: cuda" (the whole CUDA metapackage), and
# hip-dev depends on hip-runtime-amd, rocm-llvm and hsa-rocr-dev (the AMD runtime
# plus ~2 GB of LLVM). HIP on NVIDIA needs none of that: CMake's HIP language with
# CMAKE_HIP_PLATFORM=nvidia uses nvcc as compiler, adds <rocm root>/include, and
# finds that root via `hipconfig --rocmpath`. So we extract exactly three packages,
# hip-dev (headers incl. include/hip/nvidia_detail), hipcc-nvidia (hipconfig)
# and rocm-core, after verifying AMD's signed InRelease -> Packages.gz -> .deb SHA256.
#
# Version: the nvidia_detail headers must match the CUDA toolkit. With CUDA 12.6:
#   7.2.x fails (calls cudaMemcpyBatchAsync, CUDA >= 12.8 only, unguarded),
#   7.1.1 fails ("forward declaration of enum type is nonstandard"),
#   7.0.3 and 6.4.4 compile and run.  -> default 7.0.3. Re-test if you move to CUDA 12.8+.
set -euo pipefail
ROCM_VER="${1:-7.0.3}"
PREFIX="${2:-/opt/rocm-${ROCM_VER}-nvidia}"
CODENAME=$(. /etc/os-release; echo "$VERSION_CODENAME")
REPO="https://repo.radeon.com/rocm/apt/${ROCM_VER}"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cd "$W"

curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > amd.gpg
curl -fsSL -O "$REPO/dists/$CODENAME/InRelease"
gpgv --keyring ./amd.gpg InRelease
want=$(sed -n '/^SHA256:/,$p' InRelease | awk '$3=="main/binary-amd64/Packages.gz"{print $1; exit}')
curl -fsSL -o Packages.gz "$REPO/dists/$CODENAME/main/binary-amd64/Packages.gz"
echo "$want  Packages.gz" | sha256sum -c -
gunzip Packages.gz

mkdir x
for p in rocm-core hipcc-nvidia hip-dev; do
  fn=$(awk -v p="$p" '$0=="Package: "p{f=1} f&&/^Filename:/{print $2; exit}' Packages)
  sha=$(awk -v p="$p" '$0=="Package: "p{f=1} f&&/^SHA256:/{print $2; exit}' Packages)
  curl -fsSL -o "$p.deb" "$REPO/$fn"
  echo "$sha  $p.deb" | sha256sum -c -
  echo "$p $(basename "$fn") $sha" >> manifest.txt
  dpkg -x "$p.deb" x
done

rm -rf "$PREFIX"; mkdir -p "$(dirname "$PREFIX")"
mv "x/opt/rocm-${ROCM_VER}" "$PREFIX"
cp manifest.txt "$PREFIX/MANIFEST-openems.txt"
"$PREFIX/bin/hipconfig" --rocmpath 2>/dev/null; echo
test -f "$PREFIX/include/hip/nvidia_detail/nvidia_hip_runtime.h"
echo "OK: HIP-on-NVIDIA in $PREFIX"; cat "$PREFIX/MANIFEST-openems.txt"
