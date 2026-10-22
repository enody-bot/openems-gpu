# syntax=docker/dockerfile:1
# openEMS with its GPU engine (SeanMollet/openEMS GPU_merge = thliebig/openEMS PR #235: the HIP
# backend, compiled for NVIDIA by nvcc), CSXCAD and the Python bindings, for RunPod pods that join
# the tailnet. The build is the 3080 box's (openems-bench/BUILD.md) with more GPU architectures.
#
# CUDA 12.6 (README, "CUDA version"): the toolkit the 3080 box builds and validates with, the newest
# the AMD 7.0.3 HIP headers compile with, and new enough for sm_90. gpu-run only takes RunPod hosts
# whose driver supports 12.6 or later.
ARG CUDA_VERSION=12.6.3
ARG UBUNTU=ubuntu24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build
# openEMS-Project v0.37.0 (CSXCAD c46e320, fparser 4b9c845) with openEMS at sean/GPU_merge, as on the box
ARG OPENEMS_PROJECT_SHA=974294ba7d812ef5a46a4760593057aee7ce6866
ARG OPENEMS_SHA=ee45c6dfb1a5b3a0843e7c5d4293668a95567dda
ARG ROCM_VERSION=7.0.3
# GTX 1070, RTX 2080 Ti, A100, RTX 3080 / A-series, H100 (each also as PTX)
ARG HIP_ARCHITECTURES="61;75;80;86;90"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates curl gnupg gpgv \
        libboost-all-dev libcgal-dev libhdf5-dev libtinyxml-dev libvtk9-dev \
        python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

# HIP headers for the NVIDIA platform from AMD's signed repo (no ROCm runtime)
COPY docker/hip-nvidia.sh /tmp/
RUN bash /tmp/hip-nvidia.sh "$ROCM_VERSION" /opt/rocm-nvidia

RUN git clone https://github.com/thliebig/openEMS-Project.git /src/openEMS-Project \
    && cd /src/openEMS-Project \
    && git checkout --detach "$OPENEMS_PROJECT_SHA" \
    && git submodule update --init --recursive \
    && { git -C openEMS fetch https://github.com/SeanMollet/openEMS.git GPU_merge \
         || git -C openEMS fetch https://github.com/SeanMollet/openEMS.git "$OPENEMS_SHA"; } \
    && git -C openEMS checkout --detach "$OPENEMS_SHA" \
    && test "$(git -C CSXCAD rev-parse HEAD)" = c46e3206f20027b0b4605c797d07bddab76de942 \
    && test "$(git -C fparser rev-parse HEAD)" = 4b9c845b449b520c4b8c5f23c74cd04820084f81

COPY docker/localConfig.cmake /src/openEMS-Project/openEMS/localConfig.cmake
ENV CUDA_PATH=/usr/local/cuda \
    HIP_PLATFORM=nvidia \
    OPENEMS_ROCM_ROOT=/opt/rocm-nvidia \
    OPENEMS_HIP_ARCHITECTURES=${HIP_ARCHITECTURES} \
    VIRTUAL_ENV=/opt/openEMS/venv \
    PATH=/opt/openEMS/venv/bin:/usr/local/cuda/bin:/opt/rocm-nvidia/bin:${PATH}

RUN python3 -m venv /opt/openEMS/venv \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy scipy h5py matplotlib pandas

# update_openEMS.sh builds fparser, CSXCAD and openEMS into /opt/openEMS and the bindings into the
# active venv. The build must have taken the HIP backend and hold code for every architecture.
RUN cd /src/openEMS-Project \
    && { ./update_openEMS.sh /opt/openEMS --python --disable-GUI --skip-dep-check \
         || { tail -n 80 build_*.log; exit 1; }; } \
    && grep -F -- "-- HIP GPU backend enabled, architectures: ${HIP_ARCHITECTURES}" build_*.log \
    && cuobjdump --list-elf /opt/openEMS/lib/libopenEMS.so | grep -oE 'sm_[0-9]+' | sort -u > /tmp/sass.txt \
    && for a in $(echo "$HIP_ARCHITECTURES" | tr ';' ' '); do \
         grep -qx "sm_$a" /tmp/sass.txt || { echo "libopenEMS.so has no sm_$a code"; exit 1; }; \
       done \
    && { echo "openEMS-Project $OPENEMS_PROJECT_SHA"; git submodule status; \
         echo "CUDA $CUDA_VERSION, HIP headers ROCm $ROCM_VERSION, SASS: $(tr '\n' ' ' < /tmp/sass.txt)"; \
         echo; pip freeze; } > /opt/openEMS/BUILD-INFO.txt \
    && install -D -m 644 openEMS/python/Tests/GPU_Engine.py /opt/openEMS/share/openEMS/tests/GPU_Engine.py

# The runtime image installs the Debian packages that own the shared libraries the build links.
RUN find /opt/openEMS -type f \( -name '*.so*' -o -perm -u+x \) -print0 \
    | xargs -0 ldd 2>/dev/null | awk '$2 == "=>" && $3 ~ /^\// {print $3}' | sort -u \
    | grep -v -e '^/opt/openEMS/' -e '^/usr/local/cuda' \
    | while read -r lib; do \
        dpkg -S "$lib" 2>/dev/null || dpkg -S "$(readlink -f "$lib")" 2>/dev/null || dpkg -S "/usr$lib" 2>/dev/null; \
      done \
    | cut -d: -f1 | sort -u > /opt/openEMS/runtime-packages.txt \
    && grep -q vtk /opt/openEMS/runtime-packages.txt && grep -q hdf5 /opt/openEMS/runtime-packages.txt


FROM nvidia/cuda:${CUDA_VERSION}-base-${UBUNTU}
LABEL org.opencontainers.image.source=https://github.com/enody-bot/openems-gpu
LABEL org.opencontainers.image.description="openEMS GPU engine (HIP on NVIDIA) for RunPod pods on the tailnet"
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=build /opt/openEMS /opt/openEMS
RUN apt-get update \
    && xargs -a /opt/openEMS/runtime-packages.txt apt-get install -y --no-install-recommends \
         python3 rsync openssh-client curl ca-certificates tini \
    && curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg \
         -o /usr/share/keyrings/tailscale-archive-keyring.gpg \
    && curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list \
         -o /etc/apt/sources.list.d/tailscale.list \
    && apt-get update && apt-get install -y --no-install-recommends tailscale \
    && rm -rf /var/lib/apt/lists/* \
    && echo /opt/openEMS/lib > /etc/ld.so.conf.d/openems.conf && ldconfig \
    && echo "tailscale $(tailscale version | head -n 1)" >> /opt/openEMS/BUILD-INFO.txt

ENV VIRTUAL_ENV=/opt/openEMS/venv \
    PATH=/opt/openEMS/venv/bin:/opt/openEMS/bin:${PATH}
# Every library resolves (libcuda comes from the host driver at run time) and the bindings import
# (unless libopenEMS needs libcuda itself; then only a GPU host can import it).
RUN ! ldd /opt/openEMS/lib/*.so* /opt/openEMS/bin/* 2>/dev/null | grep -v 'libcuda\.so' | grep 'not found' \
    && if ldd /opt/openEMS/lib/libopenEMS.so | grep -q 'libcuda\.so'; then \
         echo "libopenEMS.so links libcuda: import not checked here"; \
       else python -B -c "import CSXCAD, openEMS; print('openEMS', openEMS.__version__)"; fi

COPY docker/openems.sh /etc/profile.d/openems.sh
COPY --chmod=755 docker/entrypoint.sh docker/gpu-job /usr/local/bin/
RUN mkdir -p /scratch
WORKDIR /scratch
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
