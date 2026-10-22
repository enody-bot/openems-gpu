# Copied to openEMS/localConfig.cmake by the Dockerfile; openEMS/CMakeLists.txt includes it
# before check_language(HIP). The 3080 box's setup/openEMS-localConfig.cmake, with the
# architecture list taken from the environment (Dockerfile ARG HIP_ARCHITECTURES).
#
# On Linux this fork builds its GPU backend with CMake's HIP language on the NVIDIA platform
# (nvcc underneath), so the list goes in CMAKE_HIP_ARCHITECTURES; CMAKE_CUDA_ARCHITECTURES is
# only read by its Windows build. check_language(HIP) configures a child project that only
# inherits CMAKE_HIP_PLATFORM, so nvcc and hipconfig must be on PATH.
set(CMAKE_HIP_PLATFORM nvidia CACHE STRING "HIP platform" FORCE)
set(CMAKE_HIP_ARCHITECTURES "$ENV{OPENEMS_HIP_ARCHITECTURES}" CACHE STRING "HIP (CUDA) architectures")
set(CMAKE_HIP_COMPILER_ROCM_ROOT "$ENV{OPENEMS_ROCM_ROOT}" CACHE PATH "HIP headers root")
