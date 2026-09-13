cmake_minimum_required(VERSION 3.22.1)
project(lfm_mobile_native LANGUAGES C CXX)
include(FetchContent)

# Prefer a conservative Android build, with optional Vulkan enabled below when
# the SDK and shader compiler are available for the target/toolchain.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_LTO ON CACHE BOOL "" FORCE)

# Android Gradle's CMake configure does not reliably translate the shell
# CMAKE_PREFIX_PATH into package search paths. Point llama.cpp at the config
# installed by CI explicitly when it is available.
if (DEFINED ENV{CMAKE_PREFIX_PATH})
    set(_lfm_prefixes $ENV{CMAKE_PREFIX_PATH})
    list(GET _lfm_prefixes 0 _lfm_spirv_root)
    if (EXISTS "${_lfm_spirv_root}/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake")
        set(SPIRV-Headers_DIR "${_lfm_spirv_root}/share/cmake/SPIRV-Headers" CACHE PATH "" FORCE)
        message(STATUS "LFM: SPIRV-Headers_DIR=${SPIRV-Headers_DIR}")
    endif()
endif()
if (DEFINED ENV{LFM_VULKAN_HEADERS_ROOT})
    set(_lfm_vulkan_include "$ENV{LFM_VULKAN_HEADERS_ROOT}/include")
    if (EXISTS "${_lfm_vulkan_include}/vulkan/vulkan.hpp")
        include_directories(SYSTEM "${_lfm_vulkan_include}")
        message(STATUS "LFM: Vulkan C++ headers=${_lfm_vulkan_include}")
    endif()
endif()

# Enable Vulkan only when the build environment contains both the Vulkan SDK
# and shader compiler. Android Mali devices can use this path, but CI and
# devices without a usable driver must keep working with CPU inference.
find_package(Vulkan QUIET)
find_program(LFM_GLSLC_EXECUTABLE glslc)
if (Vulkan_FOUND AND LFM_GLSLC_EXECUTABLE)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    set(LFM_VULKAN_AVAILABLE ON)
    message(STATUS "LFM: Vulkan backend ENABLED; glslc=${LFM_GLSLC_EXECUTABLE}")
else()
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    set(LFM_VULKAN_AVAILABLE OFF)
    message(STATUS "LFM: Vulkan backend DISABLED; Vulkan_FOUND=${Vulkan_FOUND} glslc=${LFM_GLSLC_EXECUTABLE}")
endif()

# Explicitly disable unsupported GPU / accelerator backends for this Android target.
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_MUSA OFF CACHE BOOL "" FORCE)
set(GGML_WEBGPU OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)
set(GGML_CANN OFF CACHE BOOL "" FORCE)
set(GGML_KOMPUTE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(llama_cpp GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git GIT_TAG f1793c1c4e586022efa0b1d3aa6e30ccd67f4e2d GIT_SHALLOW FALSE)
FetchContent_MakeAvailable(llama_cpp)

# Include the existing engine translation unit so Tool Calling can share its
# private model/context state without exposing globals across JNI modules.
add_library(lfm_native SHARED tool-calling.cpp)
target_compile_features(lfm_native PRIVATE cxx_std_17)
target_compile_options(lfm_native PRIVATE -O3 -fvisibility=hidden)
target_compile_definitions(lfm_native PRIVATE LFM_VULKAN_AVAILABLE=$<BOOL:${LFM_VULKAN_AVAILABLE}>)
target_include_directories(lfm_native PRIVATE ${llama_cpp_SOURCE_DIR}/include ${llama_cpp_SOURCE_DIR}/common)
target_link_libraries(lfm_native PRIVATE llama-common llama log android)
