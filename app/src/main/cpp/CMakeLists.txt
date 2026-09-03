cmake_minimum_required(VERSION 3.22.1)
project(lfm_mobile_native LANGUAGES C CXX)

include(FetchContent)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_VULKAN_CHECK_RESULTS OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN_DEBUG OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN_RUN_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_LTO ON CACHE BOOL "" FORCE)

if (ANDROID)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH CACHE STRING "" FORCE)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH CACHE STRING "" FORCE)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER CACHE STRING "" FORCE)

    # Vulkan-Hpp comes from the Khronos Vulkan-Headers checkout in the workflow.
    get_filename_component(_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
    set(_vulkan_headers "${_repo_root}/Vulkan-Headers/include")
    set(_spirv_headers "${_repo_root}/spirv")

    if (NOT EXISTS "${_vulkan_headers}/vulkan/vulkan.hpp")
        message(FATAL_ERROR "Workspace Vulkan-Headers not found: ${_vulkan_headers}/vulkan/vulkan.hpp")
    endif()
    if (NOT EXISTS "${_vulkan_headers}/vulkan/vulkan.h")
        message(FATAL_ERROR "Workspace Vulkan-Headers not found: ${_vulkan_headers}/vulkan/vulkan.h")
    endif()
    if (NOT EXISTS "${_spirv_headers}/unified1/spirv.hpp")
        message(FATAL_ERROR "Workspace SPIR-V C++ header not found: ${_spirv_headers}/unified1/spirv.hpp")
    endif()
    if (NOT EXISTS "/usr/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake")
        message(FATAL_ERROR "SPIRV-Headers CMake package not found")
    endif()

    set(Vulkan_INCLUDE_DIR "${_vulkan_headers}" CACHE PATH "Vulkan headers" FORCE)
    set(Vulkan_INCLUDE_DIRS "${_vulkan_headers}" CACHE STRING "Vulkan header directories" FORCE)
    set(SPIRV-Headers_DIR "/usr/share/cmake/SPIRV-Headers" CACHE PATH "SPIRV-Headers CMake package" FORCE)

    set(_android_ndk "${CMAKE_ANDROID_NDK}")
    if (NOT _android_ndk)
        message(FATAL_ERROR "CMAKE_ANDROID_NDK is not set")
    endif()

    # API 29 provides the Vulkan loader symbols required by ggml-vulkan.
    set(_android_vulkan_library
        "${_android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/29/libvulkan.so")
    set(_android_glslc
        "${_android_ndk}/shader-tools/linux-x86_64/glslc")

    if (NOT EXISTS "${_android_vulkan_library}")
        message(FATAL_ERROR "Android NDK Vulkan library not found: ${_android_vulkan_library}")
    endif()
    if (NOT EXISTS "${_android_glslc}")
        message(FATAL_ERROR "Android NDK glslc not found: ${_android_glslc}")
    endif()

    set(Vulkan_LIBRARY "${_android_vulkan_library}" CACHE FILEPATH "Android NDK Vulkan loader" FORCE)
    set(Vulkan_GLSLC_EXECUTABLE "${_android_glslc}" CACHE FILEPATH "Android host glslc" FORCE)

    # Android clang does not search the host include tree. The Vulkan backend
    # needs SPIR-V-Headers, so force the workspace copy into every C++ compile
    # command as a compiler flag as a final cross-build safeguard.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem${_repo_root}" CACHE STRING "" FORCE)
endif()

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG f1793c1c4e586022efa0b1d3aa6e30ccd67f4e2d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(llama_cpp)

# Also attach the workspace header trees directly to ggml-vulkan. The global
# CXX flag above handles Android's cross-compile path, while these target-level
# settings document the dependency and cover generated backend targets.
if (ANDROID AND TARGET ggml-vulkan)
    target_include_directories(ggml-vulkan SYSTEM PRIVATE
        "${_vulkan_headers}"
        "${_repo_root}"
    )
    target_compile_options(ggml-vulkan PRIVATE
        "-isystem${_vulkan_headers}"
        "-isystem${_repo_root}"
    )
endif()

add_library(lfm_native SHARED native-lib.cpp)

target_compile_features(lfm_native PRIVATE cxx_std_17)
target_compile_options(lfm_native PRIVATE -O3 -fvisibility=hidden)

target_include_directories(lfm_native PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/common
)

target_link_libraries(lfm_native PRIVATE
    llama-common
    llama
    log
    android
)
