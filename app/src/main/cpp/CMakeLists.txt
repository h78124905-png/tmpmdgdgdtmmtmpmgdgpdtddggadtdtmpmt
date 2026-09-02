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

# Android Vulkan backend.
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_VULKAN_CHECK_RESULTS OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN_DEBUG OFF CACHE BOOL "" FORCE)
set(GGML_LTO ON CACHE BOOL "" FORCE)

# CMake's FindVulkan must be pointed at the Android NDK's Vulkan loader and
# headers.  Without these hints it can discover the Linux host Vulkan SDK
# installed on the GitHub runner, which is the wrong architecture for the
# arm64-v8a Android target.  The NDK also ships the host glslc used to compile
# Vulkan shaders during the cross-build.
if (ANDROID AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(_android_ndk "${ANDROID_NDK}")
    if (NOT _android_ndk)
        set(_android_ndk "${CMAKE_ANDROID_NDK}")
    endif()

    if (_android_ndk)
        set(Vulkan_INCLUDE_DIR
            "${_android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include"
            CACHE PATH "Android NDK Vulkan headers" FORCE)
        set(Vulkan_LIBRARY
            "${_android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/26/libvulkan.so"
            CACHE FILEPATH "Android NDK Vulkan loader" FORCE)
        set(Vulkan_GLSLC_EXECUTABLE
            "${_android_ndk}/shader-tools/linux-x86_64/glslc"
            CACHE FILEPATH "Android NDK host glslc" FORCE)
    endif()
endif()

# Pin the upstream revision used by the Android host.  Use a full clone so
# CMake/Git can reliably check out an exact commit instead of depending on
# GitHub's shallow-clone handling of a raw SHA.
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG f1793c1c4e586022efa0b1d3aa6e30ccd67f4e2d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(llama_cpp)

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
