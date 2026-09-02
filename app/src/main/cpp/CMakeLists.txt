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

    if (EXISTS "/usr/include/vulkan/vulkan.hpp")
        set(Vulkan_INCLUDE_DIR "/usr/include" CACHE PATH "Vulkan headers" FORCE)
        set(Vulkan_INCLUDE_DIRS "/usr/include" CACHE STRING "Vulkan header directories" FORCE)
    else()
        message(FATAL_ERROR "Host Vulkan-Hpp header not found: /usr/include/vulkan/vulkan.hpp")
    endif()

    set(_android_ndk "${CMAKE_ANDROID_NDK}")
    if (NOT _android_ndk)
        message(FATAL_ERROR "CMAKE_ANDROID_NDK is not set")
    endif()

    set(_android_vulkan_library
        "${_android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/26/libvulkan.so")
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

    if (EXISTS "/usr/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake")
        set(SPIRV-Headers_DIR "/usr/share/cmake/SPIRV-Headers"
            CACHE PATH "SPIRV-Headers CMake package" FORCE)
    else()
        message(FATAL_ERROR "SPIRV-Headers CMake package not found")
    endif()
endif()

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG f1793c1c4e586022efa0b1d3aa6e30ccd67f4e2d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(llama_cpp)

# The Vulkan backend is created inside llama.cpp's FetchContent tree. Add the
# host Vulkan-Hpp include path directly to the backend target so Android clang
# receives it in the actual ggml-vulkan compile command.
if (ANDROID AND TARGET ggml-vulkan)
    target_include_directories(ggml-vulkan PRIVATE SYSTEM "/usr/include")
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
