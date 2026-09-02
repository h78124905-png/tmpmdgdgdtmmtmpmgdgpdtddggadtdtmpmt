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
set(GGML_VULKAN_RUN_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_LTO ON CACHE BOOL "" FORCE)

# Android's NDK supplies the Vulkan loader library, while the host Linux
# packages supply the C++ Vulkan header (vulkan.hpp) and SPIR-V headers used
# by llama.cpp's Vulkan backend. Keeping these roles separate avoids trying
# to link the Android build against a Linux Vulkan library.
if (ANDROID AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(_android_ndk "${ANDROID_NDK}")
    if (NOT _android_ndk)
        set(_android_ndk "${CMAKE_ANDROID_NDK}")
    endif()

    if (_android_ndk)
        set(Vulkan_LIBRARY
            "${_android_ndk}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/26/libvulkan.so"
            CACHE FILEPATH "Android NDK Vulkan loader" FORCE)
        set(Vulkan_GLSLC_EXECUTABLE
            "${_android_ndk}/shader-tools/linux-x86_64/glslc"
            CACHE FILEPATH "Android NDK host glslc" FORCE)
    endif()

    # ggml-vulkan needs vulkan/vulkan.hpp, which is not provided by the NDK
    # sysroot. libvulkan-dev installs it under /usr/include/vulkan.
    if (EXISTS "/usr/include/vulkan/vulkan.hpp")
        set(Vulkan_INCLUDE_DIR "/usr/include" CACHE PATH "Host Vulkan headers" FORCE)
    endif()

    # Ubuntu's spirv-headers package installs its CMake package below
    # /usr/share/cmake rather than a default CMAKE_PREFIX_PATH entry.
    if (EXISTS "/usr/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake")
        set(SPIRV-Headers_DIR "/usr/share/cmake/SPIRV-Headers"
            CACHE PATH "SPIRV-Headers CMake package" FORCE)
    endif()
endif()

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG f1793c1c4e586022efa0b1d3aa6e30ccd67f4e2d
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(llama_cpp)

# The Vulkan backend is compiled as part of ggml. Add the host Vulkan include
# directory explicitly as a final safeguard for vulkan.hpp when cross-building.
if (ANDROID AND TARGET ggml-vulkan AND EXISTS "/usr/include/vulkan/vulkan.hpp")
    target_include_directories(ggml-vulkan SYSTEM PRIVATE /usr/include)
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
