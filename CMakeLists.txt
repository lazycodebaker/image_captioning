cmake_minimum_required(VERSION 3.20)
project(image_captioning)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable position-independent code (useful for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set macOS-specific flags
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++20")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
endif()

# Add Homebrew prefix to CMake search path
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

# Find OpenCV (Homebrew installs OpenCV in /opt/homebrew/Cellar/opencv)
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found. Please ensure it is installed (e.g., via Homebrew: 'brew install opencv') and available in your environment (e.g., set OpenCV_DIR).")
endif()

# Find ONNX Runtime (Homebrew installs ONNX Runtime in /opt/homebrew/Cellar/onnxruntime)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    HINTS
        ENV ONNXRUNTIME_ROOT
        /opt/homebrew/lib
        /usr/local/lib
    NO_DEFAULT_PATH
)
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_c_api.h
    HINTS
        ENV ONNXRUNTIME_ROOT
        /opt/homebrew/include
        /usr/local/include
    PATH_SUFFIXES onnxruntime
    NO_DEFAULT_PATH
)

if(ONNXRUNTIME_LIBRARY AND ONNXRUNTIME_INCLUDE_DIR)
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )
    set(ONNXRUNTIME_FOUND TRUE)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIBRARY}")
else()
    message(FATAL_ERROR "ONNX Runtime not found. Please ensure it is installed (e.g., via Homebrew: 'brew install onnxruntime') and set ONNXRUNTIME_ROOT environment variable if installed in a custom location.")
endif()

# Find nlohmann/json (prefer system-wide installation, fall back to external/json)
find_package(nlohmann_json REQUIRED)
if(NOT nlohmann_json_FOUND)
    # If not found system-wide, check if external/json exists
    if(EXISTS "${CMAKE_SOURCE_DIR}/external/json/CMakeLists.txt")
        message(STATUS "nlohmann/json not found system-wide, using external/json")
        set(JSON_BuildTests OFF CACHE INTERNAL "")
        add_subdirectory(external/json)
    else()
        message(FATAL_ERROR "nlohmann/json not found. Please install it system-wide (e.g., via Homebrew: 'brew install nlohmann-json') or add it as a submodule in external/json.")
    endif()
else()
    message(STATUS "nlohmann/json found system-wide: ${nlohmann_json_VERSION}")
endif()

# Find spdlog (prefer system-wide installation, fall back to external/spdlog)
find_package(spdlog REQUIRED)
if(NOT spdlog_FOUND)
    # If not found system-wide, check if external/spdlog exists
    if(EXISTS "${CMAKE_SOURCE_DIR}/external/spdlog/CMakeLists.txt")
        message(STATUS "spdlog not found system-wide, using external/spdlog")
        set(SPDLOG_BUILD_SHARED ON CACHE INTERNAL "")
        add_subdirectory(external/spdlog)
    else()
        message(FATAL_ERROR "spdlog not found. Please install it system-wide (e.g., via Homebrew: 'brew install spdlog') or add it as a submodule in external/spdlog.")
    endif()
else()
    message(STATUS "spdlog found system-wide: ${spdlog_VERSION}")
endif()

# Find fmt (Homebrew installs fmt in /opt/homebrew/Cellar/fmt)
find_package(fmt REQUIRED)
if(fmt_FOUND)
    message(STATUS "fmt found: ${fmt_VERSION}")
else()
    # Fallback: Manually find fmt library if target-based approach fails
    find_library(FMT_LIBRARY NAMES fmt
        HINTS
            /opt/homebrew/lib
        NO_DEFAULT_PATH
    )
    if(FMT_LIBRARY)
        message(STATUS "fmt library found manually: ${FMT_LIBRARY}")
    else()
        message(FATAL_ERROR "fmt library not found. Please install it (e.g., via Homebrew: 'brew install fmt').")
    endif()
endif()

# Include directories for project headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files (explicitly list to avoid issues with file(GLOB ...))
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/logger.cpp
    ${CMAKE_SOURCE_DIR}/src/image_preprocessor.cpp
    ${CMAKE_SOURCE_DIR}/src/config.cpp
    ${CMAKE_SOURCE_DIR}/src/caption_generator.cpp
    ${CMAKE_SOURCE_DIR}/src/model_inference.cpp
    ${CMAKE_SOURCE_DIR}/src/vocabulary.cpp
)

# Add executable
add_executable(image_captioning ${SOURCES})

# Link libraries (using modern CMake target-based approach)
target_include_directories(image_captioning PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    $<$<BOOL:${ONNXRUNTIME_FOUND}>:${ONNXRUNTIME_INCLUDE_DIR}>
)

target_link_libraries(image_captioning
    PRIVATE
        ${OpenCV_LIBS}
        $<$<BOOL:${ONNXRUNTIME_FOUND}>:onnxruntime>
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        fmt::fmt  # Use the CMake target name
)

# Optional: Print summary of configuration
message(STATUS "Build configuration summary:")
message(STATUS "  - OpenCV: ${OpenCV_VERSION}")
message(STATUS "  - ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "  - ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "  - nlohmann/json: ${nlohmann_json_VERSION}")
message(STATUS "  - spdlog: ${spdlog_VERSION}")
message(STATUS "  - fmt: ${fmt_VERSION}")
message(STATUS "  - C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  - macOS-specific settings enabled: YES")