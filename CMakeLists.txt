cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)

# Project
project(NRC-HPM-Renderer CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

file(GLOB_RECURSE PROJECT_INCLUDE "include/*.hpp")
file(GLOB_RECURSE PROJECT_SOURCE "src/*.cpp")
file(GLOB_RECURSE PROJECT_CUDA_SOURCE "src/*.cu")
file(GLOB_RECURSE IMGUI_SOURCE "imgui/*.cpp")
file(GLOB_RECURSE STB_SOURCE "stb/*.c")

add_executable(${PROJECT_NAME} ${PROJECT_INCLUDE} ${PROJECT_SOURCE} ${PROJECT_CUDA_SOURCE} ${IMGUI_SOURCE} ${STB_SOURCE})

target_include_directories(${PROJECT_NAME} PUBLIC "include" ${CUDA_INC_PATH} "imgui" "stb")

# Compile
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_compile_features(${PROJECT_NAME} PUBLIC cuda_std_17)

# Dependencies
include("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Vulkan
find_package(Vulkan REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)

# CUDA
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# GLFW
list(APPEND CMAKE_PREFIX_PATH "C:/Program Files/glfw")
find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

# GLM
find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

# TINYEXR
find_package(tinyexr CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE unofficial::tinyexr::tinyexr)

# ASSIMP
find_package(assimp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE assimp::assimp)

# ZLIB
find_package(ZLIB REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ZLIB::ZLIB)

# TBB
find_package(TBB CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)

# OPENVDB
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/openvdb-install/include")
target_link_libraries(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/openvdb-install/lib/openvdb.lib")

# BOOST
target_include_directories(${PROJECT_NAME} PRIVATE $ENV{BOOST_INCLUDEDIR})
target_link_directories(${PROJECT_NAME} PRIVATE $ENV{BOOST_LIBRARYDIR})

# TCNN
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory("tiny-cuda-nn")
target_include_directories(${PROJECT_NAME} PRIVATE ${TCNN_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} cuda cublas tiny-cuda-nn)
