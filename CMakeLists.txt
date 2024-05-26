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

# Vulkan
find_package(Vulkan REQUIRED COMPONENTS glslc)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)
target_include_directories(${PROJECT_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)

#==============================================================================
# COMPILE SHADERS
#

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data/shader)
set(SHADER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data/shader/include)
set(SHADER_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/data/shader)

set(IMAGE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data/image)
set(IMAGE_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/data/image)

set(VOLUME_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data/volume)
set(VOLUME_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/data/volume)

file(GLOB_RECURSE SHADERS
  ${SHADER_SOURCE_DIR}/*.vert
  ${SHADER_SOURCE_DIR}/*.frag
  ${SHADER_SOURCE_DIR}/*.comp
  ${SHADER_SOURCE_DIR}/*.geom
  ${SHADER_SOURCE_DIR}/*.tesc
  ${SHADER_SOURCE_DIR}/*.tese
  ${SHADER_SOURCE_DIR}/*.mesh
  ${SHADER_SOURCE_DIR}/*.task
  ${SHADER_SOURCE_DIR}/*.rgen
  ${SHADER_SOURCE_DIR}/*.rchit
  ${SHADER_SOURCE_DIR}/*.rmiss)

add_custom_command(
  COMMAND
    ${CMAKE_COMMAND} -E make_directory ${SHADER_TARGET_DIR}
  OUTPUT ${SHADER_TARGET_DIR}
  COMMENT "Creating ${SHADER_TARGET_DIR}"
)

# COMPILE SHADERS TO SPIR-V
foreach(source IN LISTS SHADERS)
  get_filename_component(FILENAME ${source} NAME)
  get_filename_component(FILE_PATH ${source} DIRECTORY)

  # Exclude restir shaders because they are not implemented
  string (FIND ${FILE_PATH} restir EXCLUDE_DIR_FOUND)

  if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
    message(STATUS "Skipping shader ${FILE_PATH}/${FILENAME} compilation")
    continue()
  endif()

  string(REPLACE ${SHADER_SOURCE_DIR} ${SHADER_TARGET_DIR} FILE_PATH "${FILE_PATH}")

  add_custom_command(
  COMMAND
    ${CMAKE_COMMAND} -E make_directory ${FILE_PATH}
  OUTPUT ${FILE_PATH}
  COMMENT "Creating ${FILE_PATH}"
  )

  add_custom_command(
    COMMAND
      ${glslc_executable}
      -I ${SHADER_INCLUDE_DIR}
      -o ${FILE_PATH}/${FILENAME}.spv
      ${source}
    OUTPUT ${FILE_PATH}/${FILENAME}.spv
    DEPENDS ${source} ${FILE_PATH}
    COMMENT "Compiling ${FILENAME}"
  )
  list(APPEND SPV_SHADERS ${FILE_PATH}/${FILENAME}.spv)
  message(STATUS "Shader include path: ${SHADER_INCLUDE_DIR}")
  message(STATUS ${FILE_PATH}/${FILENAME}.spv)
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_SHADERS})


# COPY DATA TO BUILD FOLDER
file(COPY ${IMAGE_SOURCE_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)
file(COPY ${VOLUME_SOURCE_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)


# COPY VCPKG BUILT DLLs TO BUILD FOLDER
file(GLOB REQUIRED_DLLS 
    "${CMAKE_SOURCE_DIR}/openvdb-install/bin/*.dll"
)

file(COPY ${REQUIRED_DLLS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


# FIND PACKAGES. IF SOME PACKAGES ARE NOT FOUND, CHECK `INSTALLATION` SECTION OF README
## CUDA
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

## GLFW
find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

## GLM
find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

## TINYEXR
find_package(tinyexr CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE unofficial::tinyexr::tinyexr)

## ASSIMP
find_package(assimp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE assimp::assimp)

## ZLIB
find_package(ZLIB REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ZLIB::ZLIB)

## TBB
find_package(TBB CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)

## OPENVDB
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/openvdb-install/include")
target_link_libraries(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/openvdb-install/lib/openvdb.lib")

## TCNN
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory("tiny-cuda-nn")
target_include_directories(${PROJECT_NAME} PRIVATE ${TCNN_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} cuda cublas tiny-cuda-nn)
