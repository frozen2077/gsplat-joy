cmake_minimum_required(VERSION 3.20)
project(DiffRast LANGUAGES CUDA CXX)

# Ensure `-fPIC` is used
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")

# Get the Torch path from Python
execute_process(
    COMMAND python -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _python_prefix_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python path is: ${_python_prefix_path}")

if (_result EQUAL 0)
    set(CMAKE_PREFIX_PATH "${_python_prefix_path}" CACHE PATH "Path to Torch")
else()
    message(FATAL_ERROR "Failed to get CMAKE_PREFIX_PATH from Python")
endif()

# set(Python_FIND_STRATEGY "LOCATION")

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/third_party/glm)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

message(STATUS "PYTHON_INCLUDE_DIRS:  ${PYTHON_INCLUDE_DIRS}")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
message(STATUS "TORCH_LIBRARIES:      ${TORCH_LIBRARIES}")
message(STATUS "CMAKE_CUDA_FLAGS:     ${CMAKE_CUDA_FLAGS}")

file(GLOB SOURCES
    ${CMAKE_SOURCE_DIR}/*.cu
)
add_library(gsplat SHARED ${SOURCES})

message(STATUS "SOURCES: ${SOURCES}")

set_target_properties(gsplat PROPERTIES CUDA_ARCHITECTURES "86;89")
target_include_directories(gsplat PRIVATE "/home/joey/anaconda3/envs/gsplat/include/python3.10")
target_link_libraries(gsplat PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")

pybind11_add_module(csrc
    ext.cpp 
)

set_target_properties(csrc PROPERTIES CUDA_ARCHITECTURES "86;89")
target_link_libraries(csrc PRIVATE gsplat "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")

set(DESTINATION_FOLDER "/mnt/d/Desktop/gsplat-joy/gsplat")

add_custom_command(
    TARGET gsplat
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gsplat> ${DESTINATION_FOLDER}
    COMMENT "Copying gsplat to output folder"
)

add_custom_command(
    TARGET csrc
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:csrc> ${DESTINATION_FOLDER}
    COMMENT "Copying csrc to output folder"
)