# Experimental 3DGS Rasterization Methods on gsplat

**⚠️Warning: This project is an pre-alpha version and under active development. Do not use it in production.**

This project is a personal endeavor to port various 3D Gaussian splatting rasterization methods from the latest thesis or other rasteration backends into project `gsplat` by nerfstudio. The goal is to enhance the capabilities of `gsplat` by integrating advanced techniques for rendering 3D graphics. 


## Table of Contents

- [Features](#features)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Release Notes](#release-notes)
- [Contributing](#contributing)
- [License](#license)

## Features
It is aims to be used as:
- Test grounds for latest 3D Gaussian splatting methods. 
- Accelerator of the existing projects that uses `gsplat` as the backend.
- A place to share my thoughts and findings on my research in the field.

## Background

> [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) is the original rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields".

> [gsplat](http://www.gsplat.studio/) is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper 3D Gaussian Splatting for Real-Time Rendering of Radiance Fields, but we’ve made gsplat even faster, more memory efficient, and with a growing list of new features!

## Installation

> ⚠️Warning: This project is currently only supports on Linux.

To install the project, please ensure you have installed the [Nvida CUDA toolkit and driver](https://developer.nvidia.com/cuda-downloads). Then follow these steps:

1. Clone the repository:
   ```bash
    git clone https://github.com/frozen2077/gsplat-joy.git
    cd gsplat-joy
   ```
2. Create a new conda environment:  
   ```bash
    conda create -n gsplat-joy python=3.10 && conda activate gsplat-joy
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 pybind11 -c pytorch -c nvidia
   ```

3. Install the required dependencies:
   ```bash
     pip install -r requirements.txt
   ```

3. Build the project:
   ```bash
    # For running
    python setup.py build

    # For debugging
    mkdir gsplat/cuda/csrc/build
    cmake gsplat/cuda/csrc
    cmake --build gsplat/cuda/csrc/build -j8
   ```

4. Copy the compiled library to your gsplat directory:
   ```bash
    # For running
    cp build/lib.*/gsplat/csrc.so gsplat/
    # For debugging
    cp gsplat/cuda/csrc/build/csrc*.so gsplat/
    cp gsplat/cuda/csrc/build/lib*.so  gsplat/
   ```

## Usage

To use the project, reference the offical [gsplat](https://docs.gsplat.studio/main/examples/colmap.html) documentation.

You can run the `image_fitting.py` example to test if the installation is successful. The rendered gif could be found in the `results` folder.

Your result should look like this:

![gsplat](pic/official.gif)

## Release Notes

### Pre-Alpha v0.0.1 
This version involves the following technical implementations:

- **Per Gaussian based rasterization**: Porting from the latest released version from [Mallick and Goel et al](https://github.com/humansensinglab/taming-3dgs) which based on the original [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization).
- **Tighter splats culling**: Porting from the [Lukas Radl et al](https://github.com/r4dl/StopThePop), the original paper is [Stop the Pop! Real-Time Rendering of 3D Gaussians](https://arxiv.org/abs/2308.00719).
- **Depth Regulated rasterization**: Techniques to use depth priors to improve rendering speed and quality.

### Performance Analysis

Currently, the performance of the rasterization is 12% faster than the offical `gsplat` backend but suffers from an average 9% quality loss in PSNR and 15% in SSIM. The problem seems to be related with the optimization process of the nvcc compiler. Disabling the `-O3` and `--use-fast-math` flags will improve to the same quality as the offical version but slows down the rasterization significantly. Further investigation is needed.

Also current version on Windows is bugged due to MSVC compiler issues with default settings. Below is the result of the same code comiled on Windows platform. Further investigation is needed.

![win-bugged](pic/win-bugged.gif)

## Guides for Python and CUDA mixed debugging 

Coming Soon

## Guides for CMakeLists on CUDA debugging 

```cmake
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
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.