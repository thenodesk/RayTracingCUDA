Ray Tracing using CUDA
=====================================================================
I'm using this project to learn more about how ray tracing works, but also using CUDA to both accelerate rendering and understand how to port C++ code to CUDA.

It is based on Peter Shirley's [Ray Tracing in One Weekend](https://raytracing.github.io/) book series and [NVIDIA's blog post](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/).

Each book is implemented in a separate Git branch, allowing each stage of the series to be explored independently.

Prerequisites
---------------------------------------------------------------------
- NVIDIA CUDA Toolkit

Building the Project
---------------------------------------------------------------------
Clone the repository and follow these steps from your terminal in the project's root directory to generate and build the solution using CMake:

1. Generate the project files
```shell
cmake -B build
```
2. Build the project directly from the command line
```shell
cmake --build build --config Release
```
The executable will be in `build/Release` folder.

_This project was tested with Visual Studio on Windows._
