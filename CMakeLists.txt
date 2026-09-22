cmake_minimum_required(VERSION 3.20)
project(RT_One_Weekend VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h" "src/*.cu")

add_executable(RT_One_Weekend ${SOURCES})

set_property(TARGET RT_One_Weekend PROPERTY CUDA_ARCHITECTURES native)