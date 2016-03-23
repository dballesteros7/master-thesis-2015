cmake_minimum_required(VERSION 3.3)
project(cppsrc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories( /usr/include/eigen3 )

set(SOURCE_FILES utils.h utils.cpp train_general_fast.cpp)
add_executable(cppsrc ${SOURCE_FILES})