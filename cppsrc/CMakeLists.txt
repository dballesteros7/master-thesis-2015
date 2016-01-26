cmake_minimum_required(VERSION 3.3)
project(cppsrc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES train_general.cpp utils.h utils.cpp)
add_executable(cppsrc ${SOURCE_FILES})