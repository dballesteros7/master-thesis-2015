cmake_minimum_required(VERSION 3.3)
project(master_thesis_2015)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    src/models/train_features.cpp
    src/models/train_features.h)

add_executable(master_thesis_2015 ${SOURCE_FILES})