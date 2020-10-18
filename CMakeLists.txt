cmake_minimum_required(VERSION 3.16)
project(pattern_recognition_lab)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(pattern_recognition_lab 01-hu-moments/main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(pattern_recognition_lab ${OpenCV_LIBS})