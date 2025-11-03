cmake_minimum_required(VERSION 3.10)
project(ArmorRecognition)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(armor src/main.cpp)
target_link_libraries(armor ${OpenCV_LIBS})