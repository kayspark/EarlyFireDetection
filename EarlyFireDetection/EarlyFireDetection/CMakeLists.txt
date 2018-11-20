cmake_minimum_required(VERSION 3.12)

project(EarlyFireDetection)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#find_package(llvm)

if (win32)
    set(OpenCV_STATIC OFF)
    set(OpenCV SHARED ON)
    set(OpenCV_DIR "C:/Users/kspark/opencv/build/install")
endif ()

find_package(OpenCV REQUIRED)
include_directories(. ${OpenCV_INCLUDE_DIRS})

add_executable(EarlyFireDetection
        src/colorModel.cpp
        src/colorModel.h
        src/ds.cpp
        src/ds.h
        src/fireBehaviorAnalysis.cpp
        src/fireBehaviorAnalysis.h
        src/main.cpp
        src/motionDetection.cpp
        src/motionDetection.h
        src/opticalFlowTool.cpp
        src/opticalFlowTool.h)


target_link_libraries(EarlyFireDetection ${OpenCV_LIBS})

#include(boost_test.cmake)
#include(clang-tidy.cmake)
