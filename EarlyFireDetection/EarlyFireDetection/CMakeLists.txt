cmake_minimum_required(VERSION 3.12)

project(EarlyFireDetection)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#find_package(llvm)

if (win32)
    set(OpenCV_STATIC OFF)
    set(OpenCV SHARED ON)
    set(OpenCV_DIR "C:/Users/kspark/opencv/build/install")
    set(LIBVLC_INCLUDE_DIR "C:/vlc-winsdk/sdk/include")
	set(LIBVLC_LIBRARY_DIR "C:/vlc-winsdk/sdk/lib")  
	set(LIBVLC_LIBRARY "C:/vlc-winsdk/sdk/lib/libvlc.lib")
	set(LIBVLCCORE_LIBRARY "C:/vlc-winsdk/sdk/lib/libvlccore.lib")
elseif(APPLE)
    set(LIBVLC_INCLUDE_DIR "/Applications/VLC.app/Contents/MacOS/include/vlc")
	set(LIBVLC_LIBRARY_DIR "/Applications/VLC.app/Contents/MacOS/lib")  
	set(LIBVLC_LIBRARY "/Applications/VLC.app/Contents/MacOS/lib/libvlc.dylib")
	set(LIBVLCCORE_LIBRARY "/Applications/VLC.app/Contents/MacOS/lib/libvlccore.dylib")
endif()
find_package(OpenCV COMPONENTS opencv_core opencv_imgproc opencv_highgui opencv_objdetect opencv_tracking REQUIRED)
include_directories(. ${LIBVLC_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
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
		src/vlccap.h
		src/vlccap.cpp
        src/opticalFlowTool.cpp
        src/opticalFlowTool.h)


target_link_libraries(EarlyFireDetection opencv_core opencv_imgproc opencv_highgui opencv_objdetect opencv_tracking ${LIBVLCCORE_LIBRARY} ${LIBVLC_LIBRARY})

include(boost_test.cmake)
include(clang-tidy.cmake)
