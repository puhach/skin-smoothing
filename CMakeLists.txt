cmake_minimum_required(VERSION 2.8.12)

project(SkinSmoothing)
set(CMAKE_CXX_STANDARD 17)


######################## EDIT IF REQUIRED ####################
# Edit the line below to specify the path to OpenCV directory i.e. the path to the OpenCVConfig.cmake file. Check the examples given below.
set(OpenCV_DIR /opt/opencv/4.4.0/installation/lib/cmake/opencv4)
#set(OpenCV_DIR /home/hp/workfolder/OpenCV-Installation/installation/OpenCV-master/lib/cmake/opencv4)
#SET(OpenCV_DIR Enter-the-path-of-OpenCV-installation-on-your-system)

################### OpenCV_DIR Examples  #####################

### MACOS : /usr/local/Cellar/opencv/3.3.1_1/share/OpenCV/

### UBUNTU : /opt/opencv/4.4.0/installation/lib/cmake/opencv4

### WINDOWS : C:\Users\yourname\Documents\opencv-3.3.1\build\install

##############################################################



find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(skinsoft main.cpp imagefilter.h imagefilter.cpp skinsmoother.h skinsmoother.cpp facialskinsmoother.h facialskinsmoother.cpp)
target_link_libraries(skinsoft ${OpenCV_LIBS} )
  
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set_property(TARGET skinsoft PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

