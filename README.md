# Skin Smoothing

This application implements an image filter for smoothing the skin of a face.

## Set Up

It is assumed that OpenCV 4.x, C++17 compiler, and cmake 2.18.12 or newer are installed on the system.

### Download the Face Detection Model

Create the `models` directory in the root of the project folder. 

Download [opencv_face_detector.pbtxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector.pbtxt) and [opencv_face_detector_uint8.pb](https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb) files to the `models` directory.

Refer to [this](https://github.com/opencv/opencv/tree/master/samples/dnn) page for model details.

### Specify OpenCV_DIR in CMakeLists

Open CMakeLists.txt and set the correct OpenCV directory in the following line:

```
SET(OpenCV_DIR /home/hp/workfolder/OpenCV-Installation/installation/OpenCV-master/lib/cmake/opencv4)
```

Depending on the platform and the way OpenCV was installed, it may be needed to provide the path to cmake files explicitly. On my KUbuntu 20.04 after building OpenCV 4.4.0 from sources the working `OpenCV_DIR` looks like <OpenCV installation path>/lib/cmake/opencv4. On Windows 8.1 after installing a binary distribution of OpenCV 4.2.0 it is C:\OpenCV\build.


### Build the Project

In the root of the project folder create the `build` directory unless it already exists. Then from the terminal run the following:

```
cd build
cmake ..
```

This should generate the build files. When it's done, compile the code:

```
cmake --build . --config Release
```

