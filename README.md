# Skin Smoothing

This application implements an image filter for smoothing the skin of a face. There are three heuristics which can be used for skin detection:

*Mean color* detects the skin by calculating the mean color of the face region.

*Dominant color* detects the skin by finding the dominant hue and saturation values in the face region

*Selective sampling* detects the skin by sampling colors from the forehead, chin, and cheeks regions.
It may be more accurate than other heuristics, but works well only for appropriately detected frontal faces with typical proportions.

![skin smoothing](./assets/hillary.jpg)

## Set Up

It is assumed that OpenCV 4.x, C++17 compiler, and cmake 2.18.12 or newer are installed on the system.

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

### Download the Face Detection Model

Create the `models` directory next to the executable file. 

Download [opencv_face_detector.pbtxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector.pbtxt) and [opencv_face_detector_uint8.pb](https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb) files to the `models` directory.

Refer to [this](https://github.com/opencv/opencv/tree/master/samples/dnn) page for model details.


## Usage

The program has to be run from the command line. It takes in the image file path and several optional parameters: 

```
skinsoft -image=<image file> [-confidence=<float 0..1>] [-radius=<integer>] [-sigmac=<float>] [-sigmas=<float>] [-heuristic=<choice>] [-help]
```

Parameter    | Meaning 
------------ | --------------------------------------
help, ? | Prints the help message
confidence | Face detection confidence threshold (defaults to 0.7)
radius | Blur radius (3 by default)
sigmac | Blur sigma in the color space (default is 30)
sigmas | Blur sigma in the coordinate space (defaults is 30)
heuristic | Skin detection heuristic: 1 - Mean color, 2 - Dominant color, 3 - Selective sampling 


Sample usage (linux):
```
./skinsoft -image=../images/old_couple_1.jpg -confidence=0.5 -radius=4
```

To work correctly the program requires the face detection model files. Place the `models` folder, containing the .pbtxt and .pb files, in the directory from where the program is launched.

