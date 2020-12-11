#include "facialskinsmoother.h"

#include <opencv2/dnn.hpp>

#include <iostream>		// !TEST!




FacialSkinSmoother::FacialSkinSmoother()
	: net(cv::dnn::readNetFromTensorflow("./models/opencv_face_detector_uint8.pb", "./models/opencv_face_detector.pbtxt"))
{

}

cv::Mat FacialSkinSmoother::apply(const cv::Mat& image)
{
	CV_Assert(image.channels() == 3);

	// Detect faces
	
	// See model details here:
	// https://github.com/opencv/opencv/tree/master/samples/dnn
	cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123), true, false);
	
	net.setInput(blob);	
	cv::Mat detection = net.forward(net.getUnconnectedOutLayersNames().front());
	CV_Assert(detection.dims == 4);

	detection = detection.reshape(1, detection.size[2]);
	CV_Assert(detection.dims == 2);

	std::cout << detection.size << std::endl;
	std::cout << detection << std::endl;

	cv::Mat out;
	return out;
}