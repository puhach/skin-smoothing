#include "facialskinsmoother.h"

#include <opencv2/dnn.hpp>

#include <iostream>		// !TEST!
#include <opencv2/highgui.hpp>	// TEST!
#include <opencv2/imgproc.hpp> // TEST!


FacialSkinSmoother::FacialSkinSmoother(float faceConfThreshold, SkinDetectionHeuristic heuristic, int blurRadius, double sigmaColor, double sigmaSpace)
	: SkinSmoother(heuristic, blurRadius, sigmaColor, sigmaSpace)
	, net(cv::dnn::readNetFromTensorflow("./models/opencv_face_detector_uint8.pb", "./models/opencv_face_detector.pbtxt"))
	, confidenceThreshold(faceConfThreshold)
{

}

void FacialSkinSmoother::applyInPlace(cv::Mat& image)	// virtual
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

	//std::cout << detection.size << std::endl;
	//std::cout << detection << std::endl;

	for (int i = 0; i < detection.rows; ++i)
	{
		float conf = detection.at<float>(i, 2);

		if (conf > this->confidenceThreshold)
		{
			int x1 = static_cast<int>(detection.at<float>(i, 3) * image.cols);
			int y1 = static_cast<int>(detection.at<float>(i, 4) * image.rows);
			int x2 = static_cast<int>(detection.at<float>(i, 5) * image.cols);
			int y2 = static_cast<int>(detection.at<float>(i, 6) * image.rows);
			CV_Assert(x2 > x1 && y2 > y1);

			cv::Mat face = image(cv::Range(y1, y2), cv::Range(x1, x2));
			//cv::imshow("test", face);
			//cv::waitKey();
			SkinSmoother::applyInPlace(face);
			//cv::blur(face, face, cv::Size(3, 3));
		}
	}	// i

}	// apply

cv::Mat FacialSkinSmoother::apply(const cv::Mat& image)		// virtual
{
	cv::Mat out = image.clone();
	FacialSkinSmoother::applyInPlace(out);
	return out;
}