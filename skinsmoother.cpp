#include "skinsmoother.h"

#include <opencv2/imgproc.hpp>

#include <opencv2/highgui.hpp>	// TEST!
#include <iostream>	// TEST!

void SkinSmoother::applyInPlace(cv::Mat& image)	// virtual
{
	// If you use cvtColor with 8-bit images, the conversion will have some information lost. For many applications, this will not be noticeable 
	// but it is recommended to use 32-bit images in applications that need the full range of colors or that convert an image before an operation 
	// and then convert back.
	// https://docs.opencv.org/4.2.0/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
	cv::Mat imageHSVF;
	image.convertTo(imageHSVF, CV_32F, 1.0 / 255);
	cv::cvtColor(imageHSVF, imageHSVF, cv::COLOR_BGR2HSV);
	//std::cout << imageHSVF << std::endl;
	/*cv::Mat imageHSV;
	cv::cvtColor(image, imageHSV, cv::COLOR_BGR2HSV);*/

	cv::Mat hist;
	int channels[] = { 0, 1 };
	int histSize[] = { 360, 256 }; 
	//int histSize[] = { 180, 256 };
	const float hueRange[] = { 0, 360 }, satRange[] = {0, 1}, *ranges[] = { hueRange, satRange };
	//const float hueRange[] = { 0, 180 }, satRange[] = { 0, 256 }, * ranges[] = { hueRange, satRange };
	cv::calcHist(&imageHSVF, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
	//cv::calcHist(&imageHSV, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
	
	cv::Mat hueHist;
	cv::reduce(hist, hueHist, 1, cv::REDUCE_SUM, -1);
	
	cv::Point loc;
	double maxVal;
	cv::minMaxLoc(hueHist, nullptr, &maxVal, nullptr, &loc);
	//int dominantHue = loc.y;
	float dominantHue = hueRange[1] * loc.y / histSize[0];

	//std::cout << hist.row(dominantHue) << std::endl;
	cv::minMaxLoc(hist.row(dominantHue), nullptr, &maxVal, nullptr, &loc);
	//int dominantSat = loc.x;
	float dominantSat = satRange[1] * loc.x / histSize[1];

	//cv::Mat hist;
	//int channels[] = { 0 };
	//int histSize[] = { 180 };
	////const float hueRange[] = { 0, 180 }, satRange[] = { 0, 256 }, *ranges[] = { hueRange, satRange };
	//const float hueRange[] = { 0, 180 };
	//const float* ranges[] = { hueRange };
	//cv::calcHist(&imageHSV, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

	//// Find the dominant hue
	//double maxVal;
	////cv::minMaxLoc(hist, nullptr, &maxVal, nullptr, &maxLoc);
	////int dominantColor[2];
	//int dominantHue;
	//cv::minMaxIdx(hist, nullptr, &maxVal, nullptr, &dominantHue);

	//cv::Mat imageHSVChannels[3];
	//cv::split(imageHSV, imageHSVChannels);

	//cv::Mat mask = imageHSVChannels[0] == dominantHue;

	//channels[0] = 1;
	//histSize[0] = 256;
	//float satRange[] = { 0, 256 };
	//ranges[0] = satRange;
	//cv::calcHist(&imageHSV, 1, channels, mask, hist, 1, histSize, ranges, true, false);

	//int dominantSat;
	//cv::minMaxIdx(hist, nullptr, &maxVal, nullptr, &dominantSat);

	/*cv::Scalar mu = cv::mean(imageHSV);
	dominantColor[0] = int(mu[0]);
	dominantColor[1] = int(mu[1]);

	cv::Scalar lowerHSV = { dominantColor[0]-30.0, dominantColor[1] - 30.0, 0 }
			, upperHSV = { dominantColor[0]+30.0, dominantColor[1] + 30.0, 255 };*/

	// TODO: implement skin smoothing

	cv::blur(image, image, cv::Size(3, 3));
}

cv::Mat SkinSmoother::apply(const cv::Mat& image)	// virtual
{
	cv::Mat imageCopy = image.clone();
	SkinSmoother::applyInPlace(imageCopy);
	return imageCopy;
}

