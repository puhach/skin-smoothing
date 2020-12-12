#include "skinsmoother.h"

#include <opencv2/imgproc.hpp>
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
	int histSize[] = { 360, 256 }; //{ 180, 256 };
	const float hueRange[] = { 0, 360 }, satRange[] = {0, 1}, *ranges[] = { hueRange, satRange };
	cv::calcHist(&imageHSVF, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
	

	// TODO: implement skin smoothing

	cv::blur(image, image, cv::Size(3, 3));
}

cv::Mat SkinSmoother::apply(const cv::Mat& image)	// virtual
{
	cv::Mat imageCopy = image.clone();
	SkinSmoother::applyInPlace(imageCopy);
	return imageCopy;
}

