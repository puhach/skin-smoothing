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

	// Compute the 2D histogram of hue-saturation pairs
	cv::Mat hist;
	int channels[] = { 0, 1 };
	int histSize[] = { 360, 256 }; 
	//int histSize[] = { 180, 256 };
	const float hueRange[] = { 0, 360 }, satRange[] = {0, 1}, *ranges[] = { hueRange, satRange };
	//const float hueRange[] = { 0, 180 }, satRange[] = { 0, 256 }, * ranges[] = { hueRange, satRange };
	cv::calcHist(&imageHSVF, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
	//cv::calcHist(&imageHSV, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
		
	// Reduce the 2D hue-saturation histogram to find the dominant hue.
	// It looks like skin color tends to have rather high variance in the saturation channel, therefore if we consider hue-saturation pairs, 
	// we are likely to pick a pair which belongs to some constant color area of the background or clothes. To overcome this difficulty,
	// we are finding the dominant saturation among pixels having the dominant hue.
	cv::Mat histHue;
	cv::reduce(hist, histHue, 1, cv::REDUCE_SUM, -1);
	
	// Find the dominant hue from the hue histogram
	cv::Point loc;
	double maxVal;		// TODO: remove it
	cv::minMaxLoc(histHue, nullptr, &maxVal, nullptr, &loc);
	//int dominantHue = loc.y;
	float dominantHue = hueRange[1] * loc.y / histSize[0];

	// Find the dominant saturation among pixels having the dominant hue
	cv::minMaxLoc(hist.row(loc.y), nullptr, &maxVal, nullptr, &loc);
	//int dominantSat = loc.x;
	float dominantSat = satRange[1] * loc.x / histSize[1];

	// Find how much on average the face color differs from the dominant color (we are interested in hue and saturation only)
	cv::Mat devMat;
	cv::absdiff(imageHSVF, cv::Scalar(dominantHue, dominantSat), devMat);
	cv::Scalar meanDev = cv::mean(devMat);

	cv::Scalar lowerHSV = { dominantHue-meanDev[0], dominantSat-meanDev[1], 0 }
		, upperHSV = { dominantHue+meanDev[0], dominantSat + meanDev[1], 255 };

	cv::Mat mask, orMask;
	cv::inRange(imageHSVF, lowerHSV, upperHSV, mask);

	// Account for hue values wrapping around 360 degrees
	if (lowerHSV[0] < 0 && upperHSV[0] < hueRange[1])
	{
		lowerHSV[0] = lowerHSV[0] + hueRange[1];
		upperHSV[0] = hueRange[1];
		cv::inRange(imageHSVF, lowerHSV, upperHSV, orMask);
		cv::bitwise_or(mask, orMask, mask);
	}
	else if (lowerHSV[0] > 0 && upperHSV[0] > hueRange[1])
	{
		lowerHSV[0] = 0;
		upperHSV[0] = upperHSV[0] - hueRange[1];
		cv::inRange(imageHSVF, lowerHSV, upperHSV, orMask);
		cv::bitwise_or(mask, orMask, mask);
	}

	// Denoise the mask
	///cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
	//cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

	cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0, 0);

	cv::imshow("mask", mask);
	cv::waitKey(0);

	/*cv::Mat tmp;
	cv::cvtColor(imageHSVF, tmp, cv::COLOR_HSV2BGR);
	cv::imshow("test", tmp);
	cv::waitKey();*/


}

cv::Mat SkinSmoother::apply(const cv::Mat& image)	// virtual
{
	cv::Mat imageCopy = image.clone();
	SkinSmoother::applyInPlace(imageCopy);
	return imageCopy;
}

