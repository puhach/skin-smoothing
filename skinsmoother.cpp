#include "skinsmoother.h"

#include <opencv2/imgproc.hpp>


void SkinSmoother::applyInPlace(cv::Mat& image)	// virtual
{
	// TODO: implement skin smoothing

	cv::blur(image, image, cv::Size(3, 3));
}

cv::Mat SkinSmoother::apply(const cv::Mat& image)	// virtual
{
	cv::Mat imageCopy = image.clone();
	SkinSmoother::applyInPlace(imageCopy);
	return imageCopy;
}

