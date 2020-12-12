#ifndef IMAGEFILTER_H
#define IMAGEFILTER_H

#include <opencv2/core.hpp>

class ImageFilter
{
public:
	virtual ~ImageFilter() = default;

	virtual void applyInPlace(cv::Mat& image) = 0;

	virtual cv::Mat apply(const cv::Mat& image) = 0;
};	// ImageFilter

#endif // IMAGEFILTER_H