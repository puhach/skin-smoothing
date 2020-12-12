#ifndef SKINSMOOTHER_H
#define SKINSMOOTHER_H

#include "imagefilter.h"

class SkinSmoother : public ImageFilter
{
public:

	virtual void applyInPlace(cv::Mat& image) override;

	virtual cv::Mat apply(const cv::Mat& image) override;
};	// SkinSmoother

#endif	// SKINSMOOTHER_H