#ifndef SKINSMOOTHER_H
#define SKINSMOOTHER_H

#include "imagefilter.h"

class SkinSmoother : public ImageFilter
{
public:

	SkinSmoother(int blurRadius, double sigmaColor, double sigmaSpace)
		: blurRadius(blurRadius)
		, sigmaColor(sigmaColor)
		, sigmaSpace(sigmaSpace)
	{}

	virtual void applyInPlace(cv::Mat& image) override;

	virtual cv::Mat apply(const cv::Mat& image) override;

private:

	cv::Mat createSkinMask(const cv::Mat &imHSV, double lowerHue, double upperHue, double lowerSat, double upperSat);

	int blurRadius;
	double sigmaColor, sigmaSpace;
};	// SkinSmoother

#endif	// SKINSMOOTHER_H