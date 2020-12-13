#ifndef SKINSMOOTHER_H
#define SKINSMOOTHER_H

#include "imagefilter.h"

enum class SkinDetectionHeuristic
{
	MeanColor,
	DominantColor,
	SelectiveSampling
};

class SkinSmoother : public ImageFilter
{
public:

	SkinSmoother(SkinDetectionHeuristic heuristic, int blurRadius, double sigmaColor, double sigmaSpace)
		: heuristic(heuristic)
		, blurRadius(blurRadius)
		, sigmaColor(sigmaColor)
		, sigmaSpace(sigmaSpace)
	{}

	virtual void applyInPlace(cv::Mat& image) override;

	virtual cv::Mat apply(const cv::Mat& image) override;

private:

	cv::Mat createSkinMask(const cv::Mat &imHSVF, double lowerHue, double upperHue, double lowerSat, double upperSat);

	SkinDetectionHeuristic heuristic;
	int blurRadius;
	double sigmaColor, sigmaSpace;
};	// SkinSmoother

#endif	// SKINSMOOTHER_H