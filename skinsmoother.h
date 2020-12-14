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

	constexpr SkinSmoother(SkinDetectionHeuristic heuristic, int blurRadius, double sigmaColor, double sigmaSpace) noexcept
		: heuristic(heuristic)
		, blurRadius(blurRadius)
		, sigmaColor(sigmaColor)
		, sigmaSpace(sigmaSpace)
	{}

	constexpr SkinDetectionHeuristic getSkinDetectionHeuristic() const noexcept { return this->heuristic; }
	constexpr void setSkinDetectionHeuristic(SkinDetectionHeuristic heuristic) noexcept { this->heuristic = heuristic; }

	constexpr int getBlurRadius() const noexcept { return this->blurRadius; }
	constexpr void setBlurRadius(int blurRadius) noexcept { this->blurRadius = blurRadius; }

	constexpr double getSigmaColor() const noexcept { return this->sigmaColor; }
	constexpr void setSigmaColor(double sigmaColor) noexcept { this->sigmaColor = sigmaColor; }

	constexpr double getSigmaSpace() const noexcept { return this->sigmaSpace; }
	constexpr void setSigmaSpace(double sigmaSpace) noexcept { this->sigmaSpace = sigmaSpace; }

	virtual void applyInPlace(cv::Mat& image) override;

	virtual cv::Mat apply(const cv::Mat& image) override;

private:

	cv::Mat detectSkinByMeanColor(const cv::Mat &faceHSVF);
	cv::Mat detectSkinByDominantColor(const cv::Mat &faceHSVF);
	cv::Mat detectSkinBySampling(const cv::Mat &faceHSVF);

	cv::Mat createSkinMask(const cv::Mat &imHSVF, double lowerHue, double upperHue, double lowerSat, double upperSat);

	SkinDetectionHeuristic heuristic;
	int blurRadius;
	double sigmaColor, sigmaSpace;
};	// SkinSmoother

#endif	// SKINSMOOTHER_H