#ifndef FACIALSKINSMOOTHER_H
#define FACIALSKINSMOOTHER_H

#include "skinsmoother.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

class FacialSkinSmoother : public SkinSmoother
{
public:
	FacialSkinSmoother(float faceConfThreshold, SkinDetectionHeuristic heuristic, int blurRadius=3, double sigmaColor=30.0, double sigmaSpace=30.0);
	//FacialSkinSmoother(const FacialSkinSmoother& other) = default;
	//FacialSkinSmoother(FacialSkinSmoother&& other) = default;

	// https://stackoverflow.com/questions/54575426/what-is-the-purpose-of-marking-the-set-function-setter-as-constexpr/54575507
	constexpr float getFaceConfidenceThreshold() const noexcept { return this->confidenceThreshold; }
	constexpr void setFaceConfidenceThreshold(float faceConfThreshold) noexcept { this->confidenceThreshold = faceConfThreshold; }

	virtual void applyInPlace(cv::Mat &image) override;

	virtual cv::Mat apply(const cv::Mat& image) override;

private:
	cv::dnn::Net net;
	float confidenceThreshold;
};	// FacialSkinSmoother


#endif	// FACIALSKINSMOOTHER_H