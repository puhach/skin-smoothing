#ifndef FACIALSKINSMOOTHER_H
#define FACIALSKINSMOOTHER_H

#include "skinsmoother.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

class FacialSkinSmoother : public SkinSmoother
{
public:
	FacialSkinSmoother(float faceConfThreshold, SkinDetectionHeuristic heuristic, int blurRadius=3, double sigmaColor=30.0, double sigmaSpace=30.0);

	// For a constexpr function or constexpr constructor that is neither defaulted nor a template, if no argument values exist such that 
	// an invocation of the function or constructor could be an evaluated subexpression of a core constant expression, or, for a constructor, 
	// a constant initializer for some object, the program is ill - formed; no diagnostic required
	// https://stackoverflow.com/questions/40322579/constexpr-member-function-of-non-constexpr-constructible-class	
	float getFaceConfidenceThreshold() const noexcept { return this->confidenceThreshold; }
	void setFaceConfidenceThreshold(float faceConfThreshold) noexcept { this->confidenceThreshold = faceConfThreshold; }

	virtual void applyInPlace(cv::Mat &image) override;

	virtual cv::Mat apply(const cv::Mat& image) override;

private:
	cv::dnn::Net net;
	float confidenceThreshold;
};	// FacialSkinSmoother


#endif	// FACIALSKINSMOOTHER_H