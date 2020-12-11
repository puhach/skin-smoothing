#ifndef FACIALSKINSMOOTHER_H
#define FACIALSKINSMOOTHER_H

#include "skinsmoother.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

class FacialSkinSmoother : public SkinSmoother
{
public:
	FacialSkinSmoother(float confidenceThreshold);
	// TODO: implement copy/move constructors

	virtual cv::Mat apply(const cv::Mat& image) override;

private:
	//static const cv::String weightFile, configFile;

	cv::dnn::Net net;
	float confidenceThreshold;
};	// FacialSkinSmoother


#endif	// FACIALSKINSMOOTHER_H