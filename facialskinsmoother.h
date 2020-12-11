#ifndef FACIALSKINSMOOTHER_H
#define FACIALSKINSMOOTHER_H

#include "imagefilter.h"

class FacialSkinSmoother : public ImageFilter
{
public:
	FacialSkinSmoother() = default;
	// TODO: implement copy/move constructors

	virtual cv::Mat apply(const cv::Mat& image) override;
};	// FacialSkinSmoother


#endif	// FACIALSKINSMOOTHER_H