#include "skinsmoother.h"

#include <opencv2/imgproc.hpp>


void SkinSmoother::applyInPlace(cv::Mat& image)	// virtual
{
	// If you use cvtColor with 8-bit images, the conversion will have some information lost. For many applications, this will not be noticeable 
	// but it is recommended to use 32-bit images in applications that need the full range of colors or that convert an image before an operation 
	// and then convert back.
	// https://docs.opencv.org/4.2.0/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
	cv::Mat imageF;
	image.convertTo(imageF, CV_32F, 1.0 / 255);
	cv::cvtColor(imageF, imageF, cv::COLOR_BGR2HSV);
	
	cv::Mat mask;

	switch (this->heuristic)
	{
	case SkinDetectionHeuristic::MeanColor:
		mask = detectSkinByMeanColor(imageF);
		break;

	case SkinDetectionHeuristic::DominantColor:
		mask = detectSkinByDominantColor(imageF);
		break;

	case SkinDetectionHeuristic::SelectiveSampling:
		mask = detectSkinBySampling(imageF);
		cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));	// denoise the mask
		//cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));	// denoise the mask
		break;

	default:
		throw std::runtime_error("The heuristic is not implemented.");
	}	// heuristic

	// Scale the mask values to 0..1 range
	mask.convertTo(mask, CV_32F, 1 / 255.0);

	// Smooth the mask			
	cv::GaussianBlur(mask, mask, cv::Size(7, 7), 0, 0);


	// Blur the image using the edge-preserving filter
	cv::Mat imageBlurredHSVF;
	//cv::GaussianBlur(imageF, imageBlurredHSVF, cv::Size(5,5), 0, 0);
	cv::bilateralFilter(imageF, imageBlurredHSVF, 2*this->blurRadius, this->sigmaColor, this->sigmaSpace);

	
	// Combine the blurred and the original part of the image, making a seamless transition between these regions

	// 1) The blurred part
	cv::Mat mask3F;		// to perform arithmetic operations matrices need to have the same number of channels
	cv::merge(std::vector<cv::Mat>{ mask, mask, mask }, mask3F);
	cv::multiply(imageBlurredHSVF, mask3F, imageBlurredHSVF);

	// 2) The original part
	mask3F.convertTo(mask3F, -1, -1, 1);	// invert the mask
	cv::multiply(imageF, mask3F, imageF);

	// 3) Combined image
	imageBlurredHSVF += imageF;


	// Replace the ROI in the original image with the blurred image
	cv::cvtColor(imageBlurredHSVF, imageF, cv::COLOR_HSV2BGR);
	imageF.convertTo(image, CV_8UC3, 255.0);
}

cv::Mat SkinSmoother::apply(const cv::Mat& image)	// virtual
{
	cv::Mat imageCopy = image.clone();
	SkinSmoother::applyInPlace(imageCopy);
	return imageCopy;
}

// The mean color heuristic detects the skin by calculating the mean color of the face region
cv::Mat SkinSmoother::detectSkinByMeanColor(const cv::Mat& faceHSVF)
{
	CV_Assert(faceHSVF.depth() == CV_32F);

	cv::Scalar meanColor, stdDev;
	cv::meanStdDev(faceHSVF, meanColor, stdDev);

	return createSkinMask(faceHSVF, meanColor[0] - stdDev[0], meanColor[0] + stdDev[0], meanColor[1] - stdDev[1], meanColor[1] + stdDev[1]);
}

// The dominant color heuristic detects the skin by finding the dominant hue and saturation values in the face region
cv::Mat SkinSmoother::detectSkinByDominantColor(const cv::Mat& faceHSVF)
{
	CV_Assert(faceHSVF.depth() == CV_32F);

	// Compute the 2D histogram of hue-saturation pairs
	cv::Mat histHueSat;
	int channels[] = { 0, 1 };
	int histSize[] = { 360, 256 };
	const float hueRange[] = { 0, 360 }, satRange[] = { 0, 1 }, * ranges[] = { hueRange, satRange };
	cv::calcHist(&faceHSVF, 1, channels, cv::Mat(), histHueSat, 2, histSize, ranges, true, false);

	// Reduce the 2D hue-saturation histogram to find the dominant hue.
	// It looks like skin color tends to have rather high variance in the saturation channel, therefore if we consider hue-saturation pairs, 
	// we are likely to pick a pair which belongs to some constant color area of the background or clothes. To overcome this difficulty,
	// we are finding the dominant saturation among pixels having the dominant hue.
	cv::Mat histHue;
	cv::reduce(histHueSat, histHue, 1, cv::REDUCE_SUM, -1);

	// Find the dominant hue from the hue histogram
	cv::Point loc;
	cv::minMaxLoc(histHue, nullptr, nullptr, nullptr, &loc);
	float dominantHue = hueRange[1] * loc.y / histSize[0];

	// Find the dominant saturation among pixels having the dominant hue
	cv::minMaxLoc(histHueSat.row(loc.y), nullptr, nullptr, nullptr, &loc);
	float dominantSat = satRange[1] * loc.x / histSize[1];


	// Find how much on average the face color differs from the dominant color (we are interested in hue and saturation only)
	cv::Mat devMat;
	cv::absdiff(faceHSVF, cv::Scalar(dominantHue, dominantSat), devMat);
	cv::Scalar meanDev = cv::mean(devMat);

	// Create a skin region mask
	return createSkinMask(faceHSVF, dominantHue - meanDev[0], dominantHue + meanDev[0], dominantSat - meanDev[1], dominantSat + meanDev[1]);
}

// The selective sampling heuristic detects the skin by sampling colors from the forehead, chin, and cheeks regions.
// It may be more accurate than other heuristics, but works well only for appropriately detected frontal faces with typical proportions.
cv::Mat SkinSmoother::detectSkinBySampling(const cv::Mat& faceHSVF)
{
	CV_Assert(faceHSVF.depth() == CV_32F);
	
	// Face proportions have been found on the web and adjusted by experimentation
	cv::Mat forehead = faceHSVF({ static_cast<int>(0.1*faceHSVF.rows), static_cast<int>(0.3*faceHSVF.rows) }
		, { static_cast<int>(0.2*faceHSVF.cols), static_cast<int>(0.8*faceHSVF.cols) });

	cv::Mat leftCheek = faceHSVF({ static_cast<int>(0.5 * faceHSVF.rows), static_cast<int>(0.7 * faceHSVF.rows) } 
		, { static_cast<int>(0.1 * faceHSVF.cols), static_cast<int>(0.3 * faceHSVF.cols) });

	cv::Mat rightCheek = faceHSVF({ static_cast<int>(0.5*faceHSVF.rows), static_cast<int>(0.7*faceHSVF.rows) }
		, { static_cast<int>(0.7*faceHSVF.cols), static_cast<int>(0.9*faceHSVF.cols) });

	cv::Mat chin = faceHSVF({ static_cast<int>(0.85*faceHSVF.rows), static_cast<int>(1.0*faceHSVF.rows) }
		, { static_cast<int>(0.4*faceHSVF.cols), static_cast<int>(0.6*faceHSVF.cols) });


	std::vector<cv::Mat> samples{ forehead, leftCheek, rightCheek, chin };

	cv::Scalar meanHSV, devHSV;
	int n = 0;
	for (const cv::Mat& sample : samples)
	{
		if (!sample.empty())
		{
			cv::Scalar mu, sigma;
			cv::meanStdDev(sample, mu, sigma);

			meanHSV += mu;

			// Maximal deviation seems to work better than the average one, probably because these regions tend to have low variance in general
			//devHSV += sigma;
			devHSV[0] = cv::max(devHSV[0], sigma[0]);
			devHSV[1] = cv::max(devHSV[1], sigma[1]);

			++n;
		}
	}

	if (n > 0)
	{
		// Average the results from non-empty samples
		meanHSV /= n;
		//devHSV /= n;
	}
	else
	{
		// In case all sample regions were empty, fall back to finding the mean color over the entire face 
		cv::meanStdDev(faceHSVF, meanHSV, devHSV);
	}

	return createSkinMask(faceHSVF, meanHSV[0]-devHSV[0], meanHSV[0]+devHSV[0], meanHSV[1]-devHSV[1], meanHSV[1]+devHSV[1]);	
}

cv::Mat SkinSmoother::createSkinMask(const cv::Mat &imHSVF, double lowerHue, double upperHue, double lowerSat, double upperSat)
{
	CV_Assert(imHSVF.depth() == CV_32F);

	constexpr double maxHue = 360, maxValue = 1;

	cv::Scalar lowerHSV = { lowerHue, lowerSat, 0 }, upperHSV = { upperHue, upperSat, maxValue };

	cv::Mat mask, orMask;
	cv::inRange(imHSVF, lowerHSV, upperHSV, mask);

	// Account for hue values wrapping around 360 degrees
	if (lowerHSV[0] < 0 && upperHSV[0] < maxHue)
	{
		lowerHSV[0] = lowerHSV[0] + maxHue;
		upperHSV[0] = maxHue;
		cv::inRange(imHSVF, lowerHSV, upperHSV, orMask);
		cv::bitwise_or(mask, orMask, mask);
	}
	else if (lowerHSV[0] > 0 && upperHSV[0] > maxHue)
	{
		lowerHSV[0] = 0;
		upperHSV[0] = upperHSV[0] - maxHue;
		cv::inRange(imHSVF, lowerHSV, upperHSV, orMask);
		cv::bitwise_or(mask, orMask, mask);
	}

	return mask;
}