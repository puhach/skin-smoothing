#include "facialskinsmoother.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{

	try
	{
		// TODO: get images from the command line
		Mat im = imread("./images/hillary_clinton.jpg", IMREAD_COLOR);
		//Mat im = imread("./images/wrinkles_2.jpg", IMREAD_COLOR);
		//Mat im = imread("./images/wrinkles_1.png", IMREAD_COLOR);
		//Mat im = imread("./images/hopkins.jpg", IMREAD_COLOR);
		//Mat im = imread("./images/old_couple_2.jpg", IMREAD_COLOR);
		imshow("test", im);
		waitKey(0);

		/*Beautifier beautifier;
		beautifier.smoothSkin()*/

		/*SelfieApp app;
		app.smoothFacialSkin(im);*/

		/*SelfieApp app("selfie app");
		Mat out = app.applyFilter(im, FacialSkinSmoother{});
		app.display(out);*/

		FacialSkinSmoother filter{ 0.7f, SkinDetectionHeuristic::SelectiveSampling, 3, 30.0, 30.0 };
		Mat out = filter.apply(im);

		imshow("Skin Smoothing", out);
		waitKey(0);
	} // try
	catch (std::exception& e)
	{
		cout << e.what() << endl;
		return -1;
	}

	return 0;
}