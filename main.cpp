#include "facialskinsmoother.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;


void printUsage()
{
	cout << "Usage: skinsoft -image=<image file> "
			"[-confidence=<float 0..1>] "
			"[-radius=<integer>] "
			"[-sigmac=<float>] "
			"[-sigmas=<float>] "
			"[-heuristic=<choice>]" << endl;
}

int main(int argc, char *argv[])
{

	static const String keys =
		"{help h usage ? |      | Print the help message  }"
		"{image          |<none>| The input image file  }"
		"{confidence     |0.7   | Face detection confidence threshold }"
		"{radius         |3     | Blur radius          }"
		"{sigmac         |30    | Blur sigma in the color space }"
		"{sigmas         |30    | Blur sigma in the coordinate space }"
		"{heuristic      |2     | Skin detection heuristic: 1 - Mean color, 2 - Dominant color, 3 - Selective sampling }";

	try
	{
		CommandLineParser parser(argc, argv, keys);
		parser.about("Facial Skin Smoother\n(c) Yaroslav Pugach");

		if (parser.has("help"))
		{
			printUsage();
			return 0;
		}

		String inputFile = parser.get<String>("image");
		float faceConfThreshold = parser.get<float>("confidence");
		int blurRadius = parser.get<int>("radius");
		double sigmaColor = parser.get<double>("sigmac");
		double sigmaSpace = parser.get<double>("sigmas");
		int heuristicId = parser.get<int>("heuristic");
		SkinDetectionHeuristic heuristic{ heuristicId-1 };

		if (!parser.check())
		{
			parser.printErrors();
			printUsage();
			return -1;
		}

		Mat im = imread(inputFile, IMREAD_COLOR);
		imshow("Original", im);
		waitKey(0);

		FacialSkinSmoother filter{ faceConfThreshold, heuristic, blurRadius, sigmaColor, sigmaSpace };
		Mat out = filter.apply(im);

		imshow("Output", out);
		waitKey(0);
	} // try
	catch (std::exception& e)
	{
		cout << e.what() << endl;
		return -2;
	}

	return 0;
}