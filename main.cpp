#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	Mat im = imread("./images/hillary_clinton.jpg", IMREAD_COLOR);
	imshow("test", im);
	waitKey(0);

	return 0;
}