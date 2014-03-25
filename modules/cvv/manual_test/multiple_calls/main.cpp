#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "show_image.hpp"
#include "final_show.hpp"

/**
 * This test calls showImage for every provided image followed by a call to finalShow for each.
 * 
 * The expected behaviour is that for every Image img the debug-framework will open once with one
 * SingleImage-call and get into the final-call modus after steping forward once. After closing
 * it, the same should happen with the next Image, whereby all old state was deleted between these
 * calls.
 */
int main(int argc, char **argv)
{
	for (int i = 1; i < argc; ++i)
	{
		cvv::showImage(cv::imread(argv[i]), CVVISUAL_LOCATION);
		cvv::finalShow();
	}
}
