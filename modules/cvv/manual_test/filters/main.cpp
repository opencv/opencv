#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "filter.hpp"
#include "final_show.hpp"

void dilateFile(char *filename)
{
	auto src = cv::imread(filename);
	auto elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9),
	                                      cv::Point(4, 4));
	cv::Mat dest;

	cv::dilate(src, dest, elem);
	cvv::debugFilter(src, dest, CVVISUAL_LOCATION, filename);
	cv::erode(src, dest, elem);
	cvv::debugFilter(src, dest, CVVISUAL_LOCATION, filename);
	cv::Sobel(src, dest, -1, 1, 1);
	cvv::debugFilter(src, dest, CVVISUAL_LOCATION, filename);
	cv::morphologyEx(src, dest, cv::MORPH_GRADIENT, elem);
	cvv::debugFilter(src, dest, CVVISUAL_LOCATION, filename);
}

/**
 * @brief This test filters  all the provided images with erode(), dilate(), Sobel() and morphologyEx()
 * and visualizes these filters with the debug-framework.
 * 
 * Expected behaviour:
 * * A Mainwindow should open that shows an overview-table containing the first image and it's dilated
 *   version.
 * * Upon klicking step multiple times or '>>' once, all further images and filters should appear in
 *   the table.
 * * All calls can be opened in any existing window or in a new one. It is possible to select all the
 *   different filter-visualisations for all of them.
 * * Closing calltabs should work. Closing the last tab of a window results in the closing of the window
 * * Clicking the Close-button results in the termination of the program with 0 as exit-status.
 */
int main(int argc, char **argv)
{
	if (argc == 1)
	{
		std::cerr
		    << argv[0]
		    << " must be callled with one or more files as arguments\n";
		return 1;
	}

	for (int i = 1; i < argc; ++i)
	{
		dilateFile(argv[i]);
	}
	std::cout << "All calculation done" << std::endl;
	cvv::finalShow();
	std::cout << "Program finished" << std::endl;
}
