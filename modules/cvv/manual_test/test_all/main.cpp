#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "show_image.hpp"
#include "final_show.hpp"
#include "filter.hpp"

/**
 * @brief This test shows all the provided images and filters them using morphologyEx() using the 
 * debug-framework.
 * 
 * Expected behaviour:
 * * A Mainwindow should open that shows an overview-table containing the first imagei
 * * Upon klicking step multiple times or '>>' once, all further images and filters should appear in
 *   the table.
 * * All calls can be opened in any existing window or in a new one. For the filtered Images it is
 *   possible to select all the different filter-visualisations for all of them.
 * * Closing calltabs should work. Closing the last tab of a window results in the closing of the window
 * * Clicking the Close-button results in the termination of the program with 0 as exit-status.
 */
int main(int argc, char **argv)
{
	std::vector<cv::Mat> images;
	cvv::FinalShowCaller finalizer;
	for (int i = 1; i < argc; ++i)
	{
		auto img = cv::imread(argv[i]);
		cvv::showImage(img, CVVISUAL_LOCATION, argv[i]);
		auto elem = cv::getStructuringElement(
		    cv::MORPH_RECT, cv::Size(9, 9), cv::Point(4, 4));
		cv::Mat filteredImg;
		cv::morphologyEx(img, filteredImg, cv::MORPH_GRADIENT, elem);
		cvv::debugFilter(img, filteredImg, CVVISUAL_LOCATION, argv[i]);
		images.emplace_back(img);
	}
}
