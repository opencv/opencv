#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "show_image.hpp"
#include "final_show.hpp"

/**
 * @brief This test shows all the provided images using the debug-framework.
 * 
 * Expected behaviour:
 * * A Mainwindow should open that shows an overview-table containing the first imagei
 * * Upon klicking step multiple times or '>>' once, all further images and filters should appear in
 *   the table.
 * * All calls can be opened in any existing window or in a new one.
 * * Closing calltabs should work. Closing the last tab of a window results in the closing of the window
 * * Clicking the Close-button results in the termination of the program with 0 as exit-status.
 */
int main(int argc, char **argv)
{
	for (int i = 1; i < argc; ++i)
	{
		cvv::showImage(cv::imread(argv[i]), CVVISUAL_LOCATION);
	}
	cvv::finalShow();
}
