#include <iostream>

#include <QWidget>
#include <QApplication>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../src/impl/init.hpp"
#include "../../src/view/image_view.hpp"

/**
 * (Single)ImageView Test:
 * Give an image as the argument when running the test.
 * It should appear correctly on the right.
 * On the left should be an accordion menu consisting of only
 * one collapsable, "ImageInformation:".
 * Open it.
 * Zooming in and out of the image with both the spin box
 * and (Shift) + Strg + Mouse Wheel should work (Shift: slower).
 * The information about the image in the collapsable should be correct.
 * "Show full image" should work.
 */
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	if(argc < 2)
	{
		std::cerr << "Only execute this with filenames of images as arguments!" << std::endl;
		return -1;
	}
	auto src = cv::imread(argv[1]);

	cvv::impl::initializeFilterAndViews();
	cvv::view::ImageView view{src};
	view.setWindowTitle("Image View Test");
	view.show();

	return a.exec();
}
