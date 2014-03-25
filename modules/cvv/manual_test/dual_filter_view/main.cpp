#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <QApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QWidget>

#include "../../src/impl/init.hpp"
#include "../../src/view/dual_filter_view.hpp"
#include "../../src/view/singlefilterview.hpp"

/**
 * DualFilterView Test:
 * Give an image as the argument when running the test.

 * A window should open with an accordion menu on the left and three images on the
 * right, ordered as follows:
 * The first image should be the image given as argument and the third its
 * dilated counterpart.
 * The image in the middle should be the result of the selected filter being
 * applied to the two outer images. At the beginning no filter is selected, so
 * the middle image is the same as the left image.
 
 * The accordion menu should consist of the following collapsables:
 * - Select a filter:
 *		A comboBox allows you to select the filter (Changed Pixels, Difference,
 *		Overlay). When a filter is selected you can choose is settings below
 *		the comboBox.
 *		The Changed Pixels filter marks changed pixels as black and unchanged
 *		pixels as white.
 *		The Difference filter shows either the difference for every channel
 *		(grayscale), or the difference between the two images' hue, saturation
 *		or value. The requestes filter type can be selected through a comboBox.
 *		 The Overlay filter adds the two images (overlays them) where the weight
 *		of the right images is shown and changable by a slider.
 *		To apply the selectedFilter press the 'apply' Button. When you change the
 *		filter or the filter settings after you have pressed apply, the new filter
 *		is automatically applied to the selected images.
 * - Zoom sychronization:
 *		When opened four checkboxes should be displayed. One for every image and
 *		one to indicate that no zoom synchronization is selected.
 *		If one image is selected, zooming or scrolling in this image is mapped to
 *		the other image so they all show the same section and have the same zoom
 *		level. Zooming or scrolling in a non-selected image only changes this image,
 *		as it would if no zoom synchronization was selected.
 * - Image Information x:
 *		For each image there is a widget which displays information about the image.
 * 		Zooming in and out of an image with both the spin box in the image information
 * 		and (Shift) + Strg + Mouse Wheel should work (Shift: slower).
 * 		Unchecking "Show image" should make it disappear, checking make it reappear.
 * 		The information about the image in the collapsable should be correct.
 * 		"Show full image" should work.
 */
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	if (argc < 2)
	{
		std::cerr << "Only execute this with filenames of images as "
		             "arguments!" << std::endl;
		return -1;
	}
	auto src = cv::imread(argv[1]);
	auto elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9),
	                                      cv::Point(4, 4));
	cv::Mat dest;

	cv::dilate(src, dest, elem);

	cvv::impl::initializeFilterAndViews();
	std::array<cv::Mat, 2> inArray{ src, dest };
	cvv::view::DualFilterView view{ inArray };
	view.setWindowTitle("Dual Filter View Test");
	view.show();

	return a.exec();
}
