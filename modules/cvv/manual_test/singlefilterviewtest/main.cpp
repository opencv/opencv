#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <QApplication>
#include <QImage>

#include "../../src/view/singlefilterview.hpp"
#include "../../src/impl/init.hpp"

/**
 * SingleFilterView Test:
 * Give an image as the argument when running the test.

 * A window should open with an accordion menu on the left and four images on the
 * right, ordered as follows:
 * The top two images should be the image given as argument to the left and its
 * dilated counterpart to the right.
 * Below each of the two images the image should be shown with the selected filter
 * applied to it. At the beginning no filter is selected, so the lower images are
 * the same as the upper images.
 
 * The accordion menu should consist of the following collapsables:
 * - Select a filter:
 *		If opened a checkbox for each original image should be shown, indicating
 *		whether the selected filter is to be applied to the respective image.
 *		A comboBox allows you to select the filter (gray filter, reorder channels
 *		and soebel). When a filter is selected you can choose is settings below
 *		the comboBox.
 *		To apply the selectedFilter press the 'apply' Button. When you change the
 *		filter or the filter settings after you have pressed apply, the new filter
 *		is automatically applied to the selected images.
 * - Zoom sychronization:
 *		When opened five checkboxes should be displayed. One for every image and
 *		one to indicate that no zoom synchronization is selected.
 *		If one image is selected, zooming or scrolling in this image is mapped to
 *		the other image so they all show the same section and have the same zoom
 *		level. Zooming or scrolling in a non-selected image only changes this image,
 *		as it would if no zoom synchronization was selected.
 * - Info image x:
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
	std::vector<cv::Mat> vec{ src, dest };
	cvv::view::SingleFilterView view{ vec };
	view.setWindowTitle("Single Filter View Test");
	view.show();
	
	return a.exec();
}
