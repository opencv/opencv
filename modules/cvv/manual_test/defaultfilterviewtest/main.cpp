
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <QApplication>
#include <QImage>

#include "../../src/view/defaultfilterview.hpp"

/**
 * DefaultFilterView Test:
 * Give images as program arguments.
 *
 * A window should open with an accordion menu on the left and as many images as
 * you passed to the test on the right.
 
 * The accordion menu should consist of the following collapsables:
 * - Zoom sychronization:
 *		When opened one plus as many checkboxes as there are images should be 
 *		displayed. One for every image and one to indicate that no zoom
 *		synchronization is selected.
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
	std::vector<cv::Mat> imagelist;
	
	for(int i = 1; i < argc; i++)
	{
		auto src = cv::imread(argv[i]);
		imagelist.push_back(src);
	}
	cvv::view::DefaultFilterView dfv{ imagelist, nullptr };

	dfv.show();
	return a.exec();
}
