#include <iostream>

#include <QApplication>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/highgui/highgui.hpp>

#include "../../src/view/translationsmatchview.hpp"
#include "../../src/impl/init.hpp"

/**
 * @brief this test shows a translationview, it should only end if u close the window manual.
 * The matches should be schow as translationlinesin the images, a line should only be in one image.
 *
 * all classes like ,zoomableimage, zoomableoptpane, synczoomwidget, match Managemant,...
 * should work correctly.
 * "Match Settings":
 *	you can select matches with the selections and settings should only applied to the selected
 *	matches.
 *
 * "KeyPoint Settings":
 *	see match settings
 *
 * "Image Information":
 *	the zoomableoptpanel should show all informations of the cv::Mat correctly and the zoom
 *	should be syncron with the zoomabelImage
 *
 * "syncWidget":
 *	you can choose that the zoom in one image should do the same in the other images, or none
 *
 * @param argc number of arguments
 * @param argv arguments executablepath, image path1, imagepath 2
 * @return 0
 */
int main(int argc, char **argv)
{
	QApplication a(argc, argv);

	if (argc != 3)
	{
		std::cerr << "Only execute this test with filenames of two "
			     "images as arguments! \n";
		return -1;
	}
	auto src = cv::imread(argv[1]);
	auto train = cv::imread(argv[2]);
	std::vector<cv::KeyPoint> key1;

	for (int i = std::min(src.rows, src.cols); i > 0; i -= 30)
	{
		cv::Point2f pt{ static_cast<float>(
				    std::min(src.rows, src.cols) - i),
				static_cast<float>(i) };
		key1.emplace_back(pt, 0.0f);
	}
	std::vector<cv::KeyPoint> key2;
	for (int i = 0; i < std::min(train.rows, train.cols); i += 30)
	{
		cv::Point2f pt{ static_cast<float>(i), static_cast<float>(i) };
		key2.emplace_back(pt, 0.0f);
	}
	std::vector<cv::DMatch> match;
	for (size_t i = 0; i < std::min(key1.size() - 1, key2.size() - 1); i++)
	{
		match.emplace_back(i, i + 1, 1.0f);
	}
	cvv::impl::initializeFilterAndViews();
	cvv::view::TranslationMatchView view{ key1, key2, match, src, train };
	view.setWindowTitle("TranslationMatchView Test");
	view.show();
	return a.exec();
}
