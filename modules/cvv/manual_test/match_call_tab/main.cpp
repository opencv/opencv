#include <sstream>
#include <vector>
#include <functional>
#include <memory>

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QApplication>

#include "../../src/gui/match_call_tab.hpp"
#include "../../src/impl/match_call.hpp"
#include "../../include/opencv2/call_meta_data.hpp"
#include "../../src/view/match_view.hpp"
#include "../../src/impl/init.hpp"

#include "../../src/util/util.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

/**
 * MatchCallTab Test:
 * 1. Start the test. Two windows should appear, one showing RawView,
 * the other whichever View is currently set as default
 * (check the CVVisual file).
 * [-> all Constructors work]
 * In each window, on the top there should be (left to right)
 * a label reading "View:", a ComboBox displaying the name of the current view,
 * a "Set as default" and a "Help" button.
 * Below should be the View selected in the ComboBox
 * (refer to the respective View's test;
 * the images are entirely black and of size 1000x1000).
 * [-> GUI is complete]
 *
 * 2. Close the RawView window.
 * In the other, open a collapsable of the accordion menu.
 *
 * 3. Switch to another view. It should appear correctly.
 * [-> Switching Views works].
 * Click "Set as default".
 *
 * 4. Click "Help". The help page of this view should open in a web browser.
 * [-> Help works]
 *
 * 5. Switch back to the original view.
 * The accordion collapsable should still be open.
 * [-> History works]
 *
 * 6. Close the window. There should be no segfaults or other problems.
 * [-> Closing works]
 *
 * 7. Run the test again to see if the window showing the default View
 * now shows the one you set as default before.
 * [-> "Set as default" works]
 */
int main(int argc, char *argv[])
{
	/* Create some data for the MatchCallTab: */
	cv::Mat src{ 1000, 1000, CV_8U };
	cv::Mat train{ 1000, 1000, CV_8U };
	cvv::impl::CallMetaData data{};
	QString type{ "test_type" };

	QApplication vc{ argc, argv };

	std::vector<cv::KeyPoint> key1;

	for (int i = 0; i < std::min(src.rows, src.cols); i += 30)
	{
		cv::Point2f pt{ static_cast<float>(i), static_cast<float>(i) };
		key1.emplace_back(pt, 0.0f);
	}
	std::vector<cv::KeyPoint> key2;
	for (int i = 0; i < std::min(train.rows, train.cols); i += 30)
	{
		cv::Point2f pt{ static_cast<float>(i), static_cast<float>(i) };
		key2.emplace_back(pt, 0.0f);
	}
	std::vector<cv::DMatch> match;
	for (size_t i = 0; i < std::min(key1.size(), key2.size()); i++)
	{
		match.emplace_back(i, i, 1.0f);
	}

	cvv::impl::MatchCall mc{ src,  key1, train,              key2, match,
		                 data, type, "some description", "",   true };

	cvv::impl::initializeFilterAndViews();

	cvv::gui::MatchCallTab v{ mc };
	cvv::gui::MatchCallTab w{ mc, "RawView" };
	v.show();
	w.show();
	vc.exec();
	return 0;
}
