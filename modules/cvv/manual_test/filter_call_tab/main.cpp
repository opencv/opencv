#include <sstream>

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QApplication>

#include "../../src/gui/filter_call_tab.hpp"
#include "../../src/impl/filter_call.hpp"
#include "../../include/opencv2/call_meta_data.hpp"
#include "../../src/view/defaultfilterview.hpp"
#include "../../src/view/dual_filter_view.hpp"
#include "../../src/view/singlefilterview.hpp"
#include "../../src/util/util.hpp"

#include <opencv2/core/core.hpp>

/**
 * FilterCallTab Test:
 * 1. Start the test.
 * Two windows should appear, one showing SingleFilterView,
 * the other whichever View is currently set as default
 * (check the CVVisual file).
 * [-> all Constructors work]
 * In each window, on the top there should be (left to right)
 * a label reading "View:", a ComboBox displaying the name of the current view,
 * a "Set as default" and a "Help" button.
 * Below should be the View selected in the ComboBox
 * (refer to the respective View's test;
 * the images are entirely black and of size 100x100).
 * [-> GUI is complete]
 *
 * 2. Close the SingleFilterView window.
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

	/* Create some data for the FilterCallTab: */
	cv::Mat in{ 100, 100, CV_8U };
	cv::Mat out{ 100, 100, CV_8U };
	cvv::impl::CallMetaData data{};
	QString type{ "test_type" };
	QApplication controller{ argc, argv };
	cvv::impl::FilterCall fc{ in, out, data, type, "some description", "" };

	cvv::gui::FilterCallTab::registerFilterView<
	    cvv::view::DefaultFilterView>("DefaultFilterView");
	cvv::gui::FilterCallTab::registerFilterView<cvv::view::DualFilterView>(
	    "DualFilterView");
	cvv::gui::FilterCallTab::registerFilterView<
	    cvv::view::SingleFilterView>("SingleFilterView");

	cvv::gui::FilterCallTab u{ fc };
	cvv::gui::FilterCallTab v{ fc, "SingleFilterView" };

	u.show();
	v.show();
	controller.exec();
	return 0;
}
