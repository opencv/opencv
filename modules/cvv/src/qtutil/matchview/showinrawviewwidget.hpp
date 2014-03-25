#ifndef CVVISUAL_SHOW_IN_RAWVIEW
#define CVVISUAL_SHOW_IN_RAWVIEW

#include <QWidget>

#include "opencv2/features2d/features2d.hpp"

#include "matchmanagement.hpp"
#include "keypointmanagement.hpp"
#include "rawview_window.hpp"

namespace  cvv {namespace qtutil{
/**
 * @brief this class shows a RawViewWindow when it will be shown and hides it when this will be hidden
 * and it connect the managemts (match/keypoint) with the RawViewWindow
 */
class ShowInRawView:public QWidget{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 * @param left_key the left keypoints
	 * @param right_key th right keypoints
	 * @param matches the matches
	 * @param matchmnt the matchmanagement
	 * @param keymnt the keypointmanagement
	 * @param parent the parent Widget
	 */
	ShowInRawView(std::vector<cv::KeyPoint> left_key,
		      std::vector<cv::KeyPoint> right_key,
		      std::vector<cv::DMatch> matches,
		      MatchManagement* matchmnt,
		      KeyPointManagement* keymnt,
		      QWidget*parent=nullptr);

	/**
	 * @brief the cestructor deletes the RawViewWindow
	 */
	~ShowInRawView();

protected:
	virtual void hideEvent(QHideEvent * );

	virtual void showEvent(QShowEvent *);

private slots:

	void getcurrentSelection();

private:
	MatchManagement* matchmnt_;
	KeyPointManagement* keymnt_;
	RawviewWindow* rawViewWindow_;
	std::vector<cv::KeyPoint> left_key_;
	std::vector<cv::KeyPoint> right_key_;
	std::vector<cv::DMatch> matches_;
};

}}

#endif
