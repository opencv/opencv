#ifndef CVVISUAL_RAWVIEW_WINDOW_HPP
#define CVVISUAL_RAWVIEW_WINDOW_HPP

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <QMainWindow>
#include <QString>

#include "../../view/rawview.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * @brief A window showing the raw view, that displays the raw data of a match
 * call.
 */
class RawviewWindow : public QMainWindow
{

	Q_OBJECT

      public:
	/**
	 * @brief Constructor of this class.
	 * @param title window title
	 * @param keypoints1 left keypoints
	 * @param keypoints2 right keypoints
	 * @param matches matches between the left and the right keypoints.
	 */
	RawviewWindow(QString title,
		      const std::vector<cv::KeyPoint> keypoints1,
		      const std::vector<cv::KeyPoint> keypoints2,
		      const std::vector<cv::DMatch> matches);

	/**
	 * @brief Constructor of this class.
	 * A view will be created when the matchesSelected gets called the first
	 * time.
	 * @param title window title
	 * @param keypoints1 left keypoints
	 * @param keypoints2 right keypoints
	 */
	RawviewWindow(QString title,
		      const std::vector<cv::KeyPoint> &keypoints1,
		      const std::vector<cv::KeyPoint> &keypoints2);

signals:

	/**
	 * @brief The user selected some matches.
	 * @param matches seleted matches
	 */
	void matchesSelected(const std::vector<cv::DMatch> &matches);

	/**
	 * @brief The user selected some single key points.
	 * Single key points are key points without correspoinding match.
	 * @param keyPoints seleted single key points
	 */
	void keyPointsSelected(const std::vector<cv::KeyPoint> &keyPoints);

public slots:

	/**
	 * @brief Show only the given matches.
	 * @param matches
	 */
	void selectMatches(const std::vector<cv::DMatch> &matches);

	/**
	 * @brief Show (and filter) only the given single key points.
	 * @param keyPoints given key points
	 */
	void selectKeyPoints(const std::vector<cv::KeyPoint> &keyPoints);

      private:
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	view::Rawview *view = nullptr;
};
}
}

#endif
