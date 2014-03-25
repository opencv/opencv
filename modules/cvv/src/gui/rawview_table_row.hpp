#ifndef CVVISUAL_RAWVIEWTABLEROW_HPP
#define CVVISUAL_RAWVIEWTABLEROW_HPP

#include <vector>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <QTableWidget>
#include <QString>
#include <QList>

namespace cvv
{
namespace gui
{

/**
 * @brief A simple container wrapper for the cv::DMatch and cv::KeyPoint class.
 * See the opencv documentation for more information on the getter methods.
 */
class RawviewTableRow
{
      public:
	/**
	 * @brief Constructor of this class.
	 * @param match match that this row inherits.
	 * @param keyPoint1 "left" key point of the match
	 * @param keyPoint2 "right" key point of the match
	 */
	RawviewTableRow(cv::DMatch match, cv::KeyPoint keyPoint1,
	                cv::KeyPoint keyPoint2);

	/**
	 * @brief Constructor of this class for a single key point.
	 * The keypoint is stored as the first key point.
	 * @param keyPoint only key point of this object,
	 * @param left is the given key point is a left one?
	 */
	RawviewTableRow(cv::KeyPoint keyPoint, bool left = true);

	/**
	 * @brief Add this row to the given table.
	 * @note It does only fills the row in the table with the given index
	 * with its data.
	 * @param table given table
	 * @param row given row index
	 */
	void addToTable(QTableWidget *table, size_t row);

	float matchDistance() const
	{
		return match.distance;
	}

	int matchImgIdx() const
	{
		return match.imgIdx;
	}

	int matchQueryIdx() const
	{
		return match.queryIdx;
	}

	int matchTrainIdx() const
	{
		return match.trainIdx;
	}

	float keyPoint1XCoord() const
	{
		return keyPoint1.pt.x;
	}

	float keyPoint1YCoord() const
	{
		return keyPoint1.pt.y;
	}

	cv::Point2f keyPoint1Coords() const
	{
		return keyPoint1.pt;
	}

	float keyPoint1Size() const
	{
		return keyPoint1.size;
	}

	float keyPoint1Angle() const
	{
		return keyPoint1.angle;
	}

	float keyPoint1Response() const
	{
		return keyPoint1.response;
	}

	int keyPoint1Octave() const
	{
		return keyPoint1.octave;
	}

	int keyPoint1ClassId() const
	{
		return keyPoint1.class_id;
	}

	float keyPoint2XCoord() const
	{
		return keyPoint2.pt.x;
	}

	float keyPoint2YCoord() const
	{
		return keyPoint2.pt.y;
	}

	cv::Point2f keyPoint2Coords() const
	{
		return keyPoint2.pt;
	}

	float keyPoint2Size() const
	{
		return keyPoint2.size;
	}

	float keyPoint2Angle() const
	{
		return keyPoint2.angle;
	}

	float keyPoint2Response() const
	{
		return keyPoint2.response;
	}

	int keyPoint2Octave() const
	{
		return keyPoint2.octave;
	}

	int keyPoint2ClassId() const
	{
		return keyPoint2.class_id;
	}

	cv::DMatch getMatch() const
	{
		return match;
	}

	cv::KeyPoint getKeyPoint1() const
	{
		return keyPoint1;
	}

	cv::KeyPoint getKeyPoint2() const
	{
		return keyPoint2;
	}

	bool hasSingleKeyPoint() const
	{
		return hasLonelyKeyPoint_;
	}

	bool isLeftSingleKeyPoint() const
	{
		return left;
	}

	/**
	 * @brief Serealizes the given rows into a single block of text.
	 * The currently supported formats are:
	 *  - `CSV`   : Valid RFC 4180 CSV, with the same columns like the
	 * table.
	 *  - `JSON`  : Valid JSON (each row is an object consisting of three
	 * sub objects:
	 * 			  `match`, `keypoint 1` and `keypoint 2`).
	 *  - `PYTHON`: Valid python code (see JSON).
	 *  - `RUBY`  : Valid ruby code (see JSON).
	 *  @param rows given rows
	 *  @param format the format of the resulting representation (see above)
	 *  @param singleKeyPointRows do the given rows consist of single key
	 * point rows?
	 *  @return block representation of the given rows.
	 */
	static QString rowsToText(const std::vector<RawviewTableRow> &rows,
	                          const QString format,
	                          bool singleKeyPointRows = false);

	/**
	 * @brief Returns the currently available text formats for the
	 * rowsToText method.
	 * @return {"CSV", "JSON", "PYTHON", "RUBY"}
	 */
	static std::vector<QString> getAvailableTextFormats();

      private:
	cv::DMatch match;
	cv::KeyPoint keyPoint1;
	cv::KeyPoint keyPoint2;
	bool hasLonelyKeyPoint_;
	bool left = false;
};

/**
 * @brief Create a list of rows from the given key points and matches.
 * And one for the single key points.
 *
 * It creates a row for each match and uses the key points to get the two
 * locations of each one..
 * @param keyPoints1 given "left" key points
 * @param keyPoints2 given "right" key points
 * @param matches given matches
 * @param usesTrainDescriptor Use the trainIdx property of each match to get the
 * "right" key points?
 * @return first element is the match row list, second is the single key point
 *row list
 */
std::pair<QList<RawviewTableRow>, QList<RawviewTableRow>>
createRawviewTableRows(const std::vector<cv::KeyPoint> &keyPoints1,
                       const std::vector<cv::KeyPoint> &keyPoints2,
                       const std::vector<cv::DMatch> &matches,
					   bool usesTrainDescriptor = true);

/**
 * @brief Create a list of rows from the given key points.
 * It creates a row for each key point, that does only contain this key point.
 * @param keyPoints given key points
 * @param left are the given key points are "left" ones?
 * @return resulting list
 */
QList<RawviewTableRow>
createSingleKeyPointRawviewTableRows(const std::vector<cv::KeyPoint> &keyPoints,
                                     bool left = true);
}
}

#endif
