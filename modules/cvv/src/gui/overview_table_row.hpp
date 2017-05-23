#ifndef CVVISUAL_OVERVIEWTABLEROW_HPP
#define CVVISUAL_OVERVIEWTABLEROW_HPP

#include <vector>

#include <QTableWidget>
#include <QString>
#include <QPixmap>

#include "../impl/call.hpp"
#include "../util/util.hpp"

namespace cvv
{
namespace gui
{

/**
 * @brief A UI wrapper for an impl::Call, providing utility and UI functions.
 *
 * Also allowing it to add its data to a table.
 */
class OverviewTableRow
{

public:
	/**
	 * @brief Constructor of this class.
	 * @param call call this row is based on
	 */
	OverviewTableRow(util::Reference<const impl::Call> call);

	~OverviewTableRow()
	{
	}

	/**
	 * @brief Adds the inherited data set to the given table.
	 * @param table given table
	 * @param row row index at which the data will be shown
	 * @param showImages does the table show images?
	 * @param maxImages the maximum number of images the table shows
	 * @param imgHeight height of the shown images
	 * @param imgWidth width of the shown images
     */
	void addToTable(QTableWidget *table, size_t row, bool showImages,
	                size_t maxImages, int imgHeight = 100,
	                int imgWidth = 100);
	
	/**
	 * @brief Resizes the images in the given row.
	 * Make sure to call this after (!) you called addToTable() with the
	 * same row parameter on this object some time. 
	 * @param table given table
	 * @param row row index at which the data will be shown
	 * @param showImages does the table show images?
	 * @param maxImages the maximum number of images the table shows
	 * @param imgHeight height of the shown images
	 * @param imgWidth width of the shown images
	 */
	void resizeInTable(QTableWidget *table, size_t row, bool showImages,
				  size_t maxImages, int imgHeight = 100,
                  int imgWidth = 100);
	
	/**
	 * @brief Get the inherited call.
	 * @return the inherited call
	 */
	util::Reference<const impl::Call> call() const
	{
		return call_;
	}

	/**
	 * @brief Returns the description of the inherited call.
	 * @return description of the inherited call.
	 */
	QString description() const
	{
		return description_;
	}

	/**
	 * @brief Returns the id of the inherited call.
	 * @return id of the inherited call.
	 */
	size_t id() const
	{
		return id_;
	}

	/**
	 * @brief Returns the function name property of the inherited call.
	 * @return function name property of the inherited call.
	 */
	QString function() const
	{
		return functionStr;
	}

	/**
	 * @brief Returns the file name of the inherited call.
	 * @return file name of the inherited call.
	 */
	QString file() const
	{
		return fileStr;
	}

	/**
	 * @brief Returns the line property of the inherited call.
	 * @return line property of the inherited call.
	 */
	size_t line() const
	{
		return line_;
	}

	/**
	 * @brief Returns the type of the inherited call.
	 * @return type of the inherited call.
	 */
	QString type() const
	{
		return typeStr;
	}

private:
	util::Reference<const impl::Call> call_;
	size_t id_ = 0;
	size_t line_ = 0;
	QString idStr = "";
	QString description_ = "";
	std::vector<QPixmap> imgs{};
	QString functionStr = "";
	QString fileStr = "";
	QString lineStr = "";
	QString typeStr = "";
};
}
}

#endif
