#ifndef CVVISUAL_RAWVIEW_GROUP_SUBTABLE_HPP
#define CVVISUAL_RAWVIEW_GROUP_SUBTABLE_HPP

#include <memory>
#include <set>
#include <vector>

#include <QWidget>
#include <QTableWidget>
#include <QAction>
#include <QItemSelection>

#include "../stfl/element_group.hpp"
#include "rawview_table_row.hpp"
#include "../util/util.hpp"

namespace cvv
{
namespace controller
{
class ViewController;
}
}

namespace cvv
{
namespace gui
{

class RawviewTable;

/**
 * @brief A table for the a group of overview data sets.
 */
class RawviewGroupSubtable : public QWidget
{
	Q_OBJECT

      public:
	/**
	 * @brief Constructs an over group subtable.
	     * @param controller view controller
	     * @param parent parent table
	     * @param group the displayed group of overview data sets
	 */
	RawviewGroupSubtable(RawviewTable *parent,
	                     stfl::ElementGroup<RawviewTableRow> group);

	/**
	 * @brief Updates the displayed table UI.
	 */
	void updateUI();
	
	std::vector<cv::DMatch> getMatchSelection();

	std::vector<cv::KeyPoint> getKeyPointSelection();	

public slots:
	
	void setMatchSelection(std::vector<cv::DMatch> matches);

	void setKeyPointSelection(std::vector<cv::KeyPoint> keyPoints);

private slots:
	
	void customMenuRequested(QPoint location);
	void customMenuAction(QAction *action);
	void selectionChanged();

private:
	RawviewTable *parent;
	stfl::ElementGroup<RawviewTableRow> group;
	QTableWidget *qTable;
	std::set<int> currentRowIndexes;
	
	std::vector<RawviewTableRow> getSelectedRows();
	
	void setSelectedRows(std::set<int> rowIndexes);
};
}
}

#endif
