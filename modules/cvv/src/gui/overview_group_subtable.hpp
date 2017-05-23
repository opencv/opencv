#ifndef CVVISUAL_OVERVIEW_GROUP_SUBTABLE_HPP
#define CVVISUAL_OVERVIEW_GROUP_SUBTABLE_HPP

#include <memory>

#include <QWidget>
#include <QTableWidget>
#include <QAction>
#include <QResizeEvent>

#include "../stfl/element_group.hpp"
#include "overview_table_row.hpp"
#include "../util/util.hpp"
#include "../controller/view_controller.hpp"

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

class OverviewTable;

/**
 * @brief A table for the a group of overview data sets.
 */
class OverviewGroupSubtable : public QWidget
{
	Q_OBJECT

      public:
	/**
	 * @brief Constructs an over group subtable.
	     * @param controller view controller
	     * @param parent parent table
	     * @param group the displayed group of overview data sets
	 */
	OverviewGroupSubtable(
	    util::Reference<controller::ViewController> controller,
	    OverviewTable *parent, stfl::ElementGroup<OverviewTableRow> group);

	~OverviewGroupSubtable()
	{
	}

	/**
	 * @brief Updates the displayed table UI.
	 */
	void updateUI();

	/**
	 * @brief Remove the row with the given id.
	 * @param given table row id
	 */
	void removeRow(size_t id);

	/**
	 * @brief Checks whether or not the table shows the row with the given
	 * id.
	 * @param id given row id
	 * @return Does the table show the row with the given id?
	 */
	bool hasRow(size_t id);

	/**
	 * @brief Set the displayed rows.
	 * @note This method does some optimisations to only fully rebuild all
	 * rows if neccessary.
	 * @param newGroup new group of rows that will be displayed
	 */
	void setRowGroup(stfl::ElementGroup<OverviewTableRow> &newGroup);
	
protected:
	void resizeEvent(QResizeEvent *event);

private slots:
	void rowClicked(int row, int collumn);
	void customMenuRequested(QPoint location);
	void customMenuAction(QAction *action);

private:
	util::Reference<controller::ViewController> controller;
	OverviewTable *parent;
	stfl::ElementGroup<OverviewTableRow> group;
	QTableWidget *qTable;
	size_t currentCustomMenuCallTabId = 0;
	bool currentCustomMenuCallTabIdValid = false;
	size_t maxImages = 0;
	int imgSize = 0;
	int rowHeight = 0;

	void initUI();

	void updateMinimumSize();
};
}
}

#endif
