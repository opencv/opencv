#ifndef CVVISUAL_RAWVIEWTABLE_HPP
#define CVVISUAL_RAWVIEWTABLE_HPP

#include <vector>

#include <QWidget>
#include <QList>

#include "../view/rawview.hpp"
#include "rawview_table_row.hpp"
#include "../stfl/element_group.hpp"
#include "../qtutil/accordion.hpp"
#include "../util/util.hpp"
#include "rawview_group_subtable.hpp"

namespace cvv
{

namespace view
{
class Rawview;
}

namespace gui
{

class RawviewTableCollumn;

/**
 * @brief A table (consisting of subtables) displaying raw match data.
 */
class RawviewTable : public QWidget
{
	Q_OBJECT

public:
	/**
	 * @brief Constructor of this class.
	 * @param parent parent view
	 */
	RawviewTable(view::Rawview *parent);

	/**
	 * @brief Update the inherited groups of rows and rebuild the UI fully.
	 * @param newGroups new groups for this table
	 */
	void updateRowGroups(
	    const std::vector<stfl::ElementGroup<RawviewTableRow>> newGroups);

	/**
	 * @brief Updates the UI
	 */
	void updateUI();

	/**
	 * @brief Returns the parent view.
	 * @return parent view
	 */
	view::Rawview *getParent()
	{
		return parent;
	}

	std::vector<cv::DMatch> getMatchSelection();

	std::vector<cv::KeyPoint> getKeyPointSelection();	

public slots:
	
	void setMatchSelection(std::vector<cv::DMatch> matches);

	void setKeyPointSelection(std::vector<cv::KeyPoint> keyPoints);
	
private:
	view::Rawview *parent;
	qtutil::Accordion *subtableAccordion;
	std::vector<RawviewGroupSubtable *> subTables{};
	
};
}
}

#endif
