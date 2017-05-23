#include "rawview_table.hpp"

#include <utility>

#include <QVBoxLayout>
#include <QStringList>

#include "../stfl/element_group.hpp"
#include "rawview_table_row.hpp"
#include "rawview_group_subtable.hpp"
#include "../qtutil/accordion.hpp"

namespace cvv
{
namespace gui
{

RawviewTable::RawviewTable(view::Rawview *parent) : parent{ parent }
{
	subtableAccordion = new qtutil::Accordion{};
	auto *layout = new QVBoxLayout{};
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(subtableAccordion);
	setLayout(layout);
}

void RawviewTable::updateRowGroups(
    std::vector<stfl::ElementGroup<RawviewTableRow>> newGroups)
{
	subtableAccordion->clear();
	subTables.clear();
	for (auto &group : newGroups)
	{
		if (group.size() > 0)
		{
			auto subtable = util::make_unique<RawviewGroupSubtable>(
			    this, std::move(group));
			auto subtablePtr = subtable.get();
			auto titles = group.getTitles();
			QString title =
			    "No grouping specified, use #group to do specify";
			if (titles.size() != 0)
			{
				title = titles.join(", ");
			}
			subtableAccordion->push_back(title, std::move(subtable),
			                             false);
			subTables.push_back(subtablePtr);
		}
	}
}

void RawviewTable::updateUI()
{
	for (auto *subTable : subTables)
	{
		subTable->updateUI();
	}
}

std::vector<cv::DMatch> RawviewTable::getMatchSelection()
{
	std::vector<cv::DMatch> matches;
	for (auto subTable : subTables)
	{
		auto subMatches = subTable->getMatchSelection();
		matches.insert(matches.end(), subMatches.begin(), subMatches.end());
	}
	return matches;
}

std::vector<cv::KeyPoint> RawviewTable::getKeyPointSelection()
{
	std::vector<cv::KeyPoint> keyPoints;
	for (auto subTable : subTables)
	{
		auto subKeyPoints = subTable->getKeyPointSelection();
		keyPoints.insert(keyPoints.end(), subKeyPoints.begin(), subKeyPoints.end());
	}
	return keyPoints;
}

void RawviewTable::setMatchSelection(std::vector<cv::DMatch> matches)
{
	for (auto subTable : subTables)
	{
		subTable->setMatchSelection(matches);
	}
}

void RawviewTable::setKeyPointSelection(std::vector<cv::KeyPoint> keyPoints)
{
	for (auto subTable : subTables)
	{
		subTable->setKeyPointSelection(keyPoints);
	}
}
}
}
