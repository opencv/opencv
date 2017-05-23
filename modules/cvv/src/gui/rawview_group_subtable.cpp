#include "rawview_group_subtable.hpp"

#include <utility>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <QVBoxLayout>
#include <QStringList>
#include <QModelIndexList>
#include <QModelIndex>
#include <QMenu>
#include <QAction>
#include <QHeaderView>
#include <QApplication>
#include <QClipboard>
#include <QTableWidgetSelectionRange>

#include "call_window.hpp"
#include "../view/rawview.hpp"
#include "rawview_table.hpp"
#include "../controller/view_controller.hpp"

namespace cvv
{
namespace gui
{

RawviewGroupSubtable::RawviewGroupSubtable(
    RawviewTable *parent, stfl::ElementGroup<RawviewTableRow> group)
    : parent{ parent }, group{ std::move(group) }
{
	qTable = new QTableWidget(this);
	qTable->setSelectionBehavior(QAbstractItemView::SelectRows);
	qTable->setSelectionMode(QAbstractItemView::ExtendedSelection);
	auto horizontalHeader = qTable->horizontalHeader();
	horizontalHeader->setSectionResizeMode(QHeaderView::Stretch);
	horizontalHeader->setStretchLastSection(false);
	qTable->setContextMenuPolicy(Qt::CustomContextMenu);
	connect(qTable, SIGNAL(customContextMenuRequested(QPoint)), this,
	        SLOT(customMenuRequested(QPoint)));
	connect(qTable, SIGNAL(itemSelectionChanged()), this,
	        SLOT(selectionChanged()));

	auto *layout = new QVBoxLayout;
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(qTable);
	setLayout(layout);

	QStringList list{};
	if (group.getTitles().contains("single key point"))
	{
		list << "x"
		     << "y"
		     << "size"
		     << "angle"
		     << "response"
		     << "octave"
		     << "class id"
		     << "img number";
	}
	else
	{
		list << "match distance"
		     << "img idx"
		     << "query idx"
		     << "train idx";
		list << "key point 1 x"
		     << "y 1"
		     << "size 1"
		     << "angle 1"
		     << "response 1"
		     << "octave 1"
		     << "class id 1";
		list << "key point 2 x"
		     << "y 2"
		     << "size 2"
		     << "angle 2"
		     << "response 2"
		     << "octave 2"
		     << "class id 2";
	}
	qTable->setColumnCount(list.size());
	qTable->setHorizontalHeaderLabels(list);
	updateUI();
}

void RawviewGroupSubtable::updateUI()
{
	qTable->setRowCount(group.size());
	for (size_t i = 0; i < group.size(); i++)
	{
		group.get(i).addToTable(qTable, i);
	}
}

void RawviewGroupSubtable::selectionChanged()
{
	QModelIndexList indexList = qTable->selectionModel()->selectedIndexes();
	currentRowIndexes.clear();
	for (QModelIndex index : indexList)
	{
		if (index.isValid())
		{
			auto row = index.row();
			if (row < qTable->rowCount() && row >= 0)
			{
				currentRowIndexes.insert(row);
			}
		}
	}
}

void RawviewGroupSubtable::customMenuRequested(QPoint location)
{
	QModelIndex index = qTable->indexAt(location);
	if (!index.isValid())
	{
		return;
	}

	QMenu *menu = new QMenu(this);
	connect(menu, SIGNAL(triggered(QAction *)), this,
	        SLOT(customMenuAction(QAction *)));

	if (parent->getParent()->doesShowShowInViewMenu())
	{
		menu->addAction(new QAction("Show selected rows in view", this));
	}
	
	auto formats = RawviewTableRow::getAvailableTextFormats();
	for (auto format : formats)
	{
		menu->addAction(
		    new QAction(QString("Copy as %1").arg(format), this));
	}
	int row = index.row();
	if (currentRowIndexes.size() == 0)
	{
		currentRowIndexes = { row };
	}
	menu->popup(qTable->viewport()->mapToGlobal(location));
}

void RawviewGroupSubtable::customMenuAction(QAction *action)
{
	bool single = group.getTitles().contains("single key point");
	if (currentRowIndexes.size() > 0)
	{
		auto rows = getSelectedRows();
		QString text = action->text();
		if (text == "Show selected rows in view")
		{
			if (single)
			{
				std::vector<cv::KeyPoint> keyPoints;
				for (auto row : rows)
				{
					keyPoints.push_back(row.getKeyPoint1());
				}
				parent->getParent()->keyPointsSelected(
				    keyPoints);
				std::vector<cv::DMatch> emptyVec{};
				parent->getParent()->matchesSelected(emptyVec); //unselect matches
			}
			else
			{
				std::vector<cv::DMatch> matches;
				for (auto row : rows)
				{
					matches.push_back(row.getMatch());
				}
				parent->getParent()->matchesKeyPointsSelected(matches);
			}
		}
		else
		{
			auto formats =
			    RawviewTableRow::getAvailableTextFormats();
			for (auto format : formats)
			{
				if (text == QString("Copy as %1").arg(format))
				{
					QString formattedRows;
					formattedRows =
					    RawviewTableRow::rowsToText(
					        rows, format, single);
					QApplication::clipboard()->setText(
					    formattedRows);
					break;
				}
			}
		}
	}
}

std::vector<cv::DMatch> RawviewGroupSubtable::getMatchSelection()
{
	std::vector<cv::DMatch> matches;
	for (RawviewTableRow row : getSelectedRows())
	{
		if (!row.hasSingleKeyPoint())
		{
			matches.push_back(row.getMatch());
		}
	}
	return matches;
}

std::vector<cv::KeyPoint> RawviewGroupSubtable::getKeyPointSelection()
{
	std::vector<cv::KeyPoint> keyPoints;
	for (RawviewTableRow row : getSelectedRows())
	{
		if (row.hasSingleKeyPoint())
		{
			keyPoints.push_back(row.getKeyPoint1());
		}
	}
	return keyPoints;
}

void RawviewGroupSubtable::setMatchSelection(std::vector<cv::DMatch> matches)
{
	std::set<int> indexes;
	for (size_t i = 0; i < group.size(); i++)
	{
		RawviewTableRow elem = group.get(i);
		if (!elem.hasSingleKeyPoint())
		{
			for (auto &match : matches)
			{
				if (match.distance == elem.matchDistance() &&
						match.imgIdx == elem.matchImgIdx() &&
						match.queryIdx == elem.matchQueryIdx() &&
						match.trainIdx == elem.matchTrainIdx())
				{
					indexes.insert(i);
					break;
				}
			}
		}
	}
	setSelectedRows(indexes);
}

void RawviewGroupSubtable::setKeyPointSelection(std::vector<cv::KeyPoint> keyPoints)
{
	std::set<int> indexes;
	for (size_t i = 0; i < group.size(); i++)
	{
		RawviewTableRow elem = group.get(i);
		if (elem.hasSingleKeyPoint())
		{
			for (auto &keyPoint : keyPoints)
			{
				if (keyPoint.pt.x == elem.keyPoint1XCoord() &&
						keyPoint.pt.y == elem.keyPoint1YCoord() &&
						keyPoint.size == elem.keyPoint1Size() &&
						keyPoint.angle == elem.keyPoint1Angle() &&
						keyPoint.response == elem.keyPoint1Response() &&
						keyPoint.octave == elem.keyPoint1Octave() &&
						keyPoint.class_id == elem.keyPoint1ClassId())
				{
					indexes.insert(i);
					break;
				}
			}
		}
	}
	setSelectedRows(indexes);
}

std::vector<RawviewTableRow> RawviewGroupSubtable::getSelectedRows()
{
	std::vector<RawviewTableRow> rows;
	for (auto index : currentRowIndexes)
	{
		if (index < qTable->rowCount() && index >= 0)
		{
			rows.push_back(group.get(index));
		}
	}
	return rows;
}

void RawviewGroupSubtable::setSelectedRows(std::set<int> rowIndexes)
{
	currentRowIndexes = rowIndexes;
	QTableWidgetSelectionRange clearSelectionRange(0, 0, qTable->rowCount(), qTable->columnCount());
	qTable->setRangeSelected(clearSelectionRange, false);
	for (int i : rowIndexes)
	{
		QTableWidgetSelectionRange range(i, 0, i + 1, qTable->columnCount());
		qTable->setRangeSelected(range, true);
	}
}

}
}
