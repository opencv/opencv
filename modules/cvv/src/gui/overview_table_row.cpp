#include "overview_table_row.hpp"

#include <algorithm>
#include <memory>

#include <QTableWidgetItem>
#include <QImage>

#include "../qtutil/util.hpp"
#include "../stfl/stringutils.hpp"

namespace cvv
{
namespace gui
{

OverviewTableRow::OverviewTableRow(util::Reference<const impl::Call> call)
    : call_{ call }
{
	id_ = call_->getId();
	idStr = QString::number(call_->getId());
	for (size_t i = 0; i < 2 && i < call->matrixCount(); i++)
	{
		QPixmap img;
		std::tie(std::ignore, img) =
		    qtutil::convertMatToQPixmap(call->matrixAt(i));
		imgs.push_back(std::move(img));
	}
	description_ = QString(call_->description());
	if (call_->metaData().isKnown)
	{
		const auto &data = call_->metaData();
		line_ = data.line;
		lineStr = QString::number(data.line);
		fileStr = data.file;
		functionStr = data.function;
	}
	typeStr = QString(call_->type());
}

void OverviewTableRow::addToTable(QTableWidget *table, size_t row,
                                  bool showImages, size_t maxImages,
                                  int imgHeight, int imgWidth)
{
	std::vector<std::unique_ptr<QTableWidgetItem>> items{};
	items.push_back(util::make_unique<QTableWidgetItem>(idStr));
	if (showImages)
	{
		for (size_t i = 0; i < imgs.size() && i < maxImages; i++)
		{
			auto imgWidget = util::make_unique<QTableWidgetItem>("");
			imgWidget->setData(
			    Qt::DecorationRole,
			    imgs.at(i).scaled(imgHeight, imgWidth,
			                      Qt::KeepAspectRatio,
			                      Qt::SmoothTransformation));
			imgWidget->setTextAlignment(Qt::AlignHCenter);
			items.push_back(std::move(imgWidget));
		}
	}

	size_t emptyImagesToAdd =
	    showImages ? maxImages - std::min(maxImages, imgs.size())
	               : maxImages;

	for (size_t i = 0; i < emptyImagesToAdd; i++)
	{
		items.push_back(util::make_unique<QTableWidgetItem>(""));
	}

	items.push_back(util::make_unique<QTableWidgetItem>(description_));
	items.push_back(util::make_unique<QTableWidgetItem>(functionStr, 30));
	items.push_back(util::make_unique<QTableWidgetItem>(fileStr));
	items.push_back(util::make_unique<QTableWidgetItem>(lineStr));
	items.push_back(util::make_unique<QTableWidgetItem>(typeStr));
	for (size_t i = 0; i < items.size(); i++)
	{
		items[i]->setFlags(items[i]->flags() ^ Qt::ItemIsEditable);
		table->setItem(row, i, items[i].release());
	}
}

void OverviewTableRow::resizeInTable(QTableWidget *table, size_t row,
                                  bool showImages, size_t maxImages,
                                  int imgHeight, int imgWidth)
{
	if (showImages)
	{
		for (size_t i = 0; i < imgs.size() && i < maxImages; i++)
		{
			auto imgWidget = util::make_unique<QTableWidgetItem>("");
			imgWidget->setData(
			    Qt::DecorationRole,
			    imgs.at(i).scaled(imgHeight, imgWidth,
			                      Qt::KeepAspectRatio,
			                      Qt::SmoothTransformation));
			imgWidget->setTextAlignment(Qt::AlignHCenter);
			imgWidget->setFlags(imgWidget->flags() ^ Qt::ItemIsEditable);
			table->setItem(row, i + 1, imgWidget.release());
		}
	}
}
}
}
