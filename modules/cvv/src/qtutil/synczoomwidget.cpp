
#include <QVBoxLayout>
#include <QRadioButton>
#include <QLabel>

#include "synczoomwidget.hpp"
#include "../util/util.hpp"

namespace cvv
{
namespace qtutil
{

cvv::qtutil::SyncZoomWidget::SyncZoomWidget(
    std::vector<cvv::qtutil::ZoomableImage *> images, QWidget *parent)
    : QWidget(parent), images_{ images }, currentIdx_{ images_.size() },
      buttonGroup_{ new QButtonGroup }
{
	if (images_.size() >= 2)
	{

		auto layout = util::make_unique<QVBoxLayout>();
		auto label = util::make_unique<QLabel>("choose 'master' image");
		auto none = util::make_unique<QRadioButton>("no sync");

		buttonGroup_->setExclusive(true);
		none->setChecked(true);
		buttonGroup_->addButton(none.get(), images.size());

		layout->addWidget(label.release());
		layout->addWidget(none.release());

		for (size_t i = 0; i < images_.size(); i++)
		{

			auto checkbox = util::make_unique<QRadioButton>(QString
			{ "Image Nr. %1" }.arg(i));

			buttonGroup_->addButton(checkbox.get(), i);

			layout->addWidget(checkbox.release());

			connect(this, SIGNAL(updateArea(QRectF, qreal)),
				images_.at(i), SLOT(setArea(QRectF, qreal)));
		}
		connect(buttonGroup_, SIGNAL(buttonClicked(int)), this,
			SLOT(selectMaster(int)));

		setLayout(layout.release());
	}
}

void SyncZoomWidget::selectMaster(int id)
{
	if (currentIdx_ < images_.size())
	{
		disconnect(images_.at(currentIdx_),
			   SIGNAL(updateArea(QRectF, qreal)), this,
			   SIGNAL(updateArea(QRectF, qreal)));

		connect(this, SIGNAL(updateArea(QRectF, qreal)),
			images_.at(currentIdx_), SLOT(setArea(QRectF, qreal)));
	}
	currentIdx_ = id;
	if (currentIdx_ < images_.size())
	{
		disconnect(this, SIGNAL(updateArea(QRectF, qreal)),
			   images_.at(currentIdx_),
			   SLOT(setArea(QRectF, qreal)));

		connect(images_.at(currentIdx_),
			SIGNAL(updateArea(QRectF, qreal)), this,
			SIGNAL(updateArea(QRectF, qreal)));
	}
}
}
}
