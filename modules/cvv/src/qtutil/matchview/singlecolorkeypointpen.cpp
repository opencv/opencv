#include <QHBoxLayout>
#include <QPushButton>

#include "singlecolorkeypointpen.hpp"
#include "../../util/util.hpp"

namespace cvv
{
namespace qtutil
{

SingleColorKeyPen::SingleColorKeyPen(std::vector<cv::KeyPoint>, QWidget *parent)
    : KeyPointSettings{ parent }, colordia_{ new QColorDialog{} }
{
	auto layout = util::make_unique<QVBoxLayout>();
	auto button = util::make_unique<QPushButton>("Color Dialog");

	layout->setMargin(0);

	connect(colordia_, SIGNAL(currentColorChanged(const QColor &)), this,
		SLOT(updateColor(const QColor &)));
	connect(button.get(), SIGNAL(clicked(bool)), this,
		SLOT(colorButtonClicked()));

	layout->addWidget(button.release());
	setLayout(layout.release());
}

void SingleColorKeyPen::setSettings(CVVKeyPoint& keypoint)
{
	QPen pen=keypoint.getPen();
	pen.setColor(color_);
	keypoint.setPen(pen);

	QBrush brush=keypoint.getBrush();
	brush.setColor(color_);
	keypoint.setBrush(brush);
}

void SingleColorKeyPen::updateColor(const QColor &color)
{
	color_ = color;
	emit settingsChanged(*this);
}
}
}
