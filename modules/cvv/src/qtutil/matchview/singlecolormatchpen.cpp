
#include <QPushButton>
#include <QVBoxLayout>

#include "singlecolormatchpen.hpp"

namespace cvv
{
namespace qtutil
{

SingleColorMatchPen::SingleColorMatchPen(std::vector<cv::DMatch>, QWidget *parent)
    : MatchSettings{ parent },
      color_(Qt::red)
{
	colorDialog_ = new QColorDialog{}; // wird im Destructor zerstört
	auto layout = util::make_unique<QVBoxLayout>();
	auto button = util::make_unique<QPushButton>("Change Color");

	connect(colorDialog_, SIGNAL(currentColorChanged(const QColor &)), this,
		SLOT(updateColor(const QColor &)));

	connect(button.get(), SIGNAL(clicked(bool)), this,
		SLOT(colorButtonClicked()));

	layout->setMargin(0);
	layout->addWidget(button.release());

	setLayout(layout.release());
}

void SingleColorMatchPen::setSettings(CVVMatch &match)
{
	QPen pen=match.getPen();
	pen.setColor(color_);
	match.setPen(pen);
}

void SingleColorMatchPen::updateColor(const QColor &color)
{
	color_ = color;
	emit settingsChanged(*this);
}

}
}
