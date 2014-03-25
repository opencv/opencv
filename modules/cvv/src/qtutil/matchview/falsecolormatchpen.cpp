
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QVBoxLayout>

#include "falsecolormatchpen.hpp"
#include "colorutil.hpp"
#include "cvvmatch.hpp"
#include "../../util/util.hpp"

namespace cvv
{
namespace qtutil
{

FalseColorMatchPen::FalseColorMatchPen(std::vector<cv::DMatch> univers, QWidget *parent)
    : MatchSettings{parent},
      maxDistance_{0.0},
      minDistance_{0.0}
{
	auto layout = util::make_unique<QVBoxLayout>();
	auto button = util::make_unique<QPushButton>("use false color");

	for(auto& match:univers){
		maxDistance_=std::max(maxDistance_,static_cast<double>(match.distance));
		//min_=std::max(min_,static_cast<double>(match.distance));
	}

	connect(button.get(), SIGNAL(clicked()), this, SLOT(updateAll()));

	layout->addWidget(button.release());

	setLayout(layout.release());
}

void FalseColorMatchPen::setSettings(CVVMatch &match)
{
	QPen pen= match.getPen();
	pen.setColor(getFalseColor( static_cast<double>(match.match().distance), maxDistance_, minDistance_) );
	match.setPen(pen);
}

}
}
