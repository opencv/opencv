
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QVBoxLayout>

#include "falsecolorkeypointpen.hpp"
#include "colorutil.hpp"
#include "cvvmatch.hpp"
#include "../../util/util.hpp"

namespace cvv
{
namespace qtutil
{

FalseColorKeyPointPen::FalseColorKeyPointPen(std::vector<cv::KeyPoint> univers, QWidget *parent)
    : KeyPointSettings{ parent },
      univers_{univers},
      maxDistance_{0.0},
      minDistance_{0.0}
{
	auto layout = util::make_unique<QVBoxLayout>();
	auto valueChooser=util::make_unique<KeyPointValueChooser>();
	auto button = util::make_unique<QPushButton>("use false color");
	valueChooser_=valueChooser.get();

	connect(valueChooser.get(),SIGNAL(valueChanged()),this,SLOT(updateMinMax()));
	connect(button.get(), SIGNAL(clicked()), this, SLOT(updateAll()));

	layout->setContentsMargins(0, 0, 0, 0);

	layout->addWidget(valueChooser.release());
	layout->addWidget(button.release());

	setLayout(layout.release());

	updateMinMax();
}

void FalseColorKeyPointPen::setSettings(CVVKeyPoint &key)
{
	QPen pen= key.getPen();
	QBrush brush=key.getBrush();
	pen.setColor(getFalseColor( valueChooser_->getChoosenValue(key.keyPoint()), maxDistance_, minDistance_) );
	brush.setColor(getFalseColor( valueChooser_->getChoosenValue(key.keyPoint()), maxDistance_, minDistance_) );
	key.setPen(pen);
	key.setBrush(brush);
}

void FalseColorKeyPointPen::updateMinMax()
{
	maxDistance_=0.0;
	minDistance_=0.0;
	for(auto& key:univers_){
		maxDistance_=std::max(maxDistance_,valueChooser_->getChoosenValue(key));
		//minDistance_=std::max(minDistance_,valueChooser_->getChoosenValue(key));
	}
	updateAll();
}

}
}
