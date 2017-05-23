
#include <QHBoxLayout>
#include <QLabel>

#include "keypointvaluechooser.hpp"
#include "../../util/util.hpp"

namespace cvv{ namespace qtutil{

KeyPointValueChooser::KeyPointValueChooser(QWidget *parent):
	QWidget{parent}
{
	auto layout=util::make_unique<QHBoxLayout>();
	auto comb=util::make_unique<QComboBox>();
	auto label=util::make_unique<QLabel>("choose a value");


	combBox_=comb.get();
	comb->addItem("size");
	comb->addItem("angle");
	comb->addItem("response");
	comb->addItem("octave");
	comb->addItem("class_id");

	connect(comb.get(),SIGNAL(currentIndexChanged(int)),this,SIGNAL(valueChanged()));

	layout->setContentsMargins(0, 0, 0, 0);

	layout->addWidget(label.release());
	layout->addWidget(comb.release());

	setLayout(layout.release());
}

double KeyPointValueChooser::getChoosenValue(cv::KeyPoint keypoint)
{
	switch(combBox_->currentIndex()){
		case 0:
			return static_cast<double>(keypoint.size);
		case 1:
			return static_cast<double>(keypoint.angle);
		case 2:
			return static_cast<double>(keypoint.response);
		case 3:
			return static_cast<double>(keypoint.octave);
		case 4:
			return static_cast<double>(keypoint.class_id);
		default:
			return -1.0;
	}
}


}}
