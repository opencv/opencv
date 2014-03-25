#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <unordered_map>

#include <QComboBox>
#include <QLabel>
#include <QString>
#include <QVBoxLayout>

#include "../../util/util.hpp"
#include "diffFilterWidget.hpp"

namespace cvv
{
namespace qtutil
{

DiffFilterFunction::DiffFilterFunction(QWidget *parent)
    : FilterFunctionWidget<2, 1>{ parent },
      filterType_{ DiffFilterType::GRAYSCALE }
{
	auto layout = util::make_unique<QVBoxLayout>();
	auto comboBox = util::make_unique<QComboBox>();

	filterMap_.insert(
	    std::make_pair<std::string, std::function<void(void)>>(
	        "Hue", [this]()
	{ filterType_ = DiffFilterType::HUE; }));
	filterMap_.insert(
	    std::make_pair<std::string, std::function<void(void)>>(
	        "Saturation", [this]()
	{ filterType_ = DiffFilterType::SATURATION; }));
	filterMap_.insert(
	    std::make_pair<std::string, std::function<void(void)>>(
	        "Value", [this]()
	{ filterType_ = DiffFilterType::VALUE; }));
	filterMap_.insert(
	    std::make_pair<std::string, std::function<void(void)>>(
	        "Grayscale", [this]()
	{ filterType_ = DiffFilterType::GRAYSCALE; }));

	// Register filter names at comboBox
	comboBox->addItems(DiffFilterFunction::extractStringListfromMap());
	connect(comboBox.get(), SIGNAL(currentIndexChanged(const QString &)),
	        this, SLOT(updateFilterType(const QString &)));

	// Add title of comboBox and comboBox to the layout
	layout->addWidget(
	    util::make_unique<QLabel>("Select a filter").release());
	layout->addWidget(comboBox.release());
	setLayout(layout.release());
}

void DiffFilterFunction::applyFilter(InputArray in, OutputArray out) const
{

	auto check = checkInput(in);
	if (!check.first)
	{
		return;
	}

	if (filterType_ == DiffFilterType::GRAYSCALE)
	{
		out.at(0).get() = cv::abs(in.at(0).get() - in.at(1).get());
		return;
	}

	cv::Mat originalHSV, filteredHSV;
	cv::cvtColor(in.at(0).get(), originalHSV, CV_BGR2HSV);
	cv::cvtColor(in.at(1).get(), filteredHSV, CV_BGR2HSV);
	auto diffHSV = cv::abs(originalHSV - filteredHSV);

	std::array<cv::Mat, 3> splitVector;
	cv::split(diffHSV, splitVector.data());

	out.at(0).get() = splitVector.at(static_cast<size_t>(filterType_));

}

std::pair<bool, QString> DiffFilterFunction::checkInput(InputArray in) const
{

	if (in.at(0).get().size() != in.at(1).get().size())
	{
		return std::make_pair(false, "Images need to have same size");
	}

	size_t inChannels = in.at(0).get().channels();

	if (inChannels != static_cast<size_t>(in.at(1).get().channels()))
	{
		return std::make_pair(
		    false, "Images need to have same number of channels");
	}

	if (inChannels == 1 && filterType_ != DiffFilterType::GRAYSCALE)
	{
		return std::make_pair(false, "Images are grayscale, but "
		                             "selected Filter can only "
		                             "progress 3-channel images");
	}

	if (inChannels != 1 && inChannels != 3 && inChannels != 4)
	{
		return std::make_pair(
		    false, "Images must have one, three or four channels");
	}


	return std::make_pair(true, "Images can be converted");
}

QStringList DiffFilterFunction::extractStringListfromMap() const
{
	QStringList stringList{};
	for (auto mapElem : filterMap_)
	{
		stringList << QString::fromStdString(mapElem.first);
	}
	return stringList;
}

void DiffFilterFunction::updateFilterType(const QString &name)
{
	filterMap_.find(name.toStdString())->second();
	signalFilterSettingsChanged().emitSignal();
}
}
}
