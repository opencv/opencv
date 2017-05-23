#ifndef CVVISUAL_KEY_POINT_SELECTION_SELECTOR
#define CVVISUAL_KEY_POINT_SELECTION_SELECTOR

#include <vector>

#include "opencv2/features2d/features2d.hpp"

#include "keypointselection.hpp"
#include "../registerhelper.hpp"

namespace cvv{ namespace qtutil{

/**
 * @brief this class can use diffrent KeyPointSelection
 * you can register functions which take a std::vector<cv::KeyPoint> as argument.
 */
class KeyPointSelectionSelector:public KeyPointSelection,public RegisterHelper<KeyPointSelection,std::vector<cv::KeyPoint>>{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 */
	KeyPointSelectionSelector(const std::vector<cv::KeyPoint>& univers,QWidget * parent=nullptr);

	/**
	 * @brief select keypoint of the given selection
	 * @return the selected matches
	 */
	std::vector<cv::KeyPoint> select(const std::vector<cv::KeyPoint>& selection);

public slots:
	/**
	 * @brief emits the signal remove with this.
	 */
	void removeMe()
		{emit remove(this);}

signals:
	/**
	 * @brief this signal contains a KeyPointSelectionSelector which should be removed. Normally the argumen is this.
	 */
	void remove(KeyPointSelectionSelector*);

private slots:

	/**
	 * @brief swap the current KeyPointSelection if the user choose another.
	 */
	virtual void changeSelector();

private:
	KeyPointSelection * selection_=nullptr;
	std::vector<cv::KeyPoint> univers_;
	QLayout *layout_;

};//end class

template <class Selection>
bool registerKeyPointSelection(const QString &name)
{
	return KeyPointSelectionSelector::registerElement(
	    name, [](std::vector<cv::KeyPoint> univers)
	{
		    return std::unique_ptr<KeyPointSelection>{ new Selection{univers}};
	});
}
}}

#endif
