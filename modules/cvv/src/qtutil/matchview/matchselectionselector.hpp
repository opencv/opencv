#ifndef CVVISUAL_MATCH_SELECTION_SELECTOR
#define CVVISUAL_MATCH_SELECTION_SELECTOR

#include <vector>

#include "opencv2/features2d/features2d.hpp"

#include "matchselection.hpp"
#include "../registerhelper.hpp"

namespace cvv{ namespace qtutil{

/**
 * @brief this class can use diffrent MatchSelection
 * you can register functions which take a std::vector<cv::DMatch> as argument.
 */
class MatchSelectionSelector:public MatchSelection,public RegisterHelper<MatchSelection,std::vector<cv::DMatch>>{

	Q_OBJECT

public:
	/**
	 * @brief the constructor
	 */
	MatchSelectionSelector(const std::vector<cv::DMatch>& univers,QWidget * parent=nullptr);

	/**
	 * @brief select matches of the given selection
	 * @return the selected matches
	 */
	std::vector<cv::DMatch> select(const std::vector<cv::DMatch>& selection);

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
	void remove(MatchSelectionSelector*);

private slots:

	/**
	 * @brief swap the current MatchSelection if the user choose another.
	 */
	virtual void changeSelector();

private:
	MatchSelection * selection_=nullptr;
	std::vector<cv::DMatch> univers_;
	QLayout *layout_;
};
template <class Selection>
bool registerMatchSelection(const QString &name)
{
	return MatchSelectionSelector::registerElement(
	    name, [](std::vector<cv::DMatch> univers)
	{
		    return std::unique_ptr<MatchSelection>{ new Selection{univers}};
	});
}
}}

#endif
