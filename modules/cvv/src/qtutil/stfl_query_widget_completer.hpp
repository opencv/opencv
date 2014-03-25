#ifndef CVVISUAL_STFL_QUERY_WIDGET_COMPLETER_HPP
#define CVVISUAL_STFL_QUERY_WIDGET_COMPLETER_HPP

#include <QStringList>
#include <QStringListModel>
#include <QString>
#include <QCompleter>

namespace cvv
{
namespace qtutil
{

/**
 * @brief A simple completer for the query widget.
 */
class STFLQueryWidgetCompleter : public QCompleter
{
	Q_OBJECT

      public:
	/**
	 * @brief Constructor of this class.
	 * @param parent widget
	 */
	STFLQueryWidgetCompleter(QObject *parent) : QCompleter(parent), model()
	{
		setModel(&model);
	}

	/**
	 * @brief Update the inherited model with the given suggestions.
	 * @param suggestions given suggestions
	 */
	void update(QStringList suggestions)
	{
		model.setStringList(suggestions);
		complete();
	}

      private:
	QStringListModel model;
};
}
}
#endif
