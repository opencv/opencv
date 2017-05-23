#ifndef CVVISUAL_STFLQUERYWIDGET_HPP
#define CVVISUAL_STFLQUERYWIDGET_HPP

#include <QString>
#include <QLineEdit>
#include <QStringList>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>

#include "stfl_query_widget_lineedit.hpp"
#include "../util/util.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * @brief A simple filter widget with an input field and an help button.
 */
class STFLQueryWidget : public QWidget
{
	Q_OBJECT

      public:
	/**
	 * @brief Constructor of this class.
	 */
	STFLQueryWidget();

	/**
	 * @brief Show the given suggestions.
	 * @param suggestions given suggestions
	 */
	void showSuggestions(const QStringList &suggestions);

      private
slots:
	void returnPressed();

	void textChanged();

	void helpRequested();

signals:
	/**
	 * @brief User request filtering with the given query.
	 * @param query given query
	 */
	void filterSignal(QString query);

	/**
	 * @brief Update of the user input.
	 * @param query new user input
	 */
	void userInputUpdate(QString query);

	/**
	 * @brief User request suggestions for the given query.
	 * @param query given query
	 */
	void requestSuggestions(QString query);

	/**
	 * @brief User requests the help page for the given topic.
	 * @param topic given topic
	 */
	void showHelp(QString topic);

      private:
	STFLQueryWidgetLineEdit *lineEdit;
};
}
}

#endif
