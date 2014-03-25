#ifndef CVVISUAL_STFL_QUERY_WIDGET_LINEEDIT_HPP
#define CVVISUAL_STFL_QUERY_WIDGET_LINEEDIT_HPP

#include <QStringList>
#include <QLineEdit>
#include <QKeyEvent>
#include <QAbstractItemView>

#include "stfl_query_widget_completer.hpp"

namespace cvv
{
namespace qtutil
{

/**
 * @brief A line edit class, capable of showing suggestions.
 * @note It's heavily based on
 * http://www.qtcentre.org/archive/index.php/t-23518.html
 */
class STFLQueryWidgetLineEdit : public QLineEdit
{
	Q_OBJECT
      public:
	/**
	 * @brief Contructor of this class.
	 * @param parent widget.
	 */
	STFLQueryWidgetLineEdit(QWidget *parent = 0);

	/**
	 * @brief Gets the inherited completer.
	 * @return the inherited completer
	 */
	STFLQueryWidgetCompleter *completer();

	/**
	 * @brief Show the given suggestions in a list.
	 * @param suggestions given suggestions
	 */
	void showSuggestions(QStringList suggestions);

      protected:
	void keyPressEvent(QKeyEvent *e);

signals:
	/**
	 * @brief New suggestions are requested for the given user input.
	 * @param input given user input
	 */
	void requestSuggestions(QString input);

      private
slots:
	void insertCompletion(const QString &completion);

      private:
	STFLQueryWidgetCompleter *queryCompleter;
};
}
}
#endif
