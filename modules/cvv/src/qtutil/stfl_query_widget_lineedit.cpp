#include "stfl_query_widget_lineedit.hpp"


namespace cvv
{
namespace qtutil
{

STFLQueryWidgetLineEdit::STFLQueryWidgetLineEdit(QWidget *parent)
    : QLineEdit(parent), queryCompleter(new STFLQueryWidgetCompleter(this))
{
	queryCompleter->setWidget(this);
	connect(queryCompleter, SIGNAL(activated(const QString &)), this,
	        SLOT(insertCompletion(const QString &)));
}

STFLQueryWidgetCompleter *STFLQueryWidgetLineEdit::completer()
{
	return queryCompleter;
}

void STFLQueryWidgetLineEdit::insertCompletion(const QString &completion)
{
	setText(completion);
	selectAll();
}

void STFLQueryWidgetLineEdit::keyPressEvent(QKeyEvent *e)
{
	if (queryCompleter->popup()->isVisible())
	{
		// The following keys are forwarded by the completer to the
		// widget
		switch (e->key())
		{
		case Qt::Key_Escape:
		case Qt::Key_Backtab:
			e->ignore();
			return; // Let the completer do default behavior
		case Qt::Key_Tab:
			QLineEdit::keyPressEvent(new QKeyEvent(
			    e->type(), Qt::DownArrow, e->modifiers()));
			e->ignore();
			return;
		}
	}

	bool isShortcut =
	    (e->modifiers() & Qt::ControlModifier) && e->key() == Qt::Key_E;
	if (!isShortcut)
		QLineEdit::keyPressEvent(
		    e); // Don't send the shortcut (CTRL-E) to the text edit.

	bool ctrlOrShift =
	    e->modifiers() & (Qt::ControlModifier | Qt::ShiftModifier);
	if (!isShortcut && !ctrlOrShift && e->modifiers() != Qt::NoModifier)
	{
		queryCompleter->popup()->hide();
		return;
	}

	requestSuggestions(text());
}

void STFLQueryWidgetLineEdit::showSuggestions(QStringList suggestions)
{
	queryCompleter->update(suggestions);
	queryCompleter->popup()->setCurrentIndex(
	    queryCompleter->completionModel()->index(0, 0));
}
}
}
