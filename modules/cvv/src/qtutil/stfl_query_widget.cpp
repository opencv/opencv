#include "stfl_query_widget.hpp"


namespace cvv
{
namespace qtutil
{

STFLQueryWidget::STFLQueryWidget()
{
	lineEdit = new STFLQueryWidgetLineEdit(this);
	auto *layout = new QHBoxLayout;
	layout->addWidget(lineEdit);
	auto helpButton = new QPushButton("Help", this);
	layout->addWidget(helpButton);
	setLayout(layout);
	connect(helpButton, SIGNAL(released()), this, SLOT(helpRequested()));
	connect(lineEdit, SIGNAL(returnPressed()), this, SLOT(returnPressed()));
	connect(lineEdit, SIGNAL(textChanged(QString)), this,
	        SLOT(textChanged()));
	connect(lineEdit, SIGNAL(requestSuggestions(QString)), this,
	        SIGNAL(requestSuggestions(QString)));
}

void STFLQueryWidget::showSuggestions(const QStringList &suggestions)
{
	lineEdit->showSuggestions(suggestions);
}

void STFLQueryWidget::returnPressed()
{
	filterSignal(lineEdit->text());
}

void STFLQueryWidget::textChanged()
{
	userInputUpdate(lineEdit->text());
	requestSuggestions(lineEdit->text());
}

void STFLQueryWidget::helpRequested()
{
	showHelp("filterquery");
}
}
}
