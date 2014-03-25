#include "overview_panel.hpp"

#include <functional>
#include <math.h>
#include <memory>
#include <iostream>

#include <QMap>
#include <QSet>
#include <QString>
#include <QVBoxLayout>
#include <QWidget>
#include <QScrollArea>

#include "../controller/view_controller.hpp"
#include "../qtutil/stfl_query_widget.hpp"
#include "../qtutil/util.hpp"
#include "../stfl/element_group.hpp"

namespace cvv
{
namespace gui
{

OverviewPanel::OverviewPanel(
    util::Reference<controller::ViewController> controller)
    : controller{ controller }
{
	qtutil::setDefaultSetting("overview", "imgzoom", "20");
	QVBoxLayout *layout = new QVBoxLayout{};
	setLayout(layout);
	layout->setContentsMargins(0, 0, 0, 0);
	queryWidget = new qtutil::STFLQueryWidget();
	layout->addWidget(queryWidget);
	table = new OverviewTable(controller);
	layout->addWidget(table);

	auto bottomArea = new QWidget{ this };
	auto bottomLayout = new QHBoxLayout;
	imgSizeSliderLabel = new QLabel{ "Zoom", bottomArea };
	imgSizeSliderLabel->setMaximumWidth(50);
	imgSizeSliderLabel->setAlignment(Qt::AlignRight);
	bottomLayout->addWidget(imgSizeSliderLabel);
	imgSizeSlider = new QSlider{ Qt::Horizontal, bottomArea };
	imgSizeSlider->setMinimumWidth(50);
	imgSizeSlider->setMaximumWidth(200);
	imgSizeSlider->setMinimum(0);
	imgSizeSlider->setMaximum(100);
	imgSizeSlider->setSliderPosition(
	    qtutil::getSetting("overview", "imgzoom").toInt());
	connect(imgSizeSlider, SIGNAL(valueChanged(int)), this,
	        SLOT(imgSizeSliderAction()));
	bottomLayout->addWidget(imgSizeSlider);
	bottomArea->setLayout(bottomLayout);
	layout->addWidget(bottomArea);

	initEngine();
	connect(queryWidget, SIGNAL(showHelp(QString)), this,
	        SLOT(showHelp(QString)));
	// connect(queryWidget, SIGNAL(userInputUpdate(QString)), this,
	// SLOT(updateQuery(QString)));
	connect(queryWidget, SIGNAL(filterSignal(QString)), this,
	        SLOT(filterQuery(QString)));
	connect(queryWidget, SIGNAL(requestSuggestions(QString)), this,
	        SLOT(requestSuggestions(QString)));
}

void OverviewPanel::initEngine()
{
	// raw and description filter
	auto rawFilter = [](const OverviewTableRow &elem)
	{ 
		return elem.description(); 
	};
	queryEngine.addStringCmdFunc("raw", rawFilter, false);
	queryEngine.addStringCmdFunc("description", rawFilter, false);

	// file filter
	queryEngine.addStringCmdFunc("file", [](const OverviewTableRow &elem)
	{ 
		return elem.file(); 
	});

	// function filter
	queryEngine.addStringCmdFunc("function",
	                             [](const OverviewTableRow &elem)
	{ 
		return elem.function(); 
	});

	// line filter
	queryEngine.addIntegerCmdFunc("line", [](const OverviewTableRow &elem)
	{ return elem.line(); });

	// id filter
	queryEngine.addIntegerCmdFunc("id", [](const OverviewTableRow &elem)
	{ 
		return elem.id(); 
	});

	// type filter
	queryEngine.addStringCmdFunc("type", [](const OverviewTableRow &elem)
	{ 
		return elem.type(); 
	});

	//"number of images" filter
	queryEngine.addIntegerCmdFunc("image_count",
	                              [](const OverviewTableRow &elem)
	{ 
		return elem.call()->matrixCount(); 
	});
	
	//additional commands
	
	//open call command
	queryEngine.addAdditionalCommand("open", 
		[&](QStringList args, std::vector<stfl::ElementGroup<OverviewTableRow>>& groups)
	{
		openCommand(args, groups);
	}, {"first_of_group", "last_of_group", "shown"});
}

void OverviewPanel::openCommand(QStringList args,
	std::vector<stfl::ElementGroup<OverviewTableRow>>& groups)
{
	if (args.contains("shown"))
	{
		for (auto &group : groups)
		{
			for (auto &elem : group.getElements())
			{
				controller->openCallTab(elem.id());
			}
		}
		return;
	}
	if (args.contains("first_of_group") || args.contains("last_of_group"))
	{
		bool first = args.contains("first_of_group") ;
		for (auto &group : groups)
		{
			if (group.size() > 0)
			{
				size_t index = first ? 0 : group.size() - 1;
				size_t id = group.get(index).id();
				controller->openCallTab(id);
			}
		} 
	}
}
		

void OverviewPanel::addElement(const util::Reference<const impl::Call> newCall)
{
	OverviewTableRow row(newCall);
	queryEngine.addNewElement(row);
	table->updateRowGroups(queryEngine.reexecuteLastQuery());
}

void OverviewPanel::addElementBuffered(const util::Reference<const impl::Call> newCall)
{
	elementBuffer.push_back(newCall);
}

void OverviewPanel::flushElementBuffer()
{
	std::vector<OverviewTableRow> rows;
	for (const util::Reference<const impl::Call> call : elementBuffer)
	{
		rows.push_back(OverviewTableRow(call));
	}
	queryEngine.addElements(std::move(rows));
	table->updateRowGroups(queryEngine.reexecuteLastQuery());
	elementBuffer.clear();
}

void OverviewPanel::removeElement(size_t id)
{
	queryEngine.removeElements([id](OverviewTableRow elem)
	{ 
		return elem.id() == id; 
	});
	table->removeElement(id);
}

void OverviewPanel::filterQuery(QString query)
{
	table->updateRowGroups(queryEngine.query(query));
}

void OverviewPanel::updateQuery(QString query)
{
	filterQuery(query);
}

void OverviewPanel::requestSuggestions(QString query)
{
	queryWidget->showSuggestions(queryEngine.getSuggestions(query));
}

void OverviewPanel::imgSizeSliderAction()
{
	controller->setSetting("overview", "imgzoom",
	                       QString::number(imgSizeSlider->value()));
	table->updateUI();
}

void OverviewPanel::showHelp(QString topic)
{
	controller->openHelpBrowser(topic);
}
}
}
