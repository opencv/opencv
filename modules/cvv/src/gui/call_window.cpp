#include "call_window.hpp"

#include <QMenu>
#include <QStatusBar>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVariant>

#include "../stfl/stringutils.hpp"

namespace cvv
{

namespace controller
{
class ViewController;
}

namespace gui
{

CallWindow::CallWindow(util::Reference<controller::ViewController> controller,
                       size_t id)
    : id{ id }, controller{ controller }
{
	initTabs();
	initFooter();
	setWindowTitle(QString("CVVisual | window no. %1").arg(id));
	setMinimumWidth(600);
	setMinimumHeight(600);
}

void CallWindow::initTabs()
{
	tabWidget = new TabWidget(this);
	tabWidget->setTabsClosable(true);
	tabWidget->setMovable(true);
	setCentralWidget(tabWidget);

	auto *flowButtons = new QHBoxLayout();
	auto *flowButtonsWidget = new QWidget(this);
	tabWidget->setCornerWidget(flowButtonsWidget, Qt::TopLeftCorner);
	flowButtonsWidget->setLayout(flowButtons);
	flowButtons->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	closeButton = new QPushButton("Close", this);
	flowButtons->addWidget(closeButton);
	closeButton->setStyleSheet(
	    "QPushButton {background-color: red; color: white;}");
	closeButton->setToolTip("Close this debugging application.");
	connect(closeButton, SIGNAL(clicked()), this, SLOT(closeApp()));
	fastForwardButton = new QPushButton(">>", this);
	flowButtons->addWidget(fastForwardButton);
	fastForwardButton->setStyleSheet(
	    "QPushButton {background-color: yellow; color: blue;}");
	fastForwardButton->setToolTip(
	    "Fast forward until cvv::finalCall() gets called.");
	connect(fastForwardButton, SIGNAL(clicked()), this,
	        SLOT(fastForward()));
	stepButton = new QPushButton("Step", this);
	flowButtons->addWidget(stepButton);
	stepButton->setStyleSheet(
	    "QPushButton {background-color: green; color: white;}");
	stepButton->setToolTip(
	    "Resume program execution for a next debugging step.");
	connect(stepButton, SIGNAL(clicked()), this, SLOT(step()));
	flowButtons->setContentsMargins(0, 0, 0, 0);
	flowButtons->setSpacing(0);

	auto *tabBar = tabWidget->getTabBar();
	tabBar->setElideMode(Qt::ElideRight);
	tabBar->setContextMenuPolicy(Qt::CustomContextMenu);
	connect(tabBar, SIGNAL(customContextMenuRequested(QPoint)), this,
	        SLOT(contextMenuRequested(QPoint)));
	connect(tabBar, SIGNAL(tabCloseRequested(int)), this,
	        SLOT(tabCloseRequested(int)));
}

void CallWindow::initFooter()
{
	leftFooter = new QLabel();
	rightFooter = new QLabel();
	QStatusBar *bar = statusBar();
	bar->addPermanentWidget(leftFooter, 2);
	bar->addPermanentWidget(rightFooter, 2);
}

void CallWindow::showExitProgramButton()
{
	stepButton->setVisible(false);
	fastForwardButton->setVisible(false);
}

void CallWindow::addTab(CallTab *tab)
{
	tabMap[tab->getId()] = tab;
	QString name = QString("[%1] %2").arg(tab->getId()).arg(tab->getName());
	int index =
	    tabWidget->addTab(tab, stfl::shortenString(name, 20, true, true));
	tabWidget->getTabBar()->setTabData(index, QVariant((int)tab->getId()));
}

size_t CallWindow::getId()
{
	return id;
}

void CallWindow::removeTab(CallTab *tab)
{
	tabMap.erase(tabMap.find(tab->getId()));
	int index = tabWidget->indexOf(tab);
	tabWidget->removeTab(index);
}

void CallWindow::removeTab(size_t tabId)
{
	if (hasTab(tabId))
	{
		removeTab(tabMap[tabId]);
	}
}

void CallWindow::showTab(CallTab *tab)
{
	tabWidget->setCurrentWidget(tab);
}

void CallWindow::showTab(size_t tabId)
{
	if (hasTab(tabId))
	{
		showTab(tabMap[tabId]);
	}
}

void CallWindow::updateLeftFooter(QString newText)
{
	leftFooter->setText(newText);
}

void CallWindow::updateRightFooter(QString newText)
{
	rightFooter->setText(newText);
}

void CallWindow::step()
{
	controller->resumeProgramExecution();
}

void CallWindow::fastForward()
{
	controller->setMode(controller::Mode::FAST_FORWARD);
}

void CallWindow::closeApp()
{
	controller->setMode(controller::Mode::HIDE);
}

bool CallWindow::hasTab(size_t tabId)
{
	return tabMap.count(tabId);
}

void CallWindow::contextMenuRequested(const QPoint &location)
{
	controller->removeEmptyWindows();
	auto tabBar = tabWidget->getTabBar();
	int tabIndex = tabBar->tabAt(location);
	if (tabIndex == tabOffset - 1)
		return;
	QMenu *menu = new QMenu(this);
	connect(menu, SIGNAL(triggered(QAction *)), this,
	        SLOT(contextMenuAction(QAction *)));
	auto windows = controller->getTabWindows();
	menu->addAction(new QAction("Remove call", this));
	menu->addAction(new QAction("Close tab", this));
	menu->addAction(new QAction("Open in new window", this));
	for (auto window : windows)
	{
		if (window->getId() != id)
		{
			menu->addAction(new QAction(
			    QString("Open in '%1'").arg(window->windowTitle()),
			    this));
		}
	}
	currentContextMenuTabId = getCallTabIdByTabIndex(tabIndex);
	menu->popup(tabBar->mapToGlobal(location));
}

void CallWindow::contextMenuAction(QAction *action)
{
	if (currentContextMenuTabId == -1)
	{
		return;
	}
	auto text = action->text();
	if (text == "Open in new window")
	{
		controller->moveCallTabToNewWindow(currentContextMenuTabId);
	}
	else if (text == "Remove call")
	{
		controller->removeCallTab(currentContextMenuTabId, true, true);
	}
	else if (text == "Close tab")
	{
		controller->removeCallTab(currentContextMenuTabId);
	}
	else
	{
		auto windows = controller->getTabWindows();
		for (auto window : windows)
		{
			if (text ==
			    QString("Open in '%1'").arg(window->windowTitle()))
			{
				controller->moveCallTabToWindow(
				    currentContextMenuTabId, window->getId());
				break;
			}
		}
	}
	currentContextMenuTabId = -1;
}

size_t CallWindow::tabCount()
{
	return tabMap.size();
}

std::vector<size_t> CallWindow::getCallTabIds()
{
	std::vector<size_t> ids{};
	for (auto &elem : tabMap)
	{
		ids.push_back(elem.first);
	}
	return ids;
}

void CallWindow::closeEvent(QCloseEvent *event)
{
	controller->removeWindowFromMaps(id);
	// FIXME: tabWidget is already freed sometimes: Use-after-free Bug
	tabWidget->clear();
	for (auto &elem : tabMap)
	{
		controller->removeCallTab(elem.first, true);
	}
	event->accept();
}

void CallWindow::tabCloseRequested(int index)
{
	if (hasTabAtIndex(index))
	{
		controller->removeCallTab(getCallTabIdByTabIndex(index));
	}
	controller->removeEmptyWindows();
}

size_t CallWindow::getCallTabIdByTabIndex(int index)
{
	if (hasTabAtIndex(index))
	{
		auto tabData = tabWidget->getTabBar()->tabData(index);
		bool ok = true;
		size_t callTabId = tabData.toInt(&ok);
		if (ok && tabMap.count(callTabId) > 0)
		{
			return callTabId;
		}
	}
	return 0;
}

bool CallWindow::hasTabAtIndex(int index)
{
	auto tabData = tabWidget->getTabBar()->tabData(index);
	return tabData != 0 && !tabData.isNull() && tabData.isValid();
}
}
}
