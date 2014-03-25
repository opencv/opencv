#ifndef CVVISUAL_MANUAL_TEST_ACCORDION_ACCTESTER_HPP
#define CVVISUAL_MANUAL_TEST_ACCORDION_ACCTESTER_HPP

#include "../../src/qtutil/accordion.hpp"

#include <sstream>

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QApplication>

#define TST5                                                                   \
	if (!label5)                                                           \
	{                                                                      \
		msg.show();                                                    \
		return;                                                        \
	};

class Acctester : public QWidget
{
	Q_OBJECT
	// is required fot test to work
	// atm compiler error if used: CVVisual/manual_test/accordion/main.cpp:
	// The file contains a Q_OBJECT macro, but does not include "main.moc" !
	// if comment is removed it will break build
	// will look into it later
      public:
	Acctester(QWidget *parent = nullptr);

	cvv::qtutil::Accordion *acc;
	cvv::qtutil::Accordion::Handle label5;

	// errorlabel
	QLabel msg;

	std::string s(std::string at);

      public
slots:
	void sclear();

	void shideAll()
	{
		acc->hideAll();
	}
	void sshowAll()
	{
		acc->showAll();
	}
	void scollapseAll()
	{
		acc->collapseAll();
	}
	void sexpandAll()
	{
		acc->expandAll();
	}

	void spfront()
	{
		acc->push_front("0", cvv::util::make_unique<QLabel>("0"));
	}
	void spback()
	{
		acc->push_back("end", cvv::util::make_unique<QLabel>("end"));
	}
	void sinsert5();
	void sremove5();

	void shide5()
	{
		TST5 acc->hide(label5);
	}
	void sshow5()
	{
		TST5 acc->show(label5);
	}
	void sexpand5()
	{
		TST5 acc->expand(label5);
	}
	void scollapse5()
	{
		TST5 acc->collapse(label5);
	}
};

#endif
