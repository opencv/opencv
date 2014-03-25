#include "acctester.hpp"

#include "../../src/qtutil/accordion.hpp"

Acctester::Acctester(QWidget *parent)
    : QWidget{ parent }, acc{ new cvv::qtutil::Accordion{} }, label5{ nullptr },
      msg{ "no last element inserted at 5. (maybe already deleted)" }
{
	// create
	QPushButton *pfront = new QPushButton{ "pfront" };
	QPushButton *pback = new QPushButton{ "pback" };
	QPushButton *insert5 = new QPushButton{ "insert elem at pos 5" };
	QPushButton *remove5 =
	    new QPushButton{ "remove last element inserted at 5" };
	QPushButton *clear = new QPushButton{ "clear" };
	QPushButton *hideAll = new QPushButton{ "hideAll" };
	QPushButton *hide5 = new QPushButton{ "hide5" };
	QPushButton *showAll = new QPushButton{ "showAll" };
	QPushButton *show5 = new QPushButton{ "show5" };
	QPushButton *collapseAll = new QPushButton{ "collapseAll" };
	QPushButton *collapse5 = new QPushButton{ "collapse5" };
	QPushButton *expandAll = new QPushButton{ "expandAll" };
	QPushButton *expand5 = new QPushButton{ "expand5" };

	QVBoxLayout *l = new QVBoxLayout{};

	// build
	l->addWidget(pfront);
	l->addWidget(pback);
	l->addWidget(insert5);
	l->addWidget(remove5);
	l->addWidget(clear);
	l->addWidget(hideAll);
	l->addWidget(hide5);
	l->addWidget(showAll);
	l->addWidget(show5);
	l->addWidget(collapseAll);
	l->addWidget(collapse5);
	l->addWidget(expandAll);
	l->addWidget(expand5);
	l->addWidget(acc);

	// connect
	connect(pfront, SIGNAL(pressed()), this, SLOT(spfront()));
	connect(pback, SIGNAL(pressed()), this, SLOT(spback()));
	connect(clear, SIGNAL(pressed()), this, SLOT(sclear()));
	connect(hideAll, SIGNAL(pressed()), this, SLOT(shideAll()));
	connect(showAll, SIGNAL(pressed()), this, SLOT(sshowAll()));
	connect(collapseAll, SIGNAL(pressed()), this, SLOT(scollapseAll()));
	connect(expandAll, SIGNAL(pressed()), this, SLOT(sexpandAll()));

	connect(insert5, SIGNAL(pressed()), this, SLOT(sinsert5()));
	connect(remove5, SIGNAL(pressed()), this, SLOT(sremove5()));
	connect(hide5, SIGNAL(pressed()), this, SLOT(shide5()));
	connect(show5, SIGNAL(pressed()), this, SLOT(sshow5()));
	connect(collapse5, SIGNAL(pressed()), this, SLOT(scollapse5()));
	connect(expand5, SIGNAL(pressed()), this, SLOT(sexpand5()));

	setLayout(l);
}

std::string Acctester::s(std::string at)
{
	std::stringstream s;
	s << "inserted at " << at << "\t nr: " << acc->size();
	return s.str();
}

void Acctester::sinsert5()
{
	label5 = acc->insert("5", cvv::util::make_unique<QLabel>("5"), true, 4);
}
void Acctester::sremove5()
{
	TST5 acc->remove(label5);
	label5 = nullptr;
}

void Acctester::sclear()
{
	acc->clear();
	label5 = nullptr;
}
