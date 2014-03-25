#include "../../src/qtutil/registerhelper.hpp"

#include <QApplication>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <iostream>
#include "../../src/qtutil/signalslot.hpp"
#include "../../src/util/util.hpp"

class LabelRegister : public QWidget,
		      public cvv::qtutil::RegisterHelper<QLabel, QWidget *>
{
      public:
	LabelRegister(QWidget *parent = nullptr)
	    : QWidget{ parent }, RegisterHelper<QLabel, QWidget *>{},
	      lay{ new QVBoxLayout{} }, lab{ new QLabel{} }, s{ [this]()
	{
		std::cout << "selection updated\n";
		this->updlabel();
	} },
	      reg{ [](QString s)
	{ std::cout << "regevent\t" << s.toStdString() << std::endl; } }
	{
		std::cout << __LINE__ << "\tlabel register constr begin\n";
		lay->addWidget(comboBox_);
		lay->addWidget(lab);
		setLayout(lay);
		connect(&signalElementSelected(),
			&cvv::qtutil::SignalQString::signal, &s,
			&cvv::qtutil::Slot::slot);
		std::cout << __LINE__
			  << "\tlabel register constr connected text changed\n";
		connect(&signalElementRegistered(), SIGNAL(signal(QString)),
			&reg, SLOT(slot(QString)));
		std::cout
		    << __LINE__
		    << "\tlabel register constr connected elem registered\n";
		std::cout << __LINE__ << "\tlabel register constr end\n";
		if (this->has(this->selection()))
		{
			this->updlabel();
		}
	}

	QVBoxLayout *lay;
	QLabel *lab;

	cvv::qtutil::Slot s;
	cvv::qtutil::SlotQString reg;

	void updlabel()
	{
		std::cout << "~updatelabel\n";
		std::unique_ptr<QLabel> newl{ (*this)()(nullptr) };
		lay->removeWidget(lab);
		lab->setParent(nullptr);
		delete lab;
		lab = newl.release();
		lay->addWidget(lab);

		std::cout << "\t~current selection\t"
			  << selection().toStdString() << "\n"
			  << "\t~txt of func\t" << lab->text().toStdString()
			  << "\n";
	}
};

void regnewlabelfunc()
{
	unsigned int cnt = LabelRegister::registeredElements().size();
	std::cout << "#regnewlabelfunc " << cnt << std::endl << "\t#has?\t"
		  << LabelRegister::has(QString::number(cnt)) << "\n";
	LabelRegister::registerElement(QString::number(cnt), [=](QWidget *)
	{
		std::cout << "§label fun\n";
		std::cout << "\t§cnt in label fun\t" << cnt << "\n";
		std::cout << "\t§&cnt in label fun\t" << &cnt << "\n";
		return cvv::util::make_unique<QLabel>(QString::number(cnt));
	});
	std::cout << "\t#anz now\t"
		  << LabelRegister::registeredElements().size() << std::endl;
}

/**
 * @brief
 * - a window will pop up
 * - the window contains a button ("add") and two identical subwidgets below each other
 * - the subwidget contains a combobox (initially "A") and a text (initially "A")
 * - the cobobox starts with the options "A" and "B"
 * - every time the button "add" is pressed both comboboxes will get a new entry "X"
 * - if that enty is selected the text will be "X"
 * - X is a number. it starts at 2 and is incremented for each click
 * - if a option X in a combobox is selected the text
 * 		~current selection	X
 *		~txt of func	X
 * is printed to std::out
 * (some other debug thext is printed too)
 *
 */
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	std::cout << __LINE__ << "\tregister label A\t"
		  << LabelRegister::registerElement("A", [](QWidget *)
	{ return cvv::util::make_unique<QLabel>("A"); }) << "\n";

	std::cout << __LINE__ << "\tregister label A again\t"
		  << LabelRegister::registerElement("A", [](QWidget *)
	{ return cvv::util::make_unique<QLabel>("A"); }) << "\n";

	std::cout << __LINE__ << "\tregister label B\t"
		  << LabelRegister::registerElement("B", [](QWidget *)
	{ return cvv::util::make_unique<QLabel>("B"); }) << "\n";

	QWidget w{};
	std::cout << __LINE__ << "\twill create labelregister\n";
	LabelRegister *r1 = new LabelRegister{};
	LabelRegister *r2 = new LabelRegister{};
	std::cout << __LINE__ << "\tcreated labelregister\n";

	QVBoxLayout *lay = new QVBoxLayout{};
	QPushButton *b = new QPushButton{ "add" };
	lay->addWidget(b);
	lay->addWidget(r1);
	lay->addWidget(r2);

	cvv::qtutil::Slot bPushed{ &regnewlabelfunc };

	QObject::connect(b, SIGNAL(clicked()), &bPushed, SLOT(slot()));
	w.setLayout(lay);
	w.show();
	std::cout << "*****MAIN*****\tshowed. will now exec\n";
	return a.exec();
}
