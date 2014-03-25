#include "../../src/qtutil/signalslot.hpp"

#include <sstream>

#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <sstream>
#include <QVBoxLayout>
#include <QString>
#include <QPushButton>

class SigSlotTest : public QWidget
{
      public:
	explicit SigSlotTest(QWidget *parent = nullptr)
	    : QWidget{ parent }, i{ 0 }, l{ new QLabel{ "pressed: 0" } },
	      signEmi{}, slotPressed{ std::bind(&SigSlotTest::pressed, this) },
	      slotEmi{ std::bind(&SigSlotTest::emittedCatched, this) }

	{
		QVBoxLayout *layout = new QVBoxLayout{};
		QPushButton *b = new QPushButton{ "push me" };
		layout->addWidget(b);
		layout->addWidget(l);
		setLayout(layout);
		connect(b, SIGNAL(pressed()), &slotPressed, SLOT(slot()));
		connect(&signEmi, SIGNAL(signal()), &slotEmi, SLOT(slot()));
	}

	unsigned int i;
	QLabel *l;

	cvv::qtutil::Signal signEmi;

      private:
	void pressed()
	{
		i++;
		signEmi.emitSignal();
	}

	void emittedCatched()
	{
		std::stringstream s;
		s << "pressed: " << i; // a;
		l->setText(s.str().c_str());
	}

	cvv::qtutil::Slot slotPressed;
	cvv::qtutil::Slot slotEmi;
};

/**
 * @brief
 * - a window will pop up
 * - it will contain a button ("push me") and below that a text ("pressed: 0")
 * - if the button is pressed the nub´mber in the text will increment
 */
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	SigSlotTest w{};

	w.show();
	return a.exec();
}
