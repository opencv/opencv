#include "../../src/qtutil/collapsable.hpp"
#include "../../src/util/util.hpp"

#include <QApplication>
#include <QLabel>

/**
 * @brief
 * - a window with a button will open
 * - the button can be toggled
 * - if toggled the window will expand and display following text:
 *	some
 *	text "with
 *	newlines
 *
 *
 *
 *	loooooooooooooooooooooooooong line
 */
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	auto l = cvv::util::make_unique<QLabel>("some \ntext \"with "
						"\nnewlines\n\n\n\nlooooooooooo"
						"ooooooooooooooong line");

	cvv::qtutil::Collapsable w("TITLE GOES HERE", std::move(l));

	w.show();
	return a.exec();
}
