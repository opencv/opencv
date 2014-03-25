#include "../../src/qtutil/filterselectorwidget.hpp"

#include <QLabel>
#include <QVBoxLayout>
#include <QApplication>

#include <iostream>

#include "../../src/util/util.hpp"

class A : public cvv::qtutil::FilterFunctionWidget<1, 1>
{
      public:
	A(QWidget *parent = nullptr)
	    : cvv::qtutil::FilterFunctionWidget<1, 1>{ parent }
	{
		QVBoxLayout *lay = new QVBoxLayout{};
		lay->addWidget(new QLabel{ "A" });
		setLayout(lay);
	}

	~A()
	{
	}

	void applyFilter(cvv::qtutil::CvvInputArray<1>,
			 cvv::qtutil::CvvOutputArray<1>) const override
	{
		throw std::invalid_argument{ "cant use a" };
	}

	std::pair<bool, QString> checkInput(cvv::qtutil::CvvInputArray<1>) const
	    override
	{
		return { false, "testfilter A." };
	}
};

class B : public cvv::qtutil::FilterFunctionWidget<1, 1>
{
      public:
	B(QWidget *parent = nullptr)
	    : cvv::qtutil::FilterFunctionWidget<1, 1>{ parent }
	{
		QVBoxLayout *lay = new QVBoxLayout{};
		lay->addWidget(new QLabel{ "B" });
		setLayout(lay);
	}

	~B()
	{
	}

	void applyFilter(cvv::qtutil::CvvInputArray<1>,
			 cvv::qtutil::CvvOutputArray<1>) const override
	{
		throw std::invalid_argument{ "cant use B" };
	}

	std::pair<bool, QString> checkInput(cvv::qtutil::CvvInputArray<1>) const
	    override
	{
		return { false, "testfilter B." };
	}
};

/**
 * @brief
 * - a window will pop up
 * - it will contain a combo box (options "A", "B", initialli "A" is selected)
 *   a text ("A")
 *   and a button ("apply")
 * - if a option X from the combo box is selected the text will change to "X"
 * - if a option is selected the button will move beneath the combo box
 */
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	std::cout << "add A: "
		  << cvv::qtutil::FilterSelectorWidget<1, 1>::registerElement(
			 "A", [](QWidget *parent)
	{
				 return std::unique_ptr<
				     cvv::qtutil::FilterFunctionWidget<1, 1>>{
					 new A{ parent }
				 };
			 }) << std::endl;
	std::cout << "add B: "
		  << cvv::qtutil::FilterSelectorWidget<1, 1>::registerElement(
			 "B", [](QWidget *parent)
	{
				 return std::unique_ptr<
				     cvv::qtutil::FilterFunctionWidget<1, 1>>{
					 new B{ parent }
				 };
			 }) << std::endl;

	std::cout << "elems in cvv::qtutil::FilterSelectorWidget<1,1>:\t"
		  << cvv::qtutil::FilterSelectorWidget<
			 1, 1>::registeredElements().size() << std::endl;

	cvv::qtutil::FilterSelectorWidget<1, 1> w{};

	w.show();
	return a.exec();
}
