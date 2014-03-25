#include "../../src/qtutil/zoomableimage.hpp"

#include <sstream>
#include <iostream>

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QColor>
#include <QPixmap>
#include <QRect>
#include <QImage>

#include <QWidget>
#include <QApplication>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDoubleSpinBox>
#include <QPushButton>

#include "../../src/qtutil/matchview/colorutil.hpp"

// from
// https://github.com/Itseez/opencv/blob/master/samples/cpp/
//        tutorial_code/core/Matrix/Drawing_1.cpp
cv::Mat renderImg();

static const int w = 400;

/**
 * @brief
 * - a window with 4 columns pops up
 * - the first column contains a spin box and two toggleable buttons
 *  ("autoshow", "fullimg")
 * - the 2nd column contains a vertical blue green image with a false color
 * palete at the top left
 * - the 3rd column contains a vertical blue image
 * - the 4th column contains an image of a red rook
 * - the images can be zoomed with the spin box ctrl+scroll or ctrl+shift+scroll
 * - if "autoshow" is toggled an the image is zoomed in (60 on the spin box)
 *  the pixel values are printed
 * - if "fullimg" is pressed the images are resized and the whole image is visible
 */
int main(int argc, char *argv[])
{

	QApplication a(argc, argv);

	QWidget wid{};
	QHBoxLayout *lay = new QHBoxLayout{};

	wid.setLayout(lay);

	cv::Mat mat{ 50, 5, CV_8UC3, cv::Scalar{ 125, 125, 0 } };

	mat.at<cv::Vec<uint8_t,3>>(0,0)=cvv::qtutil::falseColor(0.0);
	mat.at<cv::Vec<uint8_t,3>>(1,0)=cvv::qtutil::falseColor(0.1);
	mat.at<cv::Vec<uint8_t,3>>(2,0)=cvv::qtutil::falseColor(0.2);
	mat.at<cv::Vec<uint8_t,3>>(3,0)=cvv::qtutil::falseColor(0.3);
	mat.at<cv::Vec<uint8_t,3>>(4,0)=cvv::qtutil::falseColor(0.4);
	mat.at<cv::Vec<uint8_t,3>>(5,0)=cvv::qtutil::falseColor(0.5);
	mat.at<cv::Vec<uint8_t,3>>(6,0)=cvv::qtutil::falseColor(0.6);
	mat.at<cv::Vec<uint8_t,3>>(7,0)=cvv::qtutil::falseColor(0.7);
	mat.at<cv::Vec<uint8_t,3>>(8,0)=cvv::qtutil::falseColor(0.8);
	mat.at<cv::Vec<uint8_t,3>>(9,0)=cvv::qtutil::falseColor(0.9);
	mat.at<cv::Vec<uint8_t,3>>(10,0)=cvv::qtutil::falseColor(1.0);

	cv::Mat mat2{ 5, 50, CV_64FC4,
		      cv::Scalar{ 1,               0.234523452345432523452,
				  0.75345340e-123, 0.5 } };
	cv::Mat img = renderImg();

	cvv::qtutil::ZoomableImage *i = new cvv::qtutil::ZoomableImage{ mat };
	cvv::qtutil::ZoomableImage *i2 = new cvv::qtutil::ZoomableImage{ mat2 };
	cvv::qtutil::ZoomableImage *i3 = new cvv::qtutil::ZoomableImage{ img };

	QVBoxLayout *layleft = new QVBoxLayout{};
	QDoubleSpinBox *spb = new QDoubleSpinBox{};
	QObject::connect(spb, SIGNAL(valueChanged(double)), i,
			 SLOT(setZoom(qreal)));
	QObject::connect(spb, SIGNAL(valueChanged(double)), i2,
			 SLOT(setZoom(qreal)));
	QObject::connect(spb, SIGNAL(valueChanged(double)), i3,
			 SLOT(setZoom(qreal)));
	layleft->addWidget(spb);

	QPushButton *bautoshow = new QPushButton{ "autoshow" };
	bautoshow->setCheckable(true);
	QObject::connect(bautoshow, SIGNAL(toggled(bool)), i,
			 SLOT(setAutoShowValues(bool)));
	QObject::connect(bautoshow, SIGNAL(toggled(bool)), i2,
			 SLOT(setAutoShowValues(bool)));
	QObject::connect(bautoshow, SIGNAL(toggled(bool)), i3,
			 SLOT(setAutoShowValues(bool)));
	bautoshow->setChecked(true);
	layleft->addWidget(bautoshow);

	QPushButton *bshowfull = new QPushButton{ "fullimg" };
	QObject::connect(bshowfull, SIGNAL(clicked()), i,
			 SLOT(showFullImage()));
	QObject::connect(bshowfull, SIGNAL(clicked()), i2,
			 SLOT(showFullImage()));
	QObject::connect(bshowfull, SIGNAL(clicked()), i3,
			 SLOT(showFullImage()));
	bautoshow->setChecked(true);
	layleft->addWidget(bshowfull);

	lay->addLayout(layleft);
	lay->addWidget(i);
	lay->addWidget(i2);
	lay->addWidget(i3);

	wid.show();
	return a.exec();
}

void MyPolygon(cv::Mat img);
void MyLine(cv::Mat img, cv::Point start, cv::Point end);

cv::Mat renderImg()
{
	cv::Mat rook_image{ w, w, CV_8UC3, cv::Scalar{ 123, 123, 123 } };
	// 2. Draw a rook
	// ------------------

	// 2.a. Create a convex polygon
	MyPolygon(rook_image);

	// 2.b. Creating rectangles
	rectangle(rook_image, cv::Point(0, 7 * w / 8), cv::Point(w, w),
		  cv::Scalar(0, 255, 255), 1, 8);

	// 2.c. Create a few lines
	MyLine(rook_image, cv::Point(0, 15 * w / 16),
	       cv::Point(w, 15 * w / 16));
	MyLine(rook_image, cv::Point(w / 4, 7 * w / 8), cv::Point(w / 4, w));
	MyLine(rook_image, cv::Point(w / 2, 7 * w / 8), cv::Point(w / 2, w));
	MyLine(rook_image, cv::Point(3 * w / 4, 7 * w / 8),
	       cv::Point(3 * w / 4, w));

	return rook_image;
}

void MyPolygon(cv::Mat img)
{
	int lineType = 8;
	cv::Point rook_points[1][20];
	rook_points[0][0] = cv::Point(w / 4, 7 * w / 8);
	rook_points[0][1] = cv::Point(3 * w / 4, 7 * w / 8);
	rook_points[0][2] = cv::Point(3 * w / 4, 13 * w / 16);
	rook_points[0][3] = cv::Point(11 * w / 16, 13 * w / 16);
	rook_points[0][4] = cv::Point(19 * w / 32, 3 * w / 8);
	rook_points[0][5] = cv::Point(3 * w / 4, 3 * w / 8);
	rook_points[0][6] = cv::Point(3 * w / 4, w / 8);
	rook_points[0][7] = cv::Point(26 * w / 40, w / 8);
	rook_points[0][8] = cv::Point(26 * w / 40, w / 4);
	rook_points[0][9] = cv::Point(22 * w / 40, w / 4);
	rook_points[0][10] = cv::Point(22 * w / 40, w / 8);
	rook_points[0][11] = cv::Point(18 * w / 40, w / 8);
	rook_points[0][12] = cv::Point(18 * w / 40, w / 4);
	rook_points[0][13] = cv::Point(14 * w / 40, w / 4);
	rook_points[0][14] = cv::Point(14 * w / 40, w / 8);
	rook_points[0][15] = cv::Point(w / 4, w / 8);
	rook_points[0][16] = cv::Point(w / 4, 3 * w / 8);
	rook_points[0][17] = cv::Point(13 * w / 32, 3 * w / 8);
	rook_points[0][18] = cv::Point(5 * w / 16, 13 * w / 16);
	rook_points[0][19] = cv::Point(w / 4, 13 * w / 16);

	const cv::Point *ppt[1] = { rook_points[0] };
	int npt[] = { 20 };

	fillPoly(img, ppt, npt, 1, cv::Scalar(0, 0, 255), lineType);
}

void MyLine(cv::Mat img, cv::Point start, cv::Point end)
{
	int thickness = 2;
	int lineType = 8;
	line(img, start, end, cv::Scalar(0, 0, 0), thickness, lineType);
}
