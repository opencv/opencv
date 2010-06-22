//IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING. 

// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.


//                          License Agreement
//               For Open Source Computer Vision Library

//Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
//Copyright (C) 2008-2010, Willow Garage Inc., all rights reserved.
//Third party copyrights are property of their respective owners.

//Redistribution and use in source and binary forms, with or without modification,
//are permitted provided that the following conditions are met:

//  * Redistribution's of source code must retain the above copyright notice,
//  this list of conditions and the following disclaimer.

//  * Redistribution's in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.

//  * The name of the copyright holders may not be used to endorse or promote products
//  derived from this software without specific prior written permission.

//This software is provided by the copyright holders and contributors "as is" and
//any express or implied warranties, including, but not limited to, the implied
//warranties of merchantability and fitness for a particular purpose are disclaimed.
//In no event shall the Intel Corporation or contributors be liable for any direct,
//indirect, incidental, special, exemplary, or consequential damages
//(including, but not limited to, procurement of substitute goods or services;
//loss of use, data, or profits; or business interruption) however caused
//and on any theory of liability, whether in contract, strict liability,
//or tort (including negligence or otherwise) arising in any way out of
//the use of this software, even if advised of the possibility of such damage.

//--------------------Google Code 2010 -- Yannick Verdie--------------------//
#ifndef __OPENCV_HIGHGUI_QT_H__
#define __OPENCV_HIGHGUI_QT_H__


#include "precomp.hpp"

#if defined(OPENCV_GL)
#include <QGLWidget>
#endif

#include <QAbstractEventDispatcher>
#include <QtGui/QApplication>
#include <QFile>
#include <QPushButton>
#include <QtGui/QGraphicsView>
#include <QSizePolicy>
#include <QInputDialog>
#include <QtGui/QBoxLayout>
#include <QSettings>
#include <qtimer.h>
#include <QtConcurrentRun>
#include <QWaitCondition>
#include <QKeyEvent>
#include <QMetaObject>
#include <QPointer>
#include <QSlider>
#include <QLabel>
#include <QIODevice>
#include <QShortcut>
#include <QStatusBar>

//Macro here
#define CV_MODE_NORMAL   0
#define CV_MODE_OPENGL   1
//end macro

class CvWindow;
class ViewPort;
//class CvTrackbar;

class GuiReceiver : public QObject
{
    Q_OBJECT

public:
    GuiReceiver();
    int start();
    bool _bTimeOut;

private:


public slots:
    void createWindow( QString name, int flags = 0 );
    void destroyWindow(QString name);
    void destroyAllWindow();
    void addSlider(QString trackbar_name, QString window_name, void* value, int count, void* on_change);
    void moveWindow(QString name, int x, int y);
    void resizeWindow(QString name, int width, int height);
    void showImage(QString name, void* arr);
    void displayInfo( QString name, QString text, int delayms );
    void displayStatusBar( QString name, QString text, int delayms );
    void refreshEvents();
    void timeOut();
    void toggleFullScreen(QString name, double flags );
    double isFullScreen(QString name);
    double getPropWindow(QString name);
    void setPropWindow(QString name, double flags );
};

class CvTrackbar : public QHBoxLayout
{
    Q_OBJECT
public:
    CvTrackbar(CvWindow* parent, QString name, int* value, int count, CvTrackbarCallback on_change = NULL);
    ~CvTrackbar();

    QString trackbar_name;
    QPointer<QSlider> slider;

private slots:
    void createDialog();
    void update(int myvalue);

private:
    void setLabel(int myvalue);

    QString createLabel();
    QPointer<QPushButton > label;
    CvTrackbarCallback callback;
    QPointer<CvWindow> parent;
    int* dataSlider;
};

class CustomLayout : public QVBoxLayout
{
    Q_OBJECT
    public:
    CustomLayout();
    int heightForWidth ( int w ) const;
    bool hasHeightForWidth () const;
};

class CvWindow : public QWidget
{
    Q_OBJECT
public:
    CvWindow(QString arg2, int flag = CV_WINDOW_NORMAL);
    ~CvWindow();
    void addSlider(QString name, int* value, int count, CvTrackbarCallback on_change = NULL);
    void setMouseCallBack(CvMouseCallback m, void* param);
    void updateImage(void* arr);
    void displayInfo(QString text, int delayms );
    void displayStatusBar(QString text, int delayms );

    QString name;
    int flags;
    QPointer<QBoxLayout> layout;
    QPointer<QStatusBar> myBar;
    QPointer<QLabel> myBar_msg;
    //QPointer<CustomLayout> layout;

protected:
    void readSettings();
    void writeSettings();

    virtual void keyPressEvent(QKeyEvent *event);

private:
    QPointer<ViewPort> myview;

    int status;//0 normal, 1 fullscreen (YV)
    QPointer<QShortcut> shortcutZ;
    QPointer<QShortcut> shortcutPlus;
    QPointer<QShortcut> shortcutMinus;
    QPointer<QShortcut> shortcutLeft;
    QPointer<QShortcut> shortcutRight;
    QPointer<QShortcut> shortcutUp;
    QPointer<QShortcut> shortcutDown;
};



class ViewPort : public QGraphicsView
{
    Q_OBJECT
public:
    ViewPort(CvWindow* centralWidget, int mode = CV_MODE_NORMAL, bool keepRatio = true);
    ~ViewPort();
    void updateImage(void* arr);
    void startDisplayInfo(QString text, int delayms);
    void setMouseCallBack(CvMouseCallback m, void* param);

    IplImage* image2Draw_ipl;
    QImage image2Draw_qt;

public slots:
    //reference:
    //http://www.qtcentre.org/wiki/index.php?title=QGraphicsView:_Smooth_Panning_and_Zooming
    //http://doc.qt.nokia.com/4.6/gestures-imagegestures-imagewidget-cpp.html
    void scaleView(qreal scaleFactor, QPointF center);
    void moveView(QPointF delta);
    void resetZoom();
    void ZoomIn();
    void ZoomOut();
    void siftWindowOnLeft();
    void siftWindowOnRight();
    void siftWindowOnUp() ;
    void siftWindowOnDown();
    void resizeEvent ( QResizeEvent * );

private:
    Qt::AspectRatioMode modeRatio;
    QPoint deltaOffset;
    QPoint computeOffset();
    QPoint mouseCoordinate;
    QPointF positionGrabbing;
    QRect   positionCorners;
    QTransform matrixWorld;
    QTransform matrixWorld_inv;
    float ratioX, ratioY;

    CvMouseCallback on_mouse;
    void* on_mouse_param;

    int mode;
    bool keepRatio;

    bool isSameSize(IplImage* img1,IplImage* img2);
    QSize sizeHint() const;
    QPointer<CvWindow> centralWidget;
    QPointer<QTimer> timerDisplay;
    bool drawInfo;
    QString infoText;
    //QImage* image;

    void paintEvent(QPaintEvent* paintEventInfo);
    void wheelEvent(QWheelEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void drawInstructions(QPainter *painter);
    void drawOverview(QPainter *painter);
    void draw2D(QPainter *painter);
    void drawStatusBar();
    void controlImagePosition();

#if defined(OPENCV_GL)
    void draw3D();
    void unsetGL();
    void initGL();
    void setGL(int width, int height);
#endif

private slots:
    void stopDisplayInfo();
};


//here css for trackbar
/* from http://thesmithfam.org/blog/2010/03/10/fancy-qslider-stylesheet */
static const QString str_Trackbar_css = QString("")
+										"QSlider::groove:horizontal {"
+										"border: 1px solid #bbb;"
+										"background: white;"
+										"height: 10px;"
+										"border-radius: 4px;"
+										"}"

+										"QSlider::sub-page:horizontal {"
+										"background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,"
+										"stop: 0 #66e, stop: 1 #bbf);"
+										"background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,"
+										"stop: 0 #bbf, stop: 1 #55f);"
+										"border: 1px solid #777;"
+										"height: 10px;"
+										"border-radius: 4px;"
+										"}"

+										"QSlider::add-page:horizontal {"
+										"background: #fff;"
+										"border: 1px solid #777;"
+										"height: 10px;"
+										"border-radius: 4px;"
+										"}"

+										"QSlider::handle:horizontal {"
+										"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
+										"stop:0 #eee, stop:1 #ccc);"
+										"border: 1px solid #777;"
+										"width: 13px;"
+										"margin-top: -2px;"
+										"margin-bottom: -2px;"
+										"border-radius: 4px;"
+										"}"

+										"QSlider::handle:horizontal:hover {"
+										"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
+										"stop:0 #fff, stop:1 #ddd);"
+										"border: 1px solid #444;"
+										"border-radius: 4px;"
+										"}"

+										"QSlider::sub-page:horizontal:disabled {"
+										"background: #bbb;"
+										"border-color: #999;"
+										"}"

+										"QSlider::add-page:horizontal:disabled {"
+										"background: #eee;"
+										"border-color: #999;"
+										"}"

+										"QSlider::handle:horizontal:disabled {"
+										"background: #eee;"
+										"border: 1px solid #aaa;"
+										"border-radius: 4px;"
+										"}";

#endif
