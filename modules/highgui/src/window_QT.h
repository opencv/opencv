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

#if defined( HAVE_QT_OPENGL )
#include <QtOpenGL>
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
#include <QVarLengthArray>
#include <QFileInfo>
#include <QDate>
#include <QFileDialog>
#include <QToolBar>
#include <QAction>
#include <QPushButton>
#include <QCheckBox>
#include <QRadioButton>
#include <QButtonGroup>
#include <QMenu>

//start private enum
enum {CV_MODE_NORMAL= 0, CV_MODE_OPENGL = 1};

//we can change the keyboard shortcuts from here !
enum {	shortcut_zoom_normal 	= Qt::CTRL + Qt::Key_Z,
		shortcut_zoom_imgRegion = Qt::CTRL + Qt::Key_X,
		shortcut_save_img		= Qt::CTRL + Qt::Key_S,
		shortcut_properties_win	= Qt::CTRL + Qt::Key_P,
		shortcut_zoom_in 		= Qt::CTRL + Qt::Key_Plus,//QKeySequence(QKeySequence::ZoomIn),
		shortcut_zoom_out		= Qt::CTRL + Qt::Key_Minus,//QKeySequence(QKeySequence::ZoomOut),
		shortcut_panning_left 	= Qt::CTRL + Qt::Key_Left,
		shortcut_panning_right 	= Qt::CTRL + Qt::Key_Right,
		shortcut_panning_up 	= Qt::CTRL + Qt::Key_Up,
		shortcut_panning_down 	= Qt::CTRL + Qt::Key_Down
	};


//end enum

class CvWindow;
class ViewPort;
#if defined( HAVE_QT_OPENGL )
class OpenGLWidget;
#endif


class GuiReceiver : public QObject
{
    Q_OBJECT

public:
    GuiReceiver();
	~GuiReceiver();
    int start();
	void isLastWindow();

    bool _bTimeOut;
	QTimer *timer;

private:
	int nb_windows;

public slots:
    void createWindow( QString name, int flags = 0 );
    void destroyWindow(QString name);
    void destroyAllWindow();
    void addSlider(QString trackbar_name, QString window_name, void* value, int count, void* on_change);
	void addSlider2(QString trackbar_name, QString window_name, void* value, int count, void* on_change, void *userdata);
    void moveWindow(QString name, int x, int y);
    void resizeWindow(QString name, int width, int height);
    void showImage(QString name, void* arr);
    void displayInfo( QString name, QString text, int delayms );
    void displayStatusBar( QString name, QString text, int delayms );
    void timeOut();
    void toggleFullScreen(QString name, double flags );
    double isFullScreen(QString name);
    double getPropWindow(QString name);
    void setPropWindow(QString name, double flags );
    double getRatioWindow(QString name);
    void setRatioWindow(QString name, double arg2 );
    void saveWindowParameters(QString name);
    void loadWindowParameters(QString name);
    void setOpenGLCallback(QString window_name, void* callbackOpenGL, void* userdata, double angle, double zmin, double zmax);
    void putText(void* arg1, QString text, QPoint org, void* font);
    void addButton(QString button_name, int button_type, int initial_button_state , void* on_change, void* userdata);
	void enablePropertiesButtonEachWindow();
};

enum typeBar{type_CvTrackbar = 0, type_CvButtonbar = 1};
class CvBar  :  public QHBoxLayout
{
public:
    typeBar type;
    QString name_bar;
    QPointer<QWidget> myparent;
};


class CvButtonbar : public CvBar
{
    Q_OBJECT
public:
    CvButtonbar(QWidget* arg, QString bar_name);
    ~CvButtonbar();
    void addButton( QString button_name, CvButtonCallback call, void* userdata,  int button_type, int initial_button_state);

private:
    void setLabel();

    QPointer<QLabel> label;
    QPointer<QButtonGroup> group_button;
};


class CvPushButton : public QPushButton
{
    Q_OBJECT
public:
    CvPushButton(CvButtonbar* par, QString button_name, CvButtonCallback call, void* userdata);

private:
    CvButtonbar* myparent;
    QString button_name ;
    CvButtonCallback callback;
    void* userdata;

private slots:
    void callCallBack(bool);
};



class CvCheckBox : public QCheckBox
{
    Q_OBJECT
public:
    CvCheckBox(CvButtonbar* par, QString button_name, CvButtonCallback call, void* userdata, int initial_button_state);

private:
    CvButtonbar* myparent;
    QString button_name ;
    CvButtonCallback callback;
    void* userdata;

private slots:
    void callCallBack(bool);
};

class CvRadioButton : public QRadioButton
{
    Q_OBJECT
public:
    CvRadioButton(CvButtonbar* par, QString button_name, CvButtonCallback call, void* userdata, int initial_button_state);

private:
    CvButtonbar* myparent;
    QString button_name ;
    CvButtonCallback callback;
    void* userdata;

private slots:
    void callCallBack(bool);
};



class CvTrackbar :  public CvBar
{
    Q_OBJECT
public:
    CvTrackbar(CvWindow* parent, QString name, int* value, int count, CvTrackbarCallback on_change);
	CvTrackbar(CvWindow* parent, QString name, int* value, int count, CvTrackbarCallback2 on_change, void* data);
    ~CvTrackbar();

    //QString trackbar_name;
    QPointer<QSlider> slider;

private slots:
    void createDialog();
    void update(int myvalue);

private:
    void setLabel(int myvalue);
	void construc_trackbar(CvWindow* arg, QString name, int* value, int count);
    QString createLabel();
    QPointer<QPushButton > label;
    CvTrackbarCallback callback;
	CvTrackbarCallback2 callback2;//look like it is use by python binding
    int* dataSlider;
	void* userdata;

};

//Both are top level window, so that a way to differenciate them.
//if (obj->metaObject ()->className () == "CvWindow") does not give me robust result

enum typeWindow{type_CvWindow = 0, type_CvWinProperties = 1};

class CvWinModel : public QWidget
{
public:
typeWindow type;
};


class CvWinProperties : public CvWinModel
{
    Q_OBJECT
public:
    CvWinProperties(QString name,QObject* parent);
    ~CvWinProperties();
    QPointer<QBoxLayout> myLayout;

private:
    void closeEvent ( QCloseEvent * e );
    void showEvent ( QShowEvent * event ) ;
    void hideEvent ( QHideEvent * event ) ;
};



class GlobalLayout : public QBoxLayout
{
    Q_OBJECT
public:
	GlobalLayout(QWidget* parent) : QBoxLayout(QBoxLayout::TopToBottom,parent){};
	bool hasHeightForWidth () {return true;};
	int heightForWidth( int w ) {qDebug()<<"yopyopypp";return w;};
};


class CvWindow : public CvWinModel
{
    Q_OBJECT
public:
    CvWindow(QString arg2, int flag = CV_WINDOW_NORMAL);
    ~CvWindow();
    static void addSlider(CvWindow* w,QString name, int* value, int count, CvTrackbarCallback on_change CV_DEFAULT(NULL));
	static void addSlider2(CvWindow* w,QString name, int* value, int count, CvTrackbarCallback2 on_change CV_DEFAULT(NULL), void* userdata CV_DEFAULT(0));
    void setMouseCallBack(CvMouseCallback m, void* param);
    void updateImage(void* arr);
    void displayInfo(QString text, int delayms );
    void displayStatusBar(QString text, int delayms );
    void readSettings();
    void writeSettings();
    void setOpenGLCallback(CvOpenGLCallback arg1,void* userdata, double angle, double zmin, double zmax);
    void hideTools();
    void showTools();
    static CvButtonbar* createButtonbar(QString bar_name);
	QSize getAvailableSize();


    ViewPort* getView();

    QPointer<QBoxLayout> myGlobalLayout;//All the widget (toolbar, view, LayoutBar, ...) are attached to it
    QPointer<QBoxLayout> myBarLayout;
    QPointer<QStatusBar> myStatusBar;
    QPointer<QToolBar> myToolBar;
    QPointer<QLabel> myStatusBar_msg;

    //parameters (will be save/load)
    QString param_name;
    int param_flags;
    int param_gui_mode;
	int param_ratio_mode;
    QVector<QAction*> vect_QActions;


protected:
    virtual void keyPressEvent(QKeyEvent *event);

private:
    QPointer<ViewPort> myview;
    QVector<QShortcut*> vect_QShortcuts;

    void icvLoadTrackbars(QSettings *settings);
    void icvSaveTrackbars(QSettings *settings);
	void icvLoadControlPanel();
	void icvSaveControlPanel();
	void icvLoadButtonbar(CvButtonbar* t,QSettings *settings);
	void icvSaveButtonbar(CvButtonbar* t,QSettings *settings);

	void createActions();
	void createShortcuts();
    void createToolBar();
    void createView(int display_mode, int ratio_mode);
    void createStatusBar();
    void createGlobalLayout();
    void createBarLayout();
    CvWinProperties* createParameterWindow();

private slots:
    void displayPropertiesWin();
};




enum type_mouse_event {mouse_up = 0, mouse_down = 1, mouse_dbclick = 2, mouse_move = 3};

static const int tableMouseButtons[][3]={
    {CV_EVENT_LBUTTONUP,CV_EVENT_RBUTTONUP,CV_EVENT_MBUTTONUP},		    	//mouse_up
    {CV_EVENT_LBUTTONDOWN,CV_EVENT_RBUTTONDOWN,CV_EVENT_MBUTTONDOWN},		//mouse_down
    {CV_EVENT_LBUTTONDBLCLK,CV_EVENT_RBUTTONDBLCLK,CV_EVENT_MBUTTONDBLCLK},	//mouse_dbclick
    {CV_EVENT_MOUSEMOVE,CV_EVENT_MOUSEMOVE,CV_EVENT_MOUSEMOVE}		    	//mouse_move
};


static const double DEFAULT_ANGLE  = 45.0;
static const double DEFAULT_ZMIN = 0.01;
static const double  DEFAULT_ZMAX = 1000.0;
class ViewPort : public QGraphicsView
{
    Q_OBJECT
public:
    ViewPort(CvWindow* centralWidget, int mode = CV_MODE_NORMAL, int keepRatio = CV_WINDOW_KEEPRATIO);
    ~ViewPort();
    void updateImage(const CvArr *arr);
    void startDisplayInfo(QString text, int delayms);
    void setMouseCallBack(CvMouseCallback m, void* param);
    void setOpenGLCallback(CvOpenGLCallback func,void* userdata, double arg3, double arg4, double arg5);
    int getRatio();
    void setRatio(int arg);
	int heightForWidth( int w );// {qDebug()<<"yopyopypp";return w;};
	bool hasHeightForWidth (){qDebug()<<"ask";return true;};

    //parameters (will be save/load)
    QTransform param_matrixWorld;

    int param_keepRatio;

    //IplImage* image2Draw_ipl;
	CvMat* image2Draw_mat;
    QImage image2Draw_qt;
    QImage image2Draw_qt_resized;
    int mode_display;//opengl or native
    int nbChannelOriginImage;

public slots:
    //reference:
    //http://www.qtcentre.org/wiki/index.php?title=QGraphicsView:_Smooth_Panning_and_Zooming
    //http://doc.qt.nokia.com/4.6/gestures-imagegestures-imagewidget-cpp.html
    void scaleView(qreal scaleFactor, QPointF center);
    void imgRegion();
    void moveView(QPointF delta);
    void resetZoom();
    void ZoomIn();
    void ZoomOut();
    void siftWindowOnLeft();
    void siftWindowOnRight();
    void siftWindowOnUp() ;
    void siftWindowOnDown();
    void resizeEvent ( QResizeEvent * );
    void saveView();
    void contextMenuEvent(QContextMenuEvent *event);


private:
    QPoint mouseCoordinate;
    QPointF positionGrabbing;
    QRect  positionCorners;
    QTransform matrixWorld_inv;
    float ratioX, ratioY;

    //for mouse callback
    CvMouseCallback on_mouse;
    void* on_mouse_param;

    //for opengl callback
    CvOpenGLCallback on_openGL_draw3D;
    void* on_openGL_param;

    bool isSameSize(IplImage* img1,IplImage* img2);
    QSize sizeHint() const;
    QPointer<CvWindow> centralWidget;
    QPointer<QTimer> timerDisplay;
    bool drawInfo;
    QString infoText;
	QRectF target;
    //QImage* image;

    void paintEvent(QPaintEvent* paintEventInfo);
    void wheelEvent(QWheelEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void drawInstructions(QPainter *painter);
    void drawViewOverview(QPainter *painter);
    void drawImgRegion(QPainter *painter);
    void draw2D(QPainter *painter);
    void drawStatusBar();
    void controlImagePosition();
    void icvmouseHandler(QMouseEvent *event, type_mouse_event category, int &cv_event, int &flags);
    void icvmouseProcessing(QPointF pt, int cv_event, int flags);

#if defined( HAVE_QT_OPENGL )
	QPointer<QGLWidget> myGL;
	double angle;
	double zmin;
	double zmax;
    void unsetGL();
    void initGL();
    void setGL(int width, int height);
    void icvgluPerspective(GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar);
#endif

private slots:
    void stopDisplayInfo();
};

#endif
