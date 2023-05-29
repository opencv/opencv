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

#include "precomp.hpp"

#if defined(HAVE_QT)

#include <memory>

#include "window_QT.h"

#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// Get GL_PERSPECTIVE_CORRECTION_HINT definition, not available in GLES 2 or
// OpenGL 3 core profile or later
#ifdef HAVE_QT_OPENGL
    #if defined Q_WS_X11 /* Qt4 */ || \
        (!defined(QT_OPENGL_ES_2) && defined Q_OS_LINUX) /* Qt5 with desktop OpenGL */
        #include <GL/gl.h>
    #endif
#endif

using namespace cv;

#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
#define Qt_MiddleButton Qt::MiddleButton
inline Qt::Orientation wheelEventOrientation(QWheelEvent *we) {
    if (std::abs(we->angleDelta().x()) < std::abs(we->angleDelta().y()))
        return Qt::Vertical;
    else
        return Qt::Horizontal;
}
inline int wheelEventDelta(QWheelEvent *we) {
    if(wheelEventOrientation(we) == Qt::Vertical)
        return we->angleDelta().y();
    else
        return we->angleDelta().x();
}
inline QPoint wheelEventPos(QWheelEvent *we) {
    return we->position().toPoint();
}
#else
#define Qt_MiddleButton Qt::MidButton
inline Qt::Orientation wheelEventOrientation(QWheelEvent *we) {
    return we->orientation();
}
inline int wheelEventDelta(QWheelEvent *we) {
    return we->delta();
}
inline QPoint wheelEventPos(QWheelEvent *we) {
    return we->pos();
}

#endif


//Static and global first
static GuiReceiver *guiMainThread = NULL;
static int parameterSystemC = 1;
static char* parameterSystemV[] = {(char*)""};
static bool multiThreads = false;
static int last_key = -1;
QWaitCondition key_pressed;
QMutex mutexKey;
static const unsigned int threshold_zoom_img_region = 30;
//the minimum zoom value to start displaying the values in the grid
//that is also the number of pixel per grid

static CvWinProperties* global_control_panel = NULL;
//end static and global

// Declaration
Qt::ConnectionType autoBlockingConnection();

// Implementation - this allows us to do blocking whilst automatically selecting the right
// behaviour for in-thread and out-of-thread launches of cv windows. Qt strangely doesn't
// cater for this, but does for strictly queued connections.
Qt::ConnectionType autoBlockingConnection() {
  return (QThread::currentThread() != QApplication::instance()->thread())
      ? Qt::BlockingQueuedConnection
      : Qt::DirectConnection;
}

CV_IMPL CvFont cvFontQt(const char* nameFont, int pointSize,CvScalar color,int weight,int style, int spacing)
{
    /*
    //nameFont   <- only Qt
    //CvScalar color   <- only Qt (blue_component, green_component, red\_component[, alpha_component])
    int         font_face;//<- style in Qt
    const int*  ascii;
    const int*  greek;
    const int*  cyrillic;
    float       hscale, vscale;
    float       shear;
    int         thickness;//<- weight in Qt
    float       dx;//spacing letter in Qt (0 default) in pixel
    int         line_type;//<- pointSize in Qt
    */
    CvFont f = {nameFont,color,style,NULL,NULL,NULL,0,0,0,weight, (float)spacing, pointSize};
    return f;
}


CV_IMPL void cvAddText(const CvArr* img, const char* text, CvPoint org, CvFont* font)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "putText",
        autoBlockingConnection(),
        Q_ARG(void*, (void*) img),
        Q_ARG(QString,QString::fromUtf8(text)),
        Q_ARG(QPoint, QPoint(org.x,org.y)),
        Q_ARG(void*,(void*) font));
}


double cvGetRatioWindow_QT(const char* name)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    double result = -1;
    QMetaObject::invokeMethod(guiMainThread,
        "getRatioWindow",
        autoBlockingConnection(),
        Q_RETURN_ARG(double, result),
        Q_ARG(QString, QString(name)));

    return result;
}

double cvGetPropVisible_QT(const char* name) {
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    double result = 0;

    QMetaObject::invokeMethod(guiMainThread,
        "getWindowVisible",
        autoBlockingConnection(),
        Q_RETURN_ARG(double, result),
        Q_ARG(QString, QString(name)));

    return result;
}

void cvSetRatioWindow_QT(const char* name,double prop_value)
{

    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "setRatioWindow",
        autoBlockingConnection(),
        Q_ARG(QString, QString(name)),
        Q_ARG(double, prop_value));
}

double cvGetPropWindow_QT(const char* name)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    double result = -1;
    QMetaObject::invokeMethod(guiMainThread,
        "getPropWindow",
        autoBlockingConnection(),
        Q_RETURN_ARG(double, result),
        Q_ARG(QString, QString(name)));

    return result;
}


void cvSetPropWindow_QT(const char* name,double prop_value)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "setPropWindow",
        autoBlockingConnection(),
        Q_ARG(QString, QString(name)),
        Q_ARG(double, prop_value));
}

void setWindowTitle_QT(const String& winname, const String& title)
{
    if (!guiMainThread)
        CV_Error(Error::StsNullPtr, "NULL guiReceiver (please create a window)");

    QMetaObject::invokeMethod(guiMainThread,
        "setWindowTitle",
        autoBlockingConnection(),
        Q_ARG(QString, QString(winname.c_str())),
        Q_ARG(QString, QString(title.c_str())));
}


void cvSetModeWindow_QT(const char* name, double prop_value)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "toggleFullScreen",
        autoBlockingConnection(),
        Q_ARG(QString, QString(name)),
        Q_ARG(double, prop_value));
}

CvRect cvGetWindowRect_QT(const char* name)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    CvRect result = cvRect(-1, -1, -1, -1);

    QMetaObject::invokeMethod(guiMainThread,
        "getWindowRect",
        autoBlockingConnection(),
        Q_RETURN_ARG(CvRect, result),
        Q_ARG(QString, QString(name)));

    return result;
}

double cvGetModeWindow_QT(const char* name)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    double result = -1;

    QMetaObject::invokeMethod(guiMainThread,
        "isFullScreen",
        autoBlockingConnection(),
        Q_RETURN_ARG(double, result),
        Q_ARG(QString, QString(name)));

    return result;
}


CV_IMPL void cvDisplayOverlay(const char* name, const char* text, int delayms)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "displayInfo",
        autoBlockingConnection(),
        Q_ARG(QString, QString(name)),
        Q_ARG(QString, QString(text)),
        Q_ARG(int, delayms));
}


CV_IMPL void cvSaveWindowParameters(const char* name)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "saveWindowParameters",
        autoBlockingConnection(),
        Q_ARG(QString, QString(name)));
}


CV_IMPL void cvLoadWindowParameters(const char* name)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "loadWindowParameters",
        autoBlockingConnection(),
        Q_ARG(QString, QString(name)));
}


CV_IMPL void cvDisplayStatusBar(const char* name, const char* text, int delayms)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "displayStatusBar",
        autoBlockingConnection(),
        Q_ARG(QString, QString(name)),
        Q_ARG(QString, QString(text)),
        Q_ARG(int, delayms));
}


CV_IMPL int cvWaitKey(int delay)
{
    int result = -1;

    if (!guiMainThread)
        return result;

    unsigned long delayms = delay <= 0 ? ULONG_MAX : delay; //in milliseconds

    if (multiThreads)
    {
        mutexKey.lock();
        if (key_pressed.wait(&mutexKey, delayms)) //false if timeout
        {
            result = last_key;
        }
        last_key = -1;
        mutexKey.unlock();
    }
    else
    {
        //cannot use wait here because events will not be distributed before processEvents (the main eventLoop is broken)
        //so I create a Thread for the QTimer

        if (delay > 0)
            guiMainThread->timer->start(delay);

        //QMutex dummy;

        while (!guiMainThread->bTimeOut)
        {
            qApp->processEvents(QEventLoop::AllEvents);

            if (!guiMainThread)//when all the windows are deleted
                return result;

            mutexKey.lock();
            if (last_key != -1)
            {
                result = last_key;
                last_key = -1;
                guiMainThread->timer->stop();
                //printf("keypressed\n");
            }
            mutexKey.unlock();

            if (result!=-1)
            {
                break;
            }
            else
            {
                /*
    * //will not work, I broke the event loop !!!!
    dummy.lock();
    QWaitCondition waitCondition;
    waitCondition.wait(&dummy, 2);
    */

                //to decrease CPU usage
                //sleep 1 millisecond
#if defined _WIN32
                Sleep(1);
#else
                usleep(1000);
#endif
            }
        }

        guiMainThread->bTimeOut = false;
    }
    return result;
}


//Yannick Verdie
//This function is experimental and some functions (such as cvSet/getWindowProperty will not work)
//We recommend not using this function for now
CV_IMPL int cvStartLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[])
{
    multiThreads = true;
    QFuture<int> future = QtConcurrent::run(pt2Func, argc, argv);
    return guiMainThread->start();
}


CV_IMPL void cvStopLoop()
{
    qApp->exit();
}


static CvWindow* icvFindWindowByName(QString name)
{
    CvWindow* window = 0;

    //This is not a very clean way to do the stuff. Indeed, QAction automatically generate toolTil (QLabel)
    //that can be grabbed here and crash the code at 'w->param_name==name'.
    foreach (QWidget* widget, QApplication::topLevelWidgets())
    {
        if (widget->isWindow() && !widget->parentWidget())//is a window without parent
        {
            CvWinModel* temp = (CvWinModel*) widget;

            if (temp->type == type_CvWindow)
            {
                CvWindow* w = (CvWindow*) temp;
                if (w->objectName() == name)
                {
                    window = w;
                    break;
                }
            }
        }
    }

    return window;
}


static CvBar* icvFindBarByName(QBoxLayout* layout, QString name_bar, typeBar type)
{
    if (!layout)
        return NULL;

    int stop_index = layout->layout()->count();

    for (int i = 0; i < stop_index; ++i)
    {
        CvBar* t = (CvBar*) layout->layout()->itemAt(i);

        if (t->type == type && t->name_bar == name_bar)
            return t;
    }

    return NULL;
}


static CvTrackbar* icvFindTrackBarByName(const char* name_trackbar, const char* name_window, QBoxLayout* layout = NULL)
{
    QString nameQt(name_trackbar);
    QString nameWinQt(name_window);

    if (nameWinQt.isEmpty() && global_control_panel) //window name is null and we have a control panel
        layout = global_control_panel->myLayout;

    if (!layout)
    {
        QPointer<CvWindow> w = icvFindWindowByName(nameWinQt);

        if (!w)
            CV_Error(CV_StsNullPtr, "NULL window handler");

        if (w->param_gui_mode == CV_GUI_NORMAL)
            return (CvTrackbar*) icvFindBarByName(w->myBarLayout, nameQt, type_CvTrackbar);

        if (w->param_gui_mode == CV_GUI_EXPANDED)
        {
            CvBar* result = icvFindBarByName(w->myBarLayout, nameQt, type_CvTrackbar);

            if (result)
                return (CvTrackbar*) result;

            return (CvTrackbar*) icvFindBarByName(global_control_panel->myLayout, nameQt, type_CvTrackbar);
        }

        return NULL;
    }
    else
    {
        //layout was specified
        return (CvTrackbar*) icvFindBarByName(layout, nameQt, type_CvTrackbar);
    }
}

/*
static CvButtonbar* icvFindButtonBarByName(const char* button_name, QBoxLayout* layout)
{
    QString nameQt(button_name);
    return (CvButtonbar*) icvFindBarByName(layout, nameQt, type_CvButtonbar);
}
*/

static int icvInitSystem(int* c, char** v)
{
    //"For any GUI application using Qt, there is precisely one QApplication object"
    if (!QApplication::instance())
    {
#if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
        QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling, true);
#endif
        new QApplication(*c, v);
        setlocale(LC_NUMERIC,"C");

        qDebug() << "init done";

#ifdef HAVE_QT_OPENGL
        qDebug() << "opengl support available";
#endif
    }

    return 0;
}


CV_IMPL int cvInitSystem(int, char**)
{
    icvInitSystem(&parameterSystemC, parameterSystemV);
    return 0;
}


CV_IMPL int cvNamedWindow(const char* name, int flags)
{
    if (!guiMainThread)
        guiMainThread = new GuiReceiver;
    if (QThread::currentThread() != QApplication::instance()->thread()) {
        multiThreads = true;
        QMetaObject::invokeMethod(guiMainThread,
        "createWindow",
        Qt::BlockingQueuedConnection,  // block so that we can do useful stuff once we confirm it is created
        Q_ARG(QString, QString(name)),
        Q_ARG(int, flags));
     } else {
        guiMainThread->createWindow(QString(name), flags);
     }

    return 1; //Dummy value - probably should return the result of the invocation.
}


CV_IMPL void cvDestroyWindow(const char* name)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "destroyWindow",
        Qt::AutoConnection,  // if another thread is controlling, let it handle it without blocking ourselves here
        Q_ARG(QString, QString(name)));
}


CV_IMPL void cvDestroyAllWindows()
{
    if (!guiMainThread)
        return;
    QMetaObject::invokeMethod(guiMainThread,
        "destroyAllWindow",
        Qt::AutoConnection  // if another thread is controlling, let it handle it without blocking ourselves here
        );
}


CV_IMPL void* cvGetWindowHandle(const char* name)
{
    if (!name)
        CV_Error( CV_StsNullPtr, "NULL name string" );

    return (void*) icvFindWindowByName(QLatin1String(name));
}


CV_IMPL const char* cvGetWindowName(void* window_handle)
{
    if( !window_handle )
        CV_Error( CV_StsNullPtr, "NULL window handler" );

    return ((CvWindow*)window_handle)->objectName().toLatin1().data();
}


CV_IMPL void cvMoveWindow(const char* name, int x, int y)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );
    QMetaObject::invokeMethod(guiMainThread,
        "moveWindow",
        autoBlockingConnection(),
        Q_ARG(QString, QString(name)),
        Q_ARG(int, x),
        Q_ARG(int, y));
}

CV_IMPL void cvResizeWindow(const char* name, int width, int height)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );
    QMetaObject::invokeMethod(guiMainThread,
        "resizeWindow",
        autoBlockingConnection(),
        Q_ARG(QString, QString(name)),
        Q_ARG(int, width),
        Q_ARG(int, height));
}


CV_IMPL int cvCreateTrackbar2(const char* name_bar, const char* window_name, int* val, int count, CvTrackbarCallback2 on_notify, void* userdata)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "addSlider2",
        autoBlockingConnection(),
        Q_ARG(QString, QString(name_bar)),
        Q_ARG(QString, QString(window_name)),
        Q_ARG(void*, (void*)val),
        Q_ARG(int, count),
        Q_ARG(void*, (void*)on_notify),
        Q_ARG(void*, (void*)userdata));

    return 1; //dummy value
}


CV_IMPL int cvStartWindowThread()
{
    return 0;
}


CV_IMPL int cvCreateTrackbar(const char* name_bar, const char* window_name, int* value, int count, CvTrackbarCallback on_change)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "addSlider",
        autoBlockingConnection(),
        Q_ARG(QString, QString(name_bar)),
        Q_ARG(QString, QString(window_name)),
        Q_ARG(void*, (void*)value),
        Q_ARG(int, count),
        Q_ARG(void*, (void*)on_change));

    return 1; //dummy value
}


CV_IMPL int cvCreateButton(const char* button_name, CvButtonCallback on_change, void* userdata, int button_type, int initial_button_state)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    if (initial_button_state < 0 || initial_button_state > 1)
        return 0;

    QMetaObject::invokeMethod(guiMainThread,
        "addButton",
        autoBlockingConnection(),
        Q_ARG(QString, QString(button_name)),
        Q_ARG(int,  button_type),
        Q_ARG(int, initial_button_state),
        Q_ARG(void*, (void*)on_change),
        Q_ARG(void*, userdata));

    return 1;//dummy value
}


CV_IMPL int cvGetTrackbarPos(const char* name_bar, const char* window_name)
{
    int result = -1;

    QPointer<CvTrackbar> t = icvFindTrackBarByName(name_bar, window_name);

    if (t)
        result = t->slider->value();

    return result;
}


CV_IMPL void cvSetTrackbarPos(const char* name_bar, const char* window_name, int pos)
{
    QPointer<CvTrackbar> t = icvFindTrackBarByName(name_bar, window_name);

    if (t)
        t->slider->setValue(pos);
}


CV_IMPL void cvSetTrackbarMax(const char* name_bar, const char* window_name, int maxval)
{
    QPointer<CvTrackbar> t = icvFindTrackBarByName(name_bar, window_name);
    if (t)
    {
        t->slider->setMaximum(maxval);
    }
}


CV_IMPL void cvSetTrackbarMin(const char* name_bar, const char* window_name, int minval)
{
    QPointer<CvTrackbar> t = icvFindTrackBarByName(name_bar, window_name);
    if (t)
    {
        t->slider->setMinimum(minval);
    }
}


/* assign callback for mouse events */
CV_IMPL void cvSetMouseCallback(const char* window_name, CvMouseCallback on_mouse, void* param)
{
    QPointer<CvWindow> w = icvFindWindowByName(QLatin1String(window_name));

    if (!w)
        CV_Error(CV_StsNullPtr, "NULL window handler");

    w->setMouseCallBack(on_mouse, param);

}


CV_IMPL void cvShowImage(const char* name, const CvArr* arr)
{
    if (!guiMainThread)
        guiMainThread = new GuiReceiver;
    if (QThread::currentThread() != QApplication::instance()->thread()) {
        multiThreads = true;
        QMetaObject::invokeMethod(guiMainThread,
            "showImage",
             autoBlockingConnection(),
             Q_ARG(QString, QString(name)),
             Q_ARG(void*, (void*)arr)
        );
     } else {
        guiMainThread->showImage(QString(name), (void*)arr);
     }
}


#ifdef HAVE_QT_OPENGL

CV_IMPL void cvSetOpenGlDrawCallback(const char* window_name, CvOpenGlDrawCallback callback, void* userdata)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "setOpenGlDrawCallback",
        autoBlockingConnection(),
        Q_ARG(QString, QString(window_name)),
        Q_ARG(void*, (void*)callback),
        Q_ARG(void*, userdata));
}


CV_IMPL void cvSetOpenGlContext(const char* window_name)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "setOpenGlContext",
        autoBlockingConnection(),
        Q_ARG(QString, QString(window_name)));
}


CV_IMPL void cvUpdateWindow(const char* window_name)
{
    if (!guiMainThread)
        CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

    QMetaObject::invokeMethod(guiMainThread,
        "updateWindow",
        autoBlockingConnection(),
        Q_ARG(QString, QString(window_name)));
}

#endif


double cvGetOpenGlProp_QT(const char* name)
{
    double result = -1;

    if (guiMainThread)
    {
        QMetaObject::invokeMethod(guiMainThread,
            "isOpenGl",
            autoBlockingConnection(),
            Q_RETURN_ARG(double, result),
            Q_ARG(QString, QString(name)));
    }

    return result;
}


//////////////////////////////////////////////////////
// GuiReceiver


GuiReceiver::GuiReceiver() : bTimeOut(false), nb_windows(0)
{
    doesExternalQAppExist = (QApplication::instance() != 0);
    icvInitSystem(&parameterSystemC, parameterSystemV);

    timer = new QTimer(this);
    QObject::connect(timer, SIGNAL(timeout()), this, SLOT(timeOut()));
    timer->setSingleShot(true);
    if ( doesExternalQAppExist ) {
        moveToThread(QApplication::instance()->thread());
    }
}


void GuiReceiver::isLastWindow()
{
    if (--nb_windows <= 0)
    {
        delete guiMainThread;//delete global_control_panel too
        guiMainThread = NULL;

        if (!doesExternalQAppExist)
        {
            qApp->quit();
        }
    }
}


GuiReceiver::~GuiReceiver()
{
    if (global_control_panel)
    {
        delete global_control_panel;
        global_control_panel = NULL;
    }
}


void GuiReceiver::putText(void* arr, QString text, QPoint org, void* arg2)
{
    CV_Assert(arr);

    CvMat* mat, stub;
    mat = cvGetMat(arr, &stub);

    int nbChannelOriginImage = cvGetElemType(mat);
    if (nbChannelOriginImage != CV_8UC3) return; //for now, font works only with 8UC3

    QImage qimg(mat->data.ptr, mat->cols, mat->rows, mat->step, QImage::Format_RGB888);

    CvFont* font = (CvFont*)arg2;

    QPainter qp(&qimg);
    if (font)
    {
        QFont f(font->nameFont, font->line_type/*PointSize*/, font->thickness/*weight*/);
        f.setStyle((QFont::Style) font->font_face/*style*/);
        f.setLetterSpacing(QFont::AbsoluteSpacing, font->dx/*spacing*/);
        //cvScalar(blue_component, green_component, red_component[, alpha_component])
        //Qt map non-transparent to 0xFF and transparent to 0
        //OpenCV scalar is the reverse, so 255-font->color.val[3]
        qp.setPen(QColor(font->color.val[0], font->color.val[1], font->color.val[2], 255 - font->color.val[3]));
        qp.setFont(f);
    }
    qp.drawText(org, text);
    qp.end();
}


void GuiReceiver::saveWindowParameters(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (w)
        w->writeSettings();
}


void GuiReceiver::loadWindowParameters(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (w)
        w->readSettings();
}


double GuiReceiver::getRatioWindow(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (!w)
        return -1;

    return w->getRatio();
}


void GuiReceiver::setRatioWindow(QString name, double arg2)
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (!w)
        return;

    int flags = (int) arg2;

    w->setRatio(flags);
}


double GuiReceiver::getPropWindow(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (!w)
        return -1;

    return (double) w->getPropWindow();
}

double GuiReceiver::getWindowVisible(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (!w)
        return 0;

    return (double) w->isVisible();
}


void GuiReceiver::setPropWindow(QString name, double arg2)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (!w)
        return;

    int flags = (int) arg2;

    w->setPropWindow(flags);
}

void GuiReceiver::setWindowTitle(QString name, QString title)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (!w)
    {
        cvNamedWindow(name.toLatin1().data());
        w = icvFindWindowByName(name);
    }

    if (!w)
        return;

    w->setWindowTitle(title);
}

CvRect GuiReceiver::getWindowRect(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (!w)
        return cvRect(-1, -1, -1, -1);

    return w->getWindowRect();
}

double GuiReceiver::isFullScreen(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (!w)
        return -1;

    return w->isFullScreen() ? CV_WINDOW_FULLSCREEN : CV_WINDOW_NORMAL;
}


void GuiReceiver::toggleFullScreen(QString name, double arg2)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (!w)
        return;

    int flags = (int) arg2;

    w->toggleFullScreen(flags);
}


void GuiReceiver::createWindow(QString name, int flags)
{
    if (!qApp)
        CV_Error(CV_StsNullPtr, "NULL session handler" );

    // Check the name in the storage
    if (icvFindWindowByName(name.toLatin1().data()))
    {
        return;
    }

    nb_windows++;
    new CvWindow(name, flags);
    cvWaitKey(1);
}


void GuiReceiver::timeOut()
{
    bTimeOut = true;
}


void GuiReceiver::displayInfo(QString name, QString text, int delayms)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (w)
        w->displayInfo(text, delayms);
}


void GuiReceiver::displayStatusBar(QString name, QString text, int delayms)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (w)
        w->displayStatusBar(text, delayms);
}


void GuiReceiver::showImage(QString name, void* arr)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (!w) //as observed in the previous implementation (W32, GTK), create a new window is the pointer returned is null
    {
        cvNamedWindow(name.toLatin1().data());
        w = icvFindWindowByName(name);
    }

    if (!w || !arr)
        return; // keep silence here.

    if (w->isOpenGl())
    {
        CvMat* mat, stub;

        mat = cvGetMat(arr, &stub);

        cv::Mat im = cv::cvarrToMat(mat);
        cv::imshow(name.toUtf8().data(), im);
    }
    else
    {
        w->updateImage(arr);
    }

    if (w->isHidden())
        w->show();
}


void GuiReceiver::destroyWindow(QString name)
{

    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (w)
    {
        w->close();

        //in not-multiThreads mode, looks like the window is hidden but not deleted
        //so I do it manually
        //otherwise QApplication do it for me if the exec command was executed (in multiThread mode)
        if (!multiThreads)
            delete w;
    }
}


void GuiReceiver::destroyAllWindow()
{
    if (!qApp)
        CV_Error(CV_StsNullPtr, "NULL session handler" );

    if (multiThreads)
    {
        // WARNING: this could even close windows from an external parent app
        //#TODO check externalQAppExists and in case it does, close windows carefully,
        //      i.e. apply the className-check from below...
        qApp->closeAllWindows();
    }
    else
    {
        bool isWidgetDeleted = true;
        while(isWidgetDeleted)
        {
            isWidgetDeleted = false;
            QWidgetList list = QApplication::topLevelWidgets();
            for (int i = 0; i < list.count(); i++)
            {
                QObject *obj = list.at(i);
                if (obj->metaObject()->className() == QString("CvWindow"))
                {
                    delete obj;
                    isWidgetDeleted = true;
                    break;
                }
            }
        }
    }
}


void GuiReceiver::moveWindow(QString name, int x, int y)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (w)
        w->move(x, y);
}


void GuiReceiver::resizeWindow(QString name, int width, int height)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (w)
    {
        w->showNormal();
        w->setViewportSize(QSize(width, height));
    }
}


void GuiReceiver::enablePropertiesButtonEachWindow()
{
    //For each window, enable window property button
    foreach (QWidget* widget, QApplication::topLevelWidgets())
    {
        if (widget->isWindow() && !widget->parentWidget()) //is a window without parent
        {
            CvWinModel* temp = (CvWinModel*) widget;
            if (temp->type == type_CvWindow)
            {
                CvWindow* w = (CvWindow*) widget;

                //active window properties button
                w->enablePropertiesButton();
            }
        }
    }
}


void GuiReceiver::addButton(QString button_name, int button_type, int initial_button_state, void* on_change, void* userdata)
{
    if (!global_control_panel)
        return;

    QPointer<CvButtonbar> b;

    if (global_control_panel->myLayout->count() == 0) //if that is the first button attach to the control panel, create a new button bar
    {
        b = CvWindow::createButtonBar(button_name); //the bar has the name of the first button attached to it
        enablePropertiesButtonEachWindow();

    }
    else
    {
        CvBar* lastbar = (CvBar*) global_control_panel->myLayout->itemAt(global_control_panel->myLayout->count() - 1);

        // if last bar is a trackbar or the user requests a new buttonbar, create a new buttonbar
        // else, attach to the current bar
        if (lastbar->type == type_CvTrackbar || cv::QT_NEW_BUTTONBAR & button_type)
            b = CvWindow::createButtonBar(button_name); //the bar has the name of the first button attached to it
        else
            b = (CvButtonbar*) lastbar;

    }

    // unset buttonbar flag
    button_type = button_type & ~cv::QT_NEW_BUTTONBAR;

    b->addButton(button_name, (CvButtonCallback) on_change, userdata, button_type, initial_button_state);
}


void GuiReceiver::addSlider2(QString bar_name, QString window_name, void* value, int count, void* on_change, void *userdata)
{
    QBoxLayout *layout = NULL;
    QPointer<CvWindow> w;

    if (!window_name.isEmpty())
    {
        w = icvFindWindowByName(window_name);

        if (!w)
            return;
    }
    else
    {
        if (global_control_panel)
            layout = global_control_panel->myLayout;
    }

    QPointer<CvTrackbar> t = icvFindTrackBarByName(bar_name.toLatin1().data(), window_name.toLatin1().data(), layout);

    if (t) //trackbar exists
        return;

    if (count <= 0) //count is the max value of the slider, so must be bigger than 0
        CV_Error(CV_StsNullPtr, "Max value of the slider must be bigger than 0" );

    CvWindow::addSlider2(w, bar_name, (int*)value, count, (CvTrackbarCallback2) on_change, userdata);
}


void GuiReceiver::addSlider(QString bar_name, QString window_name, void* value, int count, void* on_change)
{
    QBoxLayout *layout = NULL;
    QPointer<CvWindow> w;

    if (!window_name.isEmpty())
    {
        w = icvFindWindowByName(window_name);

        if (!w)
            return;
    }
    else
    {
        if (global_control_panel)
            layout = global_control_panel->myLayout;
    }

    QPointer<CvTrackbar> t = icvFindTrackBarByName(bar_name.toLatin1().data(), window_name.toLatin1().data(), layout);

    if (t) //trackbar exists
        return;

    if (!value)
        CV_Error(CV_StsNullPtr, "NULL value pointer" );

    if (count <= 0) //count is the max value of the slider, so must be bigger than 0
        CV_Error(CV_StsNullPtr, "Max value of the slider must be bigger than 0" );

    CvWindow::addSlider(w, bar_name, (int*)value, count, (CvTrackbarCallback) on_change);
}


int GuiReceiver::start()
{
    return qApp->exec();
}


void GuiReceiver::setOpenGlDrawCallback(QString name, void* callback, void* userdata)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (w)
        w->setOpenGlDrawCallback((CvOpenGlDrawCallback) callback, userdata);
}

void GuiReceiver::setOpenGlContext(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (w)
        w->makeCurrentOpenGlContext();
}

void GuiReceiver::updateWindow(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (w)
        w->updateGl();
}

double GuiReceiver::isOpenGl(QString name)
{
    double result = -1;

    QPointer<CvWindow> w = icvFindWindowByName(name);

    if (w)
        result = (double) w->isOpenGl();

    return result;
}


//////////////////////////////////////////////////////
// CvTrackbar


CvTrackbar::CvTrackbar(CvWindow* arg, QString name, int* value, int _count, CvTrackbarCallback2 on_change, void* data)
{
    callback = NULL;
    callback2 = on_change;
    userdata = data;

    create(arg, name, value, _count);
}


CvTrackbar::CvTrackbar(CvWindow* arg, QString name, int* value, int _count, CvTrackbarCallback on_change)
{
    callback = on_change;
    callback2 = NULL;
    userdata = NULL;

    create(arg, name, value, _count);
}


void CvTrackbar::create(CvWindow* arg, QString name, int* value, int _count)
{
    type = type_CvTrackbar;
    myparent = arg;
    name_bar = name;
    setObjectName(name_bar);
    dataSlider = value;

    slider = new QSlider(Qt::Horizontal);
    slider->setFocusPolicy(Qt::StrongFocus);
    slider->setMinimum(0);
    slider->setMaximum(_count);
    slider->setPageStep(5);
    if (dataSlider)
        slider->setValue(*dataSlider);
    slider->setTickPosition(QSlider::TicksBelow);


    //Change style of the Slider
    //slider->setStyleSheet(str_Trackbar_css);

    QFile qss(":/stylesheet-trackbar");
    if (qss.open(QFile::ReadOnly))
    {
        slider->setStyleSheet(QLatin1String(qss.readAll()));
        qss.close();
    }


    //this next line does not work if we change the style with a stylesheet, why ? (bug in QT ?)
    //slider->setTickPosition(QSlider::TicksBelow);
    label = new QPushButton;
    label->setFlat(true);
    setLabel(slider->value());


    QObject::connect(slider, SIGNAL(valueChanged(int)), this, SLOT(update(int)));

    QObject::connect(label, SIGNAL(clicked()), this, SLOT(createDialog()));

    //label->setStyleSheet("QPushButton:disabled {color: black}");

    addWidget(label, Qt::AlignLeft);//name + value
    addWidget(slider, Qt::AlignCenter);//slider
}


void CvTrackbar::createDialog()
{
    bool ok = false;

    //crash if I access the values directly and give them to QInputDialog, so do a copy first.
    int value = slider->value();
    int step = slider->singleStep();
    int min = slider->minimum();
    int max = slider->maximum();

    int i =
#if QT_VERSION >= 0x040500
        QInputDialog::getInt
#else
        QInputDialog::getInteger
#endif
        (this->parentWidget(),
        tr("Slider %1").arg(name_bar),
        tr("New value:"),
        value,
        min,
        max,
        step,
        &ok);

    if (ok)
        slider->setValue(i);
}


void CvTrackbar::update(int myvalue)
{
    setLabel(myvalue);

    if (dataSlider)
        *dataSlider = myvalue;
    if (callback)
    {
        callback(myvalue);
        return;
    }

    if (callback2)
    {
        callback2(myvalue, userdata);
        return;
    }
}


void CvTrackbar::setLabel(int myvalue)
{
    QString nameNormalized = name_bar.leftJustified( 10, ' ', false );
    QString valueMaximum = QString("%1").arg(slider->maximum());
    QString str = QString("%1 (%2/%3)").arg(nameNormalized).arg(myvalue,valueMaximum.length(),10,QChar('0')).arg(valueMaximum);
    label->setText(str);
}


//////////////////////////////////////////////////////
// CvButtonbar


//here CvButtonbar class
CvButtonbar::CvButtonbar(QWidget* arg,  QString arg2)
{
    type = type_CvButtonbar;
    myparent = arg;
    name_bar = arg2;
    setObjectName(name_bar);

    group_button = new QButtonGroup(this);
}


void CvButtonbar::setLabel()
{
    QString nameNormalized = name_bar.leftJustified(10, ' ', true);
    label->setText(nameNormalized);
}


void CvButtonbar::addButton(QString name, CvButtonCallback call, void* userdata,  int button_type, int initial_button_state)
{
    QString button_name = name;

    if (button_name == "")
        button_name = tr("button %1").arg(this->count());

    QPointer<QAbstractButton> button;

    if (button_type == CV_PUSH_BUTTON)
        button = (QAbstractButton*) new CvPushButton(this, button_name,call, userdata);

    if (button_type == CV_CHECKBOX)
        button = (QAbstractButton*) new CvCheckBox(this, button_name,call, userdata, initial_button_state);

    if (button_type == CV_RADIOBOX)
    {
        button = (QAbstractButton*) new CvRadioButton(this, button_name,call, userdata, initial_button_state);
        group_button->addButton(button);
    }

    if (button)
    {
        if (button_type == CV_PUSH_BUTTON)
            QObject::connect(button, SIGNAL(clicked(bool)), button, SLOT(callCallBack(bool)));
        else
            QObject::connect(button, SIGNAL(toggled(bool)), button, SLOT(callCallBack(bool)));

        addWidget(button, Qt::AlignCenter);
    }
}


//////////////////////////////////////////////////////
// Buttons


//buttons here
CvPushButton::CvPushButton(CvButtonbar* arg1, QString arg2, CvButtonCallback arg3, void* arg4)
{
    myparent = arg1;
    button_name = arg2;
    callback = arg3;
    userdata = arg4;

    setObjectName(button_name);
    setText(button_name);

    if (isChecked())
        callCallBack(true);
}


void CvPushButton::callCallBack(bool checked)
{
    if (callback)
        callback(checked, userdata);
}


CvCheckBox::CvCheckBox(CvButtonbar* arg1, QString arg2, CvButtonCallback arg3, void* arg4, int initial_button_state)
{
    myparent = arg1;
    button_name = arg2;
    callback = arg3;
    userdata = arg4;

    setObjectName(button_name);
    setCheckState((initial_button_state == 1 ? Qt::Checked : Qt::Unchecked));
    setText(button_name);

    if (isChecked())
        callCallBack(true);
}


void CvCheckBox::callCallBack(bool checked)
{
    if (callback)
        callback(checked, userdata);
}


CvRadioButton::CvRadioButton(CvButtonbar* arg1, QString arg2, CvButtonCallback arg3, void* arg4, int initial_button_state)
{
    myparent = arg1;
    button_name = arg2;
    callback = arg3;
    userdata = arg4;

    setObjectName(button_name);
    setChecked(initial_button_state);
    setText(button_name);

    if (isChecked())
        callCallBack(true);
}

void CvRadioButton::callCallBack(bool checked)
{
    if (callback)
        callback(checked, userdata);
}


//////////////////////////////////////////////////////
// CvWinProperties


//here CvWinProperties class
CvWinProperties::CvWinProperties(QString name_paraWindow, QObject* /*parent*/)
{
    //setParent(parent);
    type = type_CvWinProperties;
    setWindowFlags(Qt::Tool);
    setContentsMargins(0, 0, 0, 0);
    setWindowTitle(name_paraWindow);
    setObjectName(name_paraWindow);
    resize(100, 50);

    myLayout = new QBoxLayout(QBoxLayout::TopToBottom);
    myLayout->setObjectName(QString::fromUtf8("boxLayout"));
    myLayout->setContentsMargins(0, 0, 0, 0);
    myLayout->setSpacing(0);
#if QT_VERSION < QT_VERSION_CHECK(5, 13, 0)
    myLayout->setMargin(0);
#endif
    myLayout->setSizeConstraint(QLayout::SetFixedSize);
    setLayout(myLayout);

    hide();
}


void CvWinProperties::closeEvent(QCloseEvent* e)
{
    e->accept(); //intersept the close event (not sure I really need it)
    //an hide event is also sent. I will intercept it and do some processing
}


void CvWinProperties::showEvent(QShowEvent* evnt)
{
    //why -1,-1 ?: do this trick because the first time the code is run,
    //no value pos was saved so we let Qt move the window in the middle of its parent (event ignored).
    //then hide will save the last position and thus, we want to retrieve it (event accepted).
    QPoint mypos(-1, -1);
    QSettings settings("OpenCV2", objectName());
    mypos = settings.value("pos", mypos).toPoint();

    if (mypos.x() >= 0)
    {
        move(mypos);
        evnt->accept();
    }
    else
    {
        evnt->ignore();
    }
}


void CvWinProperties::hideEvent(QHideEvent* evnt)
{
    QSettings settings("OpenCV2", objectName());
    settings.setValue("pos", pos()); //there is an offset of 6 pixels (so the window's position is wrong -- why ?)
    evnt->accept();
}


CvWinProperties::~CvWinProperties()
{
    //clear the setting pos
    QSettings settings("OpenCV2", objectName());
    settings.remove("pos");
}


//////////////////////////////////////////////////////
// CvWindow


CvWindow::CvWindow(QString name, int arg2)
{
    type = type_CvWindow;

    param_flags = arg2 & 0x0000000F;
    param_gui_mode = arg2 & 0x000000F0;
    param_ratio_mode =  arg2 & 0x00000F00;

    //setAttribute(Qt::WA_DeleteOnClose); //in other case, does not release memory
    setContentsMargins(0, 0, 0, 0);
    setWindowTitle(name);
    setObjectName(name);

    setFocus( Qt::PopupFocusReason ); //#1695 arrow keys are not received without the explicit focus

    resize(400, 300);
    setMinimumSize(1, 1);

    //1: create control panel
    if (!global_control_panel)
        global_control_panel = createParameterWindow();

    //2: Layouts
    createBarLayout();
    createGlobalLayout();

    //3: my view
#ifndef HAVE_QT_OPENGL
    if (arg2 & CV_WINDOW_OPENGL)
        CV_Error( CV_OpenGlNotSupported, "Library was built without OpenGL support" );
    mode_display = CV_MODE_NORMAL;
#else
    mode_display = arg2 & CV_WINDOW_OPENGL ? CV_MODE_OPENGL : CV_MODE_NORMAL;
    if (mode_display == CV_MODE_OPENGL)
        param_gui_mode = CV_GUI_NORMAL;
#endif
    createView();

    //4: shortcuts and actions
    //5: toolBar and statusbar
    if (param_gui_mode == CV_GUI_EXPANDED)
    {
        createActions();
        createShortcuts();

        createToolBar();
        createStatusBar();
    }

    //Now attach everything
    if (myToolBar)
        myGlobalLayout->addWidget(myToolBar, Qt::AlignCenter);

    myGlobalLayout->addWidget(myView->getWidget(), Qt::AlignCenter);

    myGlobalLayout->addLayout(myBarLayout, Qt::AlignCenter);

    if (myStatusBar)
        myGlobalLayout->addWidget(myStatusBar, Qt::AlignCenter);

    setLayout(myGlobalLayout);
    show();
}


CvWindow::~CvWindow()
{
    if (guiMainThread)
        guiMainThread->isLastWindow();
}


void CvWindow::setMouseCallBack(CvMouseCallback callback, void* param)
{
    myView->setMouseCallBack(callback, param);
}


void CvWindow::writeSettings()
{
    //organisation and application's name
    QSettings settings("OpenCV2", QFileInfo(QApplication::applicationFilePath()).fileName());

    settings.setValue("pos", pos());
    settings.setValue("size", size());
    settings.setValue("mode_resize" ,param_flags);
    settings.setValue("mode_gui", param_gui_mode);

    myView->writeSettings(settings);

    icvSaveTrackbars(&settings);

    if (global_control_panel)
    {
        icvSaveControlPanel();
        settings.setValue("posPanel", global_control_panel->pos());
    }
}



//TODO: load CV_GUI flag (done) and act accordingly (create win property if needed and attach trackbars)
void CvWindow::readSettings()
{
    //organisation and application's name
    QSettings settings("OpenCV2", QFileInfo(QApplication::applicationFilePath()).fileName());

    QPoint _pos = settings.value("pos", QPoint(200, 200)).toPoint();
    QSize _size = settings.value("size", QSize(400, 400)).toSize();

    param_flags = settings.value("mode_resize", param_flags).toInt();
    param_gui_mode = settings.value("mode_gui", param_gui_mode).toInt();

    param_flags = settings.value("mode_resize", param_flags).toInt();

    myView->readSettings(settings);

    //trackbar here
    icvLoadTrackbars(&settings);

    resize(_size);
    move(_pos);

    if (global_control_panel)
    {
        icvLoadControlPanel();
        global_control_panel->move(settings.value("posPanel", global_control_panel->pos()).toPoint());
    }
}


double CvWindow::getRatio()
{
    return myView->getRatio();
}


void CvWindow::setRatio(int flags)
{
    myView->setRatio(flags);
}

CvRect CvWindow::getWindowRect()
{
    QWidget* view = myView->getWidget();
    QRect local_rc = view->geometry(); // http://doc.qt.io/qt-5/application-windows.html#window-geometry
    QPoint global_pos = /*view->*/mapToGlobal(QPoint(local_rc.x(), local_rc.y()));
    return cvRect(global_pos.x(), global_pos.y(), local_rc.width(), local_rc.height());
}

int CvWindow::getPropWindow()
{
    return param_flags;
}


void CvWindow::setPropWindow(int flags)
{
    if (param_flags == flags) //nothing to do
        return;

    switch(flags)
    {
    case CV_WINDOW_NORMAL:
        myGlobalLayout->setSizeConstraint(QLayout::SetMinAndMaxSize);
        param_flags = flags;

        break;

    case CV_WINDOW_AUTOSIZE:
        myGlobalLayout->setSizeConstraint(QLayout::SetFixedSize);
        param_flags = flags;

        break;

    default:
        ;
    }
}

void CvWindow::toggleFullScreen(int flags)
{
    if (isFullScreen() && flags == CV_WINDOW_NORMAL)
    {
        showTools();
        showNormal();
        return;
    }

    if (!isFullScreen() && flags == CV_WINDOW_FULLSCREEN)
    {
        hideTools();
        showFullScreen();
        return;
    }
}


void CvWindow::updateImage(void* arr)
{
    myView->updateImage(arr);
}


void CvWindow::displayInfo(QString text, int delayms)
{
    myView->startDisplayInfo(text, delayms);
}


void CvWindow::displayStatusBar(QString text, int delayms)
{
    if (myStatusBar)
        myStatusBar->showMessage(text, delayms);
}


void CvWindow::enablePropertiesButton()
{
    if (!vect_QActions.empty())
        vect_QActions[10]->setDisabled(false);
}


CvButtonbar* CvWindow::createButtonBar(QString name_bar)
{
    QPointer<CvButtonbar> t = new CvButtonbar(global_control_panel, name_bar);
    t->setAlignment(Qt::AlignHCenter);

    QPointer<QBoxLayout> myLayout = global_control_panel->myLayout;

    myLayout->insertLayout(myLayout->count(), t);

    return t;
}


void CvWindow::addSlider(CvWindow* w, QString name, int* value, int count, CvTrackbarCallback on_change)
{
    QPointer<CvTrackbar> t = new CvTrackbar(w, name, value, count, on_change);
    t->setAlignment(Qt::AlignHCenter);

    QPointer<QBoxLayout> myLayout;

    if (w)
    {
        myLayout = w->myBarLayout;
    }
    else
    {
        myLayout = global_control_panel->myLayout;

        //if first one, enable control panel
        if (myLayout->count() == 0)
            guiMainThread->enablePropertiesButtonEachWindow();
    }

    myLayout->insertLayout(myLayout->count(), t);
}


void CvWindow::addSlider2(CvWindow* w, QString name, int* value, int count, CvTrackbarCallback2 on_change, void* userdata)
{
    QPointer<CvTrackbar> t = new CvTrackbar(w, name, value, count, on_change, userdata);
    t->setAlignment(Qt::AlignHCenter);

    QPointer<QBoxLayout> myLayout;

    if (w)
    {
        myLayout = w->myBarLayout;
    }
    else
    {
        myLayout = global_control_panel->myLayout;

        //if first one, enable control panel
        if (myLayout->count() == 0)
            guiMainThread->enablePropertiesButtonEachWindow();
    }

    myLayout->insertLayout(myLayout->count(), t);
}


void CvWindow::setOpenGlDrawCallback(CvOpenGlDrawCallback callback, void* userdata)
{
    myView->setOpenGlDrawCallback(callback, userdata);
}


void CvWindow::makeCurrentOpenGlContext()
{
    myView->makeCurrentOpenGlContext();
}


void CvWindow::updateGl()
{
    myView->updateGl();
}


bool CvWindow::isOpenGl()
{
    return mode_display == CV_MODE_OPENGL;
}


void CvWindow::setViewportSize(QSize _size)
{
    resize(_size);
    myView->setSize(_size);
}


void CvWindow::createBarLayout()
{
    myBarLayout = new QBoxLayout(QBoxLayout::TopToBottom);
    myBarLayout->setObjectName(QString::fromUtf8("barLayout"));
    myBarLayout->setContentsMargins(0, 0, 0, 0);
    myBarLayout->setSpacing(0);
#if QT_VERSION < QT_VERSION_CHECK(5, 13, 0)
    myBarLayout->setMargin(0);
#endif
}


void CvWindow::createGlobalLayout()
{
    myGlobalLayout = new QBoxLayout(QBoxLayout::TopToBottom);
    myGlobalLayout->setObjectName(QString::fromUtf8("boxLayout"));
    myGlobalLayout->setContentsMargins(0, 0, 0, 0);
    myGlobalLayout->setSpacing(0);
#if QT_VERSION < QT_VERSION_CHECK(5, 13, 0)
    myGlobalLayout->setMargin(0);
#endif
    setMinimumSize(1, 1);

    if (param_flags == CV_WINDOW_AUTOSIZE)
        myGlobalLayout->setSizeConstraint(QLayout::SetFixedSize);
    else if (param_flags == CV_WINDOW_NORMAL)
        myGlobalLayout->setSizeConstraint(QLayout::SetMinAndMaxSize);
}


void CvWindow::createView()
{
#ifdef HAVE_QT_OPENGL
    if (isOpenGl())
        myView = new OpenGlViewPort(this);
    else
#endif
        myView = new DefaultViewPort(this, param_ratio_mode);
}


void CvWindow::createActions()
{
    vect_QActions.resize(11);

    QWidget* view = myView->getWidget();

    //if the shortcuts are changed in window_QT.h, we need to update the tooltip manually
    vect_QActions[0] = new QAction(QIcon(":/left-icon"), "Panning left (CTRL+arrowLEFT)", this);
    vect_QActions[0]->setIconVisibleInMenu(true);
    QObject::connect(vect_QActions[0], SIGNAL(triggered()), view, SLOT(siftWindowOnLeft()));

    vect_QActions[1] = new QAction(QIcon(":/right-icon"), "Panning right (CTRL+arrowRIGHT)", this);
    vect_QActions[1]->setIconVisibleInMenu(true);
    QObject::connect(vect_QActions[1], SIGNAL(triggered()), view, SLOT(siftWindowOnRight()));

    vect_QActions[2] = new QAction(QIcon(":/up-icon"), "Panning up (CTRL+arrowUP)", this);
    vect_QActions[2]->setIconVisibleInMenu(true);
    QObject::connect(vect_QActions[2], SIGNAL(triggered()), view, SLOT(siftWindowOnUp()));

    vect_QActions[3] = new QAction(QIcon(":/down-icon"), "Panning down (CTRL+arrowDOWN)", this);
    vect_QActions[3]->setIconVisibleInMenu(true);
    QObject::connect(vect_QActions[3], SIGNAL(triggered()), view, SLOT(siftWindowOnDown()) );

    vect_QActions[4] = new QAction(QIcon(":/zoom_x1-icon"), "Zoom x1 (CTRL+P)", this);
    vect_QActions[4]->setIconVisibleInMenu(true);
    QObject::connect(vect_QActions[4], SIGNAL(triggered()), view, SLOT(resetZoom()));

    vect_QActions[5] = new QAction(QIcon(":/imgRegion-icon"), tr("Zoom x%1 (see label) (CTRL+X)").arg(threshold_zoom_img_region), this);
    vect_QActions[5]->setIconVisibleInMenu(true);
    QObject::connect(vect_QActions[5], SIGNAL(triggered()), view, SLOT(imgRegion()));

    vect_QActions[6] = new QAction(QIcon(":/zoom_in-icon"), "Zoom in (CTRL++)", this);
    vect_QActions[6]->setIconVisibleInMenu(true);
    QObject::connect(vect_QActions[6], SIGNAL(triggered()), view, SLOT(ZoomIn()));

    vect_QActions[7] = new QAction(QIcon(":/zoom_out-icon"), "Zoom out (CTRL+-)", this);
    vect_QActions[7]->setIconVisibleInMenu(true);
    QObject::connect(vect_QActions[7], SIGNAL(triggered()), view, SLOT(ZoomOut()));

    vect_QActions[8] = new QAction(QIcon(":/save-icon"), "Save current image (CTRL+S)", this);
    vect_QActions[8]->setIconVisibleInMenu(true);
    QObject::connect(vect_QActions[8], SIGNAL(triggered()), view, SLOT(saveView()));

    vect_QActions[9] = new QAction(QIcon(":/copy_clipbrd-icon"), "Copy image to clipboard (CTRL+C)", this);
    vect_QActions[9]->setIconVisibleInMenu(true);
    QObject::connect(vect_QActions[9], SIGNAL(triggered()), view, SLOT(copy2Clipbrd()));

    vect_QActions[10] = new QAction(QIcon(":/properties-icon"), "Display properties window (CTRL+P)", this);
    vect_QActions[10]->setIconVisibleInMenu(true);
    QObject::connect(vect_QActions[10], SIGNAL(triggered()), this, SLOT(displayPropertiesWin()));

    if (global_control_panel->myLayout->count() == 0)
        vect_QActions[10]->setDisabled(true);
}


void CvWindow::createShortcuts()
{
    vect_QShortcuts.resize(11);

    QWidget* view = myView->getWidget();

    vect_QShortcuts[0] = new QShortcut(shortcut_panning_left, this);
    QObject::connect(vect_QShortcuts[0], SIGNAL(activated()), view, SLOT(siftWindowOnLeft()));

    vect_QShortcuts[1] = new QShortcut(shortcut_panning_right, this);
    QObject::connect(vect_QShortcuts[1], SIGNAL(activated()), view, SLOT(siftWindowOnRight()));

    vect_QShortcuts[2] = new QShortcut(shortcut_panning_up, this);
    QObject::connect(vect_QShortcuts[2], SIGNAL(activated()), view, SLOT(siftWindowOnUp()));

    vect_QShortcuts[3] = new QShortcut(shortcut_panning_down, this);
    QObject::connect(vect_QShortcuts[3], SIGNAL(activated()), view, SLOT(siftWindowOnDown()));

    vect_QShortcuts[4] = new QShortcut(shortcut_zoom_normal, this);
    QObject::connect(vect_QShortcuts[4], SIGNAL(activated()), view, SLOT(resetZoom()));

    vect_QShortcuts[5] = new QShortcut(shortcut_zoom_imgRegion, this);
    QObject::connect(vect_QShortcuts[5], SIGNAL(activated()), view, SLOT(imgRegion()));

    vect_QShortcuts[6] = new QShortcut(shortcut_zoom_in, this);
    QObject::connect(vect_QShortcuts[6], SIGNAL(activated()), view, SLOT(ZoomIn()));

    vect_QShortcuts[7] = new QShortcut(shortcut_zoom_out, this);
    QObject::connect(vect_QShortcuts[7], SIGNAL(activated()), view, SLOT(ZoomOut()));

    vect_QShortcuts[8] = new QShortcut(shortcut_save_img, this);
    QObject::connect(vect_QShortcuts[8], SIGNAL(activated()), view, SLOT(saveView()));

    vect_QShortcuts[9] = new QShortcut(shortcut_copy_clipbrd, this);
    QObject::connect(vect_QShortcuts[9], SIGNAL(activated()), view, SLOT(copy2Clipbrd()));

    vect_QShortcuts[10] = new QShortcut(shortcut_properties_win, this);
    QObject::connect(vect_QShortcuts[10], SIGNAL(activated()), this, SLOT(displayPropertiesWin()));
}


void CvWindow::createToolBar()
{
    myToolBar = new QToolBar(this);
    myToolBar->setFloatable(false); //is not a window

    foreach (QAction *a, vect_QActions)
        myToolBar->addAction(a);
}


void CvWindow::createStatusBar()
{
    myStatusBar = new QStatusBar(this);
    myStatusBar->setSizeGripEnabled(false);
    myStatusBar->setFixedHeight(20);
    myStatusBar->setMinimumWidth(1);
    myStatusBar_msg = new QLabel;

    //I comment this because if we change the style, myview (the picture)
    //will not be the correct size anymore (will lost 2 pixel because of the borders)

    //myStatusBar_msg->setFrameStyle(QFrame::Raised);

    myStatusBar_msg->setAlignment(Qt::AlignHCenter);
    myStatusBar->addWidget(myStatusBar_msg);
}


void CvWindow::hideTools()
{
    if (myToolBar)
        myToolBar->hide();

    if (myStatusBar)
        myStatusBar->hide();

    if (global_control_panel)
        global_control_panel->hide();
}


void CvWindow::showTools()
{
    if (myToolBar)
        myToolBar->show();

    if (myStatusBar)
        myStatusBar->show();
}


CvWinProperties* CvWindow::createParameterWindow()
{
    QString name_paraWindow = QFileInfo(QApplication::applicationFilePath()).fileName() + " settings";

    CvWinProperties* result = new CvWinProperties(name_paraWindow, guiMainThread);

    return result;
}


void CvWindow::displayPropertiesWin()
{
    if (global_control_panel->isHidden())
        global_control_panel->show();
    else
        global_control_panel->hide();
}

static bool isTranslatableKey(Qt::Key key)
{
    // https://github.com/opencv/opencv/issues/21899
    // https://doc.qt.io/qt-5/qt.html#Key-enum
    // https://doc.qt.io/qt-6/qt.html#Key-enum
    // https://github.com/qt/qtbase/blob/dev/src/testlib/qasciikey.cpp

    bool ret = false;

    switch ( key )
    {
        // Special keys
        case Qt::Key_Escape:
        case Qt::Key_Tab:
        case Qt::Key_Backtab:
        case Qt::Key_Backspace:
        case Qt::Key_Enter:
        case Qt::Key_Return:
            ret = true;
            break;

        // latin-1 keys.
        default:
        ret = (
            ( ( Qt::Key_Space        <= key ) && ( key <= Qt::Key_AsciiTilde ) ) // 0x20--0x7e
            ||
            ( ( Qt::Key_nobreakspace <= key ) && ( key <= Qt::Key_ssharp     ) ) // 0x0a0--0x0de
            ||
            ( key == Qt::Key_division )                                          // 0x0f7
            ||
            ( key == Qt::Key_ydiaeresis )                                        // 0x0ff
        );
        break;
    }

    return ret;
}

//Need more test here !
void CvWindow::keyPressEvent(QKeyEvent *evnt)
{
    int key = evnt->key();
    const Qt::Key qtkey = static_cast<Qt::Key>(key);

    if ( isTranslatableKey( qtkey ) )
        key = static_cast<int>( QTest::keyToAscii( qtkey ) );
    else
        key = evnt->nativeVirtualKey(); //same codes as returned by GTK-based backend

    //control plus (Z, +, -, up, down, left, right) are used for zoom/panning functions
    if (evnt->modifiers() != Qt::ControlModifier)
    {
        mutexKey.lock();
        last_key = key;
        mutexKey.unlock();
        key_pressed.wakeAll();
        //evnt->accept();
    }

    QWidget::keyPressEvent(evnt);
}


void CvWindow::icvLoadControlPanel()
{
    QSettings settings("OpenCV2", QFileInfo(QApplication::applicationFilePath()).fileName() + " control panel");

    int bsize = settings.beginReadArray("bars");

    if (bsize == global_control_panel->myLayout->layout()->count())
    {
        for (int i = 0; i < bsize; ++i)
        {
            CvBar* t = (CvBar*) global_control_panel->myLayout->layout()->itemAt(i);
            settings.setArrayIndex(i);
            if (t->type == type_CvTrackbar)
            {
                if (t->name_bar == settings.value("namebar").toString())
                {
                    ((CvTrackbar*)t)->slider->setValue(settings.value("valuebar").toInt());
                }
            }
            if (t->type == type_CvButtonbar)
            {
                int subsize = settings.beginReadArray(QString("buttonbar%1").arg(i));

                if ( subsize == ((CvButtonbar*)t)->layout()->count() )
                    icvLoadButtonbar((CvButtonbar*)t,&settings);

                settings.endArray();
            }
        }
    }

    settings.endArray();
}


void CvWindow::icvSaveControlPanel()
{
    QSettings settings("OpenCV2", QFileInfo(QApplication::applicationFilePath()).fileName()+" control panel");

    settings.beginWriteArray("bars");

    for (int i = 0; i < global_control_panel->myLayout->layout()->count(); ++i)
    {
        CvBar* t = (CvBar*) global_control_panel->myLayout->layout()->itemAt(i);
        settings.setArrayIndex(i);
        if (t->type == type_CvTrackbar)
        {
            settings.setValue("namebar", QString(t->name_bar));
            settings.setValue("valuebar",((CvTrackbar*)t)->slider->value());
        }
        if (t->type == type_CvButtonbar)
        {
            settings.beginWriteArray(QString("buttonbar%1").arg(i));
            icvSaveButtonbar((CvButtonbar*)t,&settings);
            settings.endArray();
        }
    }

    settings.endArray();
}


void CvWindow::icvSaveButtonbar(CvButtonbar* b, QSettings* settings)
{
    for (int i = 0, count = b->layout()->count(); i < count; ++i)
    {
        settings->setArrayIndex(i);

        QWidget* temp = (QWidget*) b->layout()->itemAt(i)->widget();
        QString myclass(QLatin1String(temp->metaObject()->className()));

        if (myclass == "CvPushButton")
        {
            CvPushButton* button = (CvPushButton*) temp;
            settings->setValue("namebutton", button->text());
            settings->setValue("valuebutton", int(button->isChecked()));
        }
        else if (myclass == "CvCheckBox")
        {
            CvCheckBox* button = (CvCheckBox*) temp;
            settings->setValue("namebutton", button->text());
            settings->setValue("valuebutton", int(button->isChecked()));
        }
        else if (myclass == "CvRadioButton")
        {
            CvRadioButton* button = (CvRadioButton*) temp;
            settings->setValue("namebutton", button->text());
            settings->setValue("valuebutton", int(button->isChecked()));
        }
    }
}


void CvWindow::icvLoadButtonbar(CvButtonbar* b, QSettings* settings)
{
    for (int i = 0, count = b->layout()->count(); i < count; ++i)
    {
        settings->setArrayIndex(i);

        QWidget* temp = (QWidget*) b->layout()->itemAt(i)->widget();
        QString myclass(QLatin1String(temp->metaObject()->className()));

        if (myclass == "CvPushButton")
        {
            CvPushButton* button = (CvPushButton*) temp;

            if (button->text() == settings->value("namebutton").toString())
                button->setChecked(settings->value("valuebutton").toInt());
        }
        else if (myclass == "CvCheckBox")
        {
            CvCheckBox* button = (CvCheckBox*) temp;

            if (button->text() == settings->value("namebutton").toString())
                button->setChecked(settings->value("valuebutton").toInt());
        }
        else if (myclass == "CvRadioButton")
        {
            CvRadioButton* button = (CvRadioButton*) temp;

            if (button->text() == settings->value("namebutton").toString())
                button->setChecked(settings->value("valuebutton").toInt());
        }

    }
}


void CvWindow::icvLoadTrackbars(QSettings* settings)
{
    int bsize = settings->beginReadArray("trackbars");

    //trackbar are saved in the same order, so no need to use icvFindTrackbarByName

    if (myBarLayout->layout()->count() == bsize) //if not the same number, the window saved and loaded is not the same (nb trackbar not equal)
    {
        for (int i = 0; i < bsize; ++i)
        {
            settings->setArrayIndex(i);

            CvTrackbar* t = (CvTrackbar*) myBarLayout->layout()->itemAt(i);

            if (t->name_bar == settings->value("name").toString())
                t->slider->setValue(settings->value("value").toInt());

        }
    }

    settings->endArray();
}


void CvWindow::icvSaveTrackbars(QSettings* settings)
{
    settings->beginWriteArray("trackbars");

    for (int i = 0; i < myBarLayout->layout()->count(); ++i)
    {
        settings->setArrayIndex(i);

        CvTrackbar* t = (CvTrackbar*) myBarLayout->layout()->itemAt(i);

        settings->setValue("name", t->name_bar);
        settings->setValue("value", t->slider->value());
    }

    settings->endArray();
}


//////////////////////////////////////////////////////
// OCVViewPort

OCVViewPort::OCVViewPort()
{
    mouseCallback = 0;
    mouseData = 0;
}

void OCVViewPort::setMouseCallBack(CvMouseCallback callback, void* param)
{
    mouseCallback = callback;
    mouseData = param;
}

void OCVViewPort::icvmouseEvent(QMouseEvent* evnt, type_mouse_event category)
{
    int cv_event = -1, flags = 0;

    icvmouseHandler(evnt, category, cv_event, flags);
    icvmouseProcessing(QPointF(evnt->pos()), cv_event, flags);
}

void OCVViewPort::icvmouseHandler(QMouseEvent* evnt, type_mouse_event category, int& cv_event, int& flags)
{
    Qt::KeyboardModifiers modifiers = evnt->modifiers();
    Qt::MouseButtons buttons = evnt->buttons();

    // This line gives excess flags flushing, with it you cannot predefine flags value.
    // icvmouseHandler called with flags == 0 where it really need.
    //flags = 0;
    if(modifiers & Qt::ShiftModifier)
        flags |= CV_EVENT_FLAG_SHIFTKEY;
    if(modifiers & Qt::ControlModifier)
        flags |= CV_EVENT_FLAG_CTRLKEY;
    if(modifiers & Qt::AltModifier)
        flags |= CV_EVENT_FLAG_ALTKEY;

    if(buttons & Qt::LeftButton)
        flags |= CV_EVENT_FLAG_LBUTTON;
    if(buttons & Qt::RightButton)
        flags |= CV_EVENT_FLAG_RBUTTON;
    if(buttons & Qt_MiddleButton)
        flags |= CV_EVENT_FLAG_MBUTTON;

    if (cv_event == -1) {
        if (category == mouse_wheel) {
            QWheelEvent *we = (QWheelEvent *) evnt;
            cv_event = ((wheelEventOrientation(we) == Qt::Vertical) ? CV_EVENT_MOUSEWHEEL : CV_EVENT_MOUSEHWHEEL);
            flags |= (wheelEventDelta(we) & 0xffff)<<16;
            return;
        }
        switch(evnt->button())
        {
        case Qt::LeftButton:
            cv_event = tableMouseButtons[category][0];
            flags |= CV_EVENT_FLAG_LBUTTON;
            break;
        case Qt::RightButton:
            cv_event = tableMouseButtons[category][1];
            flags |= CV_EVENT_FLAG_RBUTTON;
            break;
        case Qt_MiddleButton:
            cv_event = tableMouseButtons[category][2];
            flags |= CV_EVENT_FLAG_MBUTTON;
            break;
        default:
            cv_event = CV_EVENT_MOUSEMOVE;
        }
    }
}

void OCVViewPort::icvmouseProcessing(QPointF pt, int cv_event, int flags)
{
    if (mouseCallback)
        mouseCallback(cv_event, pt.x(), pt.y(), flags, mouseData);
}


//////////////////////////////////////////////////////
// DefaultViewPort


DefaultViewPort::DefaultViewPort(CvWindow* arg, int arg2) : QGraphicsView(arg), OCVViewPort(), image2Draw_mat(0)
{
    centralWidget = arg;
    param_keepRatio = arg2;

    setContentsMargins(0, 0, 0, 0);
    setMinimumSize(1, 1);
    setAlignment(Qt::AlignHCenter);

    setObjectName(QString::fromUtf8("graphicsView"));

    timerDisplay = new QTimer(this);
    timerDisplay->setSingleShot(true);
    connect(timerDisplay, SIGNAL(timeout()), this, SLOT(stopDisplayInfo()));

    drawInfo = false;
    mouseCoordinate  = QPoint(-1, -1);
    positionGrabbing = QPointF(0, 0);
    positionCorners  = QRect(0, 0, size().width(), size().height());


    //no border
    setStyleSheet( "QGraphicsView { border-style: none; }" );

    image2Draw_mat = cvCreateMat(viewport()->height(), viewport()->width(), CV_8UC3);
    cvZero(image2Draw_mat);

    nbChannelOriginImage = 0;

    setInteractive(false);
    setMouseTracking(true); //receive mouse event everytime
}


DefaultViewPort::~DefaultViewPort()
{
    if (image2Draw_mat)
        cvReleaseMat(&image2Draw_mat);
}


QWidget* DefaultViewPort::getWidget()
{
    return this;
}


void DefaultViewPort::writeSettings(QSettings& settings)
{
    settings.setValue("matrix_view.m11", param_matrixWorld.m11());
    settings.setValue("matrix_view.m12", param_matrixWorld.m12());
    settings.setValue("matrix_view.m13", param_matrixWorld.m13());
    settings.setValue("matrix_view.m21", param_matrixWorld.m21());
    settings.setValue("matrix_view.m22", param_matrixWorld.m22());
    settings.setValue("matrix_view.m23", param_matrixWorld.m23());
    settings.setValue("matrix_view.m31", param_matrixWorld.m31());
    settings.setValue("matrix_view.m32", param_matrixWorld.m32());
    settings.setValue("matrix_view.m33", param_matrixWorld.m33());
}


void DefaultViewPort::readSettings(QSettings& settings)
{
    qreal m11 = settings.value("matrix_view.m11", param_matrixWorld.m11()).toDouble();
    qreal m12 = settings.value("matrix_view.m12", param_matrixWorld.m12()).toDouble();
    qreal m13 = settings.value("matrix_view.m13", param_matrixWorld.m13()).toDouble();
    qreal m21 = settings.value("matrix_view.m21", param_matrixWorld.m21()).toDouble();
    qreal m22 = settings.value("matrix_view.m22", param_matrixWorld.m22()).toDouble();
    qreal m23 = settings.value("matrix_view.m23", param_matrixWorld.m23()).toDouble();
    qreal m31 = settings.value("matrix_view.m31", param_matrixWorld.m31()).toDouble();
    qreal m32 = settings.value("matrix_view.m32", param_matrixWorld.m32()).toDouble();
    qreal m33 = settings.value("matrix_view.m33", param_matrixWorld.m33()).toDouble();

    param_matrixWorld = QTransform(m11, m12, m13, m21, m22, m23, m31, m32, m33);
}


double DefaultViewPort::getRatio()
{
    return param_keepRatio;
}


void DefaultViewPort::setRatio(int flags)
{
    if (getRatio() == flags) //nothing to do
        return;

    //if valid flags
    if (flags == CV_WINDOW_FREERATIO || flags == CV_WINDOW_KEEPRATIO)
    {
        centralWidget->param_ratio_mode = flags;
        param_keepRatio = flags;
        updateGeometry();
        viewport()->update();
    }
}


void DefaultViewPort::updateImage(const CvArr* arr)
{
    CV_Assert(arr);

    CvMat* mat, stub;
    int origin = 0;

    if (CV_IS_IMAGE_HDR(arr))
        origin = ((IplImage*)arr)->origin;

    mat = cvGetMat(arr, &stub);

    if (!image2Draw_mat || !CV_ARE_SIZES_EQ(image2Draw_mat, mat))
    {
        if (image2Draw_mat)
            cvReleaseMat(&image2Draw_mat);

        //the image in ipl (to do a deep copy with cvCvtColor)
        image2Draw_mat = cvCreateMat(mat->rows, mat->cols, CV_8UC3);
        image2Draw_qt = QImage(image2Draw_mat->data.ptr, image2Draw_mat->cols, image2Draw_mat->rows, image2Draw_mat->step, QImage::Format_RGB888);

        //use to compute mouse coordinate, I need to update the ratio here and in resizeEvent
        ratioX = width() / float(image2Draw_mat->cols);
        ratioY = height() / float(image2Draw_mat->rows);
        updateGeometry();
    }

    nbChannelOriginImage = cvGetElemType(mat);
    CV_Assert(origin == 0);
    convertToShow(cv::cvarrToMat(mat), image2Draw_mat);
    viewport()->update();
}


void DefaultViewPort::startDisplayInfo(QString text, int delayms)
{
    if (timerDisplay->isActive())
        stopDisplayInfo();

    infoText = text;
    if (delayms > 0) timerDisplay->start(delayms);
    drawInfo = true;
}


void DefaultViewPort::setOpenGlDrawCallback(CvOpenGlDrawCallback /*callback*/, void* /*userdata*/)
{
    CV_Error(CV_OpenGlNotSupported, "Window doesn't support OpenGL");
}


void DefaultViewPort::makeCurrentOpenGlContext()
{
    CV_Error(CV_OpenGlNotSupported, "Window doesn't support OpenGL");
}


void DefaultViewPort::updateGl()
{
    CV_Error(CV_OpenGlNotSupported, "Window doesn't support OpenGL");
}


//Note: move 2 percent of the window
void DefaultViewPort::siftWindowOnLeft()
{
    float delta = 2 * width() / (100.0 * param_matrixWorld.m11());
    moveView(QPointF(delta, 0));
}


//Note: move 2 percent of the window
void DefaultViewPort::siftWindowOnRight()
{
    float delta = -2 * width() / (100.0 * param_matrixWorld.m11());
    moveView(QPointF(delta, 0));
}


//Note: move 2 percent of the window
void DefaultViewPort::siftWindowOnUp()
{
    float delta = 2 * height() / (100.0 * param_matrixWorld.m11());
    moveView(QPointF(0, delta));
}


//Note: move 2 percent of the window
void DefaultViewPort::siftWindowOnDown()
{
    float delta = -2 * height() / (100.0 * param_matrixWorld.m11());
    moveView(QPointF(0, delta));
}


void DefaultViewPort::imgRegion()
{
    scaleView((threshold_zoom_img_region / param_matrixWorld.m11() - 1) * 5, QPointF(size().width() / 2, size().height() / 2));
}


void DefaultViewPort::resetZoom()
{
    param_matrixWorld.reset();
    controlImagePosition();
}


void DefaultViewPort::ZoomIn()
{
    scaleView(0.5, QPointF(size().width() / 2, size().height() / 2));
}


void DefaultViewPort::ZoomOut()
{
    scaleView(-0.5, QPointF(size().width() / 2, size().height() / 2));
}


//can save as JPG, JPEG, BMP, PNG
void DefaultViewPort::saveView()
{
    QDate date_d = QDate::currentDate();
    QString date_s = date_d.toString("dd.MM.yyyy");
    QString name_s = centralWidget->windowTitle() + "_screenshot_" + date_s;

    QString fileName = QFileDialog::getSaveFileName(this, tr("Save File %1").arg(name_s), name_s + ".png", tr("Images (*.png *.jpg *.bmp *.jpeg)"));

    if (!fileName.isEmpty()) //save the picture
    {
        QString extension = fileName.right(3);

        // Create a new pixmap to render the viewport into
        QPixmap viewportPixmap(viewport()->size());
        viewport()->render(&viewportPixmap);

        // Save it..
        if (QString::compare(extension, "png", Qt::CaseInsensitive) == 0)
        {
            viewportPixmap.save(fileName, "PNG");
            return;
        }

        if (QString::compare(extension, "jpg", Qt::CaseInsensitive) == 0)
        {
            viewportPixmap.save(fileName, "JPG");
            return;
        }

        if (QString::compare(extension, "bmp", Qt::CaseInsensitive) == 0)
        {
            viewportPixmap.save(fileName, "BMP");
            return;
        }

        if (QString::compare(extension, "jpeg", Qt::CaseInsensitive) == 0)
        {
            viewportPixmap.save(fileName, "JPEG");
            return;
        }

        CV_Error(CV_StsNullPtr, "file extension not recognized, please choose between JPG, JPEG, BMP or PNG");
    }
}


//copy image to clipboard
void DefaultViewPort::copy2Clipbrd()
{
    // Create a new pixmap to render the viewport into
    QPixmap viewportPixmap(viewport()->size());
    viewport()->render(&viewportPixmap);

    QClipboard *pClipboard = QApplication::clipboard();
    pClipboard->setPixmap(viewportPixmap);
}


void DefaultViewPort::contextMenuEvent(QContextMenuEvent* evnt)
{
    if (centralWidget->vect_QActions.size() > 0)
    {
        QMenu menu(this);

        foreach (QAction *a, centralWidget->vect_QActions)
            menu.addAction(a);

        menu.exec(evnt->globalPos());
    }
}


void DefaultViewPort::resizeEvent(QResizeEvent* evnt)
{
    controlImagePosition();

    //use to compute mouse coordinate, I need to update the ratio here and in resizeEvent
    ratioX = width() / float(image2Draw_mat->cols);
    ratioY = height() / float(image2Draw_mat->rows);

    if (param_keepRatio == CV_WINDOW_KEEPRATIO)//to keep the same aspect ratio
    {
        QSize newSize = QSize(image2Draw_mat->cols, image2Draw_mat->rows);
        newSize.scale(evnt->size(), Qt::KeepAspectRatio);

        //imageWidth/imageHeight = newWidth/newHeight +/- epsilon
        //ratioX = ratioY +/- epsilon
        //||ratioX - ratioY|| = epsilon
        if (fabs(ratioX - ratioY) * 100 > ratioX) //avoid infinity loop / epsilon = 1% of ratioX
        {
            resize(newSize);
            viewport()->resize(newSize);

            //move to the middle
            //newSize get the delta offset to place the picture in the middle of its parent
            newSize = (evnt->size() - newSize) / 2;

            //if the toolbar is displayed, avoid drawing myview on top of it
            if (centralWidget->myToolBar)
                if(!centralWidget->myToolBar->isHidden())
                    newSize += QSize(0, centralWidget->myToolBar->height());

            move(newSize.width(), newSize.height());
        }
    }

    return QGraphicsView::resizeEvent(evnt);
}


void DefaultViewPort::wheelEvent(QWheelEvent* evnt)
{
    icvmouseEvent((QMouseEvent *)evnt, mouse_wheel);

    scaleView(wheelEventDelta(evnt) / 240.0, wheelEventPos(evnt));
    viewport()->update();

    QWidget::wheelEvent(evnt);
}


void DefaultViewPort::mousePressEvent(QMouseEvent* evnt)
{
    icvmouseEvent(evnt, mouse_down);

    if (param_matrixWorld.m11()>1)
    {
        setCursor(Qt::ClosedHandCursor);
        positionGrabbing = evnt->pos();
    }

    QWidget::mousePressEvent(evnt);
}


void DefaultViewPort::mouseReleaseEvent(QMouseEvent* evnt)
{
    icvmouseEvent(evnt, mouse_up);

    if (param_matrixWorld.m11()>1)
        setCursor(Qt::OpenHandCursor);

    QWidget::mouseReleaseEvent(evnt);
}


void DefaultViewPort::mouseDoubleClickEvent(QMouseEvent* evnt)
{
    icvmouseEvent(evnt, mouse_dbclick);
    QWidget::mouseDoubleClickEvent(evnt);
}


void DefaultViewPort::mouseMoveEvent(QMouseEvent* evnt)
{
    icvmouseEvent(evnt, mouse_move);

    if (param_matrixWorld.m11() > 1 && evnt->buttons() == Qt::LeftButton)
    {
        QPoint pt = evnt->pos();
        QPointF dxy = (pt - positionGrabbing)/param_matrixWorld.m11();
        positionGrabbing = pt;
        moveView(dxy);
    }

    //I update the statusbar here because if the user does a cvWaitkey(0) (like with inpaint.cpp)
    //the status bar will only be repaint when a click occurs.
    if (centralWidget->myStatusBar)
        viewport()->update();

    QWidget::mouseMoveEvent(evnt);
}


void DefaultViewPort::paintEvent(QPaintEvent* evnt)
{
    QPainter myPainter(viewport());
    myPainter.setWorldTransform(param_matrixWorld);

    draw2D(&myPainter);

    //Now disable matrixWorld for overlay display
    myPainter.setWorldMatrixEnabled(false);

    //overlay pixel values if zoomed in far enough
    if (param_matrixWorld.m11()*ratioX >= threshold_zoom_img_region &&
        param_matrixWorld.m11()*ratioY >= threshold_zoom_img_region)
    {
        drawImgRegion(&myPainter);
    }

    //in mode zoom/panning
    if (param_matrixWorld.m11() > 1)
    {
        drawViewOverview(&myPainter);
    }

    //for information overlay
    if (drawInfo)
        drawInstructions(&myPainter);

    //for statusbar
    if (centralWidget->myStatusBar)
        drawStatusBar();

    QGraphicsView::paintEvent(evnt);
}


void DefaultViewPort::stopDisplayInfo()
{
    timerDisplay->stop();
    drawInfo = false;
}


inline bool DefaultViewPort::isSameSize(IplImage* img1, IplImage* img2)
{
    return img1->width == img2->width && img1->height == img2->height;
}


void DefaultViewPort::controlImagePosition()
{
    qreal left, top, right, bottom;
    qreal factor = 1.0 / param_matrixWorld.m11();

    //after check top-left, bottom right corner to avoid getting "out" during zoom/panning
    param_matrixWorld.map(0,0,&left,&top);

    if (left > 0)
    {
        param_matrixWorld.translate(-left * factor, 0);
        left = 0;
    }
    if (top > 0)
    {
        param_matrixWorld.translate(0, -top * factor);
        top = 0;
    }
    //-------

    QSize sizeImage = size();
    param_matrixWorld.map(sizeImage.width(),sizeImage.height(),&right,&bottom);
    if (right < sizeImage.width())
    {
        param_matrixWorld.translate((sizeImage.width() - right) * factor, 0);
        right = sizeImage.width();
    }
    if (bottom < sizeImage.height())
    {
        param_matrixWorld.translate(0, (sizeImage.height() - bottom) * factor);
        bottom = sizeImage.height();
    }

    //save corner position
    positionCorners.setTopLeft(QPoint(left,top));
    positionCorners.setBottomRight(QPoint(right,bottom));
    //save also the inv matrix
    matrixWorld_inv = param_matrixWorld.inverted();

    //viewport()->update();
}

void DefaultViewPort::moveView(QPointF delta)
{
    param_matrixWorld.translate(delta.x(),delta.y());
    controlImagePosition();
    viewport()->update();
}

//factor is -0.5 (zoom out) or 0.5 (zoom in)
void DefaultViewPort::scaleView(qreal factor,QPointF center)
{
    factor/=5;//-0.1 <-> 0.1
    factor+=1;//0.9 <-> 1.1

    //limit zoom out ---
    if (param_matrixWorld.m11()==1 && factor < 1)
        return;

    if (param_matrixWorld.m11()*factor<1)
        factor = 1/param_matrixWorld.m11();


    //limit zoom int ---
    if (param_matrixWorld.m11()>100 && factor > 1)
        return;

    //inverse the transform
    int a, b;
    matrixWorld_inv.map(center.x(),center.y(),&a,&b);

    param_matrixWorld.translate(a-factor*a,b-factor*b);
    param_matrixWorld.scale(factor,factor);

    controlImagePosition();

    //display new zoom
    if (centralWidget->myStatusBar)
        centralWidget->displayStatusBar(tr("Zoom: %1%").arg(param_matrixWorld.m11()*100),1000);

    if (param_matrixWorld.m11()>1)
        setCursor(Qt::OpenHandCursor);
    else
        unsetCursor();
}




void DefaultViewPort::icvmouseProcessing(QPointF pt, int cv_event, int flags)
{
    //to convert mouse coordinate
    qreal pfx, pfy;
    matrixWorld_inv.map(pt.x(),pt.y(),&pfx,&pfy);

    mouseCoordinate.rx()=floor(pfx/ratioX);
    mouseCoordinate.ry()=floor(pfy/ratioY);

    OCVViewPort::icvmouseProcessing(QPointF(mouseCoordinate), cv_event, flags);
}


QSize DefaultViewPort::sizeHint() const
{
    if(image2Draw_mat)
        return QSize(image2Draw_mat->cols, image2Draw_mat->rows);
    else
        return QGraphicsView::sizeHint();
}


void DefaultViewPort::draw2D(QPainter *painter)
{
    image2Draw_qt = QImage(image2Draw_mat->data.ptr, image2Draw_mat->cols, image2Draw_mat->rows,image2Draw_mat->step,QImage::Format_RGB888);
    painter->drawImage(QRect(0,0,viewport()->width(),viewport()->height()), image2Draw_qt, QRect(0,0, image2Draw_qt.width(), image2Draw_qt.height()) );
}

//only if CV_8UC1 or CV_8UC3
void DefaultViewPort::drawStatusBar()
{
    if (nbChannelOriginImage!=CV_8UC1 && nbChannelOriginImage!=CV_8UC3)
        return;

    if (mouseCoordinate.x()>=0 &&
        mouseCoordinate.y()>=0 &&
        mouseCoordinate.x()<image2Draw_qt.width() &&
        mouseCoordinate.y()<image2Draw_qt.height())
//  if (mouseCoordinate.x()>=0 && mouseCoordinate.y()>=0)
    {
        QRgb rgbValue = image2Draw_qt.pixel(mouseCoordinate);
        const QPalette colorPalette{ QApplication::palette(this) };

        const QColor normalTextColor = colorPalette.brush(QPalette::WindowText).color();
        const QString textColorName = normalTextColor.name();


        if (nbChannelOriginImage==CV_8UC3)
        {
            const int r_half = normalTextColor.red() >> 1;
            const int g_half = normalTextColor.green() >> 1;
            const int b_half = normalTextColor.blue() >> 1;
            const QColor red = QColor(255, g_half, b_half);
            const QColor green = QColor(r_half, 255, b_half);
            const QColor blue = QColor(r_half, g_half, 255);
            centralWidget->myStatusBar_msg->setText(tr("<font color=%1>(x=%2, y=%3) ~ </font>")
                .arg(textColorName)
                .arg(mouseCoordinate.x())
                .arg(mouseCoordinate.y())+
                tr("<font color=%4>R:%5 </font>").arg(red.name()).arg(qRed(rgbValue))+
                tr("<font color=%6>G:%7 </font>").arg(green.name()).arg(qGreen(rgbValue))+
                tr("<font color=%8>B:%9</font>").arg(blue.name()).arg(qBlue(rgbValue))
                );
        }

        if (nbChannelOriginImage==CV_8UC1)
        {
            //all the channel have the same value (because of cv::cvtColor(GRAY=>RGB)), so only the r channel is dsplayed
            centralWidget->myStatusBar_msg->setText(tr("<font color=%1>(x=%2, y=%3) ~ </font>")
                .arg(textColorName)
                .arg(mouseCoordinate.x())
                .arg(mouseCoordinate.y())+
                tr("<font color='grey'>L:%4 </font>").arg(qRed(rgbValue))
                );
        }
    }
}

//accept only CV_8UC1 and CV_8UC8 image for now
void DefaultViewPort::drawImgRegion(QPainter *painter)
{
    if (nbChannelOriginImage!=CV_8UC1 && nbChannelOriginImage!=CV_8UC3)
        return;

    double pixel_width = param_matrixWorld.m11()*ratioX;
    double pixel_height = param_matrixWorld.m11()*ratioY;

    qreal offsetX = param_matrixWorld.dx()/pixel_width;
    offsetX = offsetX - floor(offsetX);
    qreal offsetY = param_matrixWorld.dy()/pixel_height;
    offsetY = offsetY - floor(offsetY);

    QSize view = size();
    QVarLengthArray<QLineF, 30> linesX;
    for (qreal _x = offsetX*pixel_width; _x < view.width(); _x += pixel_width )
        linesX.append(QLineF(_x, 0, _x, view.height()));

    QVarLengthArray<QLineF, 30> linesY;
    for (qreal _y = offsetY*pixel_height; _y < view.height(); _y += pixel_height )
        linesY.append(QLineF(0, _y, view.width(), _y));


    QFont f = painter->font();
    int original_font_size = f.pointSize();
    //change font size
    //f.setPointSize(4+(param_matrixWorld.m11()-threshold_zoom_img_region)/5);
    f.setPixelSize(10+(pixel_height-threshold_zoom_img_region)/5);
    painter->setFont(f);


    for (int j=-1;j<height()/pixel_height;j++)//-1 because display the pixels top rows left columns
    {
        for (int i=-1;i<width()/pixel_width;i++)//-1
        {
            // Calculate top left of the pixel's position in the viewport (screen space)
            QPointF pos_in_view((i+offsetX)*pixel_width, (j+offsetY)*pixel_height);

            // Calculate top left of the pixel's position in the image (image space)
            QPointF pos_in_image = matrixWorld_inv.map(pos_in_view);// Top left of pixel in view
            pos_in_image.rx() = pos_in_image.x()/ratioX;
            pos_in_image.ry() = pos_in_image.y()/ratioY;
            QPoint point_in_image(pos_in_image.x() + 0.5f,pos_in_image.y() + 0.5f);// Add 0.5 for rounding

            QRgb rgbValue;
            if (image2Draw_qt.valid(point_in_image))
                rgbValue = image2Draw_qt.pixel(point_in_image);
            else
                rgbValue = qRgb(0,0,0);

            if (nbChannelOriginImage==CV_8UC3)
            {
                //for debug
                /*
                val = tr("%1 %2").arg(point2.x()).arg(point2.y());
                painter->setPen(QPen(Qt::black, 1));
                painter->drawText(QRect(point1.x(),point1.y(),param_matrixWorld.m11(),param_matrixWorld.m11()/2),
                    Qt::AlignCenter, val);
                */
                QString val;

                val = tr("%1").arg(qRed(rgbValue));
                painter->setPen(QPen(Qt::red, 1));
                painter->drawText(QRect(pos_in_view.x(),pos_in_view.y(),pixel_width,pixel_height/3),
                    Qt::AlignCenter, val);

                val = tr("%1").arg(qGreen(rgbValue));
                painter->setPen(QPen(Qt::green, 1));
                painter->drawText(QRect(pos_in_view.x(),pos_in_view.y()+pixel_height/3,pixel_width,pixel_height/3),
                    Qt::AlignCenter, val);

                val = tr("%1").arg(qBlue(rgbValue));
                painter->setPen(QPen(Qt::blue, 1));
                painter->drawText(QRect(pos_in_view.x(),pos_in_view.y()+2*pixel_height/3,pixel_width,pixel_height/3),
                    Qt::AlignCenter, val);

            }

            if (nbChannelOriginImage==CV_8UC1)
            {
                QString val = tr("%1").arg(qRed(rgbValue));
                int pixel_brightness_value = qRed(rgbValue);
                int text_brightness_value = 0;

                text_brightness_value = pixel_brightness_value > 127 ? pixel_brightness_value - 127 : 127 + pixel_brightness_value;
                painter->setPen(QPen(QColor(text_brightness_value, text_brightness_value, text_brightness_value)));
                painter->drawText(QRect(pos_in_view.x(),pos_in_view.y(),pixel_width,pixel_height),
                    Qt::AlignCenter, val);
            }
        }
    }

    painter->setPen(QPen(Qt::black, 1));
    painter->drawLines(linesX.data(), linesX.size());
    painter->drawLines(linesY.data(), linesY.size());

    //restore font size
    f.setPointSize(original_font_size);
    painter->setFont(f);
}

void DefaultViewPort::drawViewOverview(QPainter *painter)
{
    QSize viewSize = size();
    viewSize.scale ( 100, 100,Qt::KeepAspectRatio );

    const int margin = 5;

    //draw the image's location
    painter->setBrush(QColor(0, 0, 0, 127));
    painter->setPen(Qt::darkGreen);
    painter->drawRect(QRect(width()-viewSize.width()-margin, 0,viewSize.width(),viewSize.height()));

    //daw the view's location inside the image
    qreal ratioSize = 1/param_matrixWorld.m11();
    qreal ratioWindow = (qreal)(viewSize.height())/(qreal)(size().height());
    painter->setPen(Qt::darkBlue);
    painter->drawRect(QRectF(width()-viewSize.width()-positionCorners.left()*ratioSize*ratioWindow-margin,
        -positionCorners.top()*ratioSize*ratioWindow,
        (viewSize.width()-1)*ratioSize,
        (viewSize.height()-1)*ratioSize)
        );
}

void DefaultViewPort::drawInstructions(QPainter *painter)
{
    QFontMetrics metrics = QFontMetrics(font());
    int border = qMax(4, metrics.leading());

    QRect qrect = metrics.boundingRect(0, 0, width() - 2*border, int(height()*0.125),
        Qt::AlignCenter | Qt::TextWordWrap, infoText);
    painter->setRenderHint(QPainter::TextAntialiasing);
    painter->fillRect(QRect(0, 0, width(), qrect.height() + 2*border),
        QColor(0, 0, 0, 127));
    painter->setPen(Qt::white);
    painter->fillRect(QRect(0, 0, width(), qrect.height() + 2*border),
        QColor(0, 0, 0, 127));

    painter->drawText((width() - qrect.width())/2, border,
        qrect.width(), qrect.height(),
        Qt::AlignCenter | Qt::TextWordWrap, infoText);
}


void DefaultViewPort::setSize(QSize /*size_*/)
{
}


//////////////////////////////////////////////////////
// OpenGlViewPort

#ifdef HAVE_QT_OPENGL


// QOpenGLWidget vs QGLWidget info: https://www.qt.io/blog/2014/09/10/qt-weekly-19-qopenglwidget
OpenGlViewPort::OpenGlViewPort(QWidget* _parent) : OpenCVQtWidgetBase(_parent), OCVViewPort(), size(-1, -1)
{
    glDrawCallback = 0;
    glDrawData = 0;
}

OpenGlViewPort::~OpenGlViewPort()
{
}

QWidget* OpenGlViewPort::getWidget()
{
    return this;
}


void OpenGlViewPort::writeSettings(QSettings& /*settings*/)
{
}

void OpenGlViewPort::readSettings(QSettings& /*settings*/)
{
}

double OpenGlViewPort::getRatio()
{
    return (double)width() / height();
}

void OpenGlViewPort::setRatio(int /*flags*/)
{
}

void OpenGlViewPort::updateImage(const CvArr* /*arr*/)
{
}

void OpenGlViewPort::startDisplayInfo(QString /*text*/, int /*delayms*/)
{
}

void OpenGlViewPort::setOpenGlDrawCallback(CvOpenGlDrawCallback callback, void* userdata)
{
    glDrawCallback = callback;
    glDrawData = userdata;
}

void OpenGlViewPort::makeCurrentOpenGlContext()
{
    makeCurrent();
}

void OpenGlViewPort::updateGl()
{
    #ifdef HAVE_QT6
    QOpenGLWidget::update();
    #else
    QGLWidget::updateGL();
    #endif
}

void OpenGlViewPort::initializeGL()
{
#ifdef GL_PERSPECTIVE_CORRECTION_HINT
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
#endif
}

void OpenGlViewPort::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
}

void OpenGlViewPort::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (glDrawCallback)
        glDrawCallback(glDrawData);
}


void OpenGlViewPort::wheelEvent(QWheelEvent* evnt)
{
    icvmouseEvent((QMouseEvent *)evnt, mouse_wheel);
    OpenCVQtWidgetBase::wheelEvent(evnt);
}

void OpenGlViewPort::mousePressEvent(QMouseEvent* evnt)
{
    icvmouseEvent(evnt, mouse_down);
    OpenCVQtWidgetBase::mousePressEvent(evnt);
}

void OpenGlViewPort::mouseReleaseEvent(QMouseEvent* evnt)
{
    icvmouseEvent(evnt, mouse_up);
    OpenCVQtWidgetBase::mouseReleaseEvent(evnt);
}

void OpenGlViewPort::mouseDoubleClickEvent(QMouseEvent* evnt)
{
    icvmouseEvent(evnt, mouse_dbclick);
    OpenCVQtWidgetBase::mouseDoubleClickEvent(evnt);
}

void OpenGlViewPort::mouseMoveEvent(QMouseEvent* evnt)
{
    icvmouseEvent(evnt, mouse_move);
    OpenCVQtWidgetBase::mouseMoveEvent(evnt);
}


QSize OpenGlViewPort::sizeHint() const
{
    if (size.width() > 0 && size.height() > 0)
        return size;
    return OpenCVQtWidgetBase::sizeHint();
}

void OpenGlViewPort::setSize(QSize size_)
{
    size = size_;
    updateGeometry();
}

#endif //HAVE_QT_OPENGL

#endif // HAVE_QT
