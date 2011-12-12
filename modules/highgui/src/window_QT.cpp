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


#if defined(HAVE_QT)

#include <memory>

#include <window_QT.h>

#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef HAVE_QT_OPENGL
    #ifdef Q_WS_X11
        #include <GL/glx.h>
    #endif
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
	CvFont f = {nameFont,color,style,NULL,NULL,NULL,0,0,0,weight,spacing,pointSize};
	return f;
}


CV_IMPL void cvAddText(const CvArr* img, const char* text, CvPoint org, CvFont* font)
{
	if (!guiMainThread)
		CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

	QMetaObject::invokeMethod(guiMainThread,
		"putText",
		Qt::AutoConnection,
		Q_ARG(void*, (void*) img),
		Q_ARG(QString,QString(text)),
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
		//Qt::DirectConnection,
		Qt::AutoConnection,
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
		Qt::AutoConnection,
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
		//Qt::DirectConnection,
		Qt::AutoConnection,
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
		Qt::AutoConnection,
		Q_ARG(QString, QString(name)),
		Q_ARG(double, prop_value));
}


void cvSetModeWindow_QT(const char* name, double prop_value)
{
	if (!guiMainThread)
		CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

	QMetaObject::invokeMethod(guiMainThread,
		"toggleFullScreen",
		Qt::AutoConnection,
		Q_ARG(QString, QString(name)),
		Q_ARG(double, prop_value));
}


double cvGetModeWindow_QT(const char* name)
{
	if (!guiMainThread)
		CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

	double result = -1;

	QMetaObject::invokeMethod(guiMainThread,
		"isFullScreen",
		Qt::AutoConnection,
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
		Qt::AutoConnection,
		//Qt::DirectConnection,
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
		Qt::AutoConnection,
		Q_ARG(QString, QString(name)));
}


CV_IMPL void cvLoadWindowParameters(const char* name)
{
	if (!guiMainThread)
		CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

	QMetaObject::invokeMethod(guiMainThread,
		"loadWindowParameters",
		Qt::AutoConnection,
		Q_ARG(QString, QString(name)));
}


CV_IMPL void cvDisplayStatusBar(const char* name, const char* text, int delayms)
{
	if (!guiMainThread)
		CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

	QMetaObject::invokeMethod(guiMainThread,
		"displayStatusBar",
		Qt::AutoConnection,
		//Qt::DirectConnection,
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
#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
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


CvWindow* icvFindWindowByName(QString name)
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
                if (w->windowTitle() == name)
		        {
			        window = w;
			        break;
		        }
		    }
	    }
    }	

    return window;
}


CvBar* icvFindBarByName(QBoxLayout* layout, QString name_bar, typeBar type)
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


CvTrackbar* icvFindTrackBarByName(const char* name_trackbar, const char* name_window, QBoxLayout* layout = NULL)
{
    QString nameQt(name_trackbar);

    if (!name_window && global_control_panel) //window name is null and we have a control panel
	    layout = global_control_panel->myLayout;

    if (!layout)
    {
	    QPointer<CvWindow> w = icvFindWindowByName(QLatin1String(name_window));

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


CvButtonbar* icvFindButtonBarByName(const char* button_name, QBoxLayout* layout)
{
    QString nameQt(button_name);
    return (CvButtonbar*) icvFindBarByName(layout, nameQt, type_CvButtonbar);
}


int icvInitSystem(int* c, char** v)
{
    //"For any GUI application using Qt, there is precisely one QApplication object"
    if (!QApplication::instance())
    {
	    new QApplication(*c, v);

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

	if (multiThreads)
		QMetaObject::invokeMethod(guiMainThread,
		"createWindow",
		Qt::BlockingQueuedConnection,
		Q_ARG(QString, QString(name)),
		Q_ARG(int, flags));
	else
		guiMainThread->createWindow(QString(name), flags);

	return 1; //Dummy value
}


CV_IMPL void cvDestroyWindow(const char* name)
{
	if (!guiMainThread)
		CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

	QMetaObject::invokeMethod(guiMainThread,
		"destroyWindow",
		//Qt::BlockingQueuedConnection,
		Qt::AutoConnection,
		Q_ARG(QString, QString(name)));
}


CV_IMPL void cvDestroyAllWindows()
{
	if (!guiMainThread)
		CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

	QMetaObject::invokeMethod(guiMainThread,
		"destroyAllWindow",
		//Qt::BlockingQueuedConnection,
		Qt::AutoConnection);
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

	return ((CvWindow*)window_handle)->windowTitle().toLatin1().data();
}


CV_IMPL void cvMoveWindow(const char* name, int x, int y)
{
	if (!guiMainThread)
		CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

	QMetaObject::invokeMethod(guiMainThread,
		"moveWindow",
		//Qt::BlockingQueuedConnection,
		Qt::AutoConnection,
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
		//Qt::BlockingQueuedConnection,
		Qt::AutoConnection,
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
		Qt::AutoConnection, 
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
		Qt::AutoConnection,
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
		Qt::AutoConnection,
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

	QMetaObject::invokeMethod(guiMainThread,
		"showImage",
		//Qt::BlockingQueuedConnection,
		Qt::DirectConnection,
		Q_ARG(QString, QString(name)),
		Q_ARG(void*, (void*)arr));
}


#ifdef HAVE_QT_OPENGL

CV_IMPL void cvSetOpenGlDrawCallback(const char* window_name, CvOpenGlDrawCallback callback, void* userdata)
{
	if (!guiMainThread)
		CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

	QMetaObject::invokeMethod(guiMainThread,
		"setOpenGlDrawCallback",
		Qt::AutoConnection,
		Q_ARG(QString, QString(window_name)),
		Q_ARG(void*, (void*)callback),
		Q_ARG(void*, userdata));
}


void icvSetOpenGlCleanCallback(const char* window_name, CvOpenGlCleanCallback callback, void* userdata)
{
	if (!guiMainThread)
		CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

	QMetaObject::invokeMethod(guiMainThread,
		"setOpenGlCleanCallback",
		Qt::AutoConnection,
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
		Qt::AutoConnection,
		Q_ARG(QString, QString(window_name)));
}


CV_IMPL void cvUpdateWindow(const char* window_name)
{
	if (!guiMainThread)
		CV_Error( CV_StsNullPtr, "NULL guiReceiver (please create a window)" );

	QMetaObject::invokeMethod(guiMainThread,
		"updateWindow",
		Qt::AutoConnection,
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
		    Qt::AutoConnection,
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
		qp.setPen(QColor(font->color.val[2], font->color.val[1], font->color.val[0], 255 - font->color.val[3]));
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


void GuiReceiver::setPropWindow(QString name, double arg2)
{
	QPointer<CvWindow> w = icvFindWindowByName(name);

	if (!w)
		return;

	int flags = (int) arg2;

    w->setPropWindow(flags);
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
}


void GuiReceiver::timeOut()
{
	bTimeOut = true;
}


void GuiReceiver::displayInfo(QString name, QString text, int delayms)
{
	QPointer<CvWindow> w = icvFindWindowByName(name);

	if (w && delayms > 0)
		w->displayInfo(text, delayms);
}


void GuiReceiver::displayStatusBar(QString name, QString text, int delayms)
{
	QPointer<CvWindow> w = icvFindWindowByName(name);

	if (w && delayms > 0)
		w->displayStatusBar(text, delayms);
}


void GuiReceiver::showImage(QString name, void* arr)
{
	QPointer<CvWindow> w = icvFindWindowByName(name);

	if (!w) //as observed in the previous implementation (W32, GTK or Carbon), create a new window is the pointer returned is null
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

        cv::Mat im(mat);
        cv::imshow(name.toStdString(), im);
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

		if (lastbar->type == type_CvTrackbar) //if last bar is a trackbar, create a new buttonbar, else, attach to the current bar
			b = CvWindow::createButtonBar(button_name); //the bar has the name of the first button attached to it
		else
			b = (CvButtonbar*) lastbar;

	}

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

	if (!value)
		CV_Error(CV_StsNullPtr, "NULL value pointer" );

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

void GuiReceiver::setOpenGlCleanCallback(QString name, void* callback, void* userdata)
{
	QPointer<CvWindow> w = icvFindWindowByName(name);

	if (w)
		w->setOpenGlCleanCallback((CvOpenGlCleanCallback) callback, userdata);
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


CvTrackbar::CvTrackbar(CvWindow* arg, QString name, int* value, int count, CvTrackbarCallback2 on_change, void* data)
{
	callback = NULL;
	callback2 = on_change;
	userdata = data;

	create(arg, name, value, count);
}


CvTrackbar::CvTrackbar(CvWindow* arg, QString name, int* value, int count, CvTrackbarCallback on_change)
{
	callback = on_change;
	callback2 = NULL;
	userdata = NULL;

	create(arg, name, value, count);
}


void CvTrackbar::create(CvWindow* arg, QString name, int* value, int count)
{
	type = type_CvTrackbar;
	myparent = arg;
	name_bar = name;
	setObjectName(name_bar);
	dataSlider = value;

	slider = new QSlider(Qt::Horizontal);
	slider->setFocusPolicy(Qt::StrongFocus);
	slider->setMinimum(0);
	slider->setMaximum(count);
	slider->setPageStep(5);
	slider->setValue(*value);
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
	QString nameNormalized = name_bar.leftJustified( 10, ' ', true );
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
CvWinProperties::CvWinProperties(QString name_paraWindow, QObject* parent)
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
	myLayout->setMargin(0);
	myLayout->setSizeConstraint(QLayout::SetFixedSize);
	setLayout(myLayout);

	hide();
}


void CvWinProperties::closeEvent(QCloseEvent* e)
{
	e->accept(); //intersept the close event (not sure I really need it)
	//an hide event is also sent. I will intercept it and do some processing
}


void CvWinProperties::showEvent(QShowEvent* event)
{
	//why -1,-1 ?: do this trick because the first time the code is run,
	//no value pos was saved so we let Qt move the window in the middle of its parent (event ignored).
	//then hide will save the last position and thus, we want to retreive it (event accepted).
	QPoint mypos(-1, -1);
	QSettings settings("OpenCV2", windowTitle());
	mypos = settings.value("pos", mypos).toPoint();

	if (mypos.x() >= 0)
	{
		move(mypos);
		event->accept();
	}
	else
    {
		event->ignore();
	}
}


void CvWinProperties::hideEvent(QHideEvent* event)
{
	QSettings settings("OpenCV2", windowTitle());
	settings.setValue("pos", pos()); //there is an offset of 6 pixels (so the window's position is wrong -- why ?)
	event->accept();
}


CvWinProperties::~CvWinProperties()
{
	//clear the setting pos
	QSettings settings("OpenCV2", windowTitle());
	settings.remove("pos");
}


//////////////////////////////////////////////////////
// CvWindow


CvWindow::CvWindow(QString name, int arg2)
{
	type = type_CvWindow;
	moveToThread(qApp->instance()->thread());

	param_flags = arg2 & 0x0000000F;
	param_gui_mode = arg2 & 0x000000F0;
	param_ratio_mode =  arg2 & 0x00000F00;

	//setAttribute(Qt::WA_DeleteOnClose); //in other case, does not release memory
	setContentsMargins(0, 0, 0, 0);
	setWindowTitle(name);
	setObjectName(name);

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

	QPoint pos = settings.value("pos", QPoint(200, 200)).toPoint();
	QSize size = settings.value("size", QSize(400, 400)).toSize();

	param_flags = settings.value("mode_resize", param_flags).toInt();
	param_gui_mode = settings.value("mode_gui", param_gui_mode).toInt();

	param_flags = settings.value("mode_resize", param_flags).toInt();

	myView->readSettings(settings);

	//trackbar here
	icvLoadTrackbars(&settings);

	resize(size);
	move(pos);

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
    vect_QActions[9]->setDisabled(false);
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


void CvWindow::setOpenGlCleanCallback(CvOpenGlCleanCallback callback, void* userdata)
{
	myView->setOpenGlCleanCallback(callback, userdata);
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


void CvWindow::setViewportSize(QSize size)
{
    myView->getWidget()->resize(size);
    myView->setSize(size);
}


void CvWindow::createBarLayout()
{
	myBarLayout = new QBoxLayout(QBoxLayout::TopToBottom);
	myBarLayout->setObjectName(QString::fromUtf8("barLayout"));
	myBarLayout->setContentsMargins(0, 0, 0, 0);
	myBarLayout->setSpacing(0);
	myBarLayout->setMargin(0);
}


void CvWindow::createGlobalLayout()
{
	myGlobalLayout = new QBoxLayout(QBoxLayout::TopToBottom);
	myGlobalLayout->setObjectName(QString::fromUtf8("boxLayout"));
	myGlobalLayout->setContentsMargins(0, 0, 0, 0);
	myGlobalLayout->setSpacing(0);
	myGlobalLayout->setMargin(0);
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
	vect_QActions.resize(10);

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

	vect_QActions[9] = new QAction(QIcon(":/properties-icon"), "Display properties window (CTRL+P)", this);
	vect_QActions[9]->setIconVisibleInMenu(true);
	QObject::connect(vect_QActions[9], SIGNAL(triggered()), this, SLOT(displayPropertiesWin()));

	if (global_control_panel->myLayout->count() == 0)
		vect_QActions[9]->setDisabled(true);
}


void CvWindow::createShortcuts()
{
	vect_QShortcuts.resize(10);

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

	vect_QShortcuts[9] = new QShortcut(shortcut_properties_win, this);
	QObject::connect(vect_QShortcuts[9], SIGNAL(activated()), this, SLOT(displayPropertiesWin()));
}


void CvWindow::createToolBar()
{
	myToolBar = new QToolBar(this);
	myToolBar->setFloatable(false); //is not a window
	myToolBar->setFixedHeight(28);
	myToolBar->setMinimumWidth(1);

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


//Need more test here !
void CvWindow::keyPressEvent(QKeyEvent *event)
{
	//see http://doc.trolltech.com/4.6/qt.html#Key-enum
	int key = event->key();

	//bool goodKey = false;
	bool goodKey = true;

	Qt::Key qtkey = static_cast<Qt::Key>(key);
	char asciiCode = QTest::keyToAscii(qtkey);
	if (asciiCode != 0)
	{
		key = static_cast<int>(asciiCode);
	}

	//control plus (Z, +, -, up, down, left, right) are used for zoom/panning functions
	if (event->modifiers() != Qt::ControlModifier && goodKey)
	{
		mutexKey.lock();
		last_key = key;
		//last_key = event->nativeVirtualKey ();
		mutexKey.unlock();
		key_pressed.wakeAll();
		//event->accept();
	}

	QWidget::keyPressEvent(event);
}


void CvWindow::icvLoadControlPanel()
{
	QSettings settings("OpenCV2", QFileInfo(QApplication::applicationFilePath()).fileName() + " control panel");
	
    int size = settings.beginReadArray("bars");

	if (size == global_control_panel->myLayout->layout()->count())
    {
		for (int i = 0; i < size; ++i) 
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
				int subsize = settings.beginReadArray(QString("buttonbar")+i);

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
			settings.beginWriteArray(QString("buttonbar")+i);
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
	int size = settings->beginReadArray("trackbars");

	//trackbar are saved in the same order, so no need to use icvFindTrackbarByName

	if (myBarLayout->layout()->count() == size) //if not the same number, the window saved and loaded is not the same (nb trackbar not equal)
    {
		for (int i = 0; i < size; ++i)
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
// DefaultViewPort


DefaultViewPort::DefaultViewPort(CvWindow* arg, int arg2) : QGraphicsView(arg), image2Draw_mat(0)
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
	positionGrabbing = QPointF(0, 0);
	positionCorners = QRect(0, 0, size().width(), size().height());

	on_mouse = 0;
    on_mouse_param = 0;
	mouseCoordinate = QPoint(-1, -1);

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


void DefaultViewPort::setMouseCallBack(CvMouseCallback m, void* param)
{
	on_mouse = m;
	on_mouse_param = param;
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

	cvConvertImage(mat, image2Draw_mat, (origin != 0 ? CV_CVTIMG_FLIP : 0) + CV_CVTIMG_SWAP_RB);

	viewport()->update();
}


void DefaultViewPort::startDisplayInfo(QString text, int delayms)
{
	if (timerDisplay->isActive())
		stopDisplayInfo();

	infoText = text;
	timerDisplay->start(delayms);
	drawInfo = true;
}


void DefaultViewPort::setOpenGlDrawCallback(CvOpenGlDrawCallback callback, void* userdata)
{
    CV_Error(CV_OpenGlNotSupported, "Window doesn't support OpenGL");
}


void DefaultViewPort::setOpenGlCleanCallback(CvOpenGlCleanCallback callback, void* userdata)
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

	    //   (no need anymore) create the image resized to receive the 'screenshot'
	    //    image2Draw_qt_resized = QImage(viewport()->width(), viewport()->height(),QImage::Format_RGB888);
	
	    QPainter saveimage(&image2Draw_qt_resized);
	    this->render(&saveimage);

		// Save it..
		if (QString::compare(extension, "png", Qt::CaseInsensitive) == 0)
		{
			image2Draw_qt_resized.save(fileName, "PNG");
			return;
		}

		if (QString::compare(extension, "jpg", Qt::CaseInsensitive) == 0)
		{
			image2Draw_qt_resized.save(fileName, "JPG");
			return;
		}

		if (QString::compare(extension, "bmp", Qt::CaseInsensitive) == 0)
		{
			image2Draw_qt_resized.save(fileName, "BMP");
			return;
		}

		if (QString::compare(extension, "jpeg", Qt::CaseInsensitive) == 0)
		{
			image2Draw_qt_resized.save(fileName, "JPEG");
			return;
		}

		CV_Error(CV_StsNullPtr, "file extension not recognized, please choose between JPG, JPEG, BMP or PNG");
	}
}


void DefaultViewPort::contextMenuEvent(QContextMenuEvent* event)
{
	if (centralWidget->vect_QActions.size() > 0)
	{
		QMenu menu(this);

		foreach (QAction *a, centralWidget->vect_QActions)
			menu.addAction(a);

		menu.exec(event->globalPos());
	}
}


void DefaultViewPort::resizeEvent(QResizeEvent* event)
{
    controlImagePosition();

    //use to compute mouse coordinate, I need to update the ratio here and in resizeEvent
    ratioX = width() / float(image2Draw_mat->cols);
    ratioY = height() / float(image2Draw_mat->rows);
	
    if (param_keepRatio == CV_WINDOW_KEEPRATIO)//to keep the same aspect ratio
    {
	    QSize newSize = QSize(image2Draw_mat->cols, image2Draw_mat->rows);
	    newSize.scale(event->size(), Qt::KeepAspectRatio);

	    //imageWidth/imageHeight = newWidth/newHeight +/- epsilon
	    //ratioX = ratioY +/- epsilon
	    //||ratioX - ratioY|| = epsilon
	    if (fabs(ratioX - ratioY) * 100 > ratioX) //avoid infinity loop / epsilon = 1% of ratioX
	    {
		    resize(newSize);

		    //move to the middle
		    //newSize get the delta offset to place the picture in the middle of its parent
		    newSize = (event->size() - newSize) / 2;

		    //if the toolbar is displayed, avoid drawing myview on top of it
		    if (centralWidget->myToolBar)
			    if(!centralWidget->myToolBar->isHidden())
				    newSize += QSize(0, centralWidget->myToolBar->height());

		    move(newSize.width(), newSize.height());
	    }
    }

	return QGraphicsView::resizeEvent(event);
}


void DefaultViewPort::wheelEvent(QWheelEvent* event)
{
	scaleView(event->delta() / 240.0, event->pos());
	viewport()->update();
}


void DefaultViewPort::mousePressEvent(QMouseEvent* event)
{
	int cv_event = -1, flags = 0;
	QPoint pt = event->pos();

	//icvmouseHandler: pass parameters for cv_event, flags
	icvmouseHandler(event, mouse_down, cv_event, flags);
	icvmouseProcessing(QPointF(pt), cv_event, flags);

	if (param_matrixWorld.m11()>1)
	{
		setCursor(Qt::ClosedHandCursor);
		positionGrabbing = event->pos();
	}

	QWidget::mousePressEvent(event);
}


void DefaultViewPort::mouseReleaseEvent(QMouseEvent* event)
{
	int cv_event = -1, flags = 0;
	QPoint pt = event->pos();

	//icvmouseHandler: pass parameters for cv_event, flags
	icvmouseHandler(event, mouse_up, cv_event, flags);
	icvmouseProcessing(QPointF(pt), cv_event, flags);

	if (param_matrixWorld.m11()>1)
		setCursor(Qt::OpenHandCursor);

	QWidget::mouseReleaseEvent(event);
}


void DefaultViewPort::mouseDoubleClickEvent(QMouseEvent* event)
{
	int cv_event = -1, flags = 0;
	QPoint pt = event->pos();

	//icvmouseHandler: pass parameters for cv_event, flags
	icvmouseHandler(event, mouse_dbclick, cv_event, flags);
	icvmouseProcessing(QPointF(pt), cv_event, flags);

	QWidget::mouseDoubleClickEvent(event);
}


void DefaultViewPort::mouseMoveEvent(QMouseEvent* event)
{
	int cv_event = CV_EVENT_MOUSEMOVE, flags = 0;
	QPoint pt = event->pos();

	//icvmouseHandler: pass parameters for cv_event, flags
	icvmouseHandler(event, mouse_move, cv_event, flags);
	icvmouseProcessing(QPointF(pt), cv_event, flags);

	if (param_matrixWorld.m11() > 1 && event->buttons() == Qt::LeftButton)
	{
		QPointF dxy = (pt - positionGrabbing)/param_matrixWorld.m11();
		positionGrabbing = event->pos();
		moveView(dxy);
	}

	//I update the statusbar here because if the user does a cvWaitkey(0) (like with inpaint.cpp)
	//the status bar will only be repaint when a click occurs.
	if (centralWidget->myStatusBar)
		viewport()->update();

	QWidget::mouseMoveEvent(event);
}


void DefaultViewPort::paintEvent(QPaintEvent* event)
{
	QPainter myPainter(viewport());
	myPainter.setWorldTransform(param_matrixWorld);

	draw2D(&myPainter);

	//Now disable matrixWorld for overlay display
	myPainter.setWorldMatrixEnabled(false);

	//in mode zoom/panning
	if (param_matrixWorld.m11() > 1)
	{		
		if (param_matrixWorld.m11() >= threshold_zoom_img_region)
		{
			if (centralWidget->param_flags == CV_WINDOW_NORMAL)
				startDisplayInfo("WARNING: The values displayed are the resized image's values. If you want the original image's values, use CV_WINDOW_AUTOSIZE", 1000);

			drawImgRegion(&myPainter);
		}

		drawViewOverview(&myPainter);
	}

	//for information overlay
	if (drawInfo)
		drawInstructions(&myPainter);

	//for statusbar
	if (centralWidget->myStatusBar)
		drawStatusBar();

	QGraphicsView::paintEvent(event);
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

	//after check top-left, bottom right corner to avoid getting "out" during zoom/panning
	param_matrixWorld.map(0,0,&left,&top);

	if (left > 0)
	{
		param_matrixWorld.translate(-left,0);
		left = 0;
	}
	if (top > 0)
	{
		param_matrixWorld.translate(0,-top);
		top = 0;
	}
	//-------

	QSize sizeImage = size();
	param_matrixWorld.map(sizeImage.width(),sizeImage.height(),&right,&bottom);
	if (right < sizeImage.width())
	{
		param_matrixWorld.translate(sizeImage.width()-right,0);
		right = sizeImage.width();
	}
	if (bottom < sizeImage.height())
	{
		param_matrixWorld.translate(0,sizeImage.height()-bottom);
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


//up, down, dclick, move
void DefaultViewPort::icvmouseHandler(QMouseEvent *event, type_mouse_event category, int &cv_event, int &flags)
{
	Qt::KeyboardModifiers modifiers = event->modifiers();
    Qt::MouseButtons buttons = event->buttons();
    
    flags = 0;
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
    if(buttons & Qt::MidButton)
		flags |= CV_EVENT_FLAG_MBUTTON;

    cv_event = CV_EVENT_MOUSEMOVE;
	switch(event->button())
	{
	case Qt::LeftButton:
		cv_event = tableMouseButtons[category][0];
		flags |= CV_EVENT_FLAG_LBUTTON;
		break;
	case Qt::RightButton:
		cv_event = tableMouseButtons[category][1];
		flags |= CV_EVENT_FLAG_RBUTTON;
		break;
	case Qt::MidButton:
		cv_event = tableMouseButtons[category][2];
		flags |= CV_EVENT_FLAG_MBUTTON;
		break;
	default:;
	}
}


void DefaultViewPort::icvmouseProcessing(QPointF pt, int cv_event, int flags)
{
	//to convert mouse coordinate
	qreal pfx, pfy;
	matrixWorld_inv.map(pt.x(),pt.y(),&pfx,&pfy);
	
	mouseCoordinate.rx()=floor(pfx/ratioX);
	mouseCoordinate.ry()=floor(pfy/ratioY);

	if (on_mouse)
		on_mouse( cv_event, mouseCoordinate.x(),
            mouseCoordinate.y(), flags, on_mouse_param );
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
	image2Draw_qt_resized = image2Draw_qt.scaled(viewport()->width(),viewport()->height(),Qt::IgnoreAspectRatio,Qt::FastTransformation);//Qt::SmoothTransformation);
	painter->drawImage(0,0,image2Draw_qt_resized);
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
//	if (mouseCoordinate.x()>=0 && mouseCoordinate.y()>=0)
	{
		QRgb rgbValue = image2Draw_qt.pixel(mouseCoordinate);

		if (nbChannelOriginImage==CV_8UC3 )
		{
			centralWidget->myStatusBar_msg->setText(tr("<font color='black'>(x=%1, y=%2) ~ </font>")
				.arg(mouseCoordinate.x())
				.arg(mouseCoordinate.y())+
				tr("<font color='red'>R:%3 </font>").arg(qRed(rgbValue))+//.arg(value.val[0])+
				tr("<font color='green'>G:%4 </font>").arg(qGreen(rgbValue))+//.arg(value.val[1])+
				tr("<font color='blue'>B:%5</font>").arg(qBlue(rgbValue))//.arg(value.val[2])
				);
		}

		if (nbChannelOriginImage==CV_8UC1)
		{
			//all the channel have the same value (because of cvconvertimage), so only the r channel is dsplayed
			centralWidget->myStatusBar_msg->setText(tr("<font color='black'>(x=%1, y=%2) ~ </font>")
				.arg(mouseCoordinate.x())
				.arg(mouseCoordinate.y())+
				tr("<font color='grey'>L:%3 </font>").arg(qRed(rgbValue))
				);
		}
	}
}

//accept only CV_8UC1 and CV_8UC8 image for now
void DefaultViewPort::drawImgRegion(QPainter *painter)
{

	if (nbChannelOriginImage!=CV_8UC1 && nbChannelOriginImage!=CV_8UC3)
		return;

	qreal offsetX = param_matrixWorld.dx()/param_matrixWorld.m11();
	offsetX = offsetX - floor(offsetX);
	qreal offsetY = param_matrixWorld.dy()/param_matrixWorld.m11();
	offsetY = offsetY - floor(offsetY);

	QSize view = size();
	QVarLengthArray<QLineF, 30> linesX;
	for (qreal x = offsetX*param_matrixWorld.m11(); x < view.width(); x += param_matrixWorld.m11() )
		linesX.append(QLineF(x, 0, x, view.height()));

	QVarLengthArray<QLineF, 30> linesY;
	for (qreal y = offsetY*param_matrixWorld.m11(); y < view.height(); y += param_matrixWorld.m11() )
		linesY.append(QLineF(0, y, view.width(), y));


	QFont f = painter->font();
	int original_font_size = f.pointSize();
	//change font size
	//f.setPointSize(4+(param_matrixWorld.m11()-threshold_zoom_img_region)/5);
	f.setPixelSize(10+(param_matrixWorld.m11()-threshold_zoom_img_region)/5);
	painter->setFont(f);
	QString val;
	QRgb rgbValue;

	QPointF point1;//sorry, I do not know how to name it
	QPointF point2;//idem

	for (int j=-1;j<height()/param_matrixWorld.m11();j++)//-1 because display the pixels top rows left colums
		for (int i=-1;i<width()/param_matrixWorld.m11();i++)//-1
		{
			point1.setX((i+offsetX)*param_matrixWorld.m11());
			point1.setY((j+offsetY)*param_matrixWorld.m11());

			matrixWorld_inv.map(point1.x(),point1.y(),&point2.rx(),&point2.ry());

			point2.rx()= (long) (point2.x() + 0.5);
			point2.ry()= (long) (point2.y() + 0.5);

			if (point2.x() >= 0 && point2.y() >= 0)
				rgbValue = image2Draw_qt_resized.pixel(QPoint(point2.x(),point2.y()));
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

				val = tr("%1").arg(qRed(rgbValue));
				painter->setPen(QPen(Qt::red, 1));
				painter->drawText(QRect(point1.x(),point1.y(),param_matrixWorld.m11(),param_matrixWorld.m11()/3),
					Qt::AlignCenter, val);

				val = tr("%1").arg(qGreen(rgbValue));
				painter->setPen(QPen(Qt::green, 1));
				painter->drawText(QRect(point1.x(),point1.y()+param_matrixWorld.m11()/3,param_matrixWorld.m11(),param_matrixWorld.m11()/3),
					Qt::AlignCenter, val);

				val = tr("%1").arg(qBlue(rgbValue));
				painter->setPen(QPen(Qt::blue, 1));
				painter->drawText(QRect(point1.x(),point1.y()+2*param_matrixWorld.m11()/3,param_matrixWorld.m11(),param_matrixWorld.m11()/3),
					Qt::AlignCenter, val);

			}

			if (nbChannelOriginImage==CV_8UC1)
			{

				val = tr("%1").arg(qRed(rgbValue));
				painter->drawText(QRect(point1.x(),point1.y(),param_matrixWorld.m11(),param_matrixWorld.m11()),
					Qt::AlignCenter, val);
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

	QRect rect = metrics.boundingRect(0, 0, width() - 2*border, int(height()*0.125),
		Qt::AlignCenter | Qt::TextWordWrap, infoText);
	painter->setRenderHint(QPainter::TextAntialiasing);
	painter->fillRect(QRect(0, 0, width(), rect.height() + 2*border),
		QColor(0, 0, 0, 127));
	painter->setPen(Qt::white);
	painter->fillRect(QRect(0, 0, width(), rect.height() + 2*border),
		QColor(0, 0, 0, 127));

	painter->drawText((width() - rect.width())/2, border,
		rect.width(), rect.height(),
		Qt::AlignCenter | Qt::TextWordWrap, infoText);
}


void DefaultViewPort::setSize(QSize size_)
{
}


//////////////////////////////////////////////////////
// OpenGlViewPort

#ifdef HAVE_QT_OPENGL

OpenGlViewPort::OpenGlViewPort(QWidget* parent) : QGLWidget(parent), size(-1, -1)
{
    mouseCallback = 0;
    mouseData = 0;

    glDrawCallback = 0;
    glDrawData = 0;

    glCleanCallback = 0;
    glCleanData = 0;

    glFuncTab = 0;
}

OpenGlViewPort::~OpenGlViewPort()
{
    if (glFuncTab)
        delete glFuncTab;

    setOpenGlCleanCallback(0, 0);
}

QWidget* OpenGlViewPort::getWidget()
{
    return this;
}

void OpenGlViewPort::setMouseCallBack(CvMouseCallback callback, void* param)
{
    mouseCallback = callback;
    mouseData = param;
}

void OpenGlViewPort::writeSettings(QSettings& settings)
{
}

void OpenGlViewPort::readSettings(QSettings& settings)
{
}

double OpenGlViewPort::getRatio()
{
    return (double)width() / height();
}

void OpenGlViewPort::setRatio(int flags)
{
}

void OpenGlViewPort::updateImage(const CvArr* arr)
{
}

void OpenGlViewPort::startDisplayInfo(QString text, int delayms)
{
}

void OpenGlViewPort::setOpenGlDrawCallback(CvOpenGlDrawCallback callback, void* userdata)
{
    glDrawCallback = callback;
    glDrawData = userdata;
}

void OpenGlViewPort::setOpenGlCleanCallback(CvOpenGlCleanCallback callback, void* userdata)
{
    makeCurrentOpenGlContext();

    if (glCleanCallback)
        glCleanCallback(glCleanData);

    glCleanCallback = callback;
    glCleanData = userdata;
}

void OpenGlViewPort::makeCurrentOpenGlContext()
{
    makeCurrent();
    icvSetOpenGlFuncTab(glFuncTab);
}

void OpenGlViewPort::updateGl()
{
    QGLWidget::updateGL();
}

#ifndef APIENTRY
    #define APIENTRY
#endif

#ifndef APIENTRYP
    #define APIENTRYP APIENTRY *
#endif

#ifndef GL_VERSION_1_5
    /* GL types for handling large vertex buffer objects */
    typedef ptrdiff_t GLintptr;
    typedef ptrdiff_t GLsizeiptr;
#endif

typedef void (APIENTRYP PFNGLGENBUFFERSPROC   ) (GLsizei n, GLuint *buffers);
typedef void (APIENTRYP PFNGLDELETEBUFFERSPROC) (GLsizei n, const GLuint *buffers);

typedef void (APIENTRYP PFNGLBUFFERDATAPROC   ) (GLenum target, GLsizeiptr size, const GLvoid *data, GLenum usage);
typedef void (APIENTRYP PFNGLBUFFERSUBDATAPROC) (GLenum target, GLintptr offset, GLsizeiptr size, const GLvoid* data);

typedef void (APIENTRYP PFNGLBINDBUFFERPROC   ) (GLenum target, GLuint buffer);

typedef GLvoid* (APIENTRYP PFNGLMAPBUFFERPROC) (GLenum target, GLenum access);
typedef GLboolean (APIENTRYP PFNGLUNMAPBUFFERPROC) (GLenum target);

class GlFuncTab_QT : public CvOpenGlFuncTab
{
public:
#ifdef Q_WS_WIN
    GlFuncTab_QT(HDC hDC);
#else
    GlFuncTab_QT();
#endif

    void genBuffers(int n, unsigned int* buffers) const;
    void deleteBuffers(int n, const unsigned int* buffers) const;

    void bufferData(unsigned int target, ptrdiff_t size, const void* data, unsigned int usage) const;
    void bufferSubData(unsigned int target, ptrdiff_t offset, ptrdiff_t size, const void* data) const;

    void bindBuffer(unsigned int target, unsigned int buffer) const;

    void* mapBuffer(unsigned int target, unsigned int access) const;
    void unmapBuffer(unsigned int target) const;

    void generateBitmapFont(const std::string& family, int height, int weight, bool italic, bool underline, int start, int count, int base) const;

    bool isGlContextInitialized() const;
        
    PFNGLGENBUFFERSPROC    glGenBuffersExt;
    PFNGLDELETEBUFFERSPROC glDeleteBuffersExt;

    PFNGLBUFFERDATAPROC    glBufferDataExt;
    PFNGLBUFFERSUBDATAPROC glBufferSubDataExt;

    PFNGLBINDBUFFERPROC    glBindBufferExt;

    PFNGLMAPBUFFERPROC     glMapBufferExt;
    PFNGLUNMAPBUFFERPROC   glUnmapBufferExt;

    bool initialized;

#ifdef Q_WS_WIN
    HDC hDC;
#endif
};

#ifdef Q_WS_WIN
    GlFuncTab_QT::GlFuncTab_QT(HDC hDC_) : hDC(hDC_)
#else
    GlFuncTab_QT::GlFuncTab_QT()
#endif
{
    glGenBuffersExt    = 0;
    glDeleteBuffersExt = 0;

    glBufferDataExt    = 0;
    glBufferSubDataExt = 0;

    glBindBufferExt    = 0;

    glMapBufferExt     = 0;
    glUnmapBufferExt   = 0;

    initialized = false;
}

void GlFuncTab_QT::genBuffers(int n, unsigned int* buffers) const
{
    CV_FUNCNAME( "GlFuncTab_QT::genBuffers" );

    __BEGIN__;

    if (!glGenBuffersExt)
        CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

    glGenBuffersExt(n, buffers);
    CV_CheckGlError();

    __END__;
}

void GlFuncTab_QT::deleteBuffers(int n, const unsigned int* buffers) const
{
    CV_FUNCNAME( "GlFuncTab_QT::deleteBuffers" );

    __BEGIN__;

    if (!glDeleteBuffersExt)
        CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

    glDeleteBuffersExt(n, buffers);
    CV_CheckGlError();

    __END__;
}

void GlFuncTab_QT::bufferData(unsigned int target, ptrdiff_t size, const void* data, unsigned int usage) const
{
    CV_FUNCNAME( "GlFuncTab_QT::bufferData" );

    __BEGIN__;

    if (!glBufferDataExt)
        CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

    glBufferDataExt(target, size, data, usage);
    CV_CheckGlError();

    __END__;
}

void GlFuncTab_QT::bufferSubData(unsigned int target, ptrdiff_t offset, ptrdiff_t size, const void* data) const
{
    CV_FUNCNAME( "GlFuncTab_QT::bufferSubData" );

    __BEGIN__;

    if (!glBufferSubDataExt)
        CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

    glBufferSubDataExt(target, offset, size, data);
    CV_CheckGlError();

    __END__;
}

void GlFuncTab_QT::bindBuffer(unsigned int target, unsigned int buffer) const
{
    CV_FUNCNAME( "GlFuncTab_QT::bindBuffer" );

    __BEGIN__;

    if (!glBindBufferExt)
        CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

    glBindBufferExt(target, buffer);
    CV_CheckGlError();

    __END__;
}

void* GlFuncTab_QT::mapBuffer(unsigned int target, unsigned int access) const
{
    CV_FUNCNAME( "GlFuncTab_QT::mapBuffer" );

    void* res = 0;

    __BEGIN__;

    if (!glMapBufferExt)
        CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

    res = glMapBufferExt(target, access);
    CV_CheckGlError();

    __END__;

    return res;
}

void GlFuncTab_QT::unmapBuffer(unsigned int target) const
{
    CV_FUNCNAME( "GlFuncTab_QT::unmapBuffer" );

    __BEGIN__;

    if (!glUnmapBufferExt)
        CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

    glUnmapBufferExt(target);
    CV_CheckGlError();

    __END__;
}

void GlFuncTab_QT::generateBitmapFont(const std::string& family, int height, int weight, bool italic, bool underline, int start, int count, int base) const
{
    CV_FUNCNAME( "GlFuncTab_QT::generateBitmapFont" );

    QFont font(QString(family.c_str()), height, weight, italic);

    __BEGIN__;

#ifndef Q_WS_WIN
    glXUseXFont(font.handle(), start, count, base);
#else
    SelectObject(hDC, font.handle());
    if (!wglUseFontBitmaps(hDC, start, count, base))
        CV_ERROR(CV_OpenGlApiCallError, "Can't create font");
#endif

    __END__;
}

bool GlFuncTab_QT::isGlContextInitialized() const
{
    return initialized;
}

void OpenGlViewPort::initializeGL()
{
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

#ifdef Q_WS_WIN
    std::auto_ptr<GlFuncTab_QT> glFuncTab(new GlFuncTab_QT(getDC()));
#else
    std::auto_ptr<GlFuncTab_QT> glFuncTab(new GlFuncTab_QT);
#endif

    // Load extensions

    glFuncTab->glGenBuffersExt = (PFNGLGENBUFFERSPROC)context()->getProcAddress("glGenBuffers");
    glFuncTab->glDeleteBuffersExt = (PFNGLDELETEBUFFERSPROC)context()->getProcAddress("glDeleteBuffers");
    glFuncTab->glBufferDataExt = (PFNGLBUFFERDATAPROC)context()->getProcAddress("glBufferData");
    glFuncTab->glBufferSubDataExt = (PFNGLBUFFERSUBDATAPROC)context()->getProcAddress("glBufferSubData");
    glFuncTab->glBindBufferExt = (PFNGLBINDBUFFERPROC)context()->getProcAddress("glBindBuffer");
    glFuncTab->glMapBufferExt = (PFNGLMAPBUFFERPROC)context()->getProcAddress("glMapBuffer");
    glFuncTab->glUnmapBufferExt = (PFNGLUNMAPBUFFERPROC)context()->getProcAddress("glUnmapBuffer");

    glFuncTab->initialized = true;

    this->glFuncTab = glFuncTab.release();

    icvSetOpenGlFuncTab(this->glFuncTab);
}

void OpenGlViewPort::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
}

void OpenGlViewPort::paintGL()
{
    icvSetOpenGlFuncTab(glFuncTab);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (glDrawCallback)
        glDrawCallback(glDrawData);

    CV_CheckGlError();
}

void OpenGlViewPort::mousePressEvent(QMouseEvent* event)
{
	int cv_event = -1, flags = 0;
	QPoint pt = event->pos();

	icvmouseHandler(event, mouse_down, cv_event, flags);
	icvmouseProcessing(QPointF(pt), cv_event, flags);

	QGLWidget::mousePressEvent(event);
}


void OpenGlViewPort::mouseReleaseEvent(QMouseEvent* event)
{
	int cv_event = -1, flags = 0;
	QPoint pt = event->pos();

	icvmouseHandler(event, mouse_up, cv_event, flags);
	icvmouseProcessing(QPointF(pt), cv_event, flags);

	QGLWidget::mouseReleaseEvent(event);
}


void OpenGlViewPort::mouseDoubleClickEvent(QMouseEvent* event)
{
	int cv_event = -1, flags = 0;
	QPoint pt = event->pos();

	icvmouseHandler(event, mouse_dbclick, cv_event, flags);
	icvmouseProcessing(QPointF(pt), cv_event, flags);

	QGLWidget::mouseDoubleClickEvent(event);
}


void OpenGlViewPort::mouseMoveEvent(QMouseEvent* event)
{
	int cv_event = CV_EVENT_MOUSEMOVE, flags = 0;
	QPoint pt = event->pos();

	//icvmouseHandler: pass parameters for cv_event, flags
	icvmouseHandler(event, mouse_move, cv_event, flags);
	icvmouseProcessing(QPointF(pt), cv_event, flags);

	QGLWidget::mouseMoveEvent(event);
}

void OpenGlViewPort::icvmouseHandler(QMouseEvent* event, type_mouse_event category, int& cv_event, int& flags)
{
	Qt::KeyboardModifiers modifiers = event->modifiers();
    Qt::MouseButtons buttons = event->buttons();
    
    flags = 0;
    if (modifiers & Qt::ShiftModifier)
		flags |= CV_EVENT_FLAG_SHIFTKEY;
	if (modifiers & Qt::ControlModifier)
		flags |= CV_EVENT_FLAG_CTRLKEY;
	if (modifiers & Qt::AltModifier)
		flags |= CV_EVENT_FLAG_ALTKEY;

    if (buttons & Qt::LeftButton)
		flags |= CV_EVENT_FLAG_LBUTTON;
	if (buttons & Qt::RightButton)
		flags |= CV_EVENT_FLAG_RBUTTON;
    if (buttons & Qt::MidButton)
		flags |= CV_EVENT_FLAG_MBUTTON;

    cv_event = CV_EVENT_MOUSEMOVE;
	switch (event->button())
	{
	case Qt::LeftButton:
		cv_event = tableMouseButtons[category][0];
		flags |= CV_EVENT_FLAG_LBUTTON;
		break;

	case Qt::RightButton:
		cv_event = tableMouseButtons[category][1];
		flags |= CV_EVENT_FLAG_RBUTTON;
		break;

	case Qt::MidButton:
		cv_event = tableMouseButtons[category][2];
		flags |= CV_EVENT_FLAG_MBUTTON;
		break;

	default:
        ;
	}
}


void OpenGlViewPort::icvmouseProcessing(QPointF pt, int cv_event, int flags)
{
	if (mouseCallback)
		mouseCallback(cv_event, pt.x(), pt.y(), flags, mouseData);
}


QSize OpenGlViewPort::sizeHint() const
{
    if (size.width() > 0 && size.height() > 0)
        return size;

    return QGLWidget::sizeHint();
}

void OpenGlViewPort::setSize(QSize size_)
{
    size = size_;
    updateGeometry();
}

#endif

#endif // HAVE_QT
