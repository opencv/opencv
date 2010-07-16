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


#ifdef HAVE_QT

#include <window_QT.h>

//Static and global first
static GuiReceiver guiMainThread;
static int parameterSystemC = 1;
static char* parameterSystemV[] = {""};
static bool multiThreads = false;
static int last_key = -1;
QWaitCondition key_pressed;
QMutex mutexKey;
static const unsigned int threshold_zoom_img_region = 15;
//the minimum zoom value to start displaying the values in the grid
//that is also the number of pixel per grid

//end static and global



CV_IMPL CvFont cvFont_Qt(const char* nameFont, int pointSize,CvScalar color,int weight,int style, int spacing)
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



CV_IMPL void cvAddText( CvArr* img, const char* text, CvPoint org, CvFont* font)
{
    QMetaObject::invokeMethod(&guiMainThread,
			      "putText",
			      Qt::AutoConnection,
			      Q_ARG(void*, (void*) img),
			      Q_ARG(QString,QString(text)),
			      Q_ARG(QPoint, QPoint(org.x,org.y)),
			      Q_ARG(void*,(void*) font));
}

double cvGetRatioWindow_QT(const char* name)
{
    double result = -1;
    QMetaObject::invokeMethod(&guiMainThread,
			      "getRatioWindow",
			      //Qt::DirectConnection,
			      Qt::AutoConnection,
			      Q_RETURN_ARG(double, result),
			      Q_ARG(QString, QString(name)));
    return result;
}

void cvSetRatioWindow_QT(const char* name,double prop_value)
{
    QMetaObject::invokeMethod(&guiMainThread,
			      "setRatioWindow",
			      Qt::AutoConnection,
			      Q_ARG(QString, QString(name)),
			      Q_ARG(double, prop_value));
}

double cvGetPropWindow_QT(const char* name)
{
    double result = -1;
    QMetaObject::invokeMethod(&guiMainThread,
			      "getPropWindow",
			      //Qt::DirectConnection,
			      Qt::AutoConnection,
			      Q_RETURN_ARG(double, result),
			      Q_ARG(QString, QString(name)));
    return result;
}

void cvSetPropWindow_QT(const char* name,double prop_value)
{
    QMetaObject::invokeMethod(&guiMainThread,
			      "setPropWindow",
			      Qt::AutoConnection,
			      Q_ARG(QString, QString(name)),
			      Q_ARG(double, prop_value));
}

void cvSetModeWindow_QT(const char* name, double prop_value)
{
    QMetaObject::invokeMethod(&guiMainThread,
			      "toggleFullScreen",
			      Qt::AutoConnection,
			      Q_ARG(QString, QString(name)),
			      Q_ARG(double, prop_value));
}

double cvGetModeWindow_QT(const char* name)
{
    double result = -1;

    QMetaObject::invokeMethod(&guiMainThread,
			      "isFullScreen",
			      Qt::AutoConnection,
			      Q_RETURN_ARG(double, result),
			      Q_ARG(QString, QString(name)));
    return result;
}

CV_IMPL void cvDisplayOverlay(const char* name, const char* text, int delayms)
{

    QMetaObject::invokeMethod(&guiMainThread,
			      "displayInfo",
			      Qt::AutoConnection,
			      //Qt::DirectConnection,
			      Q_ARG(QString, QString(name)),
			      Q_ARG(QString, QString(text)),
			      Q_ARG(int, delayms));

}

CV_IMPL void cvSaveWindowParameters(const char* name)
{
    QMetaObject::invokeMethod(&guiMainThread,
			      "saveWindowParameters",
			      Qt::AutoConnection,
			      Q_ARG(QString, QString(name)));
}

CV_IMPL void cvLoadWindowParameters(const char* name)
{
    QMetaObject::invokeMethod(&guiMainThread,
			      "loadWindowParameters",
			      Qt::AutoConnection,
			      Q_ARG(QString, QString(name)));
}

CV_IMPL void cvDisplayStatusBar(const char* name, const char* text, int delayms)
{

    QMetaObject::invokeMethod(&guiMainThread,
			      "displayStatusBar",
			      Qt::AutoConnection,
			      //Qt::DirectConnection,
			      Q_ARG(QString, QString(name)),
			      Q_ARG(QString, QString(text)),
			      Q_ARG(int, delayms));
}


CV_IMPL int cvInitSystem( int, char** )
{
    return 0;
}

CV_IMPL int cvWaitKey( int arg )
{
    int result = -1;

    unsigned long delayms;//in milliseconds
    if (arg<=0)
	delayms = ULONG_MAX;
    else
	delayms = arg;

    if (multiThreads)
    {
	mutexKey.lock();
	if(key_pressed.wait(&mutexKey,delayms))//false if timeout
	{
	    result = last_key;
	}
	last_key = -1;
	mutexKey.unlock();

    }else{
	//cannot use wait here because events will not be distributed before processEvents (the main eventLoop is broken)
	//so I create a Thread for the QTimer

	QTimer timer(&guiMainThread);
	QObject::connect(&timer, SIGNAL(timeout()), &guiMainThread, SLOT(timeOut()));
	timer.setSingleShot(true);

	if (arg>0)
	    timer.start(arg);

	//QMutex dummy;

	while(!guiMainThread._bTimeOut)
	{
	    qApp->processEvents(QEventLoop::AllEvents);

	    mutexKey.lock();
	    if (last_key != -1)
	    {
		result = last_key;
		last_key = -1;
		timer.stop();
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

#if defined WIN32 || defined _WIN32
		sleep(2);
#else
		usleep(2);//to decrease CPU usage
#endif

	    }
	}
	guiMainThread._bTimeOut = false;
    }

    return result;
}

//Yannick Verdie
//This function is experimental and some functions (such as cvSet/getWindowProperty will not work)
//We recommend not using this function for now
CV_IMPL int cvStartLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[])
{
    multiThreads = true;
    QFuture<int> future = QtConcurrent::run(pt2Func,argc,argv);
    return guiMainThread.start();
}

CV_IMPL void cvStopLoop()
{
    qApp->exit();
}


CV_IMPL CvWindow* icvFindWindowByName( const char* arg )
{

    QPointer<CvWindow> window = NULL;

    if( !arg )
	CV_Error( CV_StsNullPtr, "NULL name string" );

    QString name(arg);
    QPointer<CvWindow> w;
    foreach (QWidget *widget, QApplication::topLevelWidgets())
    {

	if (widget->isWindow() && !widget->parentWidget ())//is a window without parent
	{
	    w = (CvWindow*) widget;
	    if (w->param_name==name)
	    {
		window = w;
		break;
	    }
	}
    }

    return window;
}

CvTrackbar* icvFindTrackbarByName( const char* name_trackbar, const char* name_window )
{

    QPointer<CvTrackbar> result = NULL;

    QPointer<CvWindow> w = icvFindWindowByName( name_window );

    if( !w )
	CV_Error( CV_StsNullPtr, "NULL window handler" );

    QString nameQt = QString(name_trackbar);
    QPointer<CvTrackbar> t;

    int start_index;
    int stop_index;
    QPointer<QLayout> myLayout;

    if (w->param_gui_mode == CV_GUI_NORMAL)
    {
	myLayout = w->myLayout;

	start_index = 1;
	if (w->myToolBar)
	    start_index = 2;
	//Warning   ----  , asume the location 0 is toolbar, 1 is myview and max-1 the status bar
	//done three times in the code, in loadtrackbars, savetrackbar and in findtrackbar

	stop_index = myLayout->layout()->count()-1;
    }

    if (w->param_gui_mode == CV_GUI_EXPANDED)
    {
	myLayout = w->getWinProp()->myLayout;
	start_index = 0;
	stop_index = myLayout->layout()->count();
    }

    for (int i = start_index; i < stop_index; ++i)
    {
	t = (CvTrackbar*) myLayout->layout()->itemAt(i);
	if (t->trackbar_name==nameQt)
	{
	    result = t;
	    break;
	}
    }
    return result;


}

/*
CvTrackbar* icvFindButtonbarByName( const char* name_Buttonbar, const char* name_window )
{

    QPointer<CvButtonbar> result = NULL;

    QPointer<CvWindow> w = icvFindWindowByName( name_window );

    if( !w )
	CV_Error( CV_StsNullPtr, "NULL window handler" );

    QString nameQt = QString(name_Buttonbar);
    QPointer<CvButtonbar> t;

    int start_index = 1;
    if (w->myToolBar)
	start_index = 2;
    //Warning   ----  , asume the location 0 is toolbar, 1 is myview and max-1 the status bar
    //done three times in the code, in loadtrackbars, savetrackbar and in findtrackbar
    for (int i = start_index; i < w->myLayout->layout()->count()-1; ++i)
    {
	t = (CvTrackbar*) w->myLayout->layout()->itemAt(i);
	if (t->trackbar_name==nameQt)
	{
	    result = t;
	    break;
	}
    }

    return result;
}*/

int icvInitSystem()
{
    static int wasInitialized = 0;

    // check initialization status
    if( !wasInitialized)
    {
	new QApplication(parameterSystemC,parameterSystemV);

	wasInitialized = 1;
	qDebug()<<"init done";

#if defined(OPENCV_GL)//OK tested !
	qDebug()<<"opengl support available";
#endif
    }

    return 0;
}

CV_IMPL int cvNamedWindow( const char* name, int flags )
{

    if (multiThreads)
	QMetaObject::invokeMethod(&guiMainThread,
				  "createWindow",
				  //Qt::AutoConnection,
				  Qt::BlockingQueuedConnection,
				  //TypeConnection,
				  //Qt::AutoConnection,
				  Q_ARG(QString, QString(name)),
				  Q_ARG(int, flags));
    else
	guiMainThread.createWindow(QString(name),flags);

    return 1;//Dummy value
}

CV_IMPL void cvDestroyWindow( const char* name )
{
    QMetaObject::invokeMethod(&guiMainThread,
			      "destroyWindow",
			      //Qt::BlockingQueuedConnection,
			      Qt::AutoConnection,
			      Q_ARG(QString, QString(name))
			      );
}

CV_IMPL void cvDestroyAllWindows(void)
{

    QMetaObject::invokeMethod(&guiMainThread,
			      "destroyAllWindow",
			      //Qt::BlockingQueuedConnection,
			      Qt::AutoConnection
			      );

}

CV_IMPL void* cvGetWindowHandle( const char* name )
{
    if( !name )
	CV_Error( CV_StsNullPtr, "NULL name string" );

    return (void*) icvFindWindowByName( name );
}

CV_IMPL const char* cvGetWindowName( void* window_handle )
{

    if( !window_handle )
	CV_Error( CV_StsNullPtr, "NULL window handler" );

    return ((CvWindow*)window_handle)->windowTitle().toLatin1().data();
}

CV_IMPL void cvMoveWindow( const char* name, int x, int y )
{


    QMetaObject::invokeMethod(&guiMainThread,
			      "moveWindow",
			      //Qt::BlockingQueuedConnection,
			      Qt::AutoConnection,
			      Q_ARG(QString, QString(name)),
			      Q_ARG(int, x),
			      Q_ARG(int, y)
			      );

}

CV_IMPL void cvResizeWindow(const char* name, int width, int height )
{

    QMetaObject::invokeMethod(&guiMainThread,
			      "resizeWindow",
			      //Qt::BlockingQueuedConnection,
			      Qt::AutoConnection,
			      Q_ARG(QString, QString(name)),
			      Q_ARG(int, width),
			      Q_ARG(int, height)
			      );

}

//TODO: implement the real one, not a wrapper
CV_IMPL int cvCreateTrackbar2( const char* trackbar_name, const char* window_name, int* val, int count, CvTrackbarCallback2 on_notify, void* userdata )
{
    return cvCreateTrackbar( trackbar_name, window_name, val, count, (CvTrackbarCallback)on_notify );
}

CV_IMPL int cvStartWindowThread()
{
    return 0;
}

CV_IMPL int cvCreateTrackbar( const char* trackbar_name, const char* window_name, int* value, int count, CvTrackbarCallback on_change)
{

    if (multiThreads)
	QMetaObject::invokeMethod(&guiMainThread,
				  "addSlider",
				  Qt::AutoConnection,
				  Q_ARG(QString, QString(trackbar_name)),
				  Q_ARG(QString, QString(window_name)),
				  Q_ARG(void*, (void*)value),
				  Q_ARG(int, count),
				  Q_ARG(void*, (void*)on_change)
				  );
    else
	guiMainThread.addSlider(QString(trackbar_name),QString(window_name),(void*)value,count,(void*)on_change);

    return 1;//dummy value
}

CV_IMPL void cvCreateOpenGLCallback( const char* window_name, CvOpenGLCallback callbackOpenGL, void* userdata)
{
    QMetaObject::invokeMethod(&guiMainThread,
			      "setOpenGLCallback",
			      Qt::AutoConnection,
			      Q_ARG(QString, QString(window_name)),
			      Q_ARG(void*, (void*)callbackOpenGL),
			      Q_ARG(void*, userdata)
			      );
}

CV_IMPL int cvGetTrackbarPos( const char* trackbar_name, const char* window_name )
{
    int result = -1;

    QPointer<CvTrackbar> t = icvFindTrackbarByName(  trackbar_name, window_name );

    if (t)
	result = t->slider->value();

    return result;
}

CV_IMPL void cvSetTrackbarPos( const char* trackbar_name, const char* window_name, int pos )
{

    QPointer<CvTrackbar> t = icvFindTrackbarByName(  trackbar_name, window_name );

    if (t)
	t->slider->setValue(pos);

}

/* assign callback for mouse events */
CV_IMPL void cvSetMouseCallback( const char* window_name, CvMouseCallback on_mouse,void* param )
{
    QPointer<CvWindow> w = icvFindWindowByName( window_name );

    if (!w)
	CV_Error(CV_StsNullPtr, "NULL window handler" );

    w->setMouseCallBack(on_mouse, param);

}

CV_IMPL void cvShowImage( const char* name, const CvArr* arr )
{

    QMetaObject::invokeMethod(&guiMainThread,
			      "showImage",
			      //Qt::BlockingQueuedConnection,
			      Qt::DirectConnection,
			      Q_ARG(QString, QString(name)),
			      Q_ARG(void*, (void*)arr)
			      );
}


//----------OBJECT----------------

GuiReceiver::GuiReceiver() : _bTimeOut(false)
{
    icvInitSystem();
    qApp->setQuitOnLastWindowClosed ( false );//maybe the user would like to access this setting
}

void GuiReceiver::putText(void* arg1, QString text, QPoint org, void* arg2)
{
    CV_Assert(arg1)

	    IplImage* img = (IplImage*)arg1;

    //for now, only support QImage::Format_RGB888
    if (img->depth !=IPL_DEPTH_8U || img->nChannels != 3)
	return;

    CvFont* font = (CvFont*)arg2;



    QImage qimg((uchar*) img->imageData, img->width, img->height,QImage::Format_RGB888);
    QPainter qp(&qimg);
    if (font)
    {
	QFont f(font->nameFont, font->line_type/*PointSize*/, font->thickness/*weight*/);
	f.setStyle((QFont::Style)font->font_face/*style*/);
	f.setLetterSpacing ( QFont::AbsoluteSpacing, font->dx/*spacing*/ );
	//cvScalar(blue_component, green_component, red\_component[, alpha_component])
	//Qt map non-transparent to 0xFF and transparent to 0
	//OpenCV scalar is the reverse, so 255-font->color.val[3]
	qp.setPen(QColor(font->color.val[2],font->color.val[1],font->color.val[0],255-font->color.val[3]));
	qp.setFont ( f );
    }
    qp.drawText (org, text );
    qp.end();
}

void GuiReceiver::saveWindowParameters(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (w)
	w->writeSettings();
}

void GuiReceiver::loadWindowParameters(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (w)
	w->readSettings();
}

double GuiReceiver::getRatioWindow(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );


    if (!w)
	return -1;

    return (double)w->getView()->getRatio();
}

void GuiReceiver::setRatioWindow(QString name, double arg2 )
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (!w)
	return;

    int flags = (int) arg2;

    if (w->getView()->getRatio() == flags)//nothing to do
	return;

    //if valid flags
    if (flags == CV_WINDOW_FREERATIO || flags == CV_WINDOW_KEEPRATIO)
	w->getView()->setRatio(flags);

}

double GuiReceiver::getPropWindow(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );


    if (!w)
	return -1;

    return (double)w->param_flags;
}

void GuiReceiver::setPropWindow(QString name, double arg2 )
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (!w)
	return;

    int flags = (int) arg2;

    if (w->param_flags == flags)//nothing to do
	return;


    switch(flags)
    {
    case  CV_WINDOW_NORMAL:
	w->myLayout->setSizeConstraint(QLayout::SetMinAndMaxSize);
	w->param_flags = flags;
	break;
    case  CV_WINDOW_AUTOSIZE:
	w->myLayout->setSizeConstraint(QLayout::SetFixedSize);
	w->param_flags = flags;
	break;
    default:;
    }
}

double GuiReceiver::isFullScreen(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (!w)
	return -1;

    if (w->isFullScreen())
	return CV_WINDOW_FULLSCREEN;
    else
	return CV_WINDOW_NORMAL;
}

//accept CV_WINDOW_NORMAL or CV_WINDOW_FULLSCREEN
void GuiReceiver::toggleFullScreen(QString name, double flags )
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (!w)
	return;

    if (w->isFullScreen() && flags == CV_WINDOW_NORMAL)
    {
	w->showTools();
	w->showNormal();
	return;
    }

    if (!w->isFullScreen() && flags == CV_WINDOW_FULLSCREEN)
    {
	w->hideTools();
	w->showFullScreen();
	return;
    }

}

void GuiReceiver::createWindow( QString name, int flags )
{
    if (!qApp)
	CV_Error(CV_StsNullPtr, "NULL session handler" );

    // Check the name in the storage
    if( icvFindWindowByName( name.toLatin1().data() ))
    {
	return;
    }

    //QPointer<CvWindow> w1 =
    new CvWindow(name, flags);
}

void GuiReceiver::timeOut()
{
    _bTimeOut = true;
}

void GuiReceiver::displayInfo( QString name, QString text, int delayms )
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (w && delayms > 0)
	w->displayInfo(text,delayms);
}

void GuiReceiver::displayStatusBar( QString name, QString text, int delayms )
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (w && delayms > 0)
	if (w->myStatusBar)//if statusbar was created
	    w->displayStatusBar(text,delayms);
}

void GuiReceiver::showImage(QString name, void* arr)
{
    //qDebug()<<"inshowimage"<<endl;
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (!w)//as observed in the previous implementation (W32, GTK or Carbon), create a new window is the pointer returned is null
    {
	cvNamedWindow( name.toLatin1().data() );
	w = icvFindWindowByName( name.toLatin1().data() );
    }

    if( w && arr )
    {
	w->updateImage(arr);
    }
    else
    {
	qDebug()<<"Do nothing (Window or Image NULL)"<<endl;
    }
}

void GuiReceiver::destroyWindow(QString name)
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

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
	qApp->closeAllWindows();
    }else{
	QPointer<CvWindow> w;
	foreach (QWidget *widget, QApplication::topLevelWidgets())
	{
	    w = (CvWindow*) widget;
	    w->close();
	    delete w;
	}
    }

}

void GuiReceiver::setOpenGLCallback(QString window_name, void* callbackOpenGL, void* userdata)
{
    QPointer<CvWindow> w = icvFindWindowByName( window_name.toLatin1().data() );

    if (w && callbackOpenGL)
	w->setOpenGLCallback((CvOpenGLCallback) callbackOpenGL, userdata);
}

void GuiReceiver::moveWindow(QString name, int x, int y)
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (w)
	w->move(x,y);

}

void GuiReceiver::resizeWindow(QString name, int width, int height)
{
    QPointer<CvWindow> w = icvFindWindowByName( name.toLatin1().data() );

    if (w)
	w->resize(width, height);
}

void GuiReceiver::addSlider(QString trackbar_name, QString window_name, void* value, int count, void* on_change)
{
    QPointer<CvWindow> w = icvFindWindowByName( window_name.toLatin1().data()  );

    if (!w)
	return;

    if (!value)
	CV_Error(CV_StsNullPtr, "NULL value pointer" );

    if (count<= 0)//count is the max value of the slider, so must be bigger than 0
	CV_Error(CV_StsNullPtr, "Max value of the slider must be bigger than 0" );

    w->addSlider(trackbar_name,(int*)value,count,(CvTrackbarCallback) on_change);
}

int GuiReceiver::start()
{
    return qApp->exec();
}

CvTrackbar::CvTrackbar(CvWindow* arg, QString name, int* value, int count, CvTrackbarCallback on_change )
{
    setObjectName(trackbar_name);
    parent = arg;
    trackbar_name = name;
    dataSlider = value;

    callback = on_change;
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


    QObject::connect( slider, SIGNAL( valueChanged( int ) ),this, SLOT( update( int ) ) );

    QObject::connect( label, SIGNAL( clicked() ),this, SLOT( createDialog() ));

    //label->setStyleSheet("QPushButton:disabled {color: black}");

    addWidget(label);//name + value
    addWidget(slider);//slider
}

void CvTrackbar::createDialog()
{

    bool ok= false;

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
	     tr("Slider %1").arg(trackbar_name),
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
	callback(myvalue);
}

void CvTrackbar::setLabel(int myvalue)
{
    QString nameNormalized = trackbar_name.leftJustified( 10, ' ', true );
    QString valueMaximum = QString("%1").arg(slider->maximum());
    QString str = QString("%1 (%2/%3)").arg(nameNormalized).arg(myvalue,valueMaximum.length(),10,QChar('0')).arg(valueMaximum);
    label->setText(str);
}

CvTrackbar::~CvTrackbar()
{
    delete slider;
    delete label;
}



//here CvButtonbar class
/*
CvButtonbar::CvButtonbar()
{

}
*/




//here CvWinProperties class
CvWinProperties::CvWinProperties(QString name_paraWindow, QWidget* parent)
{
    setParent(parent);
    setWindowFlags(Qt::Tool);
    setContentsMargins(0,0,0,0);
    setWindowTitle(name_paraWindow);
    setObjectName(name_paraWindow);
    //parameters_window->setSizePolicy(QSizePolicy::Fixed,QSizePolicy::Fixed);
    resize(100,50);

    myLayout = new QBoxLayout(QBoxLayout::TopToBottom);
    myLayout->setObjectName(QString::fromUtf8("boxLayout"));
    myLayout->setContentsMargins(0, 0, 0, 0);
    myLayout->setSpacing(0);
    myLayout->setMargin(0);
    myLayout->setSizeConstraint(QLayout::SetFixedSize);
    setLayout(myLayout);

    hide();
}

void CvWinProperties::closeEvent ( QCloseEvent * e )
{
    e->accept();//intersept the close event (not sure I really need it)
    //an hide event is also sent. I will intercept it and do some processing
}

void CvWinProperties::showEvent ( QShowEvent * event )
{
    //why -1,-1 ?: do this trick because the first time the code is run,
    //no value pos was saved so we let Qt move the window in the middle of its parent (event ignored).
    //then hide will save the last position and thus, we want to retreive it (event accepted).
    QPoint mypos(-1,-1);
    QSettings settings("OpenCV2", this->windowTitle());
    mypos = settings.value("pos", mypos).toPoint();

    if (mypos.x()>=0)
    {
	move(mypos);
	event->accept();
    }
    else{
	event->ignore();
    }
}

void CvWinProperties::hideEvent ( QHideEvent * event )
{
    QSettings settings("OpenCV2", this->windowTitle());
    settings.setValue("pos", pos());//there is an offset of 6 pixels (so the window's position is wrong -- why ?)
    event->accept();
}

CvWinProperties::~CvWinProperties()
{
    //clear the setting pos
    QSettings settings("OpenCV2", this->windowTitle());
    settings.remove("pos");


    QLayoutItem *child;
    if (myLayout)
    {
	while ((child = myLayout->takeAt(0)) != 0)
	    delete child;

	delete myLayout;
    }
}






//Here CvWindow class
CvWindow::CvWindow(QString arg, int arg2)
{
    moveToThread(qApp->instance()->thread());
    param_name = arg;

    //the first bit is for normal or autoresize
    //CV_WINDOW_NORMAL = 0x00000000 and CV_WINDOW_AUTOSIZE = 0x00000001
    //the secont bit is for the gui mode (normal or expanded)
    //CV_GUI_EXPANDED = 0x00000000 and CV_GUI_NORMAL = 0x00000010
    param_flags = arg2 & 0x0000000F;
    param_gui_mode = arg2 & 0x000000F0;

    setAttribute(Qt::WA_DeleteOnClose);//in other case, does not release memory
    setContentsMargins(0,0,0,0);
    setWindowTitle(param_name);
    setObjectName(param_name);

    resize(400,300);

    createLayout();

    //1: my view
    int mode_display = CV_MODE_NORMAL;
#if defined(OPENCV_GL)
    mode_display = CV_MODE_OPENGL;
#endif
    createView(mode_display);

    //2: shortcuts
    createShortcuts();

    //toolBar and statusbar
    if (param_gui_mode == CV_GUI_EXPANDED)
    {
	createToolBar();
	createStatusBar();
	createParameterWindow();
    }

    //Now attach everything
    if (myToolBar)
	myLayout->addWidget(myToolBar,Qt::AlignCenter);

    myLayout->addWidget(myview,Qt::AlignCenter);

    if (myStatusBar)
	myLayout->addWidget(myStatusBar,Qt::AlignCenter);

    setLayout(myLayout);
    show();
}

CvWindow::~CvWindow()
{
    QLayoutItem *child;

    if (myLayout)
    {
	while ((child = myLayout->takeAt(0)) != 0)
	    delete child;

	delete myLayout;
    }

    if (myStatusBar)
    {
	delete myStatusBar;
	delete myStatusBar_msg;
    }

    if (myToolBar)
    {
	for (int i=0;i<vect_QActions.count();i++)
	    delete vect_QActions[i];

	delete myToolBar;
    }

    for (int i=0;i<vect_QShortcuts.count();i++)
	delete vect_QShortcuts[i];
}

CvWinProperties* CvWindow::getWinProp()
{
    return parameters_window;
}

void CvWindow::hideTools()
{
    if (myToolBar)
	myToolBar->hide();

    if (myStatusBar)
	myStatusBar->hide();

    if (parameters_window)
	parameters_window->hide();
}

void CvWindow::showTools()
{
    if (myToolBar)
	myToolBar->show();

    if (myStatusBar)
	myStatusBar->show();

    //if (parameters_window)
    //	parameters_window->show();
}

void CvWindow::createParameterWindow()
{
    QString name_paraWindow=param_name+" window parameters";
    parameters_window = new CvWinProperties(name_paraWindow,this);
}

void CvWindow::displayPropertiesWin()
{
    if (parameters_window->isHidden())
	parameters_window->show();
    else
	parameters_window->hide();
}

void CvWindow::createToolBar()
{
    myToolBar = new QToolBar;
    myToolBar->setFloatable(false);//is not a window
    myToolBar->setMaximumHeight(28);

    vect_QActions.resize(10);

    //if the shortcuts are changed in window_QT.h, we need to update the tooltip manually
    vect_QActions[0] = new QAction(QIcon(":/left-icon"),"Panning left (CTRL+arrowLEFT)",this);
    QObject::connect( vect_QActions[0],SIGNAL(triggered()),myview, SLOT( siftWindowOnLeft() ));
    myToolBar->addAction(vect_QActions[0]);

    vect_QActions[1] = new QAction(QIcon(":/right-icon"),"Panning right (CTRL+arrowRIGHT)",this);
    QObject::connect( vect_QActions[1],SIGNAL(triggered()),myview, SLOT( siftWindowOnRight() ));
    myToolBar->addAction(vect_QActions[1]);

    vect_QActions[2] = new QAction(QIcon(":/up-icon"),"Panning up (CTRL+arrowUP)",this);
    QObject::connect( vect_QActions[2],SIGNAL(triggered()),myview, SLOT( siftWindowOnUp() ));
    myToolBar->addAction(vect_QActions[2]);

    vect_QActions[3] = new QAction(QIcon(":/down-icon"),"Panning down (CTRL+arrowDOWN)",this);
    QObject::connect( vect_QActions[3],SIGNAL(triggered()),myview, SLOT( siftWindowOnDown() ));
    myToolBar->addAction(vect_QActions[3]);

    vect_QActions[4] = new QAction(QIcon(":/zoom_x1-icon"),"Zoom x1 (CTRL+P)",this);
    QObject::connect( vect_QActions[4],SIGNAL(triggered()),myview, SLOT( resetZoom() ));
    myToolBar->addAction(vect_QActions[4]);

    vect_QActions[5] = new QAction(QIcon(":/imgRegion-icon"),tr("Zoom x%1 (see label) (CTRL+X)")
				   .arg(threshold_zoom_img_region)
				   ,this);
    QObject::connect( vect_QActions[5],SIGNAL(triggered()),myview, SLOT( imgRegion() ));
    myToolBar->addAction(vect_QActions[5]);

    vect_QActions[6] = new QAction(QIcon(":/zoom_in-icon"),tr("Zoom in (CTRL++)"),this);
    QObject::connect( vect_QActions[6],SIGNAL(triggered()),myview, SLOT( ZoomIn() ));
    myToolBar->addAction(vect_QActions[6]);

    vect_QActions[7] = new QAction(QIcon(":/zoom_out-icon"),tr("Zoom out (CTRL+-)"),this);
    QObject::connect( vect_QActions[7],SIGNAL(triggered()),myview, SLOT( ZoomOut() ));
    myToolBar->addAction(vect_QActions[7]);

    vect_QActions[8] = new QAction(QIcon(":/save-icon"),tr("Save current image (CTRL+S)"),this);
    QObject::connect( vect_QActions[8],SIGNAL(triggered()),myview, SLOT( saveView() ));
    myToolBar->addAction(vect_QActions[8]);

    vect_QActions[9] = new QAction(QIcon(":/properties-icon"),tr("Display properties window (CTRL+P)"),this);
    QObject::connect( vect_QActions[9],SIGNAL(triggered()),this, SLOT( displayPropertiesWin() ));
    myToolBar->addAction(vect_QActions[9]);
}

void CvWindow::createStatusBar()
{
    myStatusBar = new QStatusBar;
    myStatusBar->setSizeGripEnabled(false);
    myStatusBar->setMaximumHeight(20);
    myStatusBar_msg = new QLabel;
    myStatusBar_msg->setFrameStyle(QFrame::Raised);
    myStatusBar_msg->setAlignment(Qt::AlignHCenter);
    myStatusBar->addWidget(myStatusBar_msg);
}

void CvWindow::createLayout()
{
    myLayout = new QBoxLayout(QBoxLayout::TopToBottom);
    myLayout->setObjectName(QString::fromUtf8("boxLayout"));
    myLayout->setContentsMargins(0, 0, 0, 0);
    myLayout->setSpacing(0);
    myLayout->setMargin(0);

    if (param_flags == CV_WINDOW_AUTOSIZE)
	myLayout->setSizeConstraint(QLayout::SetFixedSize);
}

void CvWindow::createShortcuts()
{
    vect_QShortcuts.resize(10);

    vect_QShortcuts[0] = new QShortcut(shortcut_panning_left, this);
    QObject::connect( vect_QShortcuts[0], SIGNAL( activated ()),myview, SLOT( siftWindowOnLeft() ));
    vect_QShortcuts[1] = new QShortcut(shortcut_panning_right, this);
    QObject::connect( vect_QShortcuts[1], SIGNAL( activated ()),myview, SLOT( siftWindowOnRight() ));
    vect_QShortcuts[2] = new QShortcut(shortcut_panning_up, this);
    QObject::connect(vect_QShortcuts[2], SIGNAL( activated ()),myview, SLOT( siftWindowOnUp() ));
    vect_QShortcuts[3] = new QShortcut(shortcut_panning_down, this);
    QObject::connect(vect_QShortcuts[3], SIGNAL( activated ()),myview, SLOT( siftWindowOnDown() ));

    vect_QShortcuts[4] = new QShortcut(shortcut_zoom_normal, this);
    QObject::connect( vect_QShortcuts[4], SIGNAL( activated ()),myview, SLOT( resetZoom( ) ));
    vect_QShortcuts[5] = new QShortcut(shortcut_zoom_imgRegion, this);
    QObject::connect( vect_QShortcuts[5], SIGNAL( activated ()),myview, SLOT( imgRegion( ) ));
    vect_QShortcuts[6] = new QShortcut(shortcut_zoom_in, this);
    QObject::connect( vect_QShortcuts[6], SIGNAL( activated ()),myview, SLOT( ZoomIn() ));
    vect_QShortcuts[7] = new QShortcut(shortcut_zoom_out, this);
    QObject::connect(vect_QShortcuts[7], SIGNAL( activated ()),myview, SLOT( ZoomOut() ));
    vect_QShortcuts[8] = new QShortcut(shortcut_save_img, this);
    QObject::connect( vect_QShortcuts[8], SIGNAL( activated ()),myview, SLOT( saveView( ) ));
    vect_QShortcuts[9] = new QShortcut(shortcut_properties_win, this);
    QObject::connect( vect_QShortcuts[9], SIGNAL( activated ()),this, SLOT( displayPropertiesWin() ));
}

void CvWindow::createView(int mode)
{
    //mode = CV_MODE_NORMAL or CV_MODE_OPENGL
    myview = new ViewPort(this, mode,CV_WINDOW_KEEPRATIO);//parent, mode_display, keep_aspect_ratio
    myview->setAlignment(Qt::AlignHCenter);
}

void CvWindow::setOpenGLCallback(CvOpenGLCallback func,void* userdata)
{
    myview->setOpenGLCallback(func,userdata);
}

ViewPort* CvWindow::getView()
{
    return myview;
}

void CvWindow::displayInfo(QString text,int delayms)
{
    myview->startDisplayInfo(text, delayms);
}

void CvWindow::displayStatusBar(QString text,int delayms)
{
    myStatusBar->showMessage(text,delayms);
}

void CvWindow::updateImage(void* arr)
{
    myview->updateImage(arr);
}

void CvWindow::setMouseCallBack(CvMouseCallback m, void* param)
{
    myview->setMouseCallBack(m,param);
}

void CvWindow::addSlider(QString name, int* value, int count,CvTrackbarCallback on_change)
{
    QPointer<CvTrackbar> t = new CvTrackbar(this,name,value, count, on_change);
    t->setAlignment(Qt::AlignHCenter);

    int position_insert;
    if (param_gui_mode == CV_GUI_NORMAL)
    {
	position_insert = myLayout->count();

	if (myStatusBar)
	    position_insert--;//max-1 means add trackbar between myview and statusbar

	myLayout->insertLayout(position_insert,t);
	return;
    }

    if (param_gui_mode == CV_GUI_EXPANDED)
    {
	position_insert = parameters_window->myLayout->count();
	parameters_window->myLayout->insertLayout(position_insert,t);
	return;
    }

}

//Need more test here !
void CvWindow::keyPressEvent(QKeyEvent *event)
{
    //see http://doc.trolltech.com/4.6/qt.html#Key-enum
    int key = event->key();
    bool goodKey = false;

    if (key>=20 && key<=255 )
    {
	key = (int)event->text().toLocal8Bit().at(0);
	goodKey = true;
    }

    if (key == Qt::Key_Escape)
    {
	key = 27;
	goodKey = true;
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

//TODO: load CV_GUI flag (done) and act accordingly (create win property if needed and attach trackbars)
void CvWindow::readSettings()
{
    //organisation and application's name
    QSettings settings("OpenCV2", QFileInfo(QApplication::applicationFilePath()).fileName());
    QPoint pos = settings.value("pos", QPoint(200, 200)).toPoint();
    QSize size = settings.value("size", QSize(400, 400)).toSize();
    //param_name = settings.value("name_window",param_name).toString();
    param_flags = settings.value("mode_resize",param_flags).toInt();
    param_gui_mode = settings.value("mode_gui",param_gui_mode).toInt();
    myview->param_keepRatio = settings.value("view_aspectRatio",myview->param_keepRatio).toInt();

    param_flags = settings.value("mode_resize",param_flags).toInt();
    qreal m11 = settings.value("matrix_view.m11",myview->param_matrixWorld.m11()).toDouble();
    qreal m12 = settings.value("matrix_view.m12",myview->param_matrixWorld.m12()).toDouble();
    qreal m13 = settings.value("matrix_view.m13",myview->param_matrixWorld.m13()).toDouble();
    qreal m21 = settings.value("matrix_view.m21",myview->param_matrixWorld.m21()).toDouble();
    qreal m22 = settings.value("matrix_view.m22",myview->param_matrixWorld.m22()).toDouble();
    qreal m23 = settings.value("matrix_view.m23",myview->param_matrixWorld.m23()).toDouble();
    qreal m31 = settings.value("matrix_view.m31",myview->param_matrixWorld.m31()).toDouble();
    qreal m32 = settings.value("matrix_view.m32",myview->param_matrixWorld.m32()).toDouble();
    qreal m33 = settings.value("matrix_view.m33",myview->param_matrixWorld.m33()).toDouble();
    myview->param_matrixWorld = QTransform(m11,m12,m13,m21,m22,m23,m31,m32,m33);

    //trackbar here
    icvLoadTrackbars(&settings);

    resize(size);
    move(pos);
}

void CvWindow::writeSettings()
{
    //organisation and application's name
    QSettings settings("OpenCV2", QFileInfo(QApplication::applicationFilePath()).fileName());
    //settings.setValue("name_window",param_name);
    settings.setValue("pos", pos());
    settings.setValue("size", size());
    settings.setValue("mode_resize",param_flags);
    settings.setValue("mode_gui",param_gui_mode);
    settings.setValue("view_aspectRatio",myview->param_keepRatio);

    settings.setValue("matrix_view.m11",myview->param_matrixWorld.m11());
    settings.setValue("matrix_view.m12",myview->param_matrixWorld.m12());
    settings.setValue("matrix_view.m13",myview->param_matrixWorld.m13());
    settings.setValue("matrix_view.m21",myview->param_matrixWorld.m21());
    settings.setValue("matrix_view.m22",myview->param_matrixWorld.m22());
    settings.setValue("matrix_view.m23",myview->param_matrixWorld.m23());
    settings.setValue("matrix_view.m31",myview->param_matrixWorld.m31());
    settings.setValue("matrix_view.m32",myview->param_matrixWorld.m32());
    settings.setValue("matrix_view.m33",myview->param_matrixWorld.m33());

    icvSaveTrackbars(&settings);
}

void CvWindow::icvLoadTrackbars(QSettings *settings)
{
    int size = settings->beginReadArray("trackbars");
    QPointer<CvTrackbar> t;
    //Warning   ----  , asume the location 0 is toolbar, 1 is myview and max-1 the status bar
    //done three times in the code, in loadtrackbars, savetrackbar and in findtrackbar


    //trackbar are saved in the same order, so no need to use icvFindTrackbarByName

    int start_index = 1;//index 0 is myview
    if (myToolBar)
	start_index ++;//index 0 is statusbar, 1 is myview

    int stop_index = myLayout->layout()->count() - start_index ;
    if (myStatusBar)
	stop_index --;// index max-1 is the statusbar

    //(in expended mode) nbTrackbar = count() - (toolbar + myview + statusbar) (3) = stop_index - start_index

    if (stop_index-start_index == size)//if not the same number, the window saved and loaded is not the same (nb trackbar not equal)
	for (int i = start_index; i < size+start_index; ++i)
	{
	settings->setArrayIndex(i-start_index);
	t = (CvTrackbar*)  myLayout->layout()->itemAt(i);

	if (t->trackbar_name == settings->value("name").toString())
	{
	    t->slider->setValue(settings->value("value").toInt());
	}
    }
    settings->endArray();

}

void CvWindow::icvSaveTrackbars(QSettings *settings)
{
    QPointer<CvTrackbar> t;

    //Warning   ----  , asume the location 0 is toolbar, 1 is myview and max-1 the status bar
    //done three times in the code, in loadtrackbars, savetrackbar and in findtrackbar
    settings->beginWriteArray("trackbars");

    int start_index = 2;
    if (myToolBar)
	start_index=3;

    for (int i = start_index; i < myLayout->layout()->count()-1; ++i) {
	t = (CvTrackbar*)  myLayout->layout()->itemAt(i);
	settings->setArrayIndex(i-start_index);
	settings->setValue("name", t->trackbar_name);
	settings->setValue("value", t->slider->value());
    }
    settings->endArray();
}







//Here is ViewPort class
ViewPort::ViewPort(CvWindow* arg, int arg2, int arg3)
{
    centralWidget = arg,
    setParent(centralWidget);
    mode_display = arg2;
    param_keepRatio = arg3;

    setupViewport(centralWidget);
    setContentsMargins(0,0,0,0);

    setObjectName(QString::fromUtf8("graphicsView"));
    timerDisplay = new QTimer(this);
    timerDisplay->setSingleShot(true);
    connect(timerDisplay, SIGNAL(timeout()), this, SLOT(stopDisplayInfo()));
    drawInfo = false;
    positionGrabbing = QPointF(0,0);
    positionCorners = QRect(0,0,size().width(),size().height());
    on_mouse = NULL;
    mouseCoordinate = QPoint(-1,-1);
    on_openGL_draw3D = NULL;


#if defined(OPENCV_GL)
    if ( mode_display == CV_MODE_OPENGL)
    {
	//QGLWidget* wGL = new QGLWidget(QGLFormat(QGL::SampleBuffers));
	setViewport(new QGLWidget(QGLFormat(QGL::SampleBuffers)));
	initGL();
    }
#endif

    image2Draw_ipl=cvCreateImage(cvSize(centralWidget->width(),centralWidget->height()),IPL_DEPTH_8U,3);
    image2Draw_qt = QImage((uchar*) image2Draw_ipl->imageData, image2Draw_ipl->width, image2Draw_ipl->height,QImage::Format_RGB888);
    image2Draw_qt_resized = image2Draw_qt.scaled(this->width(),this->height(),Qt::IgnoreAspectRatio,Qt::SmoothTransformation);

    nbChannelOriginImage = 0;
    cvZero(image2Draw_ipl);

    setInteractive(false);
    setMouseTracking (true);//receive mouse event everytime

}

ViewPort::~ViewPort()
{
    if (image2Draw_ipl)
	cvReleaseImage(&image2Draw_ipl);

    delete timerDisplay;
}

//can save as JPG, JPEG, BMP, PNG
void ViewPort::saveView()
{
    QDate date_d = QDate::currentDate ();
    QString date_s = date_d.toString("dd.MM.yyyy");
    QString name_s = centralWidget->param_name+"_screenshot_"+date_s;

    QString fileName = QFileDialog::getSaveFileName(this, tr("Save File %1").arg(name_s),
						    name_s+".png",
						    tr("Images (*.png *.jpg *.bmp *.jpeg)"));

    if (!fileName.isEmpty ())//save the picture
    {
	QString extension = fileName.right(3);

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

	qDebug()<<"file extension not recognized, please choose between JPG, JPEG, BMP or PNG";
    }
}

void ViewPort::setRatio(int flags)
{
    param_keepRatio = flags;
    updateGeometry();
    viewport()->update();
}

void ViewPort::imgRegion()
{
    scaleView( (threshold_zoom_img_region/param_matrixWorld.m11()-1)*5,QPointF(size().width()/2,size().height()/2));
}

int ViewPort::getRatio()
{
    return param_keepRatio;
}

void ViewPort::resetZoom()
{
    param_matrixWorld.reset();
    controlImagePosition();
}

void ViewPort::ZoomIn()
{
    scaleView( 0.5,QPointF(size().width()/2,size().height()/2));
}

void ViewPort::ZoomOut()
{
    scaleView( -0.5,QPointF(size().width()/2,size().height()/2));
}

//Note: move 2 percent of the window
void  ViewPort::siftWindowOnLeft()
{
    float delta = 2*width()/(100.0*param_matrixWorld.m11());
    moveView(QPointF(delta,0));
}

//Note: move 2 percent of the window
void  ViewPort::siftWindowOnRight()
{
    float delta = -2*width()/(100.0*param_matrixWorld.m11());
    moveView(QPointF(delta,0));
}

//Note: move 2 percent of the window
void  ViewPort::siftWindowOnUp()
{
    float delta = 2*height()/(100.0*param_matrixWorld.m11());
    moveView(QPointF(0,delta));
}

//Note: move 2 percent of the window
void  ViewPort::siftWindowOnDown()
{
    float delta = -2*height()/(100.0*param_matrixWorld.m11());
    moveView(QPointF(0,delta));
}

void ViewPort::startDisplayInfo(QString text, int delayms)
{
    if (timerDisplay->isActive())
	stopDisplayInfo();

    infoText = text;
    timerDisplay->start(delayms);
    drawInfo = true;
}

void ViewPort::stopDisplayInfo()
{
    timerDisplay->stop();
    drawInfo = false;
}

inline bool ViewPort::isSameSize(IplImage* img1,IplImage* img2)
{
    return img1->width == img2->width && img1->height == img2->height;
}

void ViewPort::updateImage(void* arr)
{
    //if (!arr)
    //CV_Error(CV_StsNullPtr, "NULL arr pointer (in showImage)" );
    CV_Assert(arr)

	    IplImage* tempImage = (IplImage*)arr;

    if (!isSameSize(image2Draw_ipl,tempImage))
    {
	cvReleaseImage(&image2Draw_ipl);
	image2Draw_ipl=cvCreateImage(cvGetSize(tempImage),IPL_DEPTH_8U,3);
	image2Draw_qt = QImage((uchar*) image2Draw_ipl->imageData, image2Draw_ipl->width, image2Draw_ipl->height,QImage::Format_RGB888);
	image2Draw_qt_resized = image2Draw_qt.scaled(this->width(),this->height(),Qt::IgnoreAspectRatio,Qt::SmoothTransformation);


	nbChannelOriginImage = tempImage->nChannels;
	updateGeometry();
    }

    cvConvertImage(tempImage,image2Draw_ipl,CV_CVTIMG_SWAP_RB );

    viewport()->update();
}

void ViewPort::setMouseCallBack(CvMouseCallback m, void* param)
{
    on_mouse = m;
    on_mouse_param = param;
}

void ViewPort::setOpenGLCallback(CvOpenGLCallback func,void* userdata)
{
    on_openGL_draw3D = func;
    on_openGL_param = userdata;
}

void ViewPort::controlImagePosition()
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

    viewport()->update();
}

void ViewPort::moveView(QPointF delta)
{
    param_matrixWorld.translate(delta.x(),delta.y());
    controlImagePosition();
}

//factor is -0.5 (zoom out) or 0.5 (zoom in)
void ViewPort::scaleView(qreal factor,QPointF center)
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

void ViewPort::wheelEvent(QWheelEvent *event)
{
    scaleView( -event->delta() / 240.0,event->pos());
}

void ViewPort::mousePressEvent(QMouseEvent *event)
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

void ViewPort::mouseReleaseEvent(QMouseEvent *event)
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

void ViewPort::mouseDoubleClickEvent(QMouseEvent *event)
{
    int cv_event = -1, flags = 0;
    QPoint pt = event->pos();

    //icvmouseHandler: pass parameters for cv_event, flags
    icvmouseHandler(event, mouse_dbclick, cv_event, flags);
    icvmouseProcessing(QPointF(pt), cv_event, flags);

    QWidget::mouseDoubleClickEvent(event);
}

void ViewPort::mouseMoveEvent(QMouseEvent *event)
{
    int cv_event = -1, flags = 0;
    QPoint pt = event->pos();

    //icvmouseHandler: pass parameters for cv_event, flags
    icvmouseHandler(event, mouse_move, cv_event, flags);
    icvmouseProcessing(QPointF(pt), cv_event, flags);


    if (param_matrixWorld.m11()>1 && event->buttons() == Qt::LeftButton)
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

//up, down, dclick, move
void ViewPort::icvmouseHandler(QMouseEvent *event, type_mouse_event category, int &cv_event, int &flags)
{

    switch(event->modifiers())
    {
    case Qt::ShiftModifier:
	flags = CV_EVENT_FLAG_SHIFTKEY;
	break;
    case Qt::ControlModifier:
	flags = CV_EVENT_FLAG_CTRLKEY;
	break;
    case Qt::AltModifier:
	flags = CV_EVENT_FLAG_ALTKEY;
	break;
    case Qt::NoModifier	:
	break;
    case Qt::MetaModifier:
	break;
    case Qt::KeypadModifier:
	break;
    default:;
    }

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

void ViewPort::icvmouseProcessing(QPointF pt, int cv_event, int flags)
{
    //to convert mouse coordinate
    qreal pfx, pfy;
    matrixWorld_inv.map(pt.x(),pt.y(),&pfx,&pfy);
    mouseCoordinate.rx()=floor(pfx);
    mouseCoordinate.ry()=floor(pfy);

    if (on_mouse)
	on_mouse( cv_event, mouseCoordinate.x(),mouseCoordinate.y(), flags, on_mouse_param );

}

QSize ViewPort::sizeHint() const
{
    if(image2Draw_ipl)
    {
	return QSize(image2Draw_ipl->width,image2Draw_ipl->height);
    } else {
	return QGraphicsView::sizeHint();
    }
}

void ViewPort::resizeEvent ( QResizeEvent *event)
{
    image2Draw_qt_resized = image2Draw_qt.scaled(this->width(),this->height(),Qt::IgnoreAspectRatio,Qt::SmoothTransformation);

    controlImagePosition();
    ratioX=width()/float(image2Draw_ipl->width);
    ratioY=height()/float(image2Draw_ipl->height);

    if(param_keepRatio == CV_WINDOW_KEEPRATIO)//to keep the same aspect ratio
    {
	QSize newSize = QSize(image2Draw_ipl->width,image2Draw_ipl->height);
	newSize.scale(event->size(),Qt::KeepAspectRatio);

	//imageWidth/imageHeight = newWidth/newHeight +/- epsilon
	//ratioX = ratioY +/- epsilon
	//||ratioX - ratioY|| = epsilon
	if (fabs(ratioX - ratioY)*100> ratioX)//avoid infinity loop / epsilon = 1% of ratioX
	{
	    resize(newSize);

	    //move to the middle
	    //newSize get the delta offset to place the picture in the middle of its parent
	    newSize= (event->size()-newSize)/2;

	    //if the toolbar is displayed, avoid drawing myview on top of it
	    if (centralWidget->myToolBar)
		if(!centralWidget->myToolBar->isHidden())
		    newSize +=QSize(0,centralWidget->myToolBar->height());

	    move(newSize.width(),newSize.height());
	}
    }

    return QGraphicsView::resizeEvent(event);
}

void ViewPort::paintEvent(QPaintEvent* event)
{
    //first paint on a file (to be able to save it if needed)
    //  ---------  START PAINTING FILE  --------------  //
    QPainter myPainter(&image2Draw_qt_resized);
    myPainter.setWorldTransform(param_matrixWorld);

    draw2D(&myPainter);

#if defined(OPENCV_GL)
    if ( mode_display == CV_MODE_OPENGL && on_openGL_draw3D)
    {
	//myPainter.beginNativePainting();

	setGL(width(),height());
	on_openGL_draw3D(on_openGL_param);
	//draw3D();
	unsetGL();

	//myPainter.endNativePainting();
    }
#endif

    //Now disable matrixWorld for overlay display
    myPainter.setWorldMatrixEnabled (false );

    //in mode zoom/panning
    if (param_matrixWorld.m11()>1)
    {
	if (param_matrixWorld.m11()>=threshold_zoom_img_region)
	    drawImgRegion(&myPainter);

	drawViewOverview(&myPainter);
    }

    //for information overlay
    if (drawInfo)
	drawInstructions(&myPainter);

    //  ---------  END PAINTING FILE  --------------  //
    myPainter.end();


    //and now display the file
    myPainter.begin(viewport());
    myPainter.drawImage(0, 0, image2Draw_qt_resized);
    //end display

    //for statusbar
    if (centralWidget->myStatusBar)
	drawStatusBar();

    QGraphicsView::paintEvent(event);
}

void ViewPort::draw2D(QPainter *painter)
{
    painter->drawImage(0,0,image2Draw_qt.scaled(this->width(),this->height(),Qt::IgnoreAspectRatio,Qt::SmoothTransformation));
}

void ViewPort::drawStatusBar()
{
    if (mouseCoordinate.x()>=0 &&
	mouseCoordinate.y()>=0 &&
	mouseCoordinate.x()<image2Draw_ipl->width &&
	mouseCoordinate.y()<image2Draw_ipl->height)
    {
	QRgb rgbValue = image2Draw_qt.pixel(mouseCoordinate);

	if (nbChannelOriginImage==3)
	{
	    centralWidget->myStatusBar_msg->setText(tr("<font color='black'>Coordinate: %1x%2 ~ </font>")
						    .arg(mouseCoordinate.x())
						    .arg(mouseCoordinate.y())+
						    tr("<font color='red'>R:%3 </font>").arg(qRed(rgbValue))+//.arg(value.val[0])+
						    tr("<font color='green'>G:%4 </font>").arg(qGreen(rgbValue))+//.arg(value.val[1])+
						    tr("<font color='blue'>B:%5</font>").arg(qBlue(rgbValue))//.arg(value.val[2])
						    );
	}else{
	    //all the channel have the same value (because of cvconvertimage), so only the r channel is dsplayed
	    centralWidget->myStatusBar_msg->setText(tr("<font color='black'>Coordinate: %1x%2 ~ </font>")
						    .arg(mouseCoordinate.x())
						    .arg(mouseCoordinate.y())+
						    tr("<font color='grey'>grey:%3 </font>").arg(qRed(rgbValue))
						    );
	}
    }
}

void ViewPort::drawImgRegion(QPainter *painter)
{
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
    f.setPixelSize(6+(param_matrixWorld.m11()-threshold_zoom_img_region)/5);
    painter->setFont(f);
    QString val;
    QRgb rgbValue;

    QPointF point1;//sorry, I do not know how to name it
    QPointF point2;//idem


    for (int j=-1;j<view.height()/param_matrixWorld.m11();j++)
	for (int i=-1;i<view.width()/param_matrixWorld.m11();i++)
	{
	point1.setX((i+offsetX)*param_matrixWorld.m11());
	point1.setY((j+offsetY)*param_matrixWorld.m11());

	matrixWorld_inv.map(point1.x(),point1.y(),&point2.rx(),&point2.ry());

	if (point2.x() >= 0 && point2.y() >= 0)
	    rgbValue = image2Draw_qt.pixel(QPoint(point2.x(),point2.y()));
	else
	    rgbValue = qRgb(0,0,0);

	if (nbChannelOriginImage==3)
	{
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
	else
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

void ViewPort::drawViewOverview(QPainter *painter)
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

void ViewPort::drawInstructions(QPainter *painter)
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














#if defined(OPENCV_GL)//all this section -> not tested

void ViewPort::initGL()
{
    glShadeModel( GL_SMOOTH );
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
    glEnable( GL_TEXTURE_2D );
    glEnable( GL_CULL_FACE );
    glEnable( GL_DEPTH_TEST );
}

//from http://steinsoft.net/index.php?site=Programming/Code%20Snippets/OpenGL/gluperspective
//do not want to link glu
void ViewPort::icvgluPerspective(GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar)
{
    GLdouble xmin, xmax, ymin, ymax;

    ymax = zNear * tan(fovy * M_PI / 360.0);
    ymin = -ymax;
    xmin = ymin * aspect;
    xmax = ymax * aspect;


    glFrustum(xmin, xmax, ymin, ymax, zNear, zFar);
}


void ViewPort::setGL(int width, int height)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    icvgluPerspective(45, float(width) / float(height), 0.01, 1000);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
}

void ViewPort::unsetGL()
{
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}

void ViewPort::draw3D()
{
    //draw scene here
    glLoadIdentity();

    glTranslated(10.0, 10.0, -1.0);
    // QVector3D p = convert(mouseCoordinate);
    //glTranslated(p.x(),p.y(),p.z());

    glRotatef( 55, 1, 0, 0 );
    glRotatef( 45, 0, 1, 0 );
    glRotatef( 0, 0, 0, 1 );

    static const int coords[6][4][3] = {
	{ { +1, -1, -1 }, { -1, -1, -1 }, { -1, +1, -1 }, { +1, +1, -1 } },
	{ { +1, +1, -1 }, { -1, +1, -1 }, { -1, +1, +1 }, { +1, +1, +1 } },
	{ { +1, -1, +1 }, { +1, -1, -1 }, { +1, +1, -1 }, { +1, +1, +1 } },
	{ { -1, -1, -1 }, { -1, -1, +1 }, { -1, +1, +1 }, { -1, +1, -1 } },
	{ { +1, -1, +1 }, { -1, -1, +1 }, { -1, -1, -1 }, { +1, -1, -1 } },
	{ { -1, -1, +1 }, { +1, -1, +1 }, { +1, +1, +1 }, { -1, +1, +1 } }
    };

    for (int i = 0; i < 6; ++i) {
	glColor3ub( i*20, 100+i*10, i*42 );
	glBegin(GL_QUADS);
	for (int j = 0; j < 4; ++j) {
	    glVertex3d(0.2 * coords[i][j][0], 0.2 * coords[i][j][1], 0.2 * coords[i][j][2]);
	}
	glEnd();
    }
}
#endif

#endif
