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
static bool multiThreads = false;
static const int myargc = 1;
static char* myargv[] = {""};
static int last_key = -1;
QWaitCondition key_pressed;
QMutex mutexKey;
//end static and global


//end declaration
void cvChangeMode_QT(const char* name, double prop_value)
{
	//CV_WINDOW_NORMAL or CV_WINDOW_FULLSCREEN 

}

double cvGetMode_QT(const char* name)
{
	
	return 0;
}

CV_IMPL int cvWaitKey( int arg )
{
	
	CV_FUNCNAME( "cvWaitKey" );

    int result = -1;

    __BEGIN__;
    
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
            timer.start(arg);//delayms);

        //QTimer::singleShot(delayms, &guiMainThread, SLOT(timeOut()));
        while(!guiMainThread._bTimeOut)
        {
            qApp->processEvents(QEventLoop::AllEvents);

            mutexKey.lock();
            if (last_key != -1)
            {
                result = last_key;
                last_key = -1;
                timer.stop();
            }
            mutexKey.unlock();

            if (result!=-1)
                break;
            else
                usleep(2);//to decrease CPU usage
        }
        guiMainThread._bTimeOut = false;
    }
    
    __END__;
    
    return result;
}


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
		
	CV_FUNCNAME( "icvFindWindowByName" );

    QPointer<CvWindow> window = NULL;

    __BEGIN__;
    
    if( !arg )
        CV_Error( CV_StsNullPtr, "NULL name string" );

    QString name(arg);
    QPointer<CvWindow> w;
    foreach (QWidget *widget, QApplication::topLevelWidgets())
    {
        w = (CvWindow*) widget;
        if (w->name==name)
        {
            window = w;
            break;
        }
    }
    
    __END__;
    return window;
}

CV_IMPL CvTrackbar* icvFindTrackbarByName( const char* name_trackbar, const char* name_window )
{

	CV_FUNCNAME( "icvFindTrackbarByName" );

    QPointer<CvTrackbar> result = NULL;
    
    __BEGIN__;

    QPointer<CvWindow> w = icvFindWindowByName( name_window );

    if( !w )
        CV_Error( CV_StsNullPtr, "NULL window handler" );

    QString nameQt = QString(name_trackbar);
    QPointer<CvTrackbar> t;

    //for now, only trackbar are added so the Mutable cast is ok.
    for (int i = 0; i < w->layout->layout()->count()-1; ++i)
    {
        t = (CvTrackbar*) w->layout->layout()->itemAt(i);
        if (t->trackbar_name==nameQt)
        {
            result = t;
            break;
        }
    }

    __END__;
    return result;
}

CV_IMPL int cvNamedWindow( const char* name, int flags )
{
	CV_FUNCNAME( "cvNamedWindow" );

    __BEGIN__;
	
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
        
    __END__;
    return 1;//Dummy value
}

CV_IMPL void cvInformation(const char* name, const char* text, int delayms)
{
	CV_FUNCNAME( "cvInformation" );

    __BEGIN__;
    QMetaObject::invokeMethod(&guiMainThread,
                              "displayInfo",
                              Qt::AutoConnection,
                              Q_ARG(QString, QString(name)),
                              Q_ARG(QString, QString(text)),
                              Q_ARG(int, delayms));
                              
    __END__;
}

CV_IMPL int icvInitSystem( int argc, char** argv )
{
	CV_FUNCNAME( "icvInitSystem" );

    __BEGIN__;
    
    static int wasInitialized = 0;

    // check initialization status
    if( !wasInitialized)
    {
        new QApplication(argc,argv);

        wasInitialized = 1;
        qDebug()<<"init done"<<endl;
    }

	__END__;
    return 0;
}

CV_IMPL void cvDestroyWindow( const char* name )
{
	CV_FUNCNAME( "cvDestroyWindow" );

    __BEGIN__;
    
    QMetaObject::invokeMethod(&guiMainThread,
                              "destroyWindow",
                              //Qt::BlockingQueuedConnection,
                              Qt::AutoConnection,
                              Q_ARG(QString, QString(name))
                              );
                              
     __END__;
}


CV_IMPL void cvDestroyAllWindows(void)
{
	CV_FUNCNAME( "cvDestroyAllWindows" );

    __BEGIN__;
    
    QMetaObject::invokeMethod(&guiMainThread,
                              "destroyAllWindow",
                              //Qt::BlockingQueuedConnection,
                              Qt::AutoConnection
                              );
                              
    __END__;
}

CV_IMPL void* cvGetWindowHandle( const char* name )
{
	CV_FUNCNAME( "cvGetWindowHandle" );

    __BEGIN__;
    if( !name )
        CV_Error( CV_StsNullPtr, "NULL name string" );

    __END__;
    return (void*) icvFindWindowByName( name );
}

CV_IMPL const char* cvGetWindowName( void* window_handle )
{
	CV_FUNCNAME( "cvGetWindowName" );

    __BEGIN__;
    
    if( !window_handle )
        CV_Error( CV_StsNullPtr, "NULL window handler" );


	__END__;
    return ((CvWindow*)window_handle)->windowTitle().toLatin1().data();
}

CV_IMPL void cvMoveWindow( const char* name, int x, int y )
{   
	CV_FUNCNAME( "cvMoveWindow" );

    __BEGIN__;
    
    QMetaObject::invokeMethod(&guiMainThread,
                              "moveWindow",
                              //Qt::BlockingQueuedConnection,
                              Qt::AutoConnection,
                              Q_ARG(QString, QString(name)),
                              Q_ARG(int, x),
                              Q_ARG(int, y)
                              );
                              
     __END__;
}

CV_IMPL void cvResizeWindow(const char* name, int width, int height )
{

	CV_FUNCNAME( "cvResizeWindow" );

    __BEGIN__;

    QMetaObject::invokeMethod(&guiMainThread,
                              "resizeWindow",
                              //Qt::BlockingQueuedConnection,
                              Qt::AutoConnection,
                              Q_ARG(QString, QString(name)),
                              Q_ARG(int, width),
                              Q_ARG(int, height)
                              );
                              
    __END__;

}

CV_IMPL int cvCreateTrackbar2( const char* trackbar_name, const char* window_name, int* val, int count, CvTrackbarCallback2 on_notify, void* userdata )
{
	//TODO: implement the real one, not a wrapper
    return cvCreateTrackbar( trackbar_name, window_name, val, count, (CvTrackbarCallback)on_notify );
}

CV_IMPL int cvStartWindowThread()
{
    return 0;
}

CV_IMPL int cvCreateTrackbar( const char* trackbar_name, const char* window_name, int* value, int count, CvTrackbarCallback on_change)
{
	CV_FUNCNAME( "cvCreateTrackbar" );

    __BEGIN__;
    
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

    __END__;
	return 1;//demmy value
}

CV_IMPL int cvGetTrackbarPos( const char* trackbar_name, const char* window_name )
{
	CV_FUNCNAME( "cvGetTrackbarPos" );
    
    int result = -1;
    
    __BEGIN__;
    
    QPointer<CvTrackbar> t = icvFindTrackbarByName(  trackbar_name, window_name );

    if (t)
        result = t->slider->value();
 
    __END__;
    return result;
}

CV_IMPL void cvSetTrackbarPos( const char* trackbar_name, const char* window_name, int pos )
{
	CV_FUNCNAME( "cvSetTrackbarPos" );

    __BEGIN__;
    
    QPointer<CvTrackbar> t = icvFindTrackbarByName(  trackbar_name, window_name );

    if (t)
        t->slider->setValue(pos);
        
    __END__;
}

/* assign callback for mouse events */
CV_IMPL void cvSetMouseCallback( const char* window_name, CvMouseCallback on_mouse,void* param )
{
	CV_FUNCNAME( "cvSetMouseCallback" );

    __BEGIN__;
    
    QPointer<CvWindow> w = icvFindWindowByName( window_name );

    if (!w)
        CV_Error(CV_StsNullPtr, "NULL window handler" );

    w->setMouseCallBack(on_mouse, param);
    
    __END__;
}

CV_IMPL void cvShowImage( const char* name, const CvArr* arr )
{
	CV_FUNCNAME( "cvShowImage" );

    __BEGIN__;
    //objects were created in GUI thread, so not using invoke method here should be fine
    guiMainThread.showImage(QString(name), (void*) arr);

    //    QMetaObject::invokeMethod(&guiMainThread,
    //                              "showImage",
    //                              //Qt::BlockingQueuedConnection,
    //                              Qt::AutoConnection,
    //                                    Q_ARG(QString, QString(name)),
    //                                    Q_ARG(void*, (void*)arr)
    //                              );
    
    __END__;
}







//----------OBJECT----------------

GuiReceiver::GuiReceiver() : _bTimeOut(false)
{  
    icvInitSystem(myargc,myargv );
    qApp->setQuitOnLastWindowClosed ( false );//maybe the user would like to access this setting
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

void GuiReceiver::refreshEvents()
{
    QAbstractEventDispatcher::instance(qApp->instance()->thread())->processEvents(QEventLoop::AllEvents);
    //qDebug()<<"refresh ?"<<endl;
    //qApp->processEvents(QEventLoop::AllEvents);
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
    //moveToThread(qApp->instance()->thread());
    setObjectName(trackbar_name);
    parent = arg;
    trackbar_name = name;
    dataSlider = value;

    callback = on_change;
    slider = new QSlider(Qt::Horizontal);
    //slider->setObjectName(trackbar_name);
    slider->setFocusPolicy(Qt::StrongFocus);
    slider->setMinimum(0);
    slider->setMaximum(count);
    slider->setPageStep(5);
    slider->setValue(*value);
    slider->setTickPosition(QSlider::TicksBelow);


    //Change style of the Slider
    slider->setStyleSheet(str_Trackbar_css);
    
    //QFile qss(PATH_QSLIDERCSS);
    //if (qss.open(QFile::ReadOnly))
    //{
    //    slider->setStyleSheet(QLatin1String(qss.readAll()));
    //    qss.close();
    //}
    

    //this next line does not work if we change the style with a stylesheet, why ? (bug in QT ?)
    //slider->setTickPosition(QSlider::TicksBelow);
    label = new QPushButton;
    label->setFlat(true);
    setLabel(slider->value());


    QObject::connect( slider, SIGNAL( valueChanged( int ) ),this, SLOT( update( int ) ) );

    //if I connect those two signals in not-multiThreads mode,
    //the code crashes when we move the trackbar and then click on the button, ... why ?
    //so I disable the button

    if (multiThreads)
        QObject::connect( label, SIGNAL( clicked() ),this, SLOT( createDialog() ));
    else
        label->setEnabled(false);

    label->setStyleSheet("QPushButton:disabled {color: black}");

    addWidget(label);//name + value
    addWidget(slider);//slider
}

void CvTrackbar::createDialog()
{

    bool ok= false;

    //crash if I access the value directly to give them to QInputDialog, so do a copy first.
    int value = slider->value();
    int step = slider->singleStep();
    int min = slider->minimum();
    int max = slider->maximum();

    int i = QInputDialog::getInt(this->parentWidget(),
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

CvWindow::CvWindow(QString arg, int arg2)
{
    moveToThread(qApp->instance()->thread());

    last_key = 0;
    name = arg;
    on_mouse = NULL;
    flags = arg2;

    setAttribute(Qt::WA_DeleteOnClose);//in other case, does not release memory
    setContentsMargins(0,0,0,0);
    setWindowTitle(name);
    setObjectName(name);

    resize(400,300);

    //CV_MODE_NORMAL or CV_MODE_OPENGL
    myview = new ViewPort(this, CV_MODE_NORMAL);
    myview->setAlignment(Qt::AlignHCenter);

    layout = new QBoxLayout(QBoxLayout::TopToBottom);
    layout->setSpacing(5);
    layout->setObjectName(QString::fromUtf8("boxLayout"));
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(myview);

    if (flags == CV_WINDOW_AUTOSIZE)
        layout->setSizeConstraint(QLayout::SetFixedSize);

    setLayout(layout);

    show();
}

CvWindow::~CvWindow()
{
    QLayoutItem *child;

    if (layout)
    {
        while ((child = layout->takeAt(0)) != 0)
            delete child;

        delete layout;
    }

}

void CvWindow::displayInfo(QString text,int delayms)
{
    myview->startDisplayInfo(text, delayms);
}

void CvWindow::updateImage(void* arr)
{
    myview->updateImage(arr);
}

void CvWindow::setMouseCallBack(CvMouseCallback m, void* param)
{
    setMouseTracking (true);//receive mouse event everytime
    myview->setMouseTracking (true);//receive mouse event everytime
    on_mouse = m;
    on_mouse_param = param;
}

void CvWindow::addSlider(QString name, int* value, int count,CvTrackbarCallback on_change)
{
    QPointer<CvTrackbar> t = new CvTrackbar(this,name,value, count, on_change);
    t->setAlignment(Qt::AlignHCenter);
    layout->insertLayout(layout->count()-1,t);
}

void CvWindow::keyPressEvent(QKeyEvent *event)
{
    mutexKey.lock();
    last_key = (int)event->text().toLocal8Bit().at(0);
    mutexKey.unlock();
    key_pressed.wakeAll();
    QWidget::keyPressEvent(event);
}

void CvWindow::mousePressEvent(QMouseEvent *event)
{
    int cv_event = -1, flags = 0;
    QPoint pt = event->pos();


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
        cv_event = CV_EVENT_LBUTTONDOWN;
        flags |= CV_EVENT_FLAG_LBUTTON;
        break;
    case Qt::RightButton:
        cv_event = CV_EVENT_RBUTTONDOWN;
        flags |= CV_EVENT_FLAG_RBUTTON;
        break;
    case Qt::MidButton:
        cv_event = CV_EVENT_MBUTTONDOWN;
        flags |= CV_EVENT_FLAG_MBUTTON;
        break;
    default:;
    }

    if (on_mouse)
        on_mouse( cv_event, pt.x(), pt.y(), flags, on_mouse_param );

    QWidget::mousePressEvent(event);
}

void CvWindow::mouseReleaseEvent(QMouseEvent *event)
{

    int cv_event = -1, flags = 0;
    QPoint pt = event->pos();


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
        cv_event = CV_EVENT_LBUTTONUP;
        flags |= CV_EVENT_FLAG_LBUTTON;
        break;
    case Qt::RightButton:
        cv_event = CV_EVENT_RBUTTONUP;
        flags |= CV_EVENT_FLAG_RBUTTON;
        break;
    case Qt::MidButton:
        cv_event = CV_EVENT_MBUTTONUP;
        flags |= CV_EVENT_FLAG_MBUTTON;
        break;
    default:;
    }

    if (on_mouse)
        on_mouse( cv_event, pt.x(), pt.y(), flags, on_mouse_param );

    QWidget::mouseReleaseEvent(event);
}

void CvWindow::mouseDoubleClickEvent(QMouseEvent *event)
{
    int cv_event = -1, flags = 0;
    QPoint pt = event->pos();

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
        cv_event = CV_EVENT_LBUTTONDBLCLK;
        flags |= CV_EVENT_FLAG_LBUTTON;
        break;
    case Qt::RightButton:
        cv_event = CV_EVENT_RBUTTONDBLCLK;
        flags |= CV_EVENT_FLAG_RBUTTON;
        break;
    case Qt::MidButton:
        cv_event = CV_EVENT_MBUTTONDBLCLK;
        flags |= CV_EVENT_FLAG_MBUTTON;
        break;
    default:;
    }

    if (on_mouse)
        on_mouse( cv_event, pt.x(), pt.y(), flags, on_mouse_param );

    QWidget::mouseDoubleClickEvent(event);
}
void CvWindow::mouseMoveEvent(QMouseEvent *event)
{
    int cv_event = -1, flags = 0;
    QPoint pt = event->pos();

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

    cv_event = CV_EVENT_MOUSEMOVE;

    if (on_mouse)
        on_mouse( cv_event, pt.x(), pt.y(), flags, on_mouse_param );

    QWidget::mouseMoveEvent(event);
}



void CvWindow::readSettings()//not tested
{
    QSettings settings("Trolltech", "Application Example");
    QPoint pos = settings.value("pos", QPoint(200, 200)).toPoint();
    QSize size = settings.value("size", QSize(400, 400)).toSize();
    resize(size);
    move(pos);
}

void CvWindow::writeSettings()//not tested
{
    QSettings settings("Trolltech", "Application Example");
    settings.setValue("pos", pos());
    settings.setValue("size", size());
}



//Here is ViewPort
ViewPort::ViewPort(QWidget* arg, int arg2)
{
    mode = arg2;
    centralWidget = arg,
    setupViewport(centralWidget);
    setUpdatesEnabled(true);
    setObjectName(QString::fromUtf8("graphicsView"));
    timerDisplay = new QTimer(this);
    timerDisplay->setSingleShot(true);
    connect(timerDisplay, SIGNAL(timeout()), this, SLOT(stopDisplayInfo()));
    drawInfo = false;

    if (mode == CV_MODE_OPENGL)
    {
#if defined(OPENCV_GL)
        setViewport(new QGLWidget(QGLFormat(QGL::SampleBuffers)));
        initGL();
#endif
    }

    image2Draw=cvCreateImage(cvSize(centralWidget->width(),centralWidget->height()),IPL_DEPTH_8U,3);
    cvZero(image2Draw);
}

ViewPort::~ViewPort()
{
    if (image2Draw)
        cvReleaseImage(&image2Draw);
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
    if (!arr)
        CV_Error(CV_StsNullPtr, "NULL arr pointer (in showImage)" );

    IplImage* tempImage = (IplImage*)arr;

    if (!isSameSize(image2Draw,tempImage))
    {
        cvReleaseImage(&image2Draw);
        image2Draw=cvCreateImage(cvGetSize(tempImage),IPL_DEPTH_8U,3);

        updateGeometry();
    }       

    cvConvertImage(tempImage,image2Draw,CV_CVTIMG_SWAP_RB );
    viewport()->update();
}

//----- implemented to redirect event to the parent ------
void ViewPort::mouseMoveEvent(QMouseEvent *event)
{
    event->ignore();
}

void ViewPort::mousePressEvent(QMouseEvent *event)
{
    event->ignore();
}

void ViewPort::mouseReleaseEvent(QMouseEvent *event)
{
    event->ignore();
}

void ViewPort::mouseDoubleClickEvent(QMouseEvent *event)
{
    event->ignore();
}
//---------------------------------------------------------

QSize ViewPort::sizeHint() const
{
    if(image2Draw)
    {
        return QSize(image2Draw->width,image2Draw->height);
    } else {
        return QGraphicsView::sizeHint();
    }
}



void ViewPort::paintEvent(QPaintEvent* event)
{
    QPainter painter(viewport());

    draw2D(&painter);

    if (mode == CV_MODE_OPENGL)
    {
#if defined(OPENCV_GL)
        setGL(this->width(),this->height());
        draw3D();
        unsetGL();
#endif
    }

    if (drawInfo)
        drawInstructions(&painter);

    QGraphicsView::paintEvent(event);
}

void ViewPort::draw2D(QPainter *painter)
{ 
    QImage image((uchar*) image2Draw->imageData, image2Draw->width, image2Draw->height,QImage::Format_RGB888);
    painter->drawImage(0,0,image.scaled(this->width(),this->height(),Qt::IgnoreAspectRatio,Qt::SmoothTransformation));
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

void ViewPort::setGL(int width, int height)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluPerspective(45, float(width) / float(height), 0.01, 1000);
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

    glTranslated(0.0, 0.0, -1.0);
    // QVector3D p = convert(positionMouse);
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
