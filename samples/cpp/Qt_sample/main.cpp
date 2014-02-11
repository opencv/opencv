/**
 * Mehdi Lauters
 * mehdilauters at gmail dot com
 */
#include <QtCore/QCoreApplication>
#include <QApplication>
#include <QImage>
#include <iostream>

#include "ImageManager.h"
#include "Gui.h"

using namespace std;

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Please pass the video device or file to the program" << endl;
        cout << argv[0] << " /path/to/video" << endl;
        cout << argv[0] << " 0 #camera id" << endl;
        return -1;
    }
    // create the main application
    QApplication a(argc, argv);

    // register types which will be used by our signals
    qRegisterMetaType<std::string>("std::string");
    qRegisterMetaType<QImage>("QImage");

    // create the imageManager which will access to cv functions and peripherics/video files
    ImageManager *imgManager = new ImageManager(argv[1]);
    // create the application window
    Gui *window =new Gui();
    // display it
    window->show();

    // connect the image manager thread signal to hmi (to update images)
    QObject::connect(imgManager, SIGNAL(log(std::string , QImage )), window, SLOT(log(std::string , QImage )));

    // start the image processing thread
    imgManager->start();


    // start the app
    return a.exec();
}







//#include "moc_qtSample.cxx"
