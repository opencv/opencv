#ifndef MAINDIALOG_H_
#define MAINDIALOG_H_

#include <QThread>
#include <QtGui/QDialog>
#include <QtGui>
#include <QImage>
#include <map>



namespace Ui {
    class Dialog;
}


#define IMG_WIDTH 400
#define IMG_HEIGHT 300

/**
* class managing the main window
*/
class Gui : public QDialog
{
    Q_OBJECT
private:
####Ui::Dialog *ui;
####// vector of already existing images
####// imageName => image
####std::map<std::string, QLabel *> m_imgLogs;
public:
####Gui(QWidget *parent = 0);
####
####public slots:
####  // log a new (or update an existing) image within the dialog
####  // this will create and append a new label to the dialog if the image name is unknown
####  // else, it will only update its image
####  void log(std::string _name, QImage _image);
};

#endif
