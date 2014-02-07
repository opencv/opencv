#include "Gui.h"
#include "ui_Dialog.h"

#include "ImageManager.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Gui::Gui(QWidget *parent)
: QDialog(parent),
  ui(new Ui::Dialog)
{
  ui->setupUi(this);

}

// log an image
void Gui::log(std::string _name, QImage _image)
{
    QLabel * label;
    map<string,QLabel *>::iterator it;
    it = this->m_imgLogs.find(_name);

    // if the image name exists on our vector
    if(it != this->m_imgLogs.end())
    {
        // get the already existing label
        label = it->second;
    }
    else
    {
        // else, create  new label and append it to the vector, and to the widget
        label = new QLabel();
        this->m_imgLogs[_name] = label;
        this->ui->widgetDebugArea->layout()->addWidget(label);
    }
    // update the image to the label
    QImage resized = _image.scaled(IMG_WIDTH, IMG_HEIGHT,Qt::KeepAspectRatio);
    label->setPixmap(QPixmap::fromImage(resized));
}
