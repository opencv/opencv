#include <fstream>
#include "camera.hpp"

using namespace std;
using namespace cv;


CameraInfo CameraInfo::load(const string &path)
{
    FileStorage fs(path, FileStorage::READ);
    CV_Assert(fs.isOpened());

    CameraInfo cam;
    if (!fs["R"].isNone())
        fs["R"] >> cam.R;
    if (!fs["K"].isNone())
        fs["K"] >> cam.K;
    return cam;
}


void CameraInfo::save(const string &path)
{
    FileStorage fs(path, FileStorage::WRITE);
    CV_Assert(fs.isOpened());

    if (!R.empty())
        fs << "R" << R;
    if (!K.empty())
        fs << "K" << K;
}
