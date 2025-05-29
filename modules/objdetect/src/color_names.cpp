// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv {

class ColorNamesImpl : public ColorNamesFeatures
{
public:
    ColorNamesImpl();
    bool read(const std::string & table_file);
    void compute(InputArray image_patch, OutputArray feature_vector) CV_OVERRIDE;
private:
    Mat table;
    const int LEN = 32768;
};

ColorNamesImpl::ColorNamesImpl()
{
}

bool ColorNamesImpl::read(const std::string & table_file)
{
    FileStorage fs(table_file, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    FileNode node = fs["ColorNames"];
    if (node.empty())
        return false;
    table = node.mat();
    if ((table.size() != Size(10, LEN)) || (table.type() != CV_32FC1))
    {
        table.release();
        return false;
    }
    return true;
}

void ColorNamesImpl::compute(InputArray image_patch, OutputArray feature_vector)
{
    CV_CheckType(image_patch.type(), image_patch.type() == CV_8UC3, "BGR image expected");
    Mat image_patch_rgb555;
    cvtColor(image_patch, image_patch_rgb555, COLOR_RGB2BGR555);
    // TODO: why RGB555 is 8UC2?
    Mat patch(image_patch.size(), CV_16UC1, image_patch_rgb555.data);
    CV_CheckType(patch.type(), patch.type() == CV_16UC1, "Internal error");

    feature_vector.create(image_patch.size(), CV_32FC(10));
    Mat feature_mat = feature_vector.getMat();

    Mat ftable(Size(1, LEN), CV_32FC(10), table.data);
    for (int i = 0; i < patch.rows; i++)
    {
        for (int j = 0; j < patch.cols; j++)
        {
            typedef uint16_t Pix;
            typedef Vec<float, 10> Feat;
            const Pix pix = patch.at<Pix>(i, j);
            const Feat val = ftable.at<Feat>(pix);
            feature_mat.at<Feat>(i, j) = val;
        }
    }
}

Ptr<ColorNamesFeatures> ColorNamesFeatures::create(const std::string & table_file)
{
    cv::Ptr<ColorNamesImpl> res = std::make_shared<ColorNamesImpl>();
    if (!res->read(table_file))
        return NULL;
    return res;
}

} // cv::
