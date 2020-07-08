#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

struct BlendMode
{
    int id;
    string name;
    static vector<BlendMode> getModes()
    {
        vector<BlendMode> res;
#define ADD_MODE(ID) res.push_back({ID, #ID})
        ADD_MODE(BLEND_MODEL_DARKEN);
        ADD_MODE(BLEND_MODEL_MULTIPY);
        ADD_MODE(BLEND_MODEL_COLOR_BURN);
        ADD_MODE(BLEND_MODEL_LINEAR_BURN);
        ADD_MODE(BLEND_MODEL_LIGHTEN);
        ADD_MODE(BLEND_MODEL_SCREEN);
        ADD_MODE(BLEND_MODEL_COLOR_DODGE);
        ADD_MODE(BLEND_MODEL_LINEAR_DODGE);
        ADD_MODE(BLEND_MODEL_OVERLAY);
        ADD_MODE(BLEND_MODEL_SOFT_LIGHT);
        ADD_MODE(BLEND_MODEL_HARD_LIGHT);
        ADD_MODE(BLEND_MODEL_VIVID_LIGHT);
        ADD_MODE(BLEND_MODEL_LINEAR_LIGHT);
        ADD_MODE(BLEND_MODEL_PIN_LIGHT);
        ADD_MODE(BLEND_MODEL_DIFFERENCE);
        ADD_MODE(BLEND_MODEL_EXCLUSION);
        ADD_MODE(BLEND_MODEL_DIVIDE);
#undef ADD_MODE
        return res;
    }
};

int main()
{
    Mat target = imread(samples::findFile("lena.jpg"));
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    for (int i = 0; i < blend.rows; ++i)
        blend.row(i) = Scalar::all((double)i / blend.rows * 255);
    imshow("Gradient", blend);
    Mat result;
    vector<BlendMode> modes = BlendMode::getModes();
    for (vector<BlendMode>::const_iterator i = modes.begin(); i != modes.end(); ++i)
    {
        layerModelBlending(target, blend, result, i->id);
        putText(result, i->name, Point(5, 20), FONT_HERSHEY_PLAIN, 1., Scalar(0, 230, 0));
        imshow("Blend", result);
        const char key = waitKey(0);
        if (key == 'q' || key == 27)
            break;
    }
}
