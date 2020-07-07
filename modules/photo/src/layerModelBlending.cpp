// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/photo.hpp"

using namespace std;
using namespace cv;

//The following macro definitions refer to
//stackoverflow.com/questions/5919663/how-does-photoshop-blend-two-images-together
#define LayerModelBlend_Darken(A,B)     ((uchar)((B > A) ? A:B))
#define LayerModelBlend_Lighten(A,B)    ((uchar)((B > A) ? B:A))
#define LayerModelBlend_Average(A,B)    ((uchar)((A + B) / 2))
#define LayerModelBlend_Add(A,B)        ((uchar)(min(255, (A + B))))
#define LayerModelBlend_Subtract(A,B)   ((uchar)((A + B < 255) ? 0:(A + B - 255)))
#define LayerModelBlend_Difference(A,B) ((uchar)(abs(A - B)))
#define LayerModelBlend_Negation(A,B)   ((uchar)(255 - abs(255 - A - B)))
#define LayerModelBlend_Screen(A,B)     ((uchar)(255 - (((255 - A) * (255 - B)) >> 8)))
#define LayerModelBlend_Exclusion(A,B)  ((uchar)(A + B - 2 * A * B / 255))
#define LayerModelBlend_Overlay(A,B)    ((uchar)((B < 128) ? (2 * A * B / 255):(255 - 2 * (255 - A) * (255 - B) / 255)))
#define LayerModelBlend_SoftLight(A,B)  ((uchar)((B < 128)?(2*((A>>1)+64))*((float)B/255):(255-(2*(255-((A>>1)+64))*(float)(255-B)/255))))
#define LayerModelBlend_HardLight(A,B)  (LayerModelBlend_Overlay(B,A))
#define LayerModelBlend_ColorDodge(A,B) ((uchar)((B == 255) ? B:min(255, ((A << 8 ) / (255 - B)))))
#define LayerModelBlend_ColorBurn(A,B)  ((uchar)((B == 0) ? B:max(0, (255 - ((255 - A) << 8 ) / B))))
#define LayerModelBlend_LinearDodge(A,B)(LayerModelBlend_Add(A,B))
#define LayerModelBlend_LinearBurn(A,B) (LayerModelBlend_Subtract(A,B))
#define LayerModelBlend_LinearLight(A,B)((uchar)(B < 128)?LayerModelBlend_LinearBurn(A,(2 * B)):LayerModelBlend_LinearDodge(A,(2 * (B - 128))))
#define LayerModelBlend_VividLight(A,B) ((uchar)(B < 128)?LayerModelBlend_ColorBurn(A,(2 * B)):LayerModelBlend_ColorDodge(A,(2 * (B - 128))))
#define LayerModelBlend_PinLight(A,B)   ((uchar)(B < 128)?LayerModelBlend_Darken(A,(2 * B)):LayerModelBlend_Lighten(A,(2 * (B - 128))))
namespace cv
{
    CV_EXPORTS_W void layerModelBlending(InputArray _target, InputArray _blend, OutputArray _dst, int flag)
    {
        CV_Assert(!_target.empty());
        CV_Assert(!_blend.empty());
        CV_Assert(_target.type() == CV_8UC3 && _blend.type() == CV_8UC3);
        CV_Assert(_target.size() == _blend.size());
        Mat target = _target.getMat();
        Mat blend = _blend.getMat();
        Size target_size = _target.size();
        _dst.create(target_size, target.type());
        Mat dst = _dst.getMat();
        int nr = target.rows;
        int nl = target.cols*target.channels();
        switch (flag)
        {
        case BLEND_MODEL_DARKEN:
            dst = min(target, blend);
        break;
        case BLEND_MODEL_MULTIPY:
            multiply(target, blend, dst, 1.0 / 255);
        break;
        case BLEND_MODEL_COLOR_BURN:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_ColorBurn(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_LINEAR_BURN:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_LinearBurn(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_LIGHTEN:
            dst = max(target, blend);
        break;
        case BLEND_MODEL_SCREEN:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_Screen(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_COLOR_DODGE:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_ColorDodge(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_LINEAR_DODGE:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_LinearDodge(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_OVERLAY:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_Overlay(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_SOFT_LIGHT:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_SoftLight(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_HARD_LIGHT:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_HardLight(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_VIVID_LIGHT:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_VividLight(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_LINEAR_LIGHT:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_LinearLight(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_PIN_LIGHT:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_PinLight(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_DIFFERENCE:
            dst = abs(target - blend);
        break;
        case BLEND_MODEL_EXCLUSION:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = LayerModelBlend_Exclusion(targetData[i], blendData[i]);
            }
        break;
        case BLEND_MODEL_DIVIDE:
            for (int k = 0; k < nr; k++)
            {
                const uchar* targetData = target.ptr<uchar>(k);
                const uchar* blendData = blend.ptr<uchar>(k);
                uchar* dstData = dst.ptr<uchar>(k);
                for (int i = 0; i < nl; i++)
                    dstData[i] = (targetData[i] / blendData[i]) * 255;
            }
        break;
    }
  }
}
