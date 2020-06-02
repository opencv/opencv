// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencv2/photo.hpp"

using namespace std;
using namespace cv;

#define EPSILON      1e-6f

#define SAFE_DIV_MIN EPSILON
#define SAFE_DIV_MAX (1.0f / SAFE_DIV_MIN)

#define	CLAMP(f,min,max)	((f)<(min)?(min):(f)>(max)?(max):(f))

namespace cv
{
/*  local function prototypes  */
static inline float	safe_div(float a, float b);
/* returns a / b, clamped to [-SAFE_DIV_MAX, SAFE_DIV_MAX].
 * if -SAFE_DIV_MIN <= a <= SAFE_DIV_MIN, returns 0.
 */
static inline float safe_div(float a, float b)
{
    float result = 0.0f;
    if (fabsf(a) > SAFE_DIV_MIN)
    {
        result = a / b;
        result = CLAMP(result, -SAFE_DIV_MAX, SAFE_DIV_MAX);
    }
    return result;
}
CV_EXPORTS_W void layerModelBlending(InputArray _target, InputArray _blend, OutputArray _dst, int flag)
{
    CV_Assert(!_target.empty());
    CV_Assert(!_blend.empty());
    CV_Assert(_target.type() == CV_32FC3 && _blend.type() == CV_32FC3 );
    CV_Assert(_target.size() == _blend.size());
    Mat target = _target.getMat();
    Mat blend = _blend.getMat();
    Size target_size = _target.size();
    _dst.create(target_size,target.type());
    Mat dst = _dst.getMat();
    int nr = target.rows;
    int nl = target.cols*target.channels();

    for (int k = 0; k < nr; k++)
    {
        const float* targetData = target.ptr<float>(k);
        const float* blendData = blend.ptr<float>(k);
        float* dstData = dst.ptr<float>(k);
        for (int i = 0; i < nl; i++)
        {
            switch (flag)
            {
                case BLEND_MODEL_DARKEN:
                dstData[i] = min(targetData[i], blendData[i]);
                break;
                case BLEND_MODEL_MULTIPY:
                dstData[i] = targetData[i] * blendData[i];
                break;
                case BLEND_MODEL_COLOR_BURN:
                dstData[i] = 1.0f - safe_div((1.0f - targetData[i]), blendData[i]);
                break;
                case BLEND_MODEL_LINEAR_BRUN:
                dstData[i] = targetData[i] + blendData[i] - 1.0f;
                break;
                case BLEND_MODEL_LIGHTEN:
                dstData[i] = max(targetData[i], blendData[i]);
                break;
                case BLEND_MODEL_SCREEN:
                dstData[i] = 1.0f - (1 - targetData[i])*(1.0f - blendData[i]);
                break;
                case BLEND_MODEL_COLOR_DODGE:
                dstData[i] = safe_div(targetData[i], 1.0f - blendData[i]);
                break;
                case BLEND_MODEL_LINEAR_DODGE:
                dstData[i] = targetData[i] + blendData[i];
                break;
                case BLEND_MODEL_OVERLAY:
                if (targetData[i] > 0.5f)
                    dstData[i] = 1.0f - (1.0f - 2.0f * (targetData[i] - 0.5f))*(1.0f - blendData[i]);
                else
                    dstData[i] = 2.0f * targetData[i] * blendData[i];
                case BLEND_MODEL_SOFT_LIGHT:
                if (targetData[i] > 0.5f)
                    dstData[i] = 1.0f - (1.0f - targetData[i]) * (1.0f - (blendData[i] - 0.5f));
                else
                    dstData[i] = targetData[i] * (blendData[i] + 0.5f);
                break;
                case BLEND_MODEL_HARD_LIGHT:
                if (targetData[i] > 0.5f)
                    dstData[i] = 1.0f - (1.0f - targetData[i])*(1.0f - 2.0f * blendData[i] - 0.5f);
                else
                    dstData[i] = 2.0f * targetData[i] * blendData[i];
                break;
                case BLEND_MODEL_VIVID_LIGHT:
                if (targetData[i] > 0.5f)
                    dstData[i] = 1.0f - safe_div((1.0f - targetData[i]), (2.0f * (blendData[i] - 0.5f)));
                else
                    dstData[i] = safe_div(targetData[i], (1.0f - 2.0f * blendData[i]));
                break;
                case BLEND_MODEL_LINEAR_LIGHT:
                if (targetData[i] > 0.5f)
                    dstData[i] = targetData[i] + (2.0f * (blendData[i] - 0.5f));
                else
                    dstData[i] = targetData[i] + 2.0f * blendData[i] - 1.0f;
                break;
                case BLEND_MODEL_PIN_LIGHT:
                if (targetData[i] > 0.5f)
                    dstData[i] = max(targetData[i], (float)(2.0f * (blendData[i] - 0.5f)));
                else
                    dstData[i] = min(targetData[i], (float)(2.0f * (blendData[i])));
                break;
                case BLEND_MODEL_DIFFERENCE:
                dstData[i] = abs(targetData[i] - blendData[i]);
                break;
                case BLEND_MODEL_EXCLUSION:
                dstData[i] = targetData[i] + blendData[i] - 2.0f * targetData[i] * blendData[i];
                break;
                case BLEND_MODEL_DIVIDE:
                dstData[i] = safe_div(targetData[i], blendData[i]);
                break;
            }
        }
    }
}
