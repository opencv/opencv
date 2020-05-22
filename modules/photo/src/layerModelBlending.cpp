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
    Mat target = _target.getMat();
    Mat blend = _blend.getMat();
    Mat dst = _dst.getMat();

    for (int index_row = 0; index_row < target.rows; index_row++)
        for (int index_col = 0; index_col < target.cols; index_col++)
            for (int index_c = 0; index_c < 3; index_c++)
                switch (flag)
                {
                case DARKEN:
                    dst.at<Vec3f>(index_row, index_col)[index_c] = min(
                        target.at<Vec3f>(index_row, index_col)[index_c],
                        blend.at<Vec3f>(index_row, index_col)[index_c]);
                    break;
                case MULTIPY:
                    dst.at<Vec3f>(index_row, index_col)[index_c] =
                        target.at<Vec3f>(index_row, index_col)[index_c] *
                        blend.at<Vec3f>(index_row, index_col)[index_c];
                    break;
                case COLOR_BURN:
                    dst.at<Vec3f>(index_row, index_col)[index_c] = 1 -
                        safe_div((1 - target.at<Vec3f>(index_row, index_col)[index_c]),
                            blend.at<Vec3f>(index_row, index_col)[index_c]);
                    break;
                case LINEAR_BRUN:
                    dst.at<Vec3f>(index_row, index_col)[index_c] =
                        target.at<Vec3f>(index_row, index_col)[index_c] +
                        blend.at<Vec3f>(index_row, index_col)[index_c] - 1;
                    break;
                case LIGHTEN:
                    dst.at<Vec3f>(index_row, index_col)[index_c] = max(
                        target.at<Vec3f>(index_row, index_col)[index_c],
                        blend.at<Vec3f>(index_row, index_col)[index_c]);
                    break;
                case SCREEN:
                    dst.at<Vec3f>(index_row, index_col)[index_c] = 1 -
                        (1 - target.at<Vec3f>(index_row, index_col)[index_c]) *
                        (1 - blend.at<Vec3f>(index_row, index_col)[index_c]);
                    break;
                case COLOR_DODGE:
                    dst.at<Vec3f>(index_row, index_col)[index_c] = safe_div
                    (target.at<Vec3f>(index_row, index_col)[index_c],
                        1 - blend.at<Vec3f>(index_row, index_col)[index_c]);
                    break;
                case LINEAR_DODGE:
                    dst.at<Vec3f>(index_row, index_col)[index_c] =
                        target.at<Vec3f>(index_row, index_col)[index_c] +
                        blend.at<Vec3f>(index_row, index_col)[index_c];
                    break;
                case OVERLAY:
                    if (target.at<Vec3f>(index_row, index_col)[index_c] > 0.5f)
                        dst.at<Vec3f>(index_row, index_col)[index_c] = 1 -
                        (1 - 2 * (target.at<Vec3f>(index_row, index_col)[index_c] - 0.5)) *
                        (1 - blend.at<Vec3f>(index_row, index_col)[index_c]);
                    else
                        dst.at<Vec3f>(index_row, index_col)[index_c] = 2 *
                        target.at<Vec3f>(index_row, index_col)[index_c] *
                        blend.at<Vec3f>(index_row, index_col)[index_c];
                    break;
                case SOFT_LIGHT:
                    if (target.at<Vec3f>(index_row, index_col)[index_c] > 0.5f)
                        dst.at<Vec3f>(index_row, index_col)[index_c] = 1 -
                        (1 - target.at<Vec3f>(index_row, index_col)[index_c]) *
                        (1 - (blend.at<Vec3f>(index_row, index_col)[index_c] - 0.5));
                    else
                        dst.at<Vec3f>(index_row, index_col)[index_c] =
                        target.at<Vec3f>(index_row, index_col)[index_c] *
                        (blend.at<Vec3f>(index_row, index_col)[index_c] + 0.5);
                    break;
                case HARD_LIGHT:
                    if (target.at<Vec3f>(index_row, index_col)[index_c] > 0.5f)
                        dst.at<Vec3f>(index_row, index_col)[index_c] = 1 -
                        (1 - target.at<Vec3f>(index_row, index_col)[index_c]) *
                        (1 - 2 * blend.at<Vec3f>(index_row, index_col)[index_c] - 0.5);
                    else
                        dst.at<Vec3f>(index_row, index_col)[index_c] =
                        target.at<Vec3f>(index_row, index_col)[index_c] *
                        (2 * blend.at<Vec3f>(index_row, index_col)[index_c]);
                    break;
                case VIVID_LIGHT:
                    if (target.at<Vec3f>(index_row, index_col)[index_c] > 0.5f)
                        dst.at<Vec3f>(index_row, index_col)[index_c] = 1 -
                        safe_div(1 - target.at<Vec3f>(index_row, index_col)[index_c],
                        (2 * (blend.at<Vec3f>(index_row, index_col)[index_c] - 0.5)));
                    else
                        dst.at<Vec3f>(index_row, index_col)[index_c] =
                        safe_div(target.at<Vec3f>(index_row, index_col)[index_c],
                        (1 - 2 * blend.at<Vec3f>(index_row, index_col)[index_c]));
                    break;
                case LINEAR_LIGHT:
                    if (target.at<Vec3f>(index_row, index_col)[index_c] > 0.5f)
                        dst.at<Vec3f>(index_row, index_col)[index_c] =
                        target.at<Vec3f>(index_row, index_col)[index_c] +
                        (2 * (blend.at<Vec3f>(index_row, index_col)[index_c] - 0.5));
                    else
                        dst.at<Vec3f>(index_row, index_col)[index_c] =
                        target.at<Vec3f>(index_row, index_col)[index_c] +
                        2 * blend.at<Vec3f>(index_row, index_col)[index_c] - 1;
                    break;
                case PIN_LIGHT:
                    if (target.at<Vec3f>(index_row, index_col)[index_c] > 0.5f)
                        dst.at<Vec3f>(index_row, index_col)[index_c] =
                        max(target.at<Vec3f>(index_row, index_col)[index_c],
                        (float)(2 * (blend.at<Vec3f>(index_row, index_col)[index_c] - 0.5)));
                    else
                        dst.at<Vec3f>(index_row, index_col)[index_c] =
                        min(target.at<Vec3f>(index_row, index_col)[index_c],
                            2 * blend.at<Vec3f>(index_row, index_col)[index_c]);
                    break;
                case DIFFERENCE:
                    dst.at<Vec3f>(index_row, index_col)[index_c] =
                        abs(target.at<Vec3f>(index_row, index_col)[index_c] -
                            blend.at<Vec3f>(index_row, index_col)[index_c]);
                    break;
                case EXCLUSION:
                    dst.at<Vec3f>(index_row, index_col)[index_c] =
                        target.at<Vec3f>(index_row, index_col)[index_c] +
                        blend.at<Vec3f>(index_row, index_col)[index_c] -
                        2 * target.at<Vec3f>(index_row, index_col)[index_c] * blend.at<Vec3f>(index_row, index_col)[index_c];
                    break;
                case DIVIDE:
                    dst.at<Vec3f>(index_row, index_col)[index_c] =
                        safe_div(target.at<Vec3f>(index_row, index_col)[index_c],
                            blend.at<Vec3f>(index_row, index_col)[index_c]);
                    break;
                }
}
}
