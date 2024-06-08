// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.detail.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

TrackerSamplerAlgorithm::~TrackerSamplerAlgorithm()
{
    // nothing
}

TrackerSamplerCSC::Params::Params()
{
    initInRad = 3;
    initMaxNegNum = 65;
    searchWinSize = 25;
    trackInPosRad = 4;
    trackMaxNegNum = 65;
    trackMaxPosNum = 100000;
}

TrackerSamplerCSC::TrackerSamplerCSC(const TrackerSamplerCSC::Params& parameters)
    : params(parameters)
{
    mode = MODE_INIT_POS;
    rng = theRNG();
}

TrackerSamplerCSC::~TrackerSamplerCSC()
{
    // nothing
}

bool TrackerSamplerCSC::sampling(const Mat& image, const Rect& boundingBox, std::vector<Mat>& sample)
{
    CV_Assert(!image.empty());

    float inrad = 0;
    float outrad = 0;
    int maxnum = 0;

    switch (mode)
    {
    case MODE_INIT_POS:
        inrad = params.initInRad;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad);
        break;
    case MODE_INIT_NEG:
        inrad = 2.0f * params.searchWinSize;
        outrad = 1.5f * params.initInRad;
        maxnum = params.initMaxNegNum;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad, outrad, maxnum);
        break;
    case MODE_TRACK_POS:
        inrad = params.trackInPosRad;
        outrad = 0;
        maxnum = params.trackMaxPosNum;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad, outrad, maxnum);
        break;
    case MODE_TRACK_NEG:
        inrad = 1.5f * params.searchWinSize;
        outrad = params.trackInPosRad + 5;
        maxnum = params.trackMaxNegNum;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad, outrad, maxnum);
        break;
    case MODE_DETECT:
        inrad = params.searchWinSize;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad);
        break;
    default:
        inrad = params.initInRad;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad);
        break;
    }
    return false;
}

void TrackerSamplerCSC::setMode(int samplingMode)
{
    mode = samplingMode;
}

std::vector<Mat> TrackerSamplerCSC::sampleImage(const Mat& img, int x, int y, int w, int h, float inrad, float outrad, int maxnum)
{
    int rowsz = img.rows - h - 1;
    int colsz = img.cols - w - 1;
    float inradsq = inrad * inrad;
    float outradsq = outrad * outrad;
    int dist;

    uint minrow = max(0, (int)y - (int)inrad);
    uint maxrow = min((int)rowsz - 1, (int)y + (int)inrad);
    uint mincol = max(0, (int)x - (int)inrad);
    uint maxcol = min((int)colsz - 1, (int)x + (int)inrad);

    //fprintf(stderr,"inrad=%f minrow=%d maxrow=%d mincol=%d maxcol=%d\n",inrad,minrow,maxrow,mincol,maxcol);

    std::vector<Mat> samples;
    samples.resize((maxrow - minrow + 1) * (maxcol - mincol + 1));
    int i = 0;

    float prob = ((float)(maxnum)) / samples.size();

    for (int r = minrow; r <= int(maxrow); r++)
        for (int c = mincol; c <= int(maxcol); c++)
        {
            dist = (y - r) * (y - r) + (x - c) * (x - c);
            if (float(rng.uniform(0.f, 1.f)) < prob && dist < inradsq && dist >= outradsq)
            {
                samples[i] = img(Rect(c, r, w, h));
                i++;
            }
        }

    samples.resize(min(i, maxnum));
    return samples;
}

}}}  // namespace cv::detail::tracking
