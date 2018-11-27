#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "HOGfeatures.h"
#include "cascadeclassifier.h"

using namespace std;
using namespace cv;

CvHOGFeatureParams::CvHOGFeatureParams()
{
    maxCatCount = 0;
    name = HOGF_NAME;
    featSize = N_BINS * N_CELLS;
}

void CvHOGEvaluator::init(const CvFeatureParams *_featureParams, int _maxSampleCount, Size _winSize)
{
    CV_Assert( _maxSampleCount > 0);
    int cols = (_winSize.width + 1) * (_winSize.height + 1);
    for (int bin = 0; bin < N_BINS; bin++)
    {
        hist.push_back(Mat(_maxSampleCount, cols, CV_32FC1));
    }
    normSum.create( (int)_maxSampleCount, cols, CV_32FC1 );
    CvFeatureEvaluator::init( _featureParams, _maxSampleCount, _winSize );
}

void CvHOGEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
    CV_DbgAssert( !hist.empty());
    CvFeatureEvaluator::setImage( img, clsLabel, idx );
    vector<Mat> integralHist;
    for (int bin = 0; bin < N_BINS; bin++)
    {
        integralHist.push_back( Mat(winSize.height + 1, winSize.width + 1, hist[bin].type(), hist[bin].ptr<float>((int)idx)) );
    }
    Mat integralNorm(winSize.height + 1, winSize.width + 1, normSum.type(), normSum.ptr<float>((int)idx));
    integralHistogram(img, integralHist, integralNorm, (int)N_BINS);
}

//void CvHOGEvaluator::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
//{
//    _writeFeatures( features, fs, featureMap );
//}

void CvHOGEvaluator::writeFeatures( FileStorage &fs, const Mat& featureMap ) const
{
    int featIdx;
    int componentIdx;
    const Mat_<int>& featureMap_ = (const Mat_<int>&)featureMap;
    fs << FEATURES << "[";
    for ( int fi = 0; fi < featureMap.cols; fi++ )
        if ( featureMap_(0, fi) >= 0 )
        {
            fs << "{";
            featIdx = fi / getFeatureSize();
            componentIdx = fi % getFeatureSize();
            features[featIdx].write( fs, componentIdx );
            fs << "}";
        }
    fs << "]";
}

void CvHOGEvaluator::generateFeatures()
{
    int offset = winSize.width + 1;
    Size blockStep;
    int x, y, t, w, h;

    for (t = 8; t <= winSize.width/2; t+=8) //t = size of a cell. blocksize = 4*cellSize
    {
        blockStep = Size(4,4);
        w = 2*t; //width of a block
        h = 2*t; //height of a block
        for (x = 0; x <= winSize.width - w; x += blockStep.width)
        {
            for (y = 0; y <= winSize.height - h; y += blockStep.height)
            {
                features.push_back(Feature(offset, x, y, t, t));
            }
        }
        w = 2*t;
        h = 4*t;
        for (x = 0; x <= winSize.width - w; x += blockStep.width)
        {
            for (y = 0; y <= winSize.height - h; y += blockStep.height)
            {
                features.push_back(Feature(offset, x, y, t, 2*t));
            }
        }
        w = 4*t;
        h = 2*t;
        for (x = 0; x <= winSize.width - w; x += blockStep.width)
        {
            for (y = 0; y <= winSize.height - h; y += blockStep.height)
            {
                features.push_back(Feature(offset, x, y, 2*t, t));
            }
        }
    }

    numFeatures = (int)features.size();
}

CvHOGEvaluator::Feature::Feature()
{
    for (int i = 0; i < N_CELLS; i++)
    {
        rect[i] = Rect(0, 0, 0, 0);
    }
}

CvHOGEvaluator::Feature::Feature( int offset, int x, int y, int cellW, int cellH )
{
    rect[0] = Rect(x, y, cellW, cellH); //cell0
    rect[1] = Rect(x+cellW, y, cellW, cellH); //cell1
    rect[2] = Rect(x, y+cellH, cellW, cellH); //cell2
    rect[3] = Rect(x+cellW, y+cellH, cellW, cellH); //cell3

    for (int i = 0; i < N_CELLS; i++)
    {
        CV_SUM_OFFSETS(fastRect[i].p0, fastRect[i].p1, fastRect[i].p2, fastRect[i].p3, rect[i], offset);
    }
}

void CvHOGEvaluator::Feature::write(FileStorage &fs) const
{
    fs << CC_RECTS << "[";
    for( int i = 0; i < N_CELLS; i++ )
    {
        fs << "[:" << rect[i].x << rect[i].y << rect[i].width << rect[i].height << "]";
    }
    fs << "]";
}

//cell and bin idx writing
//void CvHOGEvaluator::Feature::write(FileStorage &fs, int varIdx) const
//{
//    int featComponent = varIdx % (N_CELLS * N_BINS);
//    int cellIdx = featComponent / N_BINS;
//    int binIdx = featComponent % N_BINS;
//
//    fs << CC_RECTS << "[:" << rect[cellIdx].x << rect[cellIdx].y <<
//        rect[cellIdx].width << rect[cellIdx].height << binIdx << "]";
//}

//cell[0] and featComponent idx writing. By cell[0] it's possible to recover all block
//All block is necessary for block normalization
void CvHOGEvaluator::Feature::write(FileStorage &fs, int featComponentIdx) const
{
    fs << CC_RECT << "[:" << rect[0].x << rect[0].y <<
        rect[0].width << rect[0].height << featComponentIdx << "]";
}


void CvHOGEvaluator::integralHistogram(const Mat &img, vector<Mat> &histogram, Mat &norm, int nbins) const
{
    CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );
    int x, y, binIdx;

    Size gradSize(img.size());
    Size histSize(histogram[0].size());
    Mat grad(gradSize, CV_32F);
    Mat qangle(gradSize, CV_8U);

    AutoBuffer<int> mapbuf(gradSize.width + gradSize.height + 4);
    int* xmap = mapbuf.data() + 1;
    int* ymap = xmap + gradSize.width + 2;

    const int borderType = (int)BORDER_REPLICATE;

    for( x = -1; x < gradSize.width + 1; x++ )
        xmap[x] = borderInterpolate(x, gradSize.width, borderType);
    for( y = -1; y < gradSize.height + 1; y++ )
        ymap[y] = borderInterpolate(y, gradSize.height, borderType);

    int width = gradSize.width;
    AutoBuffer<float> _dbuf(width*4);
    float* dbuf = _dbuf.data();
    Mat Dx(1, width, CV_32F, dbuf);
    Mat Dy(1, width, CV_32F, dbuf + width);
    Mat Mag(1, width, CV_32F, dbuf + width*2);
    Mat Angle(1, width, CV_32F, dbuf + width*3);

    float angleScale = (float)(nbins/CV_PI);

    for( y = 0; y < gradSize.height; y++ )
    {
        const uchar* currPtr = img.ptr(ymap[y]);
        const uchar* prevPtr = img.ptr(ymap[y-1]);
        const uchar* nextPtr = img.ptr(ymap[y+1]);
        float* gradPtr = grad.ptr<float>(y);
        uchar* qanglePtr = qangle.ptr(y);

        for( x = 0; x < width; x++ )
        {
            dbuf[x] = (float)(currPtr[xmap[x+1]] - currPtr[xmap[x-1]]);
            dbuf[width + x] = (float)(nextPtr[xmap[x]] - prevPtr[xmap[x]]);
        }
        cartToPolar( Dx, Dy, Mag, Angle, false );
        for( x = 0; x < width; x++ )
        {
            float mag = dbuf[x+width*2];
            float angle = dbuf[x+width*3];
            angle = angle*angleScale - 0.5f;
            int bidx = cvFloor(angle);
            angle -= bidx;
            if( bidx < 0 )
                bidx += nbins;
            else if( bidx >= nbins )
                bidx -= nbins;

            qanglePtr[x] = (uchar)bidx;
            gradPtr[x] = mag;
        }
    }
    integral(grad, norm, grad.depth());

    float* histBuf;
    const float* magBuf;
    const uchar* binsBuf;

    int binsStep = (int)( qangle.step / sizeof(uchar) );
    int histStep = (int)( histogram[0].step / sizeof(float) );
    int magStep = (int)( grad.step / sizeof(float) );
    for( binIdx = 0; binIdx < nbins; binIdx++ )
    {
        histBuf = histogram[binIdx].ptr<float>();
        magBuf = grad.ptr<float>();
        binsBuf = qangle.ptr();

        memset( histBuf, 0, histSize.width * sizeof(histBuf[0]) );
        histBuf += histStep + 1;
        for( y = 0; y < qangle.rows; y++ )
        {
            histBuf[-1] = 0.f;
            float strSum = 0.f;
            for( x = 0; x < qangle.cols; x++ )
            {
                if( binsBuf[x] == binIdx )
                    strSum += magBuf[x];
                histBuf[x] = histBuf[-histStep + x] + strSum;
            }
            histBuf += histStep;
            binsBuf += binsStep;
            magBuf += magStep;
        }
    }
}
