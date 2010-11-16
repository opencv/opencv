/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <iterator>

/****************************************************************************************\
      The code below is implementation of HOG (Histogram-of-Oriented Gradients)
      descriptor and object detection, introduced by Navneet Dalal and Bill Triggs.

      The computed feature vectors are compatible with the
      INRIA Object Detection and Localization Toolkit
      (http://pascal.inrialpes.fr/soft/olt/)
\****************************************************************************************/

namespace cv
{

size_t HOGDescriptor::getDescriptorSize() const
{
    CV_Assert(blockSize.width % cellSize.width == 0 &&
        blockSize.height % cellSize.height == 0);
    CV_Assert((winSize.width - blockSize.width) % blockStride.width == 0 &&
        (winSize.height - blockSize.height) % blockStride.height == 0 );
    return (size_t)nbins*
        (blockSize.width/cellSize.width)*
        (blockSize.height/cellSize.height)*
        ((winSize.width - blockSize.width)/blockStride.width + 1)*
        ((winSize.height - blockSize.height)/blockStride.height + 1);
}

double HOGDescriptor::getWinSigma() const
{
    return winSigma >= 0 ? winSigma : (blockSize.width + blockSize.height)/8.;
}

bool HOGDescriptor::checkDetectorSize() const
{
    size_t detectorSize = svmDetector.size(), descriptorSize = getDescriptorSize();
    return detectorSize == 0 ||
        detectorSize == descriptorSize ||
        detectorSize == descriptorSize + 1;
}

void HOGDescriptor::setSVMDetector(const vector<float>& _svmDetector)
{
    svmDetector = _svmDetector;
    CV_Assert( checkDetectorSize() );
}

#define CV_TYPE_NAME_HOG_DESCRIPTOR "opencv-object-detector-hog"

bool HOGDescriptor::read(FileNode& obj)
{
    if( !obj.isMap() )
        return false;
    FileNodeIterator it = obj["winSize"].begin();
    it >> winSize.width >> winSize.height;
    it = obj["blockSize"].begin();
    it >> blockSize.width >> blockSize.height;
    it = obj["blockStride"].begin();
    it >> blockStride.width >> blockStride.height;
    it = obj["cellSize"].begin();
    it >> cellSize.width >> cellSize.height;
    obj["nbins"] >> nbins;
    obj["derivAperture"] >> derivAperture;
    obj["winSigma"] >> winSigma;
    obj["histogramNormType"] >> histogramNormType;
    obj["L2HysThreshold"] >> L2HysThreshold;
    obj["gammaCorrection"] >> gammaCorrection;
    
    FileNode vecNode = obj["SVMDetector"];
    if( vecNode.isSeq() )
    {
        vecNode >> svmDetector;
        CV_Assert(checkDetectorSize());
    }
    return true;
}
    
void HOGDescriptor::write(FileStorage& fs, const String& objName) const
{
    if( !objName.empty() )
        fs << objName;
    
    fs << "{" CV_TYPE_NAME_HOG_DESCRIPTOR
    << "winSize" << winSize
    << "blockSize" << blockSize
    << "blockStride" << blockStride
    << "cellSize" << cellSize
    << "nbins" << nbins
    << "derivAperture" << derivAperture
    << "winSigma" << getWinSigma()
    << "histogramNormType" << histogramNormType
    << "L2HysThreshold" << L2HysThreshold
    << "gammaCorrection" << gammaCorrection;
    if( !svmDetector.empty() )
        fs << "SVMDetector" << "[:" << svmDetector << "]";
    fs << "}";
}
    
bool HOGDescriptor::load(const String& filename, const String& objname)
{
    FileStorage fs(filename, FileStorage::READ);
    FileNode obj = !objname.empty() ? fs[objname] : fs.getFirstTopLevelNode();
    return read(obj);
}

void HOGDescriptor::save(const String& filename, const String& objName) const
{
    FileStorage fs(filename, FileStorage::WRITE);
    write(fs, !objName.empty() ? objName : FileStorage::getDefaultObjectName(filename));
}

void HOGDescriptor::copyTo(HOGDescriptor& c) const
{
    c.winSize = winSize;
    c.blockSize = blockSize;
    c.blockStride = blockStride;
    c.cellSize = cellSize;
    c.nbins = nbins;
    c.derivAperture = derivAperture;
    c.winSigma = winSigma;
    c.histogramNormType = histogramNormType;
    c.L2HysThreshold = L2HysThreshold;
    c.gammaCorrection = gammaCorrection;
    c.svmDetector = svmDetector;
}
    
void HOGDescriptor::computeGradient(const Mat& img, Mat& grad, Mat& qangle,
                                    Size paddingTL, Size paddingBR) const
{
    CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );
    
    Size gradsize(img.cols + paddingTL.width + paddingBR.width,
                  img.rows + paddingTL.height + paddingBR.height);
    grad.create(gradsize, CV_32FC2);  // <magnitude*(1-alpha), magnitude*alpha>
    qangle.create(gradsize, CV_8UC2); // [0..nbins-1] - quantized gradient orientation
    Size wholeSize;
    Point roiofs;
    img.locateROI(wholeSize, roiofs);

    int i, x, y;
    int cn = img.channels();

    Mat_<float> _lut(1, 256);
    const float* lut = &_lut(0,0);

    if( gammaCorrection )
        for( i = 0; i < 256; i++ )
            _lut(0,i) = std::sqrt((float)i);
    else
        for( i = 0; i < 256; i++ )
            _lut(0,i) = (float)i;

    AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
    int* xmap = (int*)mapbuf + 1;
    int* ymap = xmap + gradsize.width + 2;

    const int borderType = (int)BORDER_REFLECT_101;

    for( x = -1; x < gradsize.width + 1; x++ )
        xmap[x] = borderInterpolate(x - paddingTL.width + roiofs.x,
                        wholeSize.width, borderType) - roiofs.x;
    for( y = -1; y < gradsize.height + 1; y++ )
        ymap[y] = borderInterpolate(y - paddingTL.height + roiofs.y,
                        wholeSize.height, borderType) - roiofs.y;

    // x- & y- derivatives for the whole row
    int width = gradsize.width;
    AutoBuffer<float> _dbuf(width*4);
    float* dbuf = _dbuf;
    Mat Dx(1, width, CV_32F, dbuf);
    Mat Dy(1, width, CV_32F, dbuf + width);
    Mat Mag(1, width, CV_32F, dbuf + width*2);
    Mat Angle(1, width, CV_32F, dbuf + width*3);

    int _nbins = nbins;
    float angleScale = (float)(_nbins/CV_PI);

    for( y = 0; y < gradsize.height; y++ )
    {
        const uchar* imgPtr = img.data + img.step*ymap[y];
        const uchar* prevPtr = img.data + img.step*ymap[y-1];
        const uchar* nextPtr = img.data + img.step*ymap[y+1];
        float* gradPtr = (float*)grad.ptr(y);
        uchar* qanglePtr = (uchar*)qangle.ptr(y);
        
        if( cn == 1 )
        {
            for( x = 0; x < width; x++ )
            {
                int x1 = xmap[x];
                dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
                dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
            }
        }
        else
        {
            for( x = 0; x < width; x++ )
            {
                int x1 = xmap[x]*3;
                const uchar* p2 = imgPtr + xmap[x+1]*3;
                const uchar* p0 = imgPtr + xmap[x-1]*3;
                float dx0, dy0, dx, dy, mag0, mag;

                dx0 = lut[p2[2]] - lut[p0[2]];
                dy0 = lut[nextPtr[x1+2]] - lut[prevPtr[x1+2]];
                mag0 = dx0*dx0 + dy0*dy0;
                
                dx = lut[p2[1]] - lut[p0[1]];
                dy = lut[nextPtr[x1+1]] - lut[prevPtr[x1+1]];
                mag = dx*dx + dy*dy;
                
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }
                
                dx = lut[p2[0]] - lut[p0[0]];
                dy = lut[nextPtr[x1]] - lut[prevPtr[x1]];
                mag = dx*dx + dy*dy;
                
                if( mag0 < mag )
                {
                    dx0 = dx;
                    dy0 = dy;
                    mag0 = mag;
                }

                dbuf[x] = dx0;
                dbuf[x+width] = dy0;
            }
        }
        
        cartToPolar( Dx, Dy, Mag, Angle, false );

        for( x = 0; x < width; x++ )
        {
            float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;
            int hidx = cvFloor(angle);
            angle -= hidx;
            if( hidx < 0 )
                hidx += _nbins;
            else if( hidx >= _nbins )
                hidx -= _nbins;
            assert( (unsigned)hidx < (unsigned)_nbins );

            qanglePtr[x*2] = (uchar)hidx;
            hidx++;
            hidx &= hidx < _nbins ? -1 : 0;
            qanglePtr[x*2+1] = (uchar)hidx;
            gradPtr[x*2] = mag*(1.f - angle);
            gradPtr[x*2+1] = mag*angle;
        }
    }
}


struct HOGCache
{
    struct BlockData
    {
        BlockData() : histOfs(0), imgOffset() {}
        int histOfs;
        Point imgOffset;
    };

    struct PixData
    {
        size_t gradOfs, qangleOfs;
        int histOfs[4];
        float histWeights[4];
        float gradWeight;
    };

    HOGCache();
    HOGCache(const HOGDescriptor* descriptor,
        const Mat& img, Size paddingTL, Size paddingBR,
        bool useCache, Size cacheStride);
    virtual ~HOGCache() {};
    virtual void init(const HOGDescriptor* descriptor,
        const Mat& img, Size paddingTL, Size paddingBR,
        bool useCache, Size cacheStride);

    Size windowsInImage(Size imageSize, Size winStride) const;
    Rect getWindow(Size imageSize, Size winStride, int idx) const;

    const float* getBlock(Point pt, float* buf);
    virtual void normalizeBlockHistogram(float* histogram) const;
    
    vector<PixData> pixData;
    vector<BlockData> blockData;

    bool useCache;
    vector<int> ymaxCached;
    Size winSize, cacheStride;
    Size nblocks, ncells;
    int blockHistogramSize;
    int count1, count2, count4;
    Point imgoffset;
    Mat_<float> blockCache;
    Mat_<uchar> blockCacheFlags;

    Mat grad, qangle;
    const HOGDescriptor* descriptor;
};


HOGCache::HOGCache()
{
    useCache = false;
    blockHistogramSize = count1 = count2 = count4 = 0;
    descriptor = 0;
}

HOGCache::HOGCache(const HOGDescriptor* _descriptor,
        const Mat& _img, Size _paddingTL, Size _paddingBR,
        bool _useCache, Size _cacheStride)
{
    init(_descriptor, _img, _paddingTL, _paddingBR, _useCache, _cacheStride);
}

void HOGCache::init(const HOGDescriptor* _descriptor,
        const Mat& _img, Size _paddingTL, Size _paddingBR,
        bool _useCache, Size _cacheStride)
{
    descriptor = _descriptor;
    cacheStride = _cacheStride;
    useCache = _useCache;

    descriptor->computeGradient(_img, grad, qangle, _paddingTL, _paddingBR);
    imgoffset = _paddingTL;

    winSize = descriptor->winSize;
    Size blockSize = descriptor->blockSize;
    Size blockStride = descriptor->blockStride;
    Size cellSize = descriptor->cellSize;
    Size winSize = descriptor->winSize;
    int i, j, nbins = descriptor->nbins;
    int rawBlockSize = blockSize.width*blockSize.height;

    nblocks = Size((winSize.width - blockSize.width)/blockStride.width + 1,
                   (winSize.height - blockSize.height)/blockStride.height + 1);
    ncells = Size(blockSize.width/cellSize.width, blockSize.height/cellSize.height);
    blockHistogramSize = ncells.width*ncells.height*nbins;

    if( useCache )
    {
        Size cacheSize((grad.cols - blockSize.width)/cacheStride.width+1,
                       (winSize.height/cacheStride.height)+1);
        blockCache.create(cacheSize.height, cacheSize.width*blockHistogramSize);
        blockCacheFlags.create(cacheSize);
        size_t i, cacheRows = blockCache.rows;
        ymaxCached.resize(cacheRows);
        for( i = 0; i < cacheRows; i++ )
            ymaxCached[i] = -1;
    }

    Mat_<float> weights(blockSize);
    float sigma = (float)descriptor->getWinSigma();
    float scale = 1.f/(sigma*sigma*2);

    for(i = 0; i < blockSize.height; i++)
        for(j = 0; j < blockSize.width; j++)
        {
            float di = i - blockSize.height*0.5f;
            float dj = j - blockSize.width*0.5f;
            weights(i,j) = std::exp(-(di*di + dj*dj)*scale);
        }

    blockData.resize(nblocks.width*nblocks.height);
    pixData.resize(rawBlockSize*3);

    // Initialize 2 lookup tables, pixData & blockData.
    // Here is why:
    //
    // The detection algorithm runs in 4 nested loops (at each pyramid layer):
    //  loop over the windows within the input image
    //    loop over the blocks within each window
    //      loop over the cells within each block
    //        loop over the pixels in each cell
    //
    // As each of the loops runs over a 2-dimensional array,
    // we could get 8(!) nested loops in total, which is very-very slow.
    //
    // To speed the things up, we do the following:
    //   1. loop over windows is unrolled in the HOGDescriptor::{compute|detect} methods;
    //         inside we compute the current search window using getWindow() method.
    //         Yes, it involves some overhead (function call + couple of divisions),
    //         but it's tiny in fact.
    //   2. loop over the blocks is also unrolled. Inside we use pre-computed blockData[j]
    //         to set up gradient and histogram pointers.
    //   3. loops over cells and pixels in each cell are merged
    //       (since there is no overlap between cells, each pixel in the block is processed once)
    //      and also unrolled. Inside we use PixData[k] to access the gradient values and
    //      update the histogram
    //
    count1 = count2 = count4 = 0;
    for( j = 0; j < blockSize.width; j++ )
        for( i = 0; i < blockSize.height; i++ )
        {
            PixData* data = 0;
            float cellX = (j+0.5f)/cellSize.width - 0.5f;
            float cellY = (i+0.5f)/cellSize.height - 0.5f;
            int icellX0 = cvFloor(cellX);
            int icellY0 = cvFloor(cellY);
            int icellX1 = icellX0 + 1, icellY1 = icellY0 + 1;
            cellX -= icellX0;
            cellY -= icellY0;
            
            if( (unsigned)icellX0 < (unsigned)ncells.width &&
                (unsigned)icellX1 < (unsigned)ncells.width )
            {
                if( (unsigned)icellY0 < (unsigned)ncells.height &&
                    (unsigned)icellY1 < (unsigned)ncells.height )
                {
                    data = &pixData[rawBlockSize*2 + (count4++)];
                    data->histOfs[0] = (icellX0*ncells.height + icellY0)*nbins;
                    data->histWeights[0] = (1.f - cellX)*(1.f - cellY);
                    data->histOfs[1] = (icellX1*ncells.height + icellY0)*nbins;
                    data->histWeights[1] = cellX*(1.f - cellY);
                    data->histOfs[2] = (icellX0*ncells.height + icellY1)*nbins;
                    data->histWeights[2] = (1.f - cellX)*cellY;
                    data->histOfs[3] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[3] = cellX*cellY;
                }
                else
                {
                    data = &pixData[rawBlockSize + (count2++)];
                    if( (unsigned)icellY0 < (unsigned)ncells.height )
                    {
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    data->histOfs[0] = (icellX0*ncells.height + icellY1)*nbins;
                    data->histWeights[0] = (1.f - cellX)*cellY;
                    data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[1] = cellX*cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            else
            {
                if( (unsigned)icellX0 < (unsigned)ncells.width )
                {
                    icellX1 = icellX0;
                    cellX = 1.f - cellX;
                }

                if( (unsigned)icellY0 < (unsigned)ncells.height &&
                    (unsigned)icellY1 < (unsigned)ncells.height )
                {
                    data = &pixData[rawBlockSize + (count2++)];
                    data->histOfs[0] = (icellX1*ncells.height + icellY0)*nbins;
                    data->histWeights[0] = cellX*(1.f - cellY);
                    data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[1] = cellX*cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
                else
                {
                    data = &pixData[count1++];
                    if( (unsigned)icellY0 < (unsigned)ncells.height )
                    {
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    data->histOfs[0] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[0] = cellX*cellY;
                    data->histOfs[1] = data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[1] = data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            data->gradOfs = (grad.cols*i + j)*2;
            data->qangleOfs = (qangle.cols*i + j)*2;
            data->gradWeight = weights(i,j);
        }

    assert( count1 + count2 + count4 == rawBlockSize );
    // defragment pixData
    for( j = 0; j < count2; j++ )
        pixData[j + count1] = pixData[j + rawBlockSize];
    for( j = 0; j < count4; j++ )
        pixData[j + count1 + count2] = pixData[j + rawBlockSize*2];
    count2 += count1;
    count4 += count2;

    // initialize blockData
    for( j = 0; j < nblocks.width; j++ )
        for( i = 0; i < nblocks.height; i++ )
        {
            BlockData& data = blockData[j*nblocks.height + i];
            data.histOfs = (j*nblocks.height + i)*blockHistogramSize;
            data.imgOffset = Point(j*blockStride.width,i*blockStride.height);
        }
}


const float* HOGCache::getBlock(Point pt, float* buf)
{
    float* blockHist = buf;
    assert(descriptor != 0);

    Size blockSize = descriptor->blockSize;
    pt += imgoffset;

    CV_Assert( (unsigned)pt.x <= (unsigned)(grad.cols - blockSize.width) &&
               (unsigned)pt.y <= (unsigned)(grad.rows - blockSize.height) );
    
    if( useCache )
    {
        CV_Assert( pt.x % cacheStride.width == 0 &&
                   pt.y % cacheStride.height == 0 );
        Point cacheIdx(pt.x/cacheStride.width,
                      (pt.y/cacheStride.height) % blockCache.rows);
        if( pt.y != ymaxCached[cacheIdx.y] )
        {
            Mat_<uchar> cacheRow = blockCacheFlags.row(cacheIdx.y);
            cacheRow = (uchar)0;
            ymaxCached[cacheIdx.y] = pt.y;
        }

        blockHist = &blockCache[cacheIdx.y][cacheIdx.x*blockHistogramSize];
        uchar& computedFlag = blockCacheFlags(cacheIdx.y, cacheIdx.x);
        if( computedFlag != 0 )
            return blockHist;
        computedFlag = (uchar)1; // set it at once, before actual computing
    }

    int k, C1 = count1, C2 = count2, C4 = count4;
    const float* gradPtr = (const float*)(grad.data + grad.step*pt.y) + pt.x*2;
    const uchar* qanglePtr = qangle.data + qangle.step*pt.y + pt.x*2;

    CV_Assert( blockHist != 0 );

    for( k = 0; k < blockHistogramSize; k++ )
        blockHist[k] = 0.f;
    
    const PixData* _pixData = &pixData[0];

    for( k = 0; k < C1; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w = pk.gradWeight*pk.histWeights[0];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];
        float* hist = blockHist + pk.histOfs[0];
        float t0 = hist[h0] + a[0]*w;
        float t1 = hist[h1] + a[1]*w;
        hist[h0] = t0; hist[h1] = t1;
    }

    for( ; k < C2; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];
        
        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight*pk.histWeights[0];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
        
        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight*pk.histWeights[1];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
    }

    for( ; k < C4; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];
        
        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight*pk.histWeights[0];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
        
        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight*pk.histWeights[1];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[2];
        w = pk.gradWeight*pk.histWeights[2];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[3];
        w = pk.gradWeight*pk.histWeights[3];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
    }

    normalizeBlockHistogram(blockHist);

    return blockHist;
}


void HOGCache::normalizeBlockHistogram(float* _hist) const
{
    float* hist = &_hist[0];
    size_t i, sz = blockHistogramSize;
    
    float sum = 0;
    for( i = 0; i < sz; i++ )
        sum += hist[i]*hist[i];
    float scale = 1.f/(std::sqrt(sum)+sz*0.1f), thresh = (float)descriptor->L2HysThreshold;
    for( i = 0, sum = 0; i < sz; i++ )
    {
        hist[i] = std::min(hist[i]*scale, thresh);
        sum += hist[i]*hist[i];
    }
    scale = 1.f/(std::sqrt(sum)+1e-3f);
    for( i = 0; i < sz; i++ )
        hist[i] *= scale;
}
    
    
Size HOGCache::windowsInImage(Size imageSize, Size winStride) const
{
    return Size((imageSize.width - winSize.width)/winStride.width + 1,
                (imageSize.height - winSize.height)/winStride.height + 1);
}

Rect HOGCache::getWindow(Size imageSize, Size winStride, int idx) const
{
    int nwindowsX = (imageSize.width - winSize.width)/winStride.width + 1;
    int y = idx / nwindowsX;
    int x = idx - nwindowsX*y;
    return Rect( x*winStride.width, y*winStride.height, winSize.width, winSize.height );
}


void HOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
                            Size winStride, Size padding,
                            const vector<Point>& locations) const
{
    if( winStride == Size() )
        winStride = cellSize;
    Size cacheStride(gcd(winStride.width, blockStride.width),
                     gcd(winStride.height, blockStride.height));
    size_t nwindows = locations.size();
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);
    
    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();
    descriptors.resize(dsize*nwindows);

    for( size_t i = 0; i < nwindows; i++ )
    {
        float* descriptor = &descriptors[i*dsize];
       
        Point pt0;
        if( !locations.empty() )
        {
            pt0 = locations[i];
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        else
        {
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }

        for( int j = 0; j < nblocks; j++ )
        {
            const HOGCache::BlockData& bj = blockData[j];
            Point pt = pt0 + bj.imgOffset;

            float* dst = descriptor + bj.histOfs;
            const float* src = cache.getBlock(pt, dst);
            if( src != dst )
                for( int k = 0; k < blockHistogramSize; k++ )
                    dst[k] = src[k];
        }
    }
}


void HOGDescriptor::detect(const Mat& img,
    vector<Point>& hits, double hitThreshold,
    Size winStride, Size padding, const vector<Point>& locations) const
{
    hits.clear();
    if( svmDetector.empty() )
        return;
    
    if( winStride == Size() )
        winStride = cellSize;
    Size cacheStride(gcd(winStride.width, blockStride.width),
                     gcd(winStride.height, blockStride.height));
    size_t nwindows = locations.size();
    padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
    padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
    Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);
    
    HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

    if( !nwindows )
        nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

    const HOGCache::BlockData* blockData = &cache.blockData[0];

    int nblocks = cache.nblocks.area();
    int blockHistogramSize = cache.blockHistogramSize;
    size_t dsize = getDescriptorSize();

    double rho = svmDetector.size() > dsize ? svmDetector[dsize] : 0;
    vector<float> blockHist(blockHistogramSize);

    for( size_t i = 0; i < nwindows; i++ )
    {
        Point pt0;
        if( !locations.empty() )
        {
            pt0 = locations[i];
            if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                continue;
        }
        else
        {
            pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
        }
        double s = rho;
        const float* svmVec = &svmDetector[0];
        int j, k;
        for( j = 0; j < nblocks; j++, svmVec += blockHistogramSize )
        {
            const HOGCache::BlockData& bj = blockData[j];
            Point pt = pt0 + bj.imgOffset;

            const float* vec = cache.getBlock(pt, &blockHist[0]);
            for( k = 0; k <= blockHistogramSize - 4; k += 4 )
                s += vec[k]*svmVec[k] + vec[k+1]*svmVec[k+1] +
                    vec[k+2]*svmVec[k+2] + vec[k+3]*svmVec[k+3];
            for( ; k < blockHistogramSize; k++ )
                s += vec[k]*svmVec[k];
        }
        if( s >= hitThreshold )
            hits.push_back(pt0);
    }
}

    
struct HOGInvoker
{
    HOGInvoker( const HOGDescriptor* _hog, const Mat& _img,
                double _hitThreshold, Size _winStride, Size _padding,
                const double* _levelScale, ConcurrentRectVector* _vec ) 
    {
        hog = _hog;
        img = _img;
        hitThreshold = _hitThreshold;
        winStride = _winStride;
        padding = _padding;
        levelScale = _levelScale;
        vec = _vec;
    }
    
    void operator()( const BlockedRange& range ) const
    {
        int i, i1 = range.begin(), i2 = range.end();
        double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1+1] : std::max(img.cols, img.rows);
        Size maxSz(cvCeil(img.cols/minScale), cvCeil(img.rows/minScale));
        Mat smallerImgBuf(maxSz, img.type());
        vector<Point> locations;
        
        for( i = i1; i < i2; i++ )
        {
            double scale = levelScale[i];
            Size sz(cvRound(img.cols/scale), cvRound(img.rows/scale));
            Mat smallerImg(sz, img.type(), smallerImgBuf.data);
            if( sz == img.size() )
                smallerImg = Mat(sz, img.type(), img.data, img.step);
            else
                resize(img, smallerImg, sz);
            hog->detect(smallerImg, locations, hitThreshold, winStride, padding);
            Size scaledWinSize = Size(cvRound(hog->winSize.width*scale), cvRound(hog->winSize.height*scale));
            for( size_t j = 0; j < locations.size(); j++ )
                vec->push_back(Rect(cvRound(locations[j].x*scale),
                                    cvRound(locations[j].y*scale),
                                    scaledWinSize.width, scaledWinSize.height));
        }
    }
    
    const HOGDescriptor* hog;
    Mat img;
    double hitThreshold;
    Size winStride;
    Size padding;
    const double* levelScale;
    ConcurrentRectVector* vec;
};


void HOGDescriptor::detectMultiScale(
    const Mat& img, vector<Rect>& foundLocations,
    double hitThreshold, Size winStride, Size padding,
    double scale0, int groupThreshold) const
{
    double scale = 1.;
    int levels = 0;

    vector<double> levelScale;
    for( levels = 0; levels < nlevels; levels++ )
    {
        levelScale.push_back(scale);
        if( cvRound(img.cols/scale) < winSize.width ||
            cvRound(img.rows/scale) < winSize.height ||
            scale0 <= 1 )
            break;
        scale *= scale0;
    }
    levels = std::max(levels, 1);
    levelScale.resize(levels);

    ConcurrentRectVector allCandidates;
    
    parallel_for(BlockedRange(0, (int)levelScale.size()),
                 HOGInvoker(this, img, hitThreshold, winStride, padding, &levelScale[0], &allCandidates));
    
    foundLocations.resize(allCandidates.size());
    std::copy(allCandidates.begin(), allCandidates.end(), foundLocations.begin());

    groupRectangles(foundLocations, groupThreshold, 0.2);
}

    
typedef RTTIImpl<HOGDescriptor> HOGRTTI;

CvType hog_type( CV_TYPE_NAME_HOG_DESCRIPTOR, HOGRTTI::isInstance,
                 HOGRTTI::release, HOGRTTI::read, HOGRTTI::write, HOGRTTI::clone);
    
vector<float> HOGDescriptor::getDefaultPeopleDetector()
{
    static const float detector[] = {
       0.05359386f, -0.14721455f, -0.05532170f, 0.05077307f,
       0.11547081f, -0.04268804f, 0.04635834f, -0.05468199f, 0.08232084f,
       0.10424068f, -0.02294518f, 0.01108519f, 0.01378693f, 0.11193510f,
       0.01268418f, 0.08528346f, -0.06309239f, 0.13054633f, 0.08100729f,
       -0.05209739f, -0.04315529f, 0.09341384f, 0.11035026f, -0.07596218f,
       -0.05517511f, -0.04465296f, 0.02947334f, 0.04555536f,
       -3.55954492e-003f, 0.07818956f, 0.07730991f, 0.07890715f, 0.06222893f,
       0.09001380f, -0.03574381f, 0.03414327f, 0.05677258f, -0.04773581f,
       0.03746637f, -0.03521175f, 0.06955440f, -0.03849038f, 0.01052293f,
       0.01736112f, 0.10867710f, 0.08748853f, 3.29739624e-003f, 0.10907028f,
       0.07913758f, 0.10393070f, 0.02091867f, 0.11594022f, 0.13182420f,
       0.09879354f, 0.05362710f, -0.06745391f, -7.01260753e-003f,
       5.24702156e-003f, 0.03236255f, 0.01407916f, 0.02207983f, 0.02537322f,
       0.04547948f, 0.07200756f, 0.03129894f, -0.06274468f, 0.02107014f,
       0.06035208f, 0.08636236f, 4.53164103e-003f, 0.02193363f, 0.02309801f,
       0.05568166f, -0.02645093f, 0.04448695f, 0.02837519f, 0.08975694f,
       0.04461516f, 0.08975355f, 0.07514391f, 0.02306982f, 0.10410084f,
       0.06368385f, 0.05943464f, 4.58420580e-003f, 0.05220337f, 0.06675851f,
       0.08358569f, 0.06712101f, 0.06559004f, -0.03930482f, -9.15936660e-003f,
       -0.05897915f, 0.02816453f, 0.05032348f, 0.06780671f, 0.03377650f,
       -6.09417039e-004f, -0.01795146f, -0.03083684f, -0.01302475f,
       -0.02972313f, 7.88706727e-003f, -0.03525961f, -2.50397739e-003f,
       0.05245084f, 0.11791293f, -0.02167498f, 0.05299332f, 0.06640524f,
       0.05190265f, -8.27316567e-003f, 0.03033127f, 0.05842173f,
       -4.01050318e-003f, -6.25105947e-003f, 0.05862958f, -0.02465461f,
       0.05546781f, -0.08228195f, -0.07234028f, 0.04640540f, -0.01308254f,
       -0.02506191f, 0.03100746f, -0.04665651f, -0.04591486f, 0.02949927f,
       0.06035462f, 0.02244646f, -0.01698639f, 0.01040041f, 0.01131170f,
       0.05419579f, -0.02130277f, -0.04321722f, -0.03665198f, 0.01126490f,
       -0.02606488f, -0.02228328f, -0.02255680f, -0.03427236f,
       -7.75165204e-003f, -0.06195229f, 8.21638294e-003f, 0.09535975f,
       -0.03709979f, -0.06942501f, 0.14579427f, -0.05448192f, -0.02055904f,
       0.05747357f, 0.02781788f, -0.07077577f, -0.05178314f, -0.10429011f,
       -0.11235505f, 0.07529039f, -0.07559302f, -0.08786739f, 0.02983843f,
       0.02667585f, 0.01382199f, -0.01797496f, -0.03141199f, -0.02098101f,
       0.09029204f, 0.04955018f, 0.13718739f, 0.11379953f, 1.80019124e-003f,
       -0.04577610f, -1.11108483e-003f, -0.09470536f, -0.11596080f,
       0.04489342f, 0.01784211f, 3.06850672e-003f, 0.10781866f,
       3.36498418e-003f, -0.10842580f, -0.07436839f, -0.10535070f,
       -0.01866805f, 0.16057891f, -5.07316366e-003f, -0.04295658f,
       -5.90488780e-003f, 8.82003549e-003f, -0.01492646f, -0.05029279f,
       -0.12875880f, 8.78831954e-004f, -0.01297184f, -0.07592774f,
       -0.02668831f, -6.93787413e-004f, 0.02406698f, -0.01773298f,
       -0.03855745f, -0.05877856f, 0.03259695f, 0.12826584f, 0.06292590f,
       -4.10733931e-003f, 0.10996531f, 0.01332991f, 0.02088735f, 0.04037504f,
       -0.05210760f, 0.07760046f, 0.06399347f, -0.05751930f, -0.10053057f,
       0.07505023f, -0.02139782f, 0.01796176f, 2.34400877e-003f, -0.04208319f,
       0.07355055f, 0.05093350f, -0.02996780f, -0.02219072f, 0.03355330f,
       0.04418742f, -0.05580705f, -0.05037573f, -0.04548179f, 0.01379514f,
       0.02150671f, -0.02194211f, -0.13682702f, 0.05464972f, 0.01608082f,
       0.05309116f, 0.04701022f, 1.33690401e-003f, 0.07575664f, 0.09625306f,
       8.92647635e-003f, -0.02819123f, 0.10866830f, -0.03439325f,
       -0.07092371f, -0.06004780f, -0.02712298f, -7.07467366e-003f,
       -0.01637020f, 0.01336790f, -0.10313606f, 0.04906582f, -0.05732445f,
       -0.02731079f, 0.01042235f, -0.08340668f, 0.03686501f, 0.06108340f,
       0.01322748f, -0.07809529f, 0.03774724f, -0.03413248f, -0.06096525f,
       -0.04212124f, -0.07982176f, -1.25973229e-003f, -0.03045501f,
       -0.01236493f, -0.06312395f, 0.04789570f, -0.04602066f, 0.08576570f,
       0.02521080f, 0.02988098f, 0.10314583f, 0.07060035f, 0.04520544f,
       -0.04426654f, 0.13146530f, 0.08386490f, 0.02164590f, -2.12280243e-003f,
       -0.03686353f, -0.02074944f, -0.03829959f, -0.01530596f, 0.02689708f,
       0.11867401f, -0.06043470f, -0.02785023f, -0.04775074f, 0.04878745f,
       0.06350956f, 0.03494788f, 0.01467400f, 1.17890188e-003f, 0.04379614f,
       2.03681854e-003f, -0.03958609f, -0.01072688f, 6.43705716e-003f,
       0.02996500f, -0.03418507f, -0.01960307f, -0.01219154f,
       -4.37000440e-003f, -0.02549453f, 0.02646318f, -0.01632513f,
       6.46516960e-003f, -0.01929734f, 4.78711911e-003f, 0.04962371f,
       0.03809111f, 0.07265724f, 0.05758125f, -0.03741554f, 0.01648608f,
       -8.45285598e-003f, 0.03996826f, -0.08185477f, 0.02638875f,
       -0.04026615f, -0.02744674f, -0.04071517f, 1.05096330e-003f,
       -0.04741232f, -0.06733172f, 8.70434940e-003f, -0.02192543f,
       1.35350740e-003f, -0.03056974f, -0.02975521f, -0.02887780f,
       -0.01210713f, -0.04828526f, -0.09066251f, -0.09969629f, -0.03665164f,
       -8.88111943e-004f, -0.06826669f, -0.01866150f, -0.03627640f,
       -0.01408288f, 0.01874239f, -0.02075835f, 0.09145175f, -0.03547291f,
       0.05396780f, 0.04198981f, 0.01301925f, -0.03384354f, -0.12201976f,
       0.06830920f, -0.03715654f, 9.55848210e-003f, 5.05685573e-003f,
       0.05659294f, 3.90764466e-003f, 0.02808490f, -0.05518097f, -0.03711621f,
       -0.02835565f, -0.04420464f, -0.01031947f, 0.01883466f,
       -8.49525444e-003f, -0.09419250f, -0.01269387f, -0.02133371f,
       -0.10190815f, -0.07844430f, 2.43644323e-003f, -4.09610150e-003f,
       0.01202551f, -0.06452291f, -0.10593818f, -0.02464746f, -0.02199699f,
       -0.07401930f, 0.07285886f, 8.87513801e-004f, 9.97662079e-003f,
       8.46779719e-003f, 0.03730333f, -0.02905126f, 0.03573337f, -0.04393689f,
       -0.12014472f, 0.03176554f, -2.76015815e-003f, 0.10824566f, 0.05090732f,
       -3.30179278e-003f, -0.05123822f, 5.04784798e-003f, -0.05664124f,
       -5.99415926e-003f, -0.05341901f, -0.01221393f, 0.01291318f,
       9.91760660e-003f, -7.56987557e-003f, -0.06193124f, -2.24549137e-003f,
       0.01987562f, -0.02018840f, -0.06975540f, -0.06601523f, -0.03349112f,
       -0.08910118f, -0.03371435f, -0.07406893f, -0.02248047f, -0.06159951f,
       2.77751544e-003f, -0.05723337f, -0.04792468f, 0.07518548f,
       2.77279224e-003f, 0.04211938f, 0.03100502f, 0.05278448f, 0.03954679f,
       -0.03006846f, -0.03851741f, -0.02792403f, -0.02875333f, 0.01531280f,
       0.02186953f, -0.01989829f, 2.50679464e-003f, -0.10258728f,
       -0.04785743f, -0.02887216f, 3.85063468e-003f, 0.01112236f,
       8.29218887e-003f, -0.04822981f, -0.04503597f, -0.03713100f,
       -0.06988008f, -0.11002295f, -2.69209221e-003f, 1.85383670e-003f,
       -0.05921049f, -0.06105053f, -0.08458050f, -0.04527602f,
       8.90329306e-004f, -0.05875023f, -2.68602883e-003f, -0.01591195f,
       0.03631859f, 0.05493166f, 0.07300330f, 5.53333294e-003f, 0.06400407f,
       0.01847740f, -5.76280477e-003f, -0.03210877f, 4.25160583e-003f,
       0.01166520f, -1.44864211e-003f, 0.02253744f, -0.03367080f, 0.06983195f,
       -4.22323542e-003f, -8.89401045e-003f, -0.07943393f, 0.05199728f,
       0.06065201f, 0.04133492f, 1.44032843e-003f, -0.09585235f, -0.03964731f,
       0.04232114f, 0.01750465f, -0.04487902f, -7.59733608e-003f, 0.02011171f,
       0.04673622f, 0.09011173f, -0.07869188f, -0.04682482f, -0.05080139f,
       -3.99383716e-003f, -0.05346331f, 0.01085723f, -0.03599333f,
       -0.07097908f, 0.03551549f, 0.02680387f, 0.03471529f, 0.01790393f,
       0.05471273f, 9.62048303e-003f, -0.03180215f, 0.05864431f, 0.02330614f,
       0.01633144f, -0.05616681f, -0.10245429f, -0.08302189f, 0.07291322f,
       -0.01972590f, -0.02619633f, -0.02485327f, -0.04627592f,
       1.48853404e-003f, 0.05514185f, -0.01270860f, -0.01948900f, 0.06373586f,
       0.05002292f, -0.03009798f, 8.76216311e-003f, -0.02474238f,
       -0.05504891f, 1.74034527e-003f, -0.03333667f, 0.01524987f, 0.11663762f,
       -1.32344989e-003f, -0.06608453f, 0.05687166f, -6.89525274e-004f,
       -0.04402352f, 0.09450210f, -0.04222684f, -0.05360983f, 0.01779531f,
       0.02561388f, -0.11075410f, -8.77790991e-003f, -0.01099504f,
       -0.10380266f, 0.03103457f, -0.02105741f, -0.07371717f, 0.05146710f,
       0.10581432f, -0.08617968f, -0.02892107f, 0.01092199f, 0.14551543f,
       -2.24320893e-003f, -0.05818033f, -0.07390742f, 0.05701261f,
       0.12937020f, -0.04986651f, 0.10182415f, 0.05028650f, 0.12515625f,
       0.09175041f, 0.06404983f, 0.01523394f, 0.09460562f, 0.06106631f,
       -0.14266998f, -0.02926703f, 0.02762171f, 0.02164151f,
       -9.58488265e-004f, -0.04231362f, -0.09866509f, 0.04322244f,
       0.05872034f, -0.04838847f, 0.06319253f, 0.02443798f, -0.03606876f,
       9.38737206e-003f, 0.04289991f, -0.01027411f, 0.08156885f, 0.08751175f,
       -0.13191354f, 8.16054735e-003f, -0.01452161f, 0.02952677f, 0.03615945f,
       -2.09128903e-003f, 0.02246693f, 0.09623287f, 0.09412123f, -0.02924758f,
       -0.07815186f, -0.02203079f, -2.02566991e-003f, 0.01094733f,
       -0.01442332f, 0.02838561f, 0.11882371f, 7.28798332e-003f, -0.10345965f,
       0.07561217f, -0.02049661f, 4.44177445e-003f, 0.01609347f, -0.04893158f,
       -0.08758243f, -7.67420698e-003f, 0.08862378f, 0.06098121f, 0.06565887f,
       7.32981879e-003f, 0.03558407f, -0.03874352f, -0.02490055f,
       -0.06771075f, 0.09939223f, -0.01066077f, 0.01382995f, -0.07289080f,
       7.47184316e-003f, 0.10621431f, -0.02878659f, 0.02383525f, -0.03274646f,
       0.02137008f, 0.03837290f, 0.02450992f, -0.04296818f, -0.02895143f,
       0.05327370f, 0.01499020f, 0.04998732f, 0.12938657f, 0.09391870f,
       0.04292390f, -0.03359194f, -0.06809492f, 0.01125796f, 0.17290455f,
       -0.03430733f, -0.06255233f, -0.01813114f, 0.11726857f, -0.06127599f,
       -0.08677909f, -0.03429872f, 0.04684938f, 0.08161420f, 0.03538774f,
       0.01833884f, 0.11321855f, 0.03261845f, -0.04826299f, 0.01752407f,
       -0.01796414f, -0.10464549f, -3.30041884e-003f, 2.29343961e-004f,
       0.01457292f, -0.02132982f, -0.02602923f, -9.87351313e-003f,
       0.04273872f, -0.02103316f, -0.07994065f, 0.02614958f, -0.02111666f,
       -0.06964913f, -0.13453490f, -0.06861878f, -6.09341264e-003f,
       0.08251446f, 0.15612499f, 2.46531400e-003f, 8.88424646e-003f,
       -0.04152999f, 0.02054853f, 0.05277953f, -0.03087788f, 0.02817579f,
       0.13939077f, 0.07641046f, -0.03627627f, -0.03015098f, -0.04041540f,
       -0.01360690f, -0.06227205f, -0.02738223f, 0.13577610f, 0.15235767f,
       -0.05392922f, -0.11175954f, 0.02157129f, 0.01146481f, -0.05264937f,
       -0.06595174f, -0.02749175f, 0.11812254f, 0.17404149f, -0.06137035f,
       -0.11003478f, -0.01351621f, -0.01745916f, -0.08577441f, -0.04469909f,
       -0.06106115f, 0.10559758f, 0.20806813f, -0.09174948f, 7.09621934e-004f,
       0.03579374f, 0.07215115f, 0.02221742f, 0.01827742f, -7.90785067e-003f,
       0.01489554f, 0.14519960f, -0.06425831f, 0.02990399f, -1.80181325e-003f,
       -0.01401528f, -0.04171134f, -3.70530109e-003f, -0.09090481f,
       0.09520713f, 0.08845516f, -0.02651753f, -0.03016730f, 0.02562448f,
       0.03563816f, -0.03817881f, 0.01433385f, 0.02256983f, 0.02872120f,
       0.01001934f, -0.06332260f, 0.04338406f, 0.07001807f, -0.04705722f,
       -0.07318907f, 0.02630457f, 0.03106382f, 0.06648342f, 0.10913180f,
       -0.01630815f, 0.02910308f, 0.02895109f, 0.08040254f, 0.06969310f,
       0.06797734f, 6.08639978e-003f, 4.16588830e-003f, 0.08926726f,
       -0.03123648f, 0.02700146f, 0.01168734f, -0.01631594f, 4.61015804e-003f,
       8.51359498e-003f, -0.03544224f, 0.03571994f, 4.29766066e-003f,
       -0.01970077f, -8.79793242e-003f, 0.09607988f, 0.01544222f,
       -0.03923707f, 0.07308586f, 0.06061262f, 1.31683104e-004f,
       -7.98222050e-003f, 0.02399261f, -0.06084389f, -0.02743429f,
       -0.05475523f, -0.04131311f, 0.03559756f, 0.03055342f, 0.02981433f,
       0.14860515f, 0.01766787f, 0.02945257f, 0.04898238f, 0.01026922f,
       0.02811658f, 0.08267091f, 0.02732154f, -0.01237693f, 0.11760156f,
       0.03802063f, -0.03309754f, 5.24957618e-003f, -0.02460510f, 0.02691451f,
       0.05399988f, -0.10133506f, 0.06385437f, -0.01818005f, 0.02259503f,
       0.03573135f, 0.01042848f, -0.04153402f, -0.04043029f, 0.01643575f,
       0.08326677f, 4.61383024e-004f, -0.05308095f, -0.08536223f,
       -1.61011645e-003f, -0.02163720f, -0.01783352f, 0.03859637f,
       0.08498885f, -0.01725216f, 0.08625131f, 0.10995087f, 0.09177644f,
       0.08498347f, 0.07646490f, 0.05580502f, 0.02693516f, 0.09996913f,
       0.09070327f, 0.06667200f, 0.05873008f, -0.02247842f, 0.07772321f,
       0.12408436f, 0.12629253f, -8.41997913e-004f, 0.01477783f, 0.09165990f,
       -2.98401713e-003f, -0.06466447f, -0.07057302f, 2.09516948e-004f,
       0.02210209f, -0.02158809f, -0.08602506f, -0.02284836f,
       4.01876355e-003f, 9.56660323e-003f, -0.02073978f, -0.04635138f,
       -7.59423291e-003f, -0.01377393f, -0.04559359f, -0.13284740f,
       -0.08671406f, -0.03654395f, 0.01142869f, 0.03287891f, -0.04392983f,
       0.06142959f, 0.17710890f, 0.10385257f, 0.01329137f, 0.10067633f,
       0.12450829f, -0.04476709f, 0.09049144f, 0.04589312f, 0.11167907f,
       0.08587538f, 0.04767583f, 1.67188141e-003f, 0.02359802f, -0.03808852f,
       0.03126272f, -0.01919029f, -0.05698918f, -0.02365112f, -0.06519032f,
       -0.05599358f, -0.07097308f, -0.03301812f, -0.04719102f, -0.02566297f,
       0.01324074f, -0.09230672f, -0.05518232f, -0.04712864f, -0.03380903f,
       -0.06719479f, 0.01183908f, -0.09326738f, 0.01642865f, 0.03789867f,
       -6.61567831e-003f, 0.07796386f, 0.07246574f, 0.04706347f, -0.02523437f,
       -0.01696830f, -0.08068866f, 0.06030888f, 0.10527060f, -0.06611756f,
       0.02977346f, 0.02621830f, 0.01913855f, -0.08479366f, -0.06322418f,
       -0.13570616f, -0.07644490f, 9.31900274e-003f, -0.08095149f,
       -0.10197903f, -0.05204025f, 0.01413151f, -0.07800411f, -0.01885122f,
       -0.07509381f, -0.10136326f, -0.05212355f, -0.09944065f,
       -1.33606605e-003f, -0.06342617f, -0.04178550f, -0.12373723f,
       -0.02832736f, -0.06057501f, 0.05830070f, 0.07604282f, -0.06462587f,
       8.02447461e-003f, 0.11580125f, 0.12332212f, 0.01978462f,
       -2.72378162e-003f, 0.05850752f, -0.04674481f, 0.05148062f,
       -2.62542837e-003f, 0.11253355f, 0.09893716f, 0.09785093f, -0.04659257f,
       -0.01102429f, -0.07002308f, 0.03088913f, -0.02565549f, -0.07671449f,
       3.17443861e-003f, -0.10783514f, -0.02314270f, -0.11089555f,
       -0.01024768f, 0.03116021f, -0.04964825f, 0.02281825f, 5.50005678e-003f,
       -0.08427856f, -0.14685495f, -0.07719755f, -0.13342668f, -0.04525511f,
       -0.09914210f, 0.02588859f, 0.03469279f, 0.04664020f, 0.11688190f,
       0.09647275f, 0.10857815f, -0.01448726f, 0.04299758f, -0.06763151f,
       1.33257592e-003f, 0.14331576f, 0.07574340f, 0.09166205f, 0.05674926f,
       0.11325553f, -0.01106494f, 0.02062161f, -0.11484840f, -0.07492137f,
       -0.02864293f, -0.01275638f, -0.06946032f, -0.10101652f, -0.04113498f,
       -0.02214783f, -0.01273942f, -0.07480393f, -0.10556041f, -0.07622112f,
       -0.09988393f, -0.11453961f, -0.12073903f, -0.09412795f, -0.07146588f,
       -0.04054537f, -0.06127083f, 0.04221122f, 0.07688113f, 0.04099256f,
       0.12663734f, 0.14683802f, 0.21761774f, 0.12525328f, 0.18431792f,
       -1.66402373e-003f, 2.37777247e-003f, 0.01445475f, 0.03509416f,
       0.02654697f, 0.01716739f, 0.05374011f, 0.02944174f, 0.11323927f,
       -0.01485456f, -0.01611330f, -1.85554172e-003f, -0.01708549f,
       -0.05435753f, -0.05302101f, 0.05260378f, -0.03582945f,
       -3.42867890e-004f, 1.36076682e-003f, -0.04436073f, -0.04228432f,
       0.03281291f, -0.05480836f, -0.10197772f, -0.07206279f, -0.10741059f,
       -0.02366946f, 0.10278475f, -2.74783419e-003f, -0.03242477f,
       0.02308955f, 0.02835869f, 0.10348799f, 0.19580358f, 0.10252027f,
       0.08039929f, 0.05525554f, -0.13250865f, -0.14395352f, 3.13586881e-003f,
       -0.03387071f, 8.94669443e-003f, 0.05406157f, -4.97324532e-003f,
       -0.01189114f, 2.82919413e-004f, -0.03901557f, -0.04898705f,
       0.02164520f, -0.01382906f, -0.01850416f, 0.01869347f, -0.02450060f,
       0.02291678f, 0.08196463f, 0.03309153f, -0.10629974f, 0.02473924f,
       0.05344394f, -0.02404823f, -0.03243643f, -5.55244600e-003f,
       -0.08009996f, 0.02811539f, 0.04235742f, 0.01859004f, 0.04902123f,
       -0.01438252f, -0.01526853f, 0.02044195f, -0.05008660f, 0.04244113f,
       0.07611816f, 0.04950470f, -0.06020549f, -4.26026015e-003f, 0.13133512f,
       -0.01438738f, -0.01958807f, -0.04044152f, -0.12425045f,
       2.84353318e-003f, -0.05042776f, -0.09121484f, 7.34345755e-003f,
       0.09388847f, 0.11800314f, 4.72295098e-003f, 4.44378285e-003f,
       -0.07984917f, -0.03613737f, 0.04490915f, -0.02246483f, 0.04681071f,
       0.05240871f, 0.02157206f, -0.04603431f, -0.01197929f, -0.02748779f,
       0.13621049f, 0.08812155f, -0.07802048f, 4.86458559e-003f, -0.01598836f,
       0.01024450f, -0.03463517f, -0.02304239f, -0.08692665f, 0.06655128f,
       0.05785803f, -0.12640759f, 0.02307472f, 0.07337402f, 0.07525434f,
       0.04943763f, -0.02241034f, -0.09978238f, 0.14487994f, -0.06570521f,
       -0.07855482f, 0.02830222f, -5.29603509e-004f, -0.04669895f,
       -0.11822784f, -0.12246452f, -0.15365660f, -0.02969127f, 0.08078201f,
       0.13512598f, 0.11505685f, 0.04740673f, 0.01376022f, -0.05852978f,
       -0.01537809f, -0.05541119f, 0.02491065f, -0.02870786f, 0.02760978f,
       0.23836176f, 0.22347429f, 0.10306466f, -0.06919070f, -0.10132039f,
       -0.20198342f, -0.05040560f, 0.27163076f, 0.36987007f, 0.34540465f,
       0.29095781f, 0.05649706f, 0.04125737f, 0.07505883f, -0.02737836f,
       -8.43431335e-003f, 0.07368195f, 0.01653876f, -0.09402955f,
       -0.09574359f, 0.01474337f, -0.07128561f, -0.03460737f, 0.11438941f,
       0.13752601f, -0.06385452f, -0.06310338f, 8.19548313e-003f, 0.11622470f,
       5.05133113e-003f, -0.07602754f, 0.06695660f, 0.25723928f, 0.09037900f,
       0.28826267f, 0.13165380f, -0.05312614f, -0.02137198f, -0.03442232f,
       -0.06255679f, 0.03899667f, 0.18391028f, 0.26016650f, 0.03374462f,
       0.01860465f, 0.19077586f, 0.18160543f, 3.43634398e-003f, -0.03036782f,
       0.19683038f, 0.35378191f, 0.24968483f, -0.03222649f, 0.28972381f,
       0.43091634f, 0.30778357f, 0.02335266f, -0.09877399f, -6.85245218e-003f,
       0.08945240f, -0.08150686f, 0.02792493f, 0.24806842f, 0.17338486f,
       0.06231801f, -0.10432383f, -0.16653322f, -0.13197899f, -0.08531576f,
       -0.19271527f, -0.13536365f, 0.22240199f, 0.39219588f, 0.26597717f,
       -0.01231649f, 0.01016179f, 0.13379875f, 0.12018334f, -0.04852953f,
       -0.07915270f, 0.07036012f, 3.87723115e-003f, -0.06126805f,
       -0.15015170f, -0.11406515f, -0.08556531f, -0.07429333f, -0.16115491f,
       0.13214062f, 0.25691369f, 0.05697750f, 0.06861912f, -6.02903729e-003f,
       -7.94562511e-003f, 0.04799571f, 0.06695165f, -0.01926842f, 0.06206308f,
       0.13450983f, -0.06381495f, -2.98370165e-003f, -0.03482971f,
       7.53991678e-003f, 0.03895611f, 0.11464261f, 0.01669971f,
       8.27818643e-003f, -7.49160210e-003f, -0.11712562f, -0.10650621f,
       -0.10353880f, -0.04994106f, -7.65618810e-004f, 0.03023767f,
       -0.04759270f, -0.07302686f, -0.05825012f, -0.13156348f, -0.10639747f,
       -0.19393684f, -0.09973683f, -0.07918908f, 4.63177625e-004f,
       -6.61382044e-004f, 0.15853868f, 0.08561199f, -0.07660093f,
       -0.08015265f, -0.06164073f, 0.01882577f, -7.29908410e-004f,
       0.06840892f, 0.03843764f, 0.20274927f, 0.22028814f, -5.26101235e-003f,
       0.01452435f, -0.06331623f, 0.02865064f, 0.05673740f, 0.12171564f,
       0.03837196f, 0.03555467f, -0.02662914f, -0.10280123f, -0.06526285f,
       -0.11066351f, -0.08988424f, -0.10103678f, 8.10526591e-003f,
       5.95238712e-003f, 0.02617721f, -0.01705742f, -0.10897956f,
       -0.08004991f, -0.11271993f, -0.06185647f, -0.06103712f, 0.01597041f,
       -0.05923606f, 0.09410726f, 0.22858568f, 0.03263380f, 0.06772990f,
       -0.09003516f, 0.01017870f, 0.01931688f, 0.08628357f, -0.01430009f,
       0.10954945f, 0.16612452f, -0.02434544f, -0.03310068f, -0.04236627f,
       0.01212392f, -6.15046406e-003f, 0.06954194f, 0.03015283f, 0.01787957f,
       0.02781667f, -0.05561153f, -8.96244217e-003f, -0.04971489f,
       0.07510284f, 0.01775282f, 0.05889897f, -0.07981427f, 0.03647643f,
       -3.73833324e-003f, -0.08894575f, -0.06429435f, -0.08068276f,
       0.03567704f, -0.07131936f, -7.21910037e-003f, -0.09566668f,
       0.17886090f, 0.14911725f, 0.02070032f, -0.05017120f, -0.04992622f,
       0.01570143f, -0.09906903f, 0.06456193f, 0.15329507f, 0.18820767f,
       0.11689861f, -0.01178513f, -0.02225163f, -0.01905318f, 0.10271224f,
       -7.27029052e-003f, 0.11664233f, 0.14796902f, 0.07771893f, 0.02400013f,
       -0.05361797f, -0.01972888f, 0.01376177f, 0.06740040f, -0.06525395f,
       0.05726178f, -0.02404981f, -0.14018567f, -0.02074987f, -0.04621970f,
       -0.04688627f, -0.01842059f, 0.07722727f, -0.04852883f, 0.01529004f,
       -0.19639495f, 0.10817073f, 0.03795860f, -0.09435206f, -0.07984378f,
       -0.03383440f, 0.11081333f, 0.02237366f, 0.12703256f, 0.21613893f,
       0.02918790f, 4.66472283e-003f, -0.10274266f, -0.04854131f,
       -3.46305710e-003f, 0.08652268f, 0.02251546f, 0.09636052f, 0.17180754f,
       -0.09272388f, 4.59174305e-004f, -0.11723048f, -0.12210111f,
       -0.15547538f, 0.07218186f, -0.05297846f, 0.03779940f, 0.05150875f,
       -0.03802310f, 0.03870645f, -0.15250699f, -0.08696499f, -0.02021560f,
       0.04118926f, -0.15177974f, 0.01577647f, 0.10249301f, 7.50041893e-003f,
       0.01721806f, -0.06828983f, -0.02397596f, -0.06598977f, -0.04317593f,
       -0.08064980f, 6.66632550e-003f, 0.03333484f, 0.07093620f, 0.08231064f,
       -0.06577903f, -0.06698844f, -0.06984019f, -0.06508023f, -0.14145090f,
       -0.02393239f, 0.06485303f, 8.83263443e-003f, 0.09251080f, -0.07557579f,
       -0.05067699f, -0.09798748f, -0.06703258f, -0.14056294f, 0.03245994f,
       0.12554143f, 0.01761621f, 0.12980327f, -0.04081950f, -0.11906909f,
       -0.14813015f, -0.08376863f, -0.12200681f, 0.04988137f, 0.05424247f,
       -3.90952639e-003f, 0.03255733f, -0.12717837f, -0.07461493f,
       -0.05703964f, -0.01736189f, -0.08026433f, -0.05433894f, -0.01719359f,
       0.02886275f, 0.01772653f, -0.09163518f, 3.57789593e-003f, -0.10129993f,
       -0.02653764f, -0.08131415f, -0.03847986f, -7.62157550e-004f,
       0.06486648f, 0.19675669f, -0.04919156f, -0.07059129f, -0.04857785f,
       -0.01042383f, -0.08328653f, 0.03660302f, -0.03696846f, 0.04969259f,
       0.08241162f, -0.12514858f, -0.06122676f, -0.03750202f,
       6.52989605e-003f, -0.10247213f, 0.02568346f, 4.51781414e-003f,
       -0.03734229f, -0.01131264f, -0.05412074f, 8.89345480e-004f,
       -0.12388977f, -0.05959237f, -0.12418608f, -0.06151643f, -0.07310260f,
       0.02441575f, 0.07023528f, -0.07548289f, -7.57147965e-004f,
       -0.09061348f, -0.08112976f, -0.06920306f, 9.54394229e-003f,
       -0.01219902f, 1.21273217e-003f, -8.88989680e-003f, -0.08309301f,
       -0.04552661f, -0.10739882f, -0.05691034f, -0.13928030f, 0.09027749f,
       0.15123098f, 0.03175976f, 0.17763577f, 3.29913251e-004f, 0.05151888f,
       -0.09844074f, -0.09475287f, -0.08571247f, 0.16241577f, 0.19336018f,
       8.57454538e-003f, 0.11474732f, -0.01493934f, 0.03352379f, -0.08966240f,
       -0.02322310f, 0.02663568f, 0.05448750f, -0.03536883f, -0.07210463f,
       -0.06807277f, -0.03121621f, -0.05932408f, -0.17282860f, -0.15873498f,
       -0.04956378f, 0.01603377f, -0.12385946f, 0.13878587f, 0.21468069f,
       0.13510075f, 0.20992437f, 0.08845878f, 0.08104013f, 0.03754176f,
       0.12173114f, 0.11103114f, 0.10643122f, 0.13941477f, 0.11640384f,
       0.14786847f, 0.01218238f, 0.01160753f, 0.03547940f, 0.08794311f,
       -0.01695384f, -0.07692261f, -0.08236158f, 6.79194089e-003f,
       -0.02458403f, 0.13022894f, 0.10953187f, 0.09857773f, 0.04735930f,
       -0.04353498f, -0.15173385f, -0.17904443f, -0.10450364f, -0.13418166f,
       -0.06633098f, -0.03170381f, -0.06839000f, -0.11350126f, -0.06983913f,
       0.19083543f, 0.17604128f, 0.07730632f, 0.10022651f, 0.36428109f,
       0.28291923f, 0.12688625f, 0.15942036f, 0.14064661f, -0.11201853f,
       -0.13969108f, -0.09088077f, -0.14107047f, 0.05117374f,
       -2.63348082e-003f, -0.10794610f, -0.09715455f, -0.05284977f,
       0.01565668f, 0.05031200f, 0.07021113f, -0.02963028f, 0.01766960f,
       0.08333644f, -0.03211382f, 4.90096770e-003f, 0.05186674f, -0.05045737f,
       -0.09624767f, -0.02525997f, 0.06916669f, 0.01213916f, 0.05333899f,
       -0.03443280f, -0.10055527f, -0.06291115f, 5.42851724e-003f,
       -6.30360236e-003f, 0.02270257f, -0.01769792f, 0.03273688f, 0.07746078f,
       7.77099328e-003f, 0.05041346f, 0.01648103f, -0.02321534f, -0.09930186f,
       -0.02293853f, 0.02034990f, -0.08324204f, 0.08510064f, -0.03732836f,
       -0.06465405f, -0.06086946f, 0.13680504f, -0.11469388f, -0.03896406f,
       -0.07142810f, 2.67581246e-003f, -0.03639632f, -0.09849060f,
       -0.11014334f, 0.17489147f, 0.17610909f, -0.16091567f, -0.07248894f,
       0.01567141f, 0.23742996f, 0.07552249f, -0.06270349f, -0.07303379f,
       0.25442186f, 0.16903116f, -0.08168741f, -0.05913896f, -0.03954096f,
       6.81776879e-003f, -0.05615319f, -0.07303037f, -0.12176382f,
       0.12385108f, 0.22084464f, -0.05543206f, -0.03310431f, 0.05731593f,
       0.19481890f, 0.04016430f, -0.06480758f, -0.12353460f, 0.18733442f,
       -0.09631214f, -0.11192076f, 0.12404587f, 0.15671748f, 0.19256128f,
       0.10895617f, 0.03391477f, -0.13032004f, -0.05626907f, -0.09025607f,
       0.23485197f, 0.27812332f, 0.26725492f, 0.07255980f, 0.16565137f,
       0.22388470f, 0.07441066f, -0.21003133f, -0.08075339f, -0.15031935f,
       0.07023834f, 0.10872041f, 0.18156518f, 0.20037253f, 0.13571967f,
       -0.11915682f, -0.11131983f, -0.18878011f, 0.06074620f, 0.20578890f,
       0.12413109f, 0.03930207f, 0.29176015f, 0.29502738f, 0.27856228f,
       -0.01803601f, 0.16646385f, 0.19268319f, 0.01900682f, 0.06026287f,
       2.35868432e-003f, 0.01558199f, 0.02707230f, 0.11383014f, 0.12103992f,
       0.03907350f, 0.04637353f, 0.09020995f, 0.11919726f, -3.63007211e-003f,
       0.02220155f, 0.10336831f, 0.17351882f, 0.12259731f, 0.18983354f,
       0.15736865f, 0.01160725f, -0.01690723f, -9.69582412e-004f, 0.07213813f,
       0.01161613f, 0.17864859f, 0.24486147f, 0.18208991f, 0.20177495f,
       0.05972528f, -8.93934630e-003f, -0.02316955f, 0.14436610f, 0.14114498f,
       0.05520950f, 0.06353590f, -0.19124921f, 0.10174713f, 0.29414919f,
       0.26448128f, 0.09344960f, 0.15284036f, 0.19797507f, 0.11369792f,
       -0.12722753f, -0.21396367f, -0.02008235f, -0.06566695f, -0.01662150f,
       -0.03937003f, 0.04778343f, 0.05017274f, -0.02299062f, -0.20208496f,
       -0.06395898f, 0.13721776f, 0.22544557f, 0.14888357f, 0.08687132f,
       0.27088094f, 0.32206613f, 0.09782200f, -0.18523243f, -0.17232181f,
       -0.01041531f, 0.04008654f, 0.04199702f, -0.08081299f, -0.03755421f,
       -0.04809646f, -0.05222081f, -0.21709201f, -0.06622940f, 0.02945281f,
       -0.04600435f, -0.05256077f, -0.08432942f, 0.02848100f, 0.03490564f,
       8.28621630e-003f, -0.11051246f, -0.11210597f, -0.01998289f,
       -0.05369405f, -0.08869293f, -0.18799506f, -0.05436598f, -0.05011634f,
       -0.05419716f, -0.06151857f, -0.10827805f, 0.04346735f, 0.04016083f,
       0.01520820f, -0.12173316f, -0.04880285f, -0.01101406f, 0.03250847f,
       -0.06009551f, -0.03082932f, -0.02295134f, -0.06856834f, -0.08775249f,
       -0.23793389f, -0.09174541f, -0.05538322f, -0.04321031f, -0.11874759f,
       -0.04221844f, -0.06070468f, 0.01194489f, 0.02608565f, -0.03892140f,
       -0.01643151f, -0.02602034f, -0.01305472f, 0.03920100f, -0.06514261f,
       0.01126918f, -6.27710763e-003f, -0.02720047f, -0.11133634f,
       0.03300330f, 0.02398472f, 0.04079665f, -0.10564448f, 0.05966159f,
       0.01195221f, -0.03179441f, -0.01692590f, -0.06177841f, 0.01841576f,
       -5.51078189e-003f, -0.06821765f, -0.03191888f, -0.09545476f,
       0.03030550f, -0.04896152f, -0.02914624f, -0.13283344f, -0.04783419f,
       6.07836898e-003f, -0.01449538f, -0.13358212f, -0.09687774f,
       -0.02813793f, 0.01213498f, 0.06650011f, -0.02039067f, 0.13356198f,
       0.05986415f, -9.12760664e-003f, -0.18780160f, -0.11992817f,
       -0.06342237f, 0.01229534f, 0.07143231f, 0.10713009f, 0.11085765f,
       0.06569190f, -0.02956399f, -0.16288325f, -0.13993549f, -0.01292515f,
       0.03833013f, 0.09130384f, -0.05086257f, 0.05617329f, -0.03896667f,
       -0.06282311f, -0.11490010f, -0.14264110f, -0.04530499f, 0.01598189f,
       0.09167797f, 0.08663294f, 0.04885277f, -0.05741219f, -0.07565769f,
       -0.17136464f, -0.02619422f, -0.02477579f, 0.02679587f, 0.11621952f,
       0.08788391f, 0.15520640f, 0.04709549f, 0.04504483f, -0.10214074f,
       -0.12293372f, -0.04820546f, -0.05484834f, 0.05473754f, 0.07346445f,
       0.05577277f, -0.08209965f, 0.03462975f, -0.20962234f, -0.09324598f,
       3.79481679e-003f, 0.03617633f, 0.16742408f, 0.07058107f, 0.10204960f,
       -0.06795346f, 3.22807301e-003f, -0.12589309f, -0.17496960f,
       0.02078314f, -0.07694324f, 0.12184640f, 0.08997164f, 0.04793497f,
       -0.11383379f, -0.08046359f, -0.25716835f, -0.08080962f,
       6.80711539e-003f, -0.02930280f, -3.04938294e-003f, -0.11106286f,
       -0.04628860f, -0.07821649f, 7.70127494e-003f, -0.10247706f,
       1.21042714e-003f, 0.20573859f, -0.03241005f, 8.42972286e-003f,
       0.01946464f, -0.01197973f, -0.14579976f, 0.04233614f,
       -4.14096704e-003f, -0.06866436f, -0.02431862f, -0.13529138f,
       1.25891645e-003f, -0.11425111f, -0.04303651f, -0.01694815f,
       0.05720210f, -0.16040207f, 0.02772896f, 0.05498345f, -0.15010567f,
       0.01450866f, 0.02350303f, -0.04301004f, -0.04951802f, 0.21702233f,
       -0.03159155f, -0.01963303f, 0.18232647f, -0.03263875f,
       -2.88476888e-003f, 0.01587562f, -1.94303901e-003f, -0.07789494f,
       0.04674156f, -6.25576358e-003f, 0.08925962f, 0.21353747f, 0.01254677f,
       -0.06999976f, -0.05931328f, -0.01884327f, -0.04306272f, 0.11794136f,
       0.03842728f, -0.03907030f, 0.05636114f, -0.09766009f, -0.02104000f,
       8.72711372e-003f, -0.02736877f, -0.05112274f, 0.16996814f, 0.02955785f,
       0.02094014f, 0.08414304f, -0.03335762f, -0.03617457f, -0.05808248f,
       -0.08872101f, 0.02927705f, 0.27077839f, 0.06075108f, 0.07478261f,
       0.15282831f, -0.03908454f, -0.05101782f, -9.51998029e-003f,
       -0.03272416f, -0.08735625f, 0.07633440f, -0.07185312f, 0.13841286f,
       0.07812646f, -0.12901451f, -0.05488589f, -0.05644578f, -0.03290703f,
       -0.11184757f, 0.03751570f, -0.05978153f, -0.09155276f, 0.05657315f,
       -0.04328186f, -0.03047933f, -0.01413135f, -0.10181040f, -0.01384013f,
       0.20132534f, -0.01536873f, -0.07641169f, 0.05906778f, -0.07833145f,
       -0.01523801f, -0.07502609f, -0.09461885f, -0.15013233f, 0.16050665f,
       0.09021381f, 0.08473236f, 0.03386267f, -0.09147339f, -0.09170618f,
       -0.08498498f, -0.05119187f, -0.10431040f, 0.01041618f, -0.03064913f,
       0.09340212f, 0.06448522f, -0.03881054f, -0.04985436f, -0.14794017f,
       -0.05200112f, -0.02144495f, 0.04000821f, 0.12420804f, -0.01851651f,
       -0.04116732f, -0.11951703f, -0.04879033f, -0.08722515f, -0.08454733f,
       -0.10549165f, 0.11251976f, 0.10766345f, 0.19201984f, 0.06128913f,
       -0.02734615f, -0.08834923f, -0.16999826f, -0.03548348f,
       -5.36092324e-003f, 0.08297954f, 0.07226378f, 0.04194529f, 0.04668673f,
       8.73902347e-003f, 0.06980139f, 0.05652480f, 0.05879445f, 0.02477076f,
       0.02451423f, 0.12433673f, 0.05600227f, 0.06886370f, 0.03863076f,
       0.07459056f, 0.02264139f, 0.01495469f, 0.06344220f, 0.06945208f,
       0.02931899f, 0.11719371f, 0.04527427f, 0.03248192f, 2.08271481e-003f,
       0.02044626f, 0.11403449f, 0.04303892f, 0.06444661f, 0.04959024f,
       0.08174094f, 0.09240247f, 0.04894639f, 0.02252937f, -0.01652530f,
       0.07587013f, 0.06064249f, 0.13954395f, 0.02772832f, 0.07093039f,
       0.08501238f, 0.01701301f, 0.09055722f, 0.33421436f, 0.20163782f,
       0.09821030f, 0.07951369f, 0.08695120f, -0.12757730f, -0.13865978f,
       -0.06610068f, -0.10985506f, 0.03406816f, -0.01116336f, -0.07281768f,
       -0.13525715f, -0.12844718f, 0.08956250f, 0.09171610f, 0.10092317f,
       0.23385370f, 0.34489515f, 0.09901748f, 0.02002922f, 0.12335990f,
       0.07606190f, -0.14899330f, -0.15634622f, -0.06494618f, -0.01760547f,
       0.03404277f, -0.13208845f, -0.12101169f, -0.18294574f, -0.16560709f,
       0.02183887f, -0.02752613f, 0.01813638f, 0.02000757f, 0.01319924f,
       0.08030242f, 0.01220535f, 2.98233377e-003f, -0.01307070f, 0.05970297f,
       -0.05345284f, -0.03381982f, -9.87543724e-003f, -0.06869387f,
       0.03956730f, -0.03108176f, -0.05732809f, 0.02172386f, 0.04159765f,
       2.62783933e-003f, 0.04813229f, 0.09358983f, -8.18389002e-003f,
       0.01724574f, -0.02547474f, -0.04967288f, -0.02390376f, 0.06640504f,
       -0.06306566f, 0.01137518f, 0.05589378f, -0.08237787f, 0.02455001f,
       -0.03059422f, -0.08953978f, 0.06851497f, 0.07190268f, -0.07610799f,
       7.87237938e-003f, -7.85830803e-003f, 0.06006952f, -0.01126728f,
       -2.85743061e-003f, -0.04772895f, 0.01884944f, 0.15005857f,
       -0.06268821f, -0.01989072f, 0.01138399f, 0.08760451f, 0.03879007f,
       -9.66926850e-003f, -0.08012961f, 0.06414555f, -0.01362950f,
       -0.09135523f, 0.01755159f, 0.04459474f, 0.09650917f, 0.05219948f,
       -2.19440833e-003f, -0.07037939f, -0.01599054f, 0.13103317f,
       -0.02492603f, -0.01032540f, -0.02903307f, 0.04489160f, 0.05148086f,
       0.01858173f, -0.02919228f, 0.08299296f, -0.04590359f, -0.15745632f,
       -0.09068198f, -0.02972453f, 0.12985018f, 0.22320485f, 0.24261914f,
       0.03642650f, -0.05506422f, 2.67413049e-003f, -0.03834032f, 0.06449424f,
       0.03834866f, 0.03816991f, 0.25039271f, 0.34212017f, 0.32433882f,
       0.18824573f, -0.08599839f, -0.17599408f, -0.15317015f, -0.09913155f,
       -0.02856072f, -0.05304699f, -1.06437842e-003f, -0.06641813f,
       -0.07509298f, 0.01463361f, -0.07551918f, -0.04510373f,
       -8.44620075e-003f, 0.01772176f, 0.04068235f, 0.20295307f, 0.15719447f,
       0.05712103f, 0.26296997f, 0.14657754f, 0.01547317f, -0.05052776f,
       -0.03881342f, -0.01437883f, -0.04930177f, 0.11719568f, 0.24098417f,
       0.26468599f, 0.31698579f, 0.10103608f, -0.01096375f, -0.01367013f,
       0.17104232f, 0.20065314f, 2.67622480e-003f, -0.01190034f, 0.18301608f,
       0.09459770f, -0.06357619f, -0.06473801f, 0.01377906f, -0.10032775f,
       -0.06388740f, 3.80393048e-003f, 0.06206078f, 0.10349120f, 0.26804337f,
       8.17918684e-003f, -0.02314351f, 9.34422202e-003f, 0.09198381f,
       0.03681326f, -8.77339672e-003f, -0.09662418f, -0.02715708f,
       0.13503517f, 0.08962728f, -6.57071499e-003f, -0.03201199f, 0.28510824f,
       0.32095715f, 0.18512695f, -0.14230858f, -0.14048551f, -0.07181299f,
       -0.08575408f, -0.08661680f, -0.17416079f, 7.54326640e-004f,
       0.05601677f, 0.13585392f, -0.04960437f, -0.07708392f, 0.10676333f,
       -0.04407546f, -0.07209078f, 0.03663663f, 0.28949317f, 0.41127121f,
       0.27431169f, -0.06900328f, -0.21474190f, -0.15578632f, -0.19555484f,
       -0.15209621f, -0.11269179f, 0.07416003f, 0.18991330f, 0.26858172f,
       0.01952259f, 0.01017922f, 0.02159843f, -4.95165400e-003f, -0.04368168f,
       -0.12721671f, -0.06673957f, -0.11275250f, 0.04413409f, 0.05578312f,
       0.03896771f, 0.03566417f, -0.05871816f, -0.07388090f, -0.17965563f,
       -0.08570268f, -0.15273231f, -0.06022318f, -0.06999847f,
       -6.81510568e-003f, 0.06294262f, -6.54901436e-004f, -0.01128654f,
       -0.02289657f, 0.04849290f, 0.04140804f, 0.23681939f, 0.14545733f,
       0.01989965f, 0.12032662f, 3.87463090e-003f, -6.02597650e-003f,
       -0.05919775f, -0.03067224f, -0.07787777f, 0.10834727f, 0.02153730f,
       0.02765649f, 0.03975543f, -0.12182906f, -0.04900113f, -0.09940100f,
       -0.06453611f, -0.13757215f, -0.03721382f, 0.02827376f, -0.04351249f,
       0.01907038f, -0.10284120f, -0.05671160f, -0.10760647f, -0.09624009f,
       -0.09565596f, -0.01303654f, 0.03080539f, 0.01416511f, 0.05846142f,
       -5.42971538e-003f, 0.06221476f, -0.03320325f, -0.06791797f,
       -0.05791342f, 0.12851369f, 0.14990346f, 0.03634374f, 0.14262885f,
       0.04330391f, 0.05032569f, -0.05631914f, 0.01606137f, 0.04387223f,
       0.22344995f, 0.15722635f, -0.04693628f, 0.03006579f, -2.52882647e-003f,
       0.05717621f, -0.07529724f, -0.02848588f, -0.06868757f,
       -4.51729307e-003f, 0.06466042f, -0.05935378f, -0.04704857f,
       -0.07363959f, 0.04843248f, -0.13421375f, -0.09789340f, -0.10255270f,
       0.03509852f, 0.04751543f, -0.03822323f, 0.09740467f, 0.04762916f,
       0.03940146f, -0.08283259f, 0.09552965f, 0.05038739f, 0.21258622f,
       0.09646992f, 0.03241193f, 0.05167701f, 0.04614570f, 0.04330090f,
       -0.02671840f, -0.06259909f, -0.02301898f, 0.18829170f, 0.10522786f,
       0.04313190f, 0.01670948f, -0.08421925f, 0.05911417f, -0.10582602f,
       -0.04855484f, -0.08373898f, 0.07775915f, 0.03723533f, -0.12047344f,
       4.86345543e-003f, -0.10520902f, 0.06571782f, -0.07528137f,
       -0.03245651f, -0.09869066f, -0.02917477f, -0.18293270f, 0.14810945f,
       9.24033765e-003f, -0.04354914f, 0.02266885f, -0.11872729f,
       -0.04016589f, 0.02830229f, 0.22539048f, 0.20565644f, 0.16701797f,
       0.09019924f, 0.01300652f, 0.09760600f, -0.03675831f, -0.01935448f,
       -0.06894835f, 0.08077277f, 0.19047537f, 0.11312226f, 0.04106043f,
       -0.11187182f, 0.04312806f, -0.18548580f, -0.11287174f, -0.08794551f,
       0.02078281f, -0.15295486f, 0.11806386f, -0.01103218f, -0.15971117f,
       0.02153538f, -0.05232147f, -0.10835317f, -0.13910367f, 0.05920752f,
       -0.10122602f, 0.20174250f, 0.09105796f, -0.01881348f, 0.09559010f,
       -0.03725745f, -0.09442931f, -0.09763174f, 0.05854454f, 0.08287182f,
       0.12919849f, 0.08594352f, -2.49806582e-003f, 0.02398440f,
       5.67950122e-003f, -0.06296340f, -0.12993270f, 0.03855852f, 0.05186560f,
       0.10839908f, -0.03380463f, -0.12654832f, -0.05399339f, -0.07456800f,
       -0.04736232f, -0.10164231f, 0.07496139f, 0.08125214f, 0.07656177f,
       -0.04999603f, -0.12823077f, -0.07692395f, -0.11317524f, -0.09118655f,
       -0.05695669f, 0.10477209f, 0.07468581f, 0.01630048f, -8.00961629e-003f,
       -0.06582128f, -0.04019095f, -0.04682907f, -0.01907842f, -0.10997720f,
       0.04911406f, 0.02931030f, 0.04197735f, -0.05773980f, -0.09670641f,
       -0.03594951f, -0.03402121f, -0.07149299f, -0.10566200f, 0.10601286f,
       0.06340689f, -0.01518632f, -5.96402306e-003f, -0.07628012f,
       -3.52779147e-003f, -0.02683854f, -0.10265494f, -0.02680815f,
       0.16338381f, 0.03103515f, 0.02296976f, 0.01624348f, -0.10831620f,
       -0.02314233f, -0.04789969f, -0.05530700f, -0.06461314f, 0.10494506f,
       0.04642856f, -0.07592955f, -0.06197905f, -0.09042154f, -0.01445521f,
       -0.04297818f, -0.11262015f, -0.11430512f, 0.03174541f, -0.03677487f,
       -0.02963996f, -0.06610169f, -0.13292049f, -0.07059067f, -0.08444111f,
       -0.02640536f, -0.07136250f, 0.04559967f, 0.01459980f, 0.17989251f,
       0.04435328f, -0.12464730f, -0.02871115f, -0.10752209f, -0.03393742f,
       -0.03791408f, 0.02548251f, 0.01956050f, 0.19245651f, 0.13963254f,
       -0.05904696f, -0.07424626f, -0.10411884f, 1.54176133e-003f,
       0.01797429f, 0.13025844f, 0.04547642f, -0.05710349f, -0.10697161f,
       -0.13489437f, -0.06515755f, -0.06406886f, -4.08572936e-003f,
       -0.01336483f, 0.04368737f, -0.11259720f, -0.05701635f, -0.06469971f,
       -0.08346602f, -0.04166770f, -0.05795543f, -0.08247511f, -0.05742628f,
       0.08452254f, -0.03350224f, 0.13980860f, 0.13252275f, 0.07589617f,
       0.07539988f, 0.12155797f, 0.19087289f, 0.15050751f, 0.21250245f,
       0.14206800f, 0.01298489f, 0.07450245f, 0.06559097f, 0.01700557f,
       0.04512971f, 0.16950700f, 0.10261577f, 0.16389982f, 0.05505059f,
       -0.03453077f, 0.08622462f, 0.07935954f, 0.03976260f, 0.02036091f,
       3.95744899e-003f, 0.03267065f, 0.15235919f, 0.01297494f, -0.08109194f,
       0.01407558f, 4.40693414e-003f, -0.15157418f, -0.11390478f,
       -0.07487597f, -7.81322457e-003f, -0.02749545f, -0.10181408f,
       0.13755716f, 0.14007211f, 0.13482562f, 0.27517235f, 0.34251109f,
       0.07639657f, 0.07268607f, 0.19823882f, 0.16135791f, -0.04186463f,
       -0.12784107f, -0.09846287f, 0.03169041f, 0.10974082f, -0.15051922f,
       -0.08916726f, -0.07138767f, -0.04153349f, 6.25418453e-003f,
       0.01266654f, 0.10533249f, 0.12749144f, 0.15148053f, 0.01498513f,
       0.06305949f, -0.01247123f, -0.08778401f, -0.08551880f, -0.11955146f,
       -0.08493572f, -0.02901620f, -0.02394859f, -0.13427313f, -0.11053200f,
       -0.14413260f, -0.15203285f, 0.03972760f, -3.72127310e-004f,
       -0.04200919f, 0.06105104f, 0.01904975f, -0.01106191f,
       -7.27445772e-003f, -0.01520341f, 1.10228511e-003f, -0.04949187f,
       -0.08013099f, 5.72071038e-003f, 0.08415454f, -0.06523152f, 0.03664081f,
       -0.02673042f, -0.12066154f, -0.03702074f, 0.06006580f, 0.01628682f,
       -6.17772620e-003f, 0.08192339f, -3.41629819e-003f, 0.02870512f,
       0.05807141f, 0.04959986f, 0.04618251f, -0.04901629f, -0.10579574f,
       0.02274442f, 0.12070961f, 2.23597488e-003f, 0.09831765f, -0.03019848f,
       -0.11181970f, -0.04961075f, 0.02498928f, -0.03714991f, -0.01619653f,
       0.02643486f, -7.62964319e-003f, -0.02882290f, -0.06242594f,
       -0.08439861f, 0.07220893f, 0.07263952f, 0.01561574f, 0.03091968f,
       0.01708712f, -0.03797151f, -3.18561122e-003f, 0.01624021f,
       -0.02828573f, 0.11284444f, -1.32280716e-003f, -0.07784860f,
       -0.07209100f, 0.03372242f, 0.12154529f, 0.02278104f, -0.05275500f,
       -0.01918484f, 0.12989293f, 0.05424401f, 0.02333086f, 0.04029022f,
       0.12392918f, 0.09495489f, 0.09190340f, 0.07935889f, 8.76816828e-003f,
       0.17148446f, -8.51302687e-003f, -0.08011249f, -0.06796283f,
       0.04884845f, 0.01112272f, -0.07835306f, -1.14811445e-003f,
       -0.03440760f, 0.02845243f, 0.07695542f, -0.07069533f, -0.01151784f,
       -8.53884313e-003f, -0.01662786f, -0.04163864f, 0.05400505f,
       0.02859163f, 0.02921852f, 0.05003135f, -6.85718050e-003f, -0.01632611f,
       0.07780217f, 0.04042810f, -0.01216440f, 3.60914599e-003f, -0.06322435f,
       0.09516726f, 0.12877031f, -9.69162490e-003f, 0.01031179f, 0.05180895f,
       -9.34659224e-003f, -0.01644533f, -0.04849347f, -0.04343236f,
       0.10514783f, 0.08046635f, -0.04615205f, -0.03975486f, -0.01485525f,
       0.13096830f, -0.01517950f, -0.06571898f, -0.04016372f, 0.01849786f,
       0.02439670f, 0.08067258f, 1.74824719e-003f, 0.07053747f, 0.08819518f,
       -5.08352555e-003f, -0.06550863f, -0.08266170f, -0.07780605f,
       0.01453450f, -0.08756890f, 0.01096501f, -8.71319138e-003f, 0.10110464f,
       0.02420769f, -0.06708383f, 0.02007811f, 5.93133038e-003f, 0.05398923f,
       0.07538138f, 0.02049227f, 0.02242589f, 0.04011070f, -1.44875818e-003f,
       -4.19115182e-003f, 0.06367654f, 0.02506934f, 0.02434536f, 0.05879405f,
       -8.22952855e-003f, -0.01242441f, 0.04224926f, -0.01754923f,
       0.05958161f, 0.03818886f, -0.01830363f, -0.04308917f, -0.04422197f,
       -0.02432721f, 0.02264866f, 2.03751423e-003f, 0.01197031f, 0.04439203f,
       0.12169247f, 0.03602713f, -0.02599251f, -1.98226492e-003f, 0.02046336f,
       -0.02639058f, -1.91242550e-003f, -0.09334669f, -0.03595153f,
       -9.88179818e-003f, -0.06848445f, -0.04666303f, -0.09955736f,
       -0.04206430f, 0.02609075f, 9.09005292e-003f, -0.07138551f,
       -4.22313227e-004f, 0.01766645f, 0.02756404f, 0.01308276f, 0.04052891f,
       0.02387515f, 0.05337298f, 0.02500631f, -0.04970853f, -0.12467445f,
       0.17604403f, 0.12256411f, -0.07512254f, 8.70451052e-003f, -0.05697548f,
       -0.03626474f, -8.76623299e-003f, -0.01210897f, -0.09451522f,
       0.07490732f, -0.02008001f, -0.02681278f, -0.06463405f, -0.01517507f,
       7.33757764e-003f, 6.07147906e-003f, -0.09316964f, -0.04575328f,
       0.13261597f, 0.15424870f, -0.01655918f, -0.02772390f, -0.05243644f,
       -0.02356456f, -0.02351753f, -0.10211615f, -0.12873036f, 0.14549787f,
       0.12519856f, 4.38762689e-003f, 0.02795992f, 0.05170322f, 0.09223596f,
       0.05890015f, 0.02376701f, -0.02777346f, 0.09506908f, 0.02328936f,
       -0.02319928f, -0.03218696f, -0.01527841f, -0.01016694f, -0.02674719f,
       0.05137179f, 0.01980666f, 0.06544447f, -0.01746171f, 0.01026380f,
       0.01561806f, 7.97004555e-004f, 0.07601810f, 0.01907250f, -0.03083035f,
       -0.05987392f, 0.09242783f, 0.14555025f, 0.01035827f, 0.03092401f,
       -0.09562709f, -0.03802354f, 0.02531144f, 0.03079449f, -0.07100715f,
       0.03330721f, -2.69116857e-003f, 0.03167490f, 0.05744999f, 0.03259895f,
       1.91266940e-003f, 0.03194578f, 0.07389776f, 0.02198060f, 0.07633314f,
       0.03293105f, -0.09103648f, 0.04718142f, 0.06102672f, -0.01003063f,
       5.85481385e-003f, -0.01522574f, 0.02323526f, 0.10584345f,
       4.35879454e-003f, 0.06107873f, 0.05868603f, -0.03115531f, 0.01214679f,
       0.08567052f, 3.93926632e-003f, -0.02521488f, -1.88425183e-003f,
       0.02038053f, -6.26854831e-004f, 0.04897438f, -0.04280585f,
       -0.04819689f, -0.04812867f, -0.01451186f, 0.05101469f,
       -9.01125465e-003f, -0.03333859f, 0.03917955f, 0.04196448f, 0.04292135f,
       0.02809529f, 0.02999715f, 0.04081348f, 9.10039060e-003f, 0.09703232f,
       0.10379741f, 0.02348725f, -4.72756615e-003f, 0.01027325f, 0.10402658f,
       0.12071823f, 0.09817299f, -0.02612033f, 0.03638414f, 0.05896405f,
       0.04865025f, 0.04793910f, -0.03882321f, -0.02962117f, -0.01222268f,
       0.04071597f, 0.01922777f, -0.02287866f, 0.03328381f, 0.01859092f,
       0.09024994f, 0.03804455f, -0.01424510f, 0.01953739f, 0.02509617f,
       -0.03390914f, -0.05663941f, -0.01641979f, 0.05848591f, 0.04639670f,
       0.02092116f, 0.12911791f, 0.19918139f, 0.07739855f, -7.25806039e-003f,
       0.04074838f, 0.03183993f, 1.39251316e-003f, -0.01428625f, 0.01865480f,
       0.08529541f, 0.13547510f, 0.11189661f, 0.03998901f, 0.09575938f,
       -0.02631102f, -0.03458253f, -0.04749985f, -0.06070716f,
       4.71884012e-003f, 0.06445789f, -0.02450038f, -0.05483776f,
       -0.04657237f, -0.02030717f, -0.03480766f, -0.09397731f, -0.06399718f,
       -0.01804585f, 5.62348310e-003f, -6.64811488e-003f, -0.06517869f,
       6.96210237e-003f, -0.01860148f, -0.04245830f, -0.05850367f,
       -3.24417115e-003f, 0.07700698f, 0.11290991f, 0.09923030f, -0.02970599f,
       0.05592411f, 0.04813979f, -0.09811195f, -0.09357996f, -0.03276114f,
       0.05218338f, 0.04141375f, 3.92977800e-003f, -0.05047480f, 0.15960084f,
       0.04612800f, -0.03114098f, -0.04650044f, -0.03249795f, -0.02425641f,
       -0.04311355f, 0.04307659f, -0.09401883f, -0.04742785f, -0.01254499f,
       -0.06598741f, 3.41369561e-003f, -0.05620445f, -7.28127593e-003f,
       -0.05998361f, -0.03274450f, -0.07376868f, 3.19015374e-003f,
       -0.07733069f, 0.05815864f, -0.02471071f, 0.03850617f, 0.13838784f,
       0.15399861f, 0.01731321f, -0.01477586f, 0.10393341f, 0.05159833f,
       -0.01945555f, -0.03427503f, -0.04867341f, 0.09237480f, 0.10732719f,
       0.06071450f, -0.01355071f, 0.01844356f, -0.03480803f, -0.03796671f,
       2.15628621e-004f, -0.05440186f, 0.01889855f, -0.01443413f,
       -0.02607902f, -0.02938001f, 0.02720689f, -0.06228397f, -0.02970936f,
       -0.03426210f, -0.10280876f, -0.06739304f, -0.05227850f, 0.03360292f,
       -0.11278441f, -0.06966180f, -0.13937433f, 9.10932291e-003f,
       2.52020749e-004f, -4.07359656e-003f, 0.12310639f, 0.09343060f,
       0.07302511f, 0.03222093f, 0.07532879f, 0.03792387f, -0.04985180f,
       0.01804602f, 0.02694195f, 0.13481498f, 0.04601225f, 0.04106982f,
       0.08511057f, 0.12314661f, 0.01320830f, 0.05044121f, -5.52943908e-003f,
       -0.08992624f, -0.02249301f, -0.08181777f, 0.06165213f, -0.03256603f,
       -0.01068920f, -0.01323473f, -0.11970232f, -0.04616347f, -0.12088681f,
       -0.06762606f, -0.08676834f, -0.06434575f, 0.01772529f, 0.03469615f,
       -0.10926618f, 0.03013873f, 0.14030397f, 0.16130108f, 0.17985588f,
       0.11281928f, 0.10530639f, 0.08905948f, 0.07733764f, 0.06695238f,
       0.02142088f, 0.06438877f, 0.09794453f, 0.05745072f, 0.02788557f,
       0.02632830f, 0.07985807f, 4.24902979e-003f, 8.47890321e-003f,
       -0.02679466f, -5.28812688e-003f, -0.02162580f, -0.07490715f,
       -0.08251337f, -0.02056576f, -0.01026194f, -1.15492963e-003f,
       -5.75720915e-004f, -0.07210591f, -0.07320981f, -0.04883312f,
       -0.10897151f, -0.07477258f, -0.08867134f, -0.09222437f, -0.10924666f,
       -0.10430276f, 0.07953499f, 0.02767959f, 0.11393359f, 0.18779543f,
       0.03313421f, 0.02143700f, 0.05852016f, -2.12067598e-003f,
       -3.76984011e-003f, 0.02774167f, -0.03124610f, 0.01465141f, 0.01616004f,
       -0.01391913f, -0.04404102f, -0.05444227f, -0.14684731f, -0.15016587f,
       0.04509468f, 1.29563001e-003f, 0.01398350f, 0.05610404f, -0.04868806f,
       -0.04776716f, -8.16873740e-003f, -2.30126386e-003f, -0.02286313f,
       0.11983398f, -0.04703261f, -0.08814441f, -0.07585249f, -0.10799607f,
       -0.03232087f, 0.01509786f, -0.04843464f, -0.03967846f, 0.09589416f,
       0.01352560f, -0.01458119f, 0.01050829f, -0.03038946f, 0.01608388f,
       1.11975556e-003f, -0.01250656f, 2.86211423e-003f, 0.04333691f,
       -0.14603497f, -0.01946543f, -0.02327525f, -0.01973944f, 0.07944400f,
       -0.02224544f, -0.06701808f, 0.03476532f, 0.11505594f, -0.02712801f,
       -0.01665113f, 0.06315716f, -0.08205860f, 0.07431999f, 0.04915778f,
       -0.04468752f, -0.01490402f, 0.07400476f, -0.11650901f, 0.05102430f,
       0.04559118f, -0.05916039f, 0.08840760f, -0.01587902f, -0.14890194f,
       0.07857784f, 0.04710254f, -0.05381983f, -0.07331945f, -0.03604643f,
       0.15611970f, 0.07649943f, -0.05959348f, -0.02776607f, 0.11098688f,
       0.03758875f, -0.04446875f, 0.04933187f, 0.01345535f, 0.06921103f,
       0.07364785f, 0.05518956f, 0.02899585f, 0.09375840f, 0.10518434f,
       -0.04420241f, 0.01915282f, -3.56386811e-003f, 0.14586878f, 0.10286101f,
       -0.04360626f, -0.12723237f, 0.09076386f, 0.11119842f, -0.06035013f,
       0.09674817f, 0.08938243f, 0.07065924f, 0.02603180f, 5.84815582e-003f,
       -0.05922065f, 0.12360309f, 3.59695964e-003f, 2.99844006e-003f,
       0.03697936f, 0.02043072f, 0.04168725f, 0.01025975f, -0.01359980f,
       -0.01600920f, 0.02581056f, 0.02329250f, 2.98100687e-003f, 0.01629762f,
       0.06652115f, 0.05855627f, 0.01237463f, -0.01297135f, 0.01761587f,
       0.05090865f, 0.06549342f, -0.04425945f, 2.43203156e-003f,
       3.07327788e-003f, 0.06678630f, -0.04303836f, 0.01082393f, -0.06476044f,
       0.04077786f, 0.12441979f, 0.08237778f, 0.07424165f, 0.04065890f,
       0.06905543f, 0.09556347f, 0.12724875f, -0.02132082f, 0.08514154f,
       -0.04175328f, -0.02666954f, 0.01897836f, 0.03317382f, 9.45465732e-003f,
       -0.01238974f, -0.04242500f, -0.01419479f, -0.03545213f, -0.02440874f,
       0.08684119f, 0.04212951f, 0.02462858f, -0.01104825f, -5.01706870e-003f,
       0.02968982f, 0.02597476f, -0.01568939f, 0.04514892f, 0.06974549f,
       0.08670278f, 0.06828108f, 0.10238872f, 0.05405957f, 0.06548470f,
       -0.03763957f, 0.01366090f, 0.07069602f, 0.05363748f, 0.04798120f,
       0.11706422f, 0.05466456f, -0.01869259f, 0.06344382f, 0.03106543f,
       0.08432506f, -0.02061096f, 0.03821088f, -6.92190882e-003f,
       6.40467042e-003f, -0.01271779f, 6.89014705e-005f, 0.04541415f,
       -0.01899539f, -0.05020239f, 0.03000903f, 0.01090422f, 4.52452758e-003f,
       0.02573632f, -0.02388454f, -0.04200457f, 1.72783900e-003f,
       -0.05978370f, -0.02720562f, 0.06573715f, 0.01154317f, 0.01265615f,
       0.07375994f, -9.19828378e-003f, -0.04914120f, 0.02124831f, 0.06455322f,
       0.04372910f, -0.03310043f, 0.03605788f, -6.78055827e-003f,
       9.36202332e-003f, 0.01747596f, -0.06406314f, -0.06812935f, 0.08080816f,
       -0.02778088f, 0.02735260f, 0.06393493f, 0.06652229f, 0.05676993f,
       0.08640018f, -7.59188086e-003f, -0.02012847f, -0.04741159f,
       -0.01657069f, -0.01624399f, 0.05547778f, -2.33309763e-003f,
       0.01120033f, 0.06141156f, -0.06285004f, -0.08732341f, -0.09313398f,
       -0.04267832f, 5.57443965e-003f, 0.04809862f, 0.01773641f,
       5.37361018e-003f, 0.14842421f, -0.06298012f, -0.02935147f, 0.11443478f,
       -0.05034208f, 5.65494271e-003f, 0.02076526f, -0.04577984f,
       -0.04735741f, 0.02961071f, -0.09307127f, -0.04417921f, -0.04990027f,
       -0.03940028f, 0.01306016f, 0.06267900f, 0.03758737f, 0.08460117f,
       0.13858789f, 0.04862388f, -0.06319809f, -0.05655516f, 0.01885816f,
       -0.03285607f, 0.03371567f, -0.07040928f, -0.04514049f, 0.01392166f,
       0.08184422f, -0.07230316f, 0.02386871f, 0.02184591f, 0.02605764f,
       -0.01033954f, 9.29878280e-003f, 7.67351175e-003f, 0.15189242f,
       0.02069071f, -0.09738296f, -0.08894105f, -0.07768748f, 0.02332268f,
       -0.01778995f, -0.03258888f, -0.08180822f, -0.08492987f, 0.02290156f,
       -0.11368170f, -0.03554465f, -0.04533844f, -0.02861580f, 0.06782424f,
       0.01113123f, 0.02453644f, 0.12721945f, 0.08084814f, -0.03607795f,
       0.01109122f, 0.04803548f, -0.03489929f, 0.03399536f, -0.05682014f,
       8.59533902e-003f, -4.27904585e-003f, 0.03230887f, -0.01300198f,
       -0.01038137f, -0.07930113f, 8.33097473e-003f, 0.02296994f,
       -0.01306500f, -0.01881626f, 0.04413369f, 0.05729880f, -0.03761553f,
       0.01942326f, 1.64540811e-003f, -0.03811319f, 0.04190650f, -0.14978096f,
       -0.04514487f, 0.01209545f, -5.46460645e-003f, -0.01647195f,
       7.63064111e-003f, -0.07494587f, 0.08415288f, 0.10020141f, -0.01228561f,
       0.06553826f, 0.04554005f, 0.07890417f, 0.03041138f, 0.01752007f,
       0.09208256f, -3.74419295e-004f, 0.10549527f, 0.04686913f, 0.01894833f,
       -0.02651412f, -4.34682379e-003f, 5.44942822e-003f, 0.01444484f,
       0.05882156f, -0.03336544f, 0.04603891f, -0.10432546f, 0.01923928f,
       0.01842845f, -0.01712168f, -0.02222766f, 0.04693324f, -0.06202956f,
       -0.01422159f, 0.08732220f, -0.07706107f, 0.02661049f, -0.04300238f,
       -0.03092422f, -0.03552184f, -0.01886088f, -0.04979934f, 0.03906401f,
       0.04608644f, 0.04966111f, 0.04275464f, -0.04621769f, -0.02653212f,
       8.57011229e-003f, 0.03839684f, 0.05818764f, 0.03880796f,
       -2.76100676e-004f, 0.03076511f, -0.03266929f, -0.05374557f,
       0.04986527f, -9.45429131e-003f, 0.03582499f, -2.64564669e-003f,
       -1.07461517e-003f, 0.02962313f, -0.01483363f, 0.03060869f, 0.02448327f,
       0.01845641f, 0.03282966f, -0.03534438f, -0.01084059f, -0.01119136f,
       -1.85360224e-003f, -5.94652840e-004f, -0.04451817f, 2.98327743e-003f,
       0.06272484f, -0.02152076f, -3.05971340e-003f, -0.05070828f,
       0.01531762f, 0.01282815f, 0.05167150f, 9.46266949e-003f,
       -3.34558333e-003f, 0.11442288f, -0.03906701f, -2.67325155e-003f,
       0.03069184f, -0.01134165f, 0.02949462f, 0.02879886f, 0.03855566f,
       -0.03450781f, 0.09142872f, -0.02156654f, 0.06075062f, -0.06220816f,
       0.01944680f, 6.68372354e-003f, -0.06656796f, 8.70784000e-003f,
       0.03456013f, 0.02434320f, -0.13236357f, -0.04177035f, -0.02069627f,
       0.01068112f, 0.01505432f, -0.07517391f, -3.83571628e-003f,
       -0.06298508f, -0.02881260f, -0.13101046f, -0.07221562f,
       -5.79945277e-003f, -8.57300125e-003f, 0.03782469f, 0.02762164f,
       0.04942456f, -0.02936396f, 0.09597211f, 0.01921411f, 0.06101191f,
       -0.04787507f, -0.01379578f, -7.40224449e-003f, -0.02220136f,
       -0.01313756f, 7.77558051e-003f, 0.12296968f, 0.02939998f, 0.03594062f,
       -0.07788624f, -0.01133144f, 3.99316690e-004f, -0.06090347f,
       -0.01122066f, -4.68682544e-003f, 0.07633100f, -0.06748922f,
       -0.05640298f, -0.05265681f, -0.01139122f, -0.01624347f, -0.04715714f,
       -0.01099092f, 0.01048561f, 3.28499987e-003f, -0.05810167f,
       -0.07699911f, -0.03330683f, 0.04185145f, 0.03478536f, 0.02275165f,
       0.02304766f, 6.66040834e-003f, 0.10968148f, -5.93013782e-003f,
       -0.04858336f, -0.04203213f, -0.09316786f, -6.13074889e-003f,
       -0.02544625f, 0.01366201f, 9.18555818e-003f, -0.01846578f,
       -0.05622401f, -0.03989377f, -0.07810296f, 6.91275718e-003f,
       0.05957597f, -0.03901334f, 0.01572002f, -0.01193903f,
       -6.89400872e-003f, -0.03093356f, -0.04136098f, -0.01562869f,
       -0.04604580f, 0.02865234f, -0.08678447f, -0.03232484f, -0.05364593f,
       -0.01445016f, -0.07003860f, -0.08669746f, -0.04520775f, 0.04274122f,
       0.03117515f, 0.08175703f, 0.01081109f, 0.06379741f, 0.06199206f,
       0.02865988f, 0.02360346f, 0.06725410f, -0.03248780f, -9.37702879e-003f,
       0.08265898f, -0.02245839f, 0.05125763f, -0.01862395f, 0.01973453f,
       -0.01994494f, -0.10770868f, 0.03180375f, 3.23935156e-003f,
       -0.02142080f, -0.04256190f, 0.04760900f, 0.04282863f, 0.05635953f,
       -0.01870849f, 0.05540622f, -0.03042666f, 0.01455277f, -0.06630179f,
       -0.05843807f, -0.03739681f, -0.09739155f, -0.03220233f, -0.05620182f,
       -0.10381401f, 0.07400211f, 4.20676917e-003f, 0.03258535f,
       2.14308966e-003f, 0.05121966f, -0.01274337f, 0.02384761f, 0.06335578f,
       -0.07905591f, 0.08375625f, -0.07898903f, -0.06508528f, -0.02498444f,
       0.06535810f, 0.03970535f, 0.04895468f, -0.01169566f, -0.03980601f,
       0.05682293f, 0.05925463f, -0.01165808f, -0.07936699f, -0.04208954f,
       0.01333987f, 0.09051196f, 0.10098671f, -0.03974256f, 0.01238771f,
       -0.07501741f, -0.03655440f, -0.04301528f, 0.09216860f,
       4.63579083e-004f, 0.02851115f, 0.02142735f, 1.28244064e-004f,
       0.02879687f, -0.08554889f, -0.04838862f, 0.08135369f, -0.05756533f,
       0.01413900f, 0.03451880f, -0.06619488f, -0.03053130f, 0.02961676f,
       -0.07384635f, 0.01135692f, 0.05283910f, -0.07778034f, -0.02107482f,
       -0.05511716f, -0.13473752f, 0.03030157f, 0.06722020f, -0.06218817f,
       -0.05826827f, 0.06254654f, 0.02895772f, -0.01664000f, -0.03620280f,
       -0.01612278f, -1.46097376e-003f, 0.14013411f, -8.96181818e-003f,
       -0.03250246f, 3.38630192e-003f, 2.64779478e-003f, 0.03359732f,
       -0.02411991f, -0.04229729f, 0.10666174f, -6.66579151f };
    return vector<float>(detector, detector + sizeof(detector)/sizeof(detector[0]));
}

}
