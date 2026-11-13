/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#include "opencv2/imgproc/detail/gcgraph.hpp"
#include <limits>

using namespace cv;
using namespace detail;

namespace {

/*
This is implementation of image segmentation algorithm GrabCut described in
"GrabCut - Interactive Foreground Extraction using Iterated Graph Cuts".
Carsten Rother, Vladimir Kolmogorov, Andrew Blake.
 */

/*
 GMM - Gaussian Mixture Model
*/
class GMM
{
public:
    static const int componentsCount = 5;

    GMM( Mat& _model );
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int whichComponent( const Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();

private:
    void calcInverseCovAndDeterm(int ci, double singularFix);
    Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

GMM::GMM( Mat& _model )
{
    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( cv::Error::StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

    model = _model;

    coefs = model.ptr<double>(0);
    mean = coefs + componentsCount;
    cov = mean + 3*componentsCount;

    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
             calcInverseCovAndDeterm(ci, 0.0);
    totalSampleCount = 0;
}

double GMM::operator()( const Vec3d color ) const
{
    double res = 0;
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}

double GMM::operator()( int ci, const Vec3d color ) const
{
    double res = 0;
    if( coefs[ci] > 0 )
    {
        CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
        Vec3d diff = color;
        double* m = mean + 3*ci;
        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
                   + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
                   + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}

int GMM::whichComponent( const Vec3d color ) const
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}

void GMM::initLearning()
{
    for( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

void GMM::addSample( int ci, const Vec3d color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

void GMM::endLearning()
{
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
            CV_Assert(totalSampleCount > 0);
            double inv_n = 1.0 / n;
            coefs[ci] = (double)n/totalSampleCount;

            double* m = mean + 3*ci;
            m[0] = sums[ci][0] * inv_n; m[1] = sums[ci][1] * inv_n; m[2] = sums[ci][2] * inv_n;

            double* c = cov + 9*ci;
            c[0] = prods[ci][0][0] * inv_n - m[0]*m[0]; c[1] = prods[ci][0][1] * inv_n - m[0]*m[1]; c[2] = prods[ci][0][2] * inv_n - m[0]*m[2];
            c[3] = prods[ci][1][0] * inv_n - m[1]*m[0]; c[4] = prods[ci][1][1] * inv_n - m[1]*m[1]; c[5] = prods[ci][1][2] * inv_n - m[1]*m[2];
            c[6] = prods[ci][2][0] * inv_n - m[2]*m[0]; c[7] = prods[ci][2][1] * inv_n - m[2]*m[1]; c[8] = prods[ci][2][2] * inv_n - m[2]*m[2];

            calcInverseCovAndDeterm(ci, 0.01);
        }
    }
}

void GMM::calcInverseCovAndDeterm(int ci, const double singularFix)
{
    if( coefs[ci] > 0 )
    {
        double *c = cov + 9*ci;
        double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
        if (dtrm <= 1e-6 && singularFix > 0)
        {
            // Adds the white noise to avoid singular covariance matrix.
            c[0] += singularFix;
            c[4] += singularFix;
            c[8] += singularFix;
            dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
        }
        covDeterms[ci] = dtrm;

        CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
        double inv_dtrm = 1.0 / dtrm;
        inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) * inv_dtrm;
        inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) * inv_dtrm;
        inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) * inv_dtrm;
        inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) * inv_dtrm;
        inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) * inv_dtrm;
        inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) * inv_dtrm;
        inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) * inv_dtrm;
        inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) * inv_dtrm;
        inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) * inv_dtrm;
    }
}

} // namespace

/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
static double calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x>0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

    return beta;
}

/*
  Calculate weights of noterminal vertices of graph.
  beta and gamma - parameters of GrabCut algorithm.
 */
static inline int colorSqDist(const Vec3b& a, const Vec3b& b)
{
    const int db = int(a[0]) - int(b[0]);
    const int dg = int(a[1]) - int(b[1]);
    const int dr = int(a[2]) - int(b[2]);
    return db * db + dg * dg + dr * dr;
}

struct NWeightLUT
{
    static const int MAX_SQ_DIST = 3 * 255 * 255; // 195075
    std::vector<double> ortho;
    std::vector<double> diag;
    double beta = 0.0;
    double gamma = 0.0;
    double gammaDivSqrt2 = 0.0;
    bool useTable = false;

    void init(Size imgSize, int iterCount, double beta_, double gamma_)
    {
        beta = beta_;
        gamma = gamma_;
        gammaDivSqrt2 = gamma_ / std::sqrt(2.0f);
        const size_t totalEdges = size_t(4) * size_t(imgSize.width) * size_t(imgSize.height) * size_t(std::max(1, iterCount));
        useTable = (totalEdges >= size_t(2 * (MAX_SQ_DIST + 1)));
        if (useTable)
        {
            ortho.resize(MAX_SQ_DIST + 1);
            diag.resize(MAX_SQ_DIST + 1);
            for (int d2 = 0; d2 <= MAX_SQ_DIST; ++d2)
            {
                const double e = std::exp(-beta * double(d2));
                ortho[d2] = gamma * e;
                diag[d2] = gammaDivSqrt2 * e;
            }
        }
    }

    inline double weightOrtho(const Vec3b& a, const Vec3b& b) const
    {
        const int d2 = colorSqDist(a, b);
        return useTable ? ortho[d2] : (gamma * std::exp(-beta * double(d2)));
    }

    inline double weightDiag(const Vec3b& a, const Vec3b& b) const
    {
        const int d2 = colorSqDist(a, b);
        return useTable ? diag[d2] : (gammaDivSqrt2 * std::exp(-beta * double(d2)));
    }
};

/*
  Check size, type and element values of mask matrix.
 */
static void checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( cv::Error::StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( cv::Error::StsBadArg, "mask must have CV_8UC1 type" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( cv::Error::StsBadArg, "mask must have as many rows and cols as img" );
    for( int y = 0; y < mask.rows; y++ )
    {
        for( int x = 0; x < mask.cols; x++ )
        {
            uchar val = mask.at<uchar>(y,x);
            if( val!=GC_BGD && val!=GC_FGD && val!=GC_PR_BGD && val!=GC_PR_FGD )
                CV_Error( cv::Error::StsBadArg, "mask element value must be equal "
                    "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
        }
    }
}

/*
  Initialize mask using rectangular.
*/
static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo( GC_BGD );

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width-rect.x);
    rect.height = std::min(rect.height, imgSize.height-rect.y);

    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
static void learnSamplesFromMat( const Mat& samples, const Mat& labels, GMM& gmm )
{
    gmm.initLearning();
    const int nSamples = samples.rows;
    const int* lblPtr = labels.ptr<int>(0);
    for( int i = 0; i < nSamples; i++ )
    {
        const float* s = samples.ptr<float>(i);
        gmm.addSample( lblPtr[i], Vec3d(s[0], s[1], s[2]) );
    }
    gmm.endLearning();
}

static void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    int bgdCount = 0, fgdCount = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        const uchar* maskRow = mask.ptr<uchar>(y);
        for( int x = 0; x < img.cols; x++ )
        {
            if( maskRow[x] == GC_BGD || maskRow[x] == GC_PR_BGD )
                ++bgdCount;
            else // GC_FGD | GC_PR_FGD
                ++fgdCount;
        }
    }
    CV_Assert( bgdCount > 0 && fgdCount > 0 );

    {
        Mat bgdSamples( bgdCount, 3, CV_32FC1 );
        Mat bgdLabels;
        int idx = 0;
        for( int y = 0; y < img.rows; y++ )
        {
            const Vec3b* imgRow = img.ptr<Vec3b>(y);
            const uchar* maskRow = mask.ptr<uchar>(y);
            for( int x = 0; x < img.cols; x++ )
            {
                if( maskRow[x] == GC_BGD || maskRow[x] == GC_PR_BGD )
                {
                    float* dst = bgdSamples.ptr<float>(idx++);
                    const Vec3b c = imgRow[x];
                    dst[0] = (float)c[0];
                    dst[1] = (float)c[1];
                    dst[2] = (float)c[2];
                }
            }
        }
        int num_clusters = std::min(GMM::componentsCount, bgdCount);
        kmeans( bgdSamples, num_clusters, bgdLabels,
                TermCriteria( TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, kMeansType );
        learnSamplesFromMat( bgdSamples, bgdLabels, bgdGMM );
    }
    {
        Mat fgdSamples( fgdCount, 3, CV_32FC1 );
        Mat fgdLabels;
        int idx = 0;
        for( int y = 0; y < img.rows; y++ )
        {
            const Vec3b* imgRow = img.ptr<Vec3b>(y);
            const uchar* maskRow = mask.ptr<uchar>(y);
            for( int x = 0; x < img.cols; x++ )
            {
                if( maskRow[x] != GC_BGD && maskRow[x] != GC_PR_BGD )
                {
                    float* dst = fgdSamples.ptr<float>(idx++);
                    const Vec3b c = imgRow[x];
                    dst[0] = (float)c[0];
                    dst[1] = (float)c[1];
                    dst[2] = (float)c[2];
                }
            }
        }
        int num_clusters = std::min(GMM::componentsCount, fgdCount);
        kmeans( fgdSamples, num_clusters, fgdLabels,
                TermCriteria( TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, kMeansType );
        learnSamplesFromMat( fgdSamples, fgdLabels, fgdGMM );
    }
}

/*
  Assign GMMs components for each pixel.
*/
static void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs )
{
    for( int y = 0; y < img.rows; y++ )
    {
        const Vec3b* imgRow = img.ptr<Vec3b>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        uchar* compRow = compIdxs.ptr<uchar>(y);
        for( int x = 0; x < img.cols; x++ )
        {
            const Vec3d color = imgRow[x];
            compRow[x] = (uchar)(maskRow[x] == GC_BGD || maskRow[x] == GC_PR_BGD ?
                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color));
        }
    }
}

/*
  Learn GMMs parameters.
*/
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    for( int y = 0; y < img.rows; y++ )
    {
        const Vec3b* imgRow = img.ptr<Vec3b>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        const uchar* compRow = compIdxs.ptr<uchar>(y);
        for( int x = 0; x < img.cols; x++ )
        {
            const int ci = compRow[x];
            if( maskRow[x] == GC_BGD || maskRow[x] == GC_PR_BGD )
                bgdGMM.addSample( ci, imgRow[x] );
            else
                fgdGMM.addSample( ci, imgRow[x] );
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

static inline bool isUncertainMask( uchar m )
{
    return m == GC_PR_BGD || m == GC_PR_FGD;
}

/*
  Construct GCGraph over uncertain (GC_PR_BGD / GC_PR_FGD) pixels only.
  Definite pixels (GC_BGD / GC_FGD) have t-weight lambda = 450 strictly greater
  than the maximum possible n-weight degree (4*gamma + 4*gamma/sqrt(2) = 341.42),
  so they never cross the min-cut; incident boundary n-weights fold directly into
  adjacent uncertain vertices' t-weights. Returns true if maxFlow() must be run.
*/
static bool constructGCGraph( const Mat& img, Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM,
                              const NWeightLUT& nweights, GCGraph<double>& graph )
{
    const int rows = img.rows;
    const int cols = img.cols;
    int vtxCount = 0;
    int edgePairCount = 0;
    for( int y = 0; y < rows; y++ )
    {
        const uchar* maskRow = mask.ptr<uchar>(y);
        const uchar* prevMaskRow = (y > 0) ? mask.ptr<uchar>(y - 1) : nullptr;
        for( int x = 0; x < cols; x++ )
        {
            if( !isUncertainMask(maskRow[x]) )
                continue;
            ++vtxCount;
            if( x > 0 && isUncertainMask(maskRow[x - 1]) )
                ++edgePairCount;
            if( y > 0 )
            {
                if( x > 0 && isUncertainMask(prevMaskRow[x - 1]) )
                    ++edgePairCount;
                if( isUncertainMask(prevMaskRow[x]) )
                    ++edgePairCount;
                if( x < cols - 1 && isUncertainMask(prevMaskRow[x + 1]) )
                    ++edgePairCount;
            }
        }
    }

    if( vtxCount == 0 )
        return false;

    if( edgePairCount == 0 )
    {
        for( int y = 0; y < rows; y++ )
        {
            const Vec3b* imgRow = img.ptr<Vec3b>(y);
            uchar* maskRow = mask.ptr<uchar>(y);
            for( int x = 0; x < cols; x++ )
            {
                if( !isUncertainMask(maskRow[x]) )
                    continue;
                const Vec3b color = imgRow[x];
                double fromSource = -log( bgdGMM(color) );
                double toSink = -log( fgdGMM(color) );
                for( int dy = -1; dy <= 1; dy++ )
                {
                    if( y + dy < 0 || y + dy >= rows )
                        continue;
                    const Vec3b* nbImgRow = img.ptr<Vec3b>(y + dy);
                    const uchar* nbMaskRow = mask.ptr<uchar>(y + dy);
                    for( int dx = -1; dx <= 1; dx++ )
                    {
                        if( (dy == 0 && dx == 0) || x + dx < 0 || x + dx >= cols )
                            continue;
                        const double w = (dy == 0 || dx == 0) ?
                            nweights.weightOrtho(color, nbImgRow[x + dx]) :
                            nweights.weightDiag(color, nbImgRow[x + dx]);
                        if( nbMaskRow[x + dx] == GC_FGD )
                            fromSource += w;
                        else
                            toSink += w;
                    }
                }
                maskRow[x] = (fromSource >= toSink) ? GC_PR_FGD : GC_PR_BGD;
            }
        }
        return false;
    }

    graph.create( vtxCount, 2 * edgePairCount );
    std::vector<int> prevRowVtx(cols, -1);
    std::vector<int> curRowVtx(cols, -1);
    for( int y = 0; y < rows; y++ )
    {
        const Vec3b* imgRow = img.ptr<Vec3b>(y);
        const Vec3b* prevRow = (y > 0) ? img.ptr<Vec3b>(y - 1) : nullptr;
        const uchar* maskRow = mask.ptr<uchar>(y);
        const uchar* prevMaskRow = (y > 0) ? mask.ptr<uchar>(y - 1) : nullptr;
        for( int x = 0; x < cols; x++ )
        {
            const Vec3b color = imgRow[x];
            const uchar mval = maskRow[x];
            if( isUncertainMask(mval) )
            {
                const int vtxIdx = graph.addVtx();
                curRowVtx[x] = vtxIdx;
                double fromSource = -log( bgdGMM(color) );
                double toSink = -log( fgdGMM(color) );

                auto handleNb = [&](int nbVtx, uchar mNb, double w) {
                    if( nbVtx >= 0 )
                        graph.addEdges( vtxIdx, nbVtx, w, w );
                    else if( mNb == GC_FGD )
                        fromSource += w;
                    else
                        toSink += w;
                };
                if( x > 0 )
                    handleNb( curRowVtx[x - 1], maskRow[x - 1], nweights.weightOrtho(color, imgRow[x - 1]) );
                if( x > 0 && y > 0 )
                    handleNb( prevRowVtx[x - 1], prevMaskRow[x - 1], nweights.weightDiag(color, prevRow[x - 1]) );
                if( y > 0 )
                    handleNb( prevRowVtx[x], prevMaskRow[x], nweights.weightOrtho(color, prevRow[x]) );
                if( x < cols - 1 && y > 0 )
                    handleNb( prevRowVtx[x + 1], prevMaskRow[x + 1], nweights.weightDiag(color, prevRow[x + 1]) );

                graph.addTermWeights( vtxIdx, fromSource, toSink );
            }
            else
            {
                curRowVtx[x] = -1;
                const double srcW_mask = (mval == GC_FGD) ? 1.0 : 0.0;
                const double snkW_mask = (mval == GC_BGD) ? 1.0 : 0.0;
                if( x > 0 && curRowVtx[x - 1] >= 0 )
                {
                    const double w = nweights.weightOrtho(color, imgRow[x - 1]);
                    graph.addTermWeights( curRowVtx[x - 1], srcW_mask * w, snkW_mask * w );
                }
                if( x > 0 && y > 0 && prevRowVtx[x - 1] >= 0 )
                {
                    const double w = nweights.weightDiag(color, prevRow[x - 1]);
                    graph.addTermWeights( prevRowVtx[x - 1], srcW_mask * w, snkW_mask * w );
                }
                if( y > 0 && prevRowVtx[x] >= 0 )
                {
                    const double w = nweights.weightOrtho(color, prevRow[x]);
                    graph.addTermWeights( prevRowVtx[x], srcW_mask * w, snkW_mask * w );
                }
                if( x < cols - 1 && y > 0 && prevRowVtx[x + 1] >= 0 )
                {
                    const double w = nweights.weightDiag(color, prevRow[x + 1]);
                    graph.addTermWeights( prevRowVtx[x + 1], srcW_mask * w, snkW_mask * w );
                }
            }
        }
        std::swap(prevRowVtx, curRowVtx);
    }
    return true;
}

/*
  Estimate segmentation using MaxFlow algorithm
*/
static void estimateSegmentation( GCGraph<double>& graph, Mat& mask )
{
    graph.maxFlow();
    int vtxIdx = 0;
    for( int y = 0; y < mask.rows; y++ )
    {
        uchar* maskRow = mask.ptr<uchar>(y);
        for( int x = 0; x < mask.cols; x++ )
        {
            if( isUncertainMask(maskRow[x]) )
                maskRow[x] = graph.inSourceSegment( vtxIdx++ ) ? GC_PR_FGD : GC_PR_BGD;
        }
    }
}

void cv::grabCut( InputArray _img, InputOutputArray _mask, Rect rect,
                  InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                  int iterCount, int mode )
{
    CV_INSTRUMENT_REGION();

    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();

    if( img.empty() )
        CV_Error( cv::Error::StsBadArg, "image is empty" );
    if( img.type() != CV_8UC3 )
        CV_Error( cv::Error::StsBadArg, "image must have CV_8UC3 type" );

    GMM bgdGMM( bgdModel ), fgdGMM( fgdModel );

    if( mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK )
    {
        if( mode == GC_INIT_WITH_RECT )
            initMaskWithRect( mask, img.size(), rect );
        else // flag == GC_INIT_WITH_MASK
            checkMask( img, mask );
        initGMMs( img, mask, bgdGMM, fgdGMM );
    }

    if( iterCount <= 0)
        return;

    if( mode == GC_EVAL_FREEZE_MODEL )
        iterCount = 1;

    if( mode == GC_EVAL || mode == GC_EVAL_FREEZE_MODEL )
        checkMask( img, mask );

    const double gamma = 50;
    const double beta = calcBeta( img );

    NWeightLUT nweights;
    nweights.init(img.size(), iterCount, beta, gamma);

    Mat compIdxs( img.size(), CV_8UC1 );
    GCGraph<double> graph;
    for( int i = 0; i < iterCount; i++ )
    {
        assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
        if( mode != GC_EVAL_FREEZE_MODEL )
            learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
        if( constructGCGraph(img, mask, bgdGMM, fgdGMM, nweights, graph) )
            estimateSegmentation( graph, mask );
    }
}
