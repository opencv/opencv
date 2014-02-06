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
#include "opencv2/calib3d.hpp"
#include <stdio.h>

namespace cv
{

/*
  The code below implements keypoint detector, fern-based point classifier and a planar object detector.

  References:
   1. Mustafa Özuysal, Michael Calonder, Vincent Lepetit, Pascal Fua,
      "Fast KeyPoint Recognition Using Random Ferns,"
      IEEE Transactions on Pattern Analysis and Machine Intelligence, 15 Jan. 2009.

   2. Vincent Lepetit, Pascal Fua,
      "Towards Recognizing Feature Points Using Classification Trees,"
      Technical Report IC/2004/74, EPFL, 2004.
*/

const int progressBarSize = 50;

//////////////////////////// Patch Generator //////////////////////////////////

static const double DEFAULT_BACKGROUND_MIN = 0;
static const double DEFAULT_BACKGROUND_MAX = 256;
static const double DEFAULT_NOISE_RANGE = 5;
static const double DEFAULT_LAMBDA_MIN = 0.6;
static const double DEFAULT_LAMBDA_MAX = 1.5;
static const double DEFAULT_THETA_MIN = -CV_PI;
static const double DEFAULT_THETA_MAX = CV_PI;
static const double DEFAULT_PHI_MIN = -CV_PI;
static const double DEFAULT_PHI_MAX = CV_PI;

PatchGenerator::PatchGenerator()
: backgroundMin(DEFAULT_BACKGROUND_MIN), backgroundMax(DEFAULT_BACKGROUND_MAX),
noiseRange(DEFAULT_NOISE_RANGE), randomBlur(true), lambdaMin(DEFAULT_LAMBDA_MIN),
lambdaMax(DEFAULT_LAMBDA_MAX), thetaMin(DEFAULT_THETA_MIN),
thetaMax(DEFAULT_THETA_MAX), phiMin(DEFAULT_PHI_MIN),
phiMax(DEFAULT_PHI_MAX)
{
}


PatchGenerator::PatchGenerator(double _backgroundMin, double _backgroundMax,
                               double _noiseRange, bool _randomBlur,
                               double _lambdaMin, double _lambdaMax,
                               double _thetaMin, double _thetaMax,
                               double _phiMin, double _phiMax )
: backgroundMin(_backgroundMin), backgroundMax(_backgroundMax),
noiseRange(_noiseRange), randomBlur(_randomBlur),
lambdaMin(_lambdaMin), lambdaMax(_lambdaMax),
thetaMin(_thetaMin), thetaMax(_thetaMax),
phiMin(_phiMin), phiMax(_phiMax)
{
}


void PatchGenerator::generateRandomTransform(Point2f srcCenter, Point2f dstCenter,
                                             Mat& transform, RNG& rng, bool inverse) const
{
    double lambda1 = rng.uniform(lambdaMin, lambdaMax);
    double lambda2 = rng.uniform(lambdaMin, lambdaMax);
    double theta = rng.uniform(thetaMin, thetaMax);
    double phi = rng.uniform(phiMin, phiMax);

    // Calculate random parameterized affine transformation A,
    // A = T(patch center) * R(theta) * R(phi)' *
    //     S(lambda1, lambda2) * R(phi) * T(-pt)
    double st = sin(theta);
    double ct = cos(theta);
    double sp = sin(phi);
    double cp = cos(phi);
    double c2p = cp*cp;
    double s2p = sp*sp;

    double A = lambda1*c2p + lambda2*s2p;
    double B = (lambda2 - lambda1)*sp*cp;
    double C = lambda1*s2p + lambda2*c2p;

    double Ax_plus_By = A*srcCenter.x + B*srcCenter.y;
    double Bx_plus_Cy = B*srcCenter.x + C*srcCenter.y;

    transform.create(2, 3, CV_64F);
    Mat_<double>& T = (Mat_<double>&)transform;
    T(0,0) = A*ct - B*st;
    T(0,1) = B*ct - C*st;
    T(0,2) = -ct*Ax_plus_By + st*Bx_plus_Cy + dstCenter.x;
    T(1,0) = A*st + B*ct;
    T(1,1) = B*st + C*ct;
    T(1,2) = -st*Ax_plus_By - ct*Bx_plus_Cy + dstCenter.y;

    if( inverse )
        invertAffineTransform(T, T);
}


void PatchGenerator::operator ()(const Mat& image, Point2f pt, Mat& patch, Size patchSize, RNG& rng) const
{
    double buffer[6];
    Mat_<double> T(2, 3, buffer);

    generateRandomTransform(pt, Point2f((patchSize.width-1)*0.5f, (patchSize.height-1)*0.5f), T, rng);
    (*this)(image, T, patch, patchSize, rng);
}


void PatchGenerator::operator ()(const Mat& image, const Mat& T,
                                 Mat& patch, Size patchSize, RNG& rng) const
{
    patch.create( patchSize, image.type() );
    if( backgroundMin != backgroundMax )
    {
        rng.fill(patch, RNG::UNIFORM, Scalar::all(backgroundMin), Scalar::all(backgroundMax));
        warpAffine(image, patch, T, patchSize, INTER_LINEAR, BORDER_TRANSPARENT);
    }
    else
        warpAffine(image, patch, T, patchSize, INTER_LINEAR, BORDER_CONSTANT, Scalar::all(backgroundMin));

    int ksize = randomBlur ? (unsigned)rng % 9 - 5 : 0;
    if( ksize > 0 )
    {
        ksize = ksize*2 + 1;
        GaussianBlur(patch, patch, Size(ksize, ksize), 0, 0);
    }

    if( noiseRange > 0 )
    {
        AutoBuffer<uchar> _noiseBuf( patchSize.width*patchSize.height*image.elemSize() );
        Mat noise(patchSize, image.type(), (uchar*)_noiseBuf);
        int delta = image.depth() == CV_8U ? 128 : image.depth() == CV_16U ? 32768 : 0;
        rng.fill(noise, RNG::NORMAL, Scalar::all(delta), Scalar::all(noiseRange));
        if( backgroundMin != backgroundMax )
            addWeighted(patch, 1, noise, 1, -delta, patch);
        else
        {
            for( int i = 0; i < patchSize.height; i++ )
            {
                uchar* prow = patch.ptr<uchar>(i);
                const uchar* nrow =  noise.ptr<uchar>(i);
                for( int j = 0; j < patchSize.width; j++ )
                    if( prow[j] != backgroundMin )
                        prow[j] = saturate_cast<uchar>(prow[j] + nrow[j] - delta);
            }
        }
    }
}

void PatchGenerator::warpWholeImage(const Mat& image, Mat& matT, Mat& buf,
                                    Mat& warped, int border, RNG& rng) const
{
    Mat_<double> T = matT;
    Rect roi(INT_MAX, INT_MAX, INT_MIN, INT_MIN);

    for( int k = 0; k < 4; k++ )
    {
        Point2f pt0, pt1;
        pt0.x = (float)(k == 0 || k == 3 ? 0 : image.cols);
        pt0.y = (float)(k < 2 ? 0 : image.rows);
        pt1.x = (float)(T(0,0)*pt0.x + T(0,1)*pt0.y + T(0,2));
        pt1.y = (float)(T(1,0)*pt0.x + T(1,1)*pt0.y + T(1,2));

        roi.x = std::min(roi.x, cvFloor(pt1.x));
        roi.y = std::min(roi.y, cvFloor(pt1.y));
        roi.width = std::max(roi.width, cvCeil(pt1.x));
        roi.height = std::max(roi.height, cvCeil(pt1.y));
    }

    roi.width -= roi.x - 1;
    roi.height -= roi.y - 1;
    int dx = border - roi.x;
    int dy = border - roi.y;

    if( (roi.width+border*2)*(roi.height+border*2) > buf.cols )
        buf.create(1, (roi.width+border*2)*(roi.height+border*2), image.type());

    warped = Mat(roi.height + border*2, roi.width + border*2,
                 image.type(), buf.data);

    T(0,2) += dx;
    T(1,2) += dy;
    (*this)(image, T, warped, warped.size(), rng);

    if( T.data != matT.data )
        T.convertTo(matT, matT.type());
}


// Params are assumed to be symmetrical: lambda w.r.t. 1, theta and phi w.r.t. 0
void PatchGenerator::setAffineParam(double lambda, double theta, double phi)
{
   lambdaMin = 1. - lambda;
   lambdaMax = 1. + lambda;
   thetaMin = -theta;
   thetaMax = theta;
   phiMin = -phi;
   phiMax = phi;
}


/////////////////////////////////////// LDetector //////////////////////////////////////////////

LDetector::LDetector() : radius(7), threshold(20), nOctaves(3), nViews(1000),
    verbose(false), baseFeatureSize(32), clusteringDistance(2)
{
}

LDetector::LDetector(int _radius, int _threshold, int _nOctaves, int _nViews,
                     double _baseFeatureSize, double _clusteringDistance)
: radius(_radius), threshold(_threshold), nOctaves(_nOctaves), nViews(_nViews),
    verbose(false), baseFeatureSize(_baseFeatureSize), clusteringDistance(_clusteringDistance)
{
}

static void getDiscreteCircle(int R, std::vector<Point>& circle, std::vector<int>& filledHCircle)
{
    int x = R, y = 0;
    for( ;;y++ )
    {
        x = cvRound(std::sqrt((double)R*R - y*y));
        if( x < y )
            break;
        circle.push_back(Point(x,y));
        if( x == y )
            break;
    }

    int i, n8 = (int)circle.size() - (x == y), n8_ = n8 - (x != y), n4 = n8 + n8_, n = n4*4;
    CV_Assert(n8 > 0);
    circle.resize(n);

    for( i = 0; i < n8; i++ )
    {
        Point p = circle[i];
        circle[i+n4] = Point(-p.y, p.x);
        circle[i+n4*2] = Point(-p.x, -p.y);
        circle[i+n4*3] = Point(p.y, -p.x);
    }

    for( i = n8; i < n4; i++ )
    {
        Point p = circle[n4 - i], q = Point(p.y, p.x);
        circle[i] = q;
        circle[i+n4] = Point(-q.y, q.x);
        circle[i+n4*2] = Point(-q.x, -q.y);
        circle[i+n4*3] = Point(q.y, -q.x);
    }

    // the filled upper half of the circle is encoded as sequence of integers,
    // i-th element is the coordinate of right-most circle point in each horizontal line y=i.
    // the left-most point will be -filledHCircle[i].
    for( i = 0, y = -1; i < n4; i++ )
    {
        Point p = circle[i];
        if( p.y != y )
        {
            filledHCircle.push_back(p.x);
            y = p.y;
            if( y == R )
                break;
        }
    }
}


struct CmpKeypointScores
{
    bool operator ()(const KeyPoint& a, const KeyPoint& b) const { return std::abs(a.response) > std::abs(b.response); }
};


void LDetector::getMostStable2D(const Mat& image, std::vector<KeyPoint>& keypoints,
                                int maxPoints, const PatchGenerator& _patchGenerator) const
{
    PatchGenerator patchGenerator = _patchGenerator;
    patchGenerator.backgroundMin = patchGenerator.backgroundMax = 128;

    Mat warpbuf, warped;
    Mat matM(2, 3, CV_64F), _iM(2, 3, CV_64F);
    double *M = (double*)matM.data, *iM = (double*)_iM.data;
    RNG& rng = theRNG();
    int i, k;
    std::vector<KeyPoint> tempKeypoints;
    double d2 = clusteringDistance*clusteringDistance;
    keypoints.clear();

    // TODO: this loop can be run in parallel, for that we need
    // a separate accumulator keypoint lists for different threads.
    for( i = 0; i < nViews; i++ )
    {
        // 1. generate random transform
        // 2. map the source image corners and compute the ROI in canvas
        // 3. select the ROI in canvas, adjust the transformation matrix
        // 4. apply the transformation
        // 5. run keypoint detector in pyramids
        // 6. map each point back and update the lists of most stable points

        if(verbose && (i+1)*progressBarSize/nViews != i*progressBarSize/nViews)
            putchar('.');

        if( i > 0 )
            patchGenerator.generateRandomTransform(Point2f(), Point2f(), matM, rng);
        else
        {
            // identity transformation
            M[0] = M[4] = 1;
            M[1] = M[3] = M[2] = M[5] = 0;
        }

        patchGenerator.warpWholeImage(image, matM, warpbuf, warped, cvCeil(baseFeatureSize*0.5+radius), rng);
        (*this)(warped, tempKeypoints, maxPoints*3);
        invertAffineTransform(matM, _iM);

        int j, sz0 = (int)tempKeypoints.size(), sz1;
        for( j = 0; j < sz0; j++ )
        {
            KeyPoint kpt1 = tempKeypoints[j];
            KeyPoint kpt0((float)(iM[0]*kpt1.pt.x + iM[1]*kpt1.pt.y + iM[2]),
                          (float)(iM[3]*kpt1.pt.x + iM[4]*kpt1.pt.y + iM[5]),
                          kpt1.size, -1.f, 1.f, kpt1.octave);
            float r = kpt1.size*0.5f;
            if( kpt0.pt.x < r || kpt0.pt.x >= image.cols - r ||
               kpt0.pt.y < r || kpt0.pt.y >= image.rows - r )
                continue;

            sz1 = (int)keypoints.size();
            for( k = 0; k < sz1; k++ )
            {
                KeyPoint kpt = keypoints[k];
                if( kpt.octave != kpt0.octave )
                    continue;
                double dx = kpt.pt.x - kpt0.pt.x, dy = kpt.pt.y - kpt0.pt.y;
                if( dx*dx + dy*dy <= d2*(1 << kpt.octave*2) )
                {
                    keypoints[k] = KeyPoint((kpt.pt.x*kpt.response + kpt0.pt.x)/(kpt.response+1),
                                            (kpt.pt.y*kpt.response + kpt0.pt.y)/(kpt.response+1),
                                            kpt.size, -1.f, kpt.response + 1, kpt.octave);
                    break;
                }
            }
            if( k == sz1 )
                keypoints.push_back(kpt0);
        }
    }

    if( verbose )
        putchar('\n');

    if( (int)keypoints.size() > maxPoints )
    {
        std::sort(keypoints.begin(), keypoints.end(), CmpKeypointScores());
        keypoints.resize(maxPoints);
    }
}


static inline int computeLResponse(const uchar* ptr, const int* cdata, int csize)
{
    int i, csize2 = csize/2, sum = -ptr[0]*csize;
    for( i = 0; i < csize2; i++ )
    {
        int ofs = cdata[i];
        sum += ptr[ofs] + ptr[-ofs];
    }
    return sum;
}


static Point2f adjustCorner(const float* fval, float& fvaln)
{
    double bx = (fval[3] - fval[5])*0.5;
    double by = (fval[2] - fval[7])*0.5;
    double Axx = fval[3] - fval[4]*2 + fval[5];
    double Axy = (fval[0] - fval[2] - fval[6] + fval[8])*0.25;
    double Ayy = fval[1] - fval[4]*2 + fval[7];
    double D = Axx*Ayy - Axy*Axy;
    D = D != 0 ? 1./D : 0;
    double dx = (bx*Ayy - by*Axy)*D;
    double dy = (by*Axx - bx*Axy)*D;
    dx = std::min(std::max(dx, -1.), 1.);
    dy = std::min(std::max(dy, -1.), 1.);
    fvaln = (float)(fval[4] + (bx*dx + by*dy)*0.5);
    if(fvaln*fval[4] < 0 || std::abs(fvaln) < std::abs(fval[4]))
        fvaln = fval[4];

    return Point2f((float)dx, (float)dy);
}

void LDetector::operator()(const Mat& image, std::vector<KeyPoint>& keypoints, int maxCount, bool scaleCoords) const
{
    std::vector<Mat> pyr;
    buildPyramid(image, pyr, std::max(nOctaves-1, 0));
    (*this)(pyr, keypoints, maxCount, scaleCoords);
}

void LDetector::operator()(const std::vector<Mat>& pyr, std::vector<KeyPoint>& keypoints, int maxCount, bool scaleCoords) const
{
    const int lthreshold = 3;
    int L, x, y, i, j, k, tau = lthreshold;
    Mat scoreBuf(pyr[0].size(), CV_16S), maskBuf(pyr[0].size(), CV_8U);
    int scoreElSize = (int)scoreBuf.elemSize();
    std::vector<Point> circle0;
    std::vector<int> fhcircle0, circle, fcircle_s, fcircle;
    getDiscreteCircle(radius, circle0, fhcircle0);
    CV_Assert(fhcircle0.size() == (size_t)(radius+1) && circle0.size() % 2 == 0);
    keypoints.clear();

    for( L = 0; L < nOctaves; L++ )
    {
        //  Pyramidal keypoint detector body:
        //    1. build next pyramid layer
        //    2. scan points, check the circular neighborhood, compute the score
        //    3. do non-maxima suppression
        //    4. adjust the corners (sub-pix)
        double cscale = scaleCoords ? 1 << L : 1;
        Size layerSize = pyr[L].size();
        if( layerSize.width < radius*2 + 3 || layerSize.height < radius*2 + 3 )
            break;
        Mat scoreLayer(layerSize, scoreBuf.type(), scoreBuf.data);
        Mat maskLayer(layerSize, maskBuf.type(), maskBuf.data);
        const Mat& pyrLayer = pyr[L];
        int sstep = (int)(scoreLayer.step/sizeof(short));
        int mstep = (int)maskLayer.step;

        int csize = (int)circle0.size(), csize2 = csize/2;
        circle.resize(csize*3);
        for( i = 0; i < csize; i++ )
            circle[i] = circle[i+csize] = circle[i+csize*2] = (int)((-circle0[i].y)*pyrLayer.step + circle0[i].x);
        fcircle.clear();
        fcircle_s.clear();
        for( i = -radius; i <= radius; i++ )
        {
            x = fhcircle0[std::abs(i)];
            for( j = -x; j <= x; j++ )
            {
                fcircle_s.push_back(i*sstep + j);
                fcircle.push_back((int)(i*pyrLayer.step + j));
            }
        }
        int nsize = (int)fcircle.size();
        const int* cdata = &circle[0];
        const int* ndata = &fcircle[0];
        const int* ndata_s = &fcircle_s[0];

        for( y = 0; y < radius; y++ )
        {
            memset( scoreLayer.ptr<short>(y), 0, layerSize.width*scoreElSize );
            memset( scoreLayer.ptr<short>(layerSize.height-y-1), 0, layerSize.width*scoreElSize );
            memset( maskLayer.ptr<uchar>(y), 0, layerSize.width );
            memset( maskLayer.ptr<uchar>(layerSize.height-y-1), 0, layerSize.width );
        }

        int vradius = (int)(radius*pyrLayer.step);

        for( y = radius; y < layerSize.height - radius; y++ )
        {
            const uchar* img = pyrLayer.ptr<uchar>(y) + radius;
            short* scores = scoreLayer.ptr<short>(y);
            uchar* mask = maskLayer.ptr<uchar>(y);

            for( x = 0; x < radius; x++ )
            {
                scores[x] = scores[layerSize.width - 1 - x] = 0;
                mask[x] = mask[layerSize.width - 1 - x] = 0;
            }

            for( x = radius; x < layerSize.width - radius; x++, img++ )
            {
                int val0 = *img;
                if( (std::abs(val0 - img[radius]) < tau && std::abs(val0 - img[-radius]) < tau) ||
                   (std::abs(val0 - img[vradius]) < tau && std::abs(val0 - img[-vradius]) < tau))
                {
                    scores[x] = 0;
                    mask[x] = 0;
                    continue;
                }

                for( k = 0; k < csize; k++ )
                {
                    if( std::abs(val0 - img[cdata[k]]) < tau &&
                       (std::abs(val0 - img[cdata[k + csize2]]) < tau ||
                        std::abs(val0 - img[cdata[k + csize2 - 1]]) < tau ||
                        std::abs(val0 - img[cdata[k + csize2 + 1]]) < tau ||
                        std::abs(val0 - img[cdata[k + csize2 - 2]]) < tau ||
                        std::abs(val0 - img[cdata[k + csize2 + 2]]) < tau/* ||
                     std::abs(val0 - img[cdata[k + csize2 - 3]]) < tau ||
                     std::abs(val0 - img[cdata[k + csize2 + 3]]) < tau*/) )
                        break;
                }

                if( k < csize )
                {
                    scores[x] = 0;
                    mask[x] = 0;
                }
                else
                {
                    scores[x] = (short)computeLResponse(img, cdata, csize);
                    mask[x] = 1;
                }
            }
        }

        for( y = radius+1; y < layerSize.height - radius-1; y++ )
        {
            const uchar* img = pyrLayer.ptr<uchar>(y) + radius+1;
            short* scores = scoreLayer.ptr<short>(y) + radius+1;
            const uchar* mask = maskLayer.ptr<uchar>(y) + radius+1;

            for( x = radius+1; x < layerSize.width - radius-1; x++, img++, scores++, mask++ )
            {
                int val0 = *scores;
                if( !*mask || std::abs(val0) < lthreshold ||
                   (mask[-1] + mask[1] + mask[-mstep-1] + mask[-mstep] + mask[-mstep+1]+
                    mask[mstep-1] + mask[mstep] + mask[mstep+1] < 3))
                    continue;
                bool recomputeZeroScores = radius*2 < y && y < layerSize.height - radius*2 &&
                radius*2 < x && x < layerSize.width - radius*2;

                if( val0 > 0 )
                {
                    for( k = 0; k < nsize; k++ )
                    {
                        int val = scores[ndata_s[k]];
                        if( val == 0 && recomputeZeroScores )
                            scores[ndata_s[k]] = (short)(val =
                                computeLResponse(img + ndata[k], cdata, csize));
                        if( val0 < val )
                            break;
                    }
                }
                else
                {
                    for( k = 0; k < nsize; k++ )
                    {
                        int val = scores[ndata_s[k]];
                        if( val == 0 && recomputeZeroScores )
                            scores[ndata_s[k]] = (short)(val =
                                computeLResponse(img + ndata[k], cdata, csize));
                        if( val0 > val )
                            break;
                    }
                }
                if( k < nsize )
                    continue;
                float fval[9], fvaln = 0;
                for( int i1 = -1; i1 <= 1; i1++ )
                    for( int j1 = -1; j1 <= 1; j1++ )
                    {
                        fval[(i1+1)*3 + j1 + 1] = (float)(scores[sstep*i1+j1] ? scores[sstep*i1+j1] :
                            computeLResponse(img + pyrLayer.step*i1 + j1, cdata, csize));
                    }
                Point2f pt = adjustCorner(fval, fvaln);
                pt.x += x;
                pt.y += y;
                keypoints.push_back(KeyPoint((float)(pt.x*cscale), (float)(pt.y*cscale),
                                             (float)(baseFeatureSize*cscale), -1, fvaln, L));
            }
        }
    }

    if( maxCount > 0 && keypoints.size() > (size_t)maxCount )
    {
        std::sort(keypoints.begin(), keypoints.end(), CmpKeypointScores());
        keypoints.resize(maxCount);
    }
}

void LDetector::read(const FileNode& objnode)
{
    radius = (int)objnode["radius"];
    threshold = (int)objnode["threshold"];
    nOctaves = (int)objnode["noctaves"];
    nViews = (int)objnode["nviews"];
    baseFeatureSize = (int)objnode["base-feature-size"];
    clusteringDistance = (int)objnode["clustering-distance"];
}

void LDetector::write(FileStorage& fs, const String& name) const
{
    internal::WriteStructContext ws(fs, name, CV_NODE_MAP);

    fs << "radius" << radius
    << "threshold" << threshold
    << "noctaves" << nOctaves
    << "nviews" << nViews
    << "base-feature-size" << baseFeatureSize
    << "clustering-distance" << clusteringDistance;
}

void LDetector::setVerbose(bool _verbose)
{
    verbose = _verbose;
}

/////////////////////////////////////// FernClassifier ////////////////////////////////////////////

FernClassifier::FernClassifier()
{
    verbose = false;
    clear();
}


FernClassifier::FernClassifier(const FileNode& node)
{
    verbose = false;
    clear();
    read(node);
}

FernClassifier::~FernClassifier()
{
}


int FernClassifier::getClassCount() const
{
    return nclasses;
}


int FernClassifier::getStructCount() const
{
    return nstructs;
}


int FernClassifier::getStructSize() const
{
    return structSize;
}


int FernClassifier::getSignatureSize() const
{
    return signatureSize;
}


int FernClassifier::getCompressionMethod() const
{
    return compressionMethod;
}


Size FernClassifier::getPatchSize() const
{
    return patchSize;
}


FernClassifier::FernClassifier(const std::vector<std::vector<Point2f> >& points,
                               const std::vector<Mat>& refimgs,
                               const std::vector<std::vector<int> >& labels,
                               int _nclasses, int _patchSize,
                               int _signatureSize, int _nstructs,
                               int _structSize, int _nviews, int _compressionMethod,
                               const PatchGenerator& patchGenerator)
{
    verbose = false;
    clear();
    train(points, refimgs, labels, _nclasses, _patchSize,
          _signatureSize, _nstructs, _structSize, _nviews,
          _compressionMethod, patchGenerator);
}


void FernClassifier::write(FileStorage& fs, const String& objname) const
{
    internal::WriteStructContext ws(fs, objname, CV_NODE_MAP);

    cv::write(fs, "nstructs", nstructs);
    cv::write(fs, "struct-size", structSize);
    cv::write(fs, "nclasses", nclasses);
    cv::write(fs, "signature-size", signatureSize);
    cv::write(fs, "compression-method", compressionMethod);
    cv::write(fs, "patch-size", patchSize.width);
    {
        internal::WriteStructContext wsf(fs, "features", CV_NODE_SEQ + CV_NODE_FLOW);
        int i, nfeatures = (int)features.size();
        for( i = 0; i < nfeatures; i++ )
        {
            cv::write(fs, features[i].y1*patchSize.width + features[i].x1);
            cv::write(fs, features[i].y2*patchSize.width + features[i].x2);
        }
    }
    {
        internal::WriteStructContext wsp(fs, "posteriors", CV_NODE_SEQ + CV_NODE_FLOW);
        cv::write(fs, posteriors);
    }
}


void FernClassifier::read(const FileNode& objnode)
{
    clear();

    nstructs = (int)objnode["nstructs"];
    structSize = (int)objnode["struct-size"];
    nclasses = (int)objnode["nclasses"];
    signatureSize = (int)objnode["signature-size"];
    compressionMethod = (int)objnode["compression-method"];
    patchSize.width = patchSize.height = (int)objnode["patch-size"];
    leavesPerStruct = 1 << structSize;

    FileNode _nodes = objnode["features"];
    int i, nfeatures = structSize*nstructs;
    features.resize(nfeatures);
    FileNodeIterator it = _nodes.begin(), it_end = _nodes.end();
    for( i = 0; i < nfeatures && it != it_end; i++ )
    {
        int ofs1, ofs2;
        it >> ofs1 >> ofs2;
        features[i] = Feature(ofs1%patchSize.width, ofs1/patchSize.width,
                              ofs2%patchSize.width, ofs2/patchSize.width);
    }

    FileNode _posteriors = objnode["posteriors"];
    int psz = leavesPerStruct*nstructs*signatureSize;
    posteriors.reserve(psz);
    _posteriors >> posteriors;
}


void FernClassifier::clear()
{
    signatureSize = nclasses = nstructs = structSize = compressionMethod = leavesPerStruct = 0;
    std::vector<Feature>().swap(features);
    std::vector<float>().swap(posteriors);
}

bool FernClassifier::empty() const
{
    return features.empty();
}

int FernClassifier::getLeaf(int fern, const Mat& _patch) const
{
    assert( 0 <= fern && fern < nstructs );
    size_t fofs = fern*structSize, idx = 0;
    const Mat_<uchar>& patch = (const Mat_<uchar>&)_patch;

    for( int i = 0; i < structSize; i++ )
    {
        const Feature& f = features[fofs + i];
        idx = (idx << 1) + f(patch);
    }

    return (int)(fern*leavesPerStruct + idx);
}


void FernClassifier::prepare(int _nclasses, int _patchSize, int _signatureSize,
                             int _nstructs, int _structSize,
                             int _nviews, int _compressionMethod)
{
    clear();

    CV_Assert( _nclasses > 1 && _patchSize >= 5 && _nstructs > 0 &&
              _nviews > 0 && _structSize > 0 &&
              (_compressionMethod == COMPRESSION_NONE ||
               _compressionMethod == COMPRESSION_RANDOM_PROJ ||
               _compressionMethod == COMPRESSION_PCA) );

    nclasses = _nclasses;
    patchSize = Size(_patchSize, _patchSize);
    nstructs = _nstructs;
    structSize = _structSize;
    signatureSize = _compressionMethod == COMPRESSION_NONE ? nclasses : std::min(_signatureSize, nclasses);
    compressionMethod = signatureSize == nclasses ? COMPRESSION_NONE : _compressionMethod;

    leavesPerStruct = 1 << structSize;

    int i, nfeatures = structSize*nstructs;

    features = std::vector<Feature>( nfeatures );
    posteriors = std::vector<float>( leavesPerStruct*nstructs*nclasses, 1.f );
    classCounters = std::vector<int>( nclasses, leavesPerStruct );

    CV_Assert( patchSize.width <= 256 && patchSize.height <= 256 );
    RNG& rng = theRNG();

    for( i = 0; i < nfeatures; i++ )
    {
        int x1 = (unsigned)rng % patchSize.width;
        int y1 = (unsigned)rng % patchSize.height;
        int x2 = (unsigned)rng % patchSize.width;
        int y2 = (unsigned)rng % patchSize.height;
        features[i] = Feature(x1, y1, x2, y2);
    }
}

static int calcNumPoints( const std::vector<std::vector<Point2f> >& points )
{
    size_t count = 0;
    for( size_t i = 0; i < points.size(); i++ )
        count += points[i].size();
    return (int)count;
}

void FernClassifier::train(const std::vector<std::vector<Point2f> >& points,
                           const std::vector<Mat>& refimgs,
                           const std::vector<std::vector<int> >& labels,
                           int _nclasses, int _patchSize,
                           int _signatureSize, int _nstructs,
                           int _structSize, int _nviews, int _compressionMethod,
                           const PatchGenerator& patchGenerator)
{
    CV_Assert( points.size() == refimgs.size() );
    int numPoints = calcNumPoints( points );
    _nclasses = (!labels.empty() && _nclasses>0) ? _nclasses : numPoints;
    CV_Assert( labels.empty() || labels.size() == points.size() );


    prepare(_nclasses, _patchSize, _signatureSize, _nstructs,
            _structSize, _nviews, _compressionMethod);

    // pass all the views of all the samples through the generated trees and accumulate
    // the statistics (posterior probabilities) in leaves.
    Mat patch;
    RNG& rng = theRNG();

    int globalPointIdx = 0;
    for( size_t imgIdx = 0; imgIdx < points.size(); imgIdx++ )
    {
        const Point2f* imgPoints = &points[imgIdx][0];
        const int* imgLabels = labels.empty() ? 0 : &labels[imgIdx][0];
        for( size_t pointIdx = 0; pointIdx < points[imgIdx].size(); pointIdx++, globalPointIdx++ )
        {
            Point2f pt = imgPoints[pointIdx];
            const Mat& src = refimgs[imgIdx];
            int classId = imgLabels==0 ? globalPointIdx : imgLabels[pointIdx];
            if( verbose && (globalPointIdx+1)*progressBarSize/numPoints != globalPointIdx*progressBarSize/numPoints )
                putchar('.');
            CV_Assert( 0 <= classId && classId < nclasses );
            classCounters[classId] += _nviews;
            for( int v = 0; v < _nviews; v++ )
            {
                patchGenerator(src, pt, patch, patchSize, rng);
                for( int f = 0; f < nstructs; f++ )
                    posteriors[getLeaf(f, patch)*nclasses + classId]++;
            }
        }
    }
    if( verbose )
        putchar('\n');

    finalize(rng);
}


void FernClassifier::trainFromSingleView(const Mat& image,
                                         const std::vector<KeyPoint>& keypoints,
                                         int _patchSize, int _signatureSize,
                                         int _nstructs, int _structSize,
                                         int _nviews, int _compressionMethod,
                                         const PatchGenerator& patchGenerator)
{
    prepare((int)keypoints.size(), _patchSize, _signatureSize, _nstructs,
            _structSize, _nviews, _compressionMethod);
    int i, j, k, nsamples = (int)keypoints.size(), maxOctave = 0;
    for( i = 0; i < nsamples; i++ )
    {
        classCounters[i] = _nviews;
        maxOctave = std::max(maxOctave, keypoints[i].octave);
    }

    double maxScale = patchGenerator.lambdaMax*2;
    Mat canvas(cvRound(std::max(image.cols,image.rows)*maxScale + patchSize.width*2 + 10),
               cvRound(std::max(image.cols,image.rows)*maxScale + patchSize.width*2 + 10), image.type());
    Mat noisebuf;
    std::vector<Mat> pyrbuf(maxOctave+1), pyr(maxOctave+1);
    Point2f center0((image.cols-1)*0.5f, (image.rows-1)*0.5f),
    center1((canvas.cols - 1)*0.5f, (canvas.rows - 1)*0.5f);
    Mat matM(2, 3, CV_64F);
    double *M = (double*)matM.data;
    RNG& rng = theRNG();

    Mat patch(patchSize, CV_8U);

    for( i = 0; i < _nviews; i++ )
    {
        patchGenerator.generateRandomTransform(center0, center1, matM, rng);

        CV_Assert(matM.type() == CV_64F);
        Rect roi(INT_MAX, INT_MAX, INT_MIN, INT_MIN);

        for( k = 0; k < 4; k++ )
        {
            Point2f pt0, pt1;
            pt0.x = (float)(k == 0 || k == 3 ? 0 : image.cols);
            pt0.y = (float)(k < 2 ? 0 : image.rows);
            pt1.x = (float)(M[0]*pt0.x + M[1]*pt0.y + M[2]);
            pt1.y = (float)(M[3]*pt0.x + M[4]*pt0.y + M[5]);

            roi.x = std::min(roi.x, cvFloor(pt1.x));
            roi.y = std::min(roi.y, cvFloor(pt1.y));
            roi.width = std::max(roi.width, cvCeil(pt1.x));
            roi.height = std::max(roi.height, cvCeil(pt1.y));
        }

        roi.width -= roi.x + 1;
        roi.height -= roi.y + 1;

        Mat canvas_roi(canvas, roi);
        M[2] -= roi.x;
        M[5] -= roi.y;

        Size size = canvas_roi.size();
        rng.fill(canvas_roi, RNG::UNIFORM, Scalar::all(0), Scalar::all(256));
        warpAffine( image, canvas_roi, matM, size, INTER_LINEAR, BORDER_TRANSPARENT);

        pyr[0] = canvas_roi;
        for( j = 1; j <= maxOctave; j++ )
        {
            size = Size((size.width+1)/2, (size.height+1)/2);
            if( pyrbuf[j].cols < size.width*size.height )
                pyrbuf[j].create(1, size.width*size.height, image.type());
            pyr[j] = Mat(size, image.type(), pyrbuf[j].data);
            pyrDown(pyr[j-1], pyr[j]);
        }

        if( patchGenerator.noiseRange > 0 )
        {
            const int noiseDelta = 128;
            if( noisebuf.cols < pyr[0].cols*pyr[0].rows )
                noisebuf.create(1, pyr[0].cols*pyr[0].rows, image.type());
            for( j = 0; j <= maxOctave; j++ )
            {
                Mat noise(pyr[j].size(), image.type(), noisebuf.data);
                rng.fill(noise, RNG::UNIFORM, Scalar::all(-patchGenerator.noiseRange + noiseDelta),
                         Scalar::all(patchGenerator.noiseRange + noiseDelta));
                addWeighted(pyr[j], 1, noise, 1, -noiseDelta, pyr[j]);
            }
        }

        for( j = 0; j < nsamples; j++ )
        {
            KeyPoint kpt = keypoints[j];
            float scale = 1.f/(1 << kpt.octave);
            Point2f pt((float)((M[0]*kpt.pt.x + M[1]*kpt.pt.y + M[2])*scale),
                       (float)((M[3]*kpt.pt.x + M[4]*kpt.pt.y + M[5])*scale));
            getRectSubPix(pyr[kpt.octave], patchSize, pt, patch, patch.type());
            for( int f = 0; f < nstructs; f++ )
                posteriors[getLeaf(f, patch)*nclasses + j]++;
        }

        if( verbose && (i+1)*progressBarSize/_nviews != i*progressBarSize/_nviews )
            putchar('.');
    }
    if( verbose )
        putchar('\n');

    finalize(rng);
}


int FernClassifier::operator()(const Mat& img, Point2f pt, std::vector<float>& signature) const
{
    Mat patch;
    getRectSubPix(img, patchSize, pt, patch, img.type());
    return (*this)(patch, signature);
}


int FernClassifier::operator()(const Mat& patch, std::vector<float>& signature) const
{
    if( posteriors.empty() )
        CV_Error(CV_StsNullPtr,
                 "The descriptor has not been trained or "
                 "the floating-point posteriors have been deleted");
    CV_Assert(patch.size() == patchSize);

    int i, j, sz = signatureSize;
    signature.resize(sz);
    float* s = &signature[0];

    for( j = 0; j < sz; j++ )
        s[j] = 0;

    for( i = 0; i < nstructs; i++ )
    {
        int lf = getLeaf(i, patch);
        const float* ldata = &posteriors[lf*signatureSize];
        for( j = 0; j <= sz - 4; j += 4 )
        {
            float t0 = s[j] + ldata[j];
            float t1 = s[j+1] + ldata[j+1];
            s[j] = t0; s[j+1] = t1;
            t0 = s[j+2] + ldata[j+2];
            t1 = s[j+3] + ldata[j+3];
            s[j+2] = t0; s[j+3] = t1;
        }
        for( ; j < sz; j++ )
            s[j] += ldata[j];
    }

    j = 0;
    if( signatureSize == nclasses && compressionMethod == COMPRESSION_NONE )
    {
        for( i = 1; i < nclasses; i++ )
            if( s[j] < s[i] )
                j = i;
    }
    return j;
}


void FernClassifier::finalize(RNG&)
{
    int i, j, k, n = nclasses;
    std::vector<double> invClassCounters(n);
    Mat_<double> _temp(1, n);
    double* temp = &_temp(0,0);

    for( i = 0; i < n; i++ )
        invClassCounters[i] = 1./classCounters[i];

    for( i = 0; i < nstructs; i++ )
    {
        for( j = 0; j < leavesPerStruct; j++ )
        {
            float* P = &posteriors[(i*leavesPerStruct + j)*nclasses];
            double sum = 0;
            for( k = 0; k < n; k++ )
                sum += P[k]*invClassCounters[k];
            sum = 1./sum;
            for( k = 0; k < n; k++ )
                temp[k] = P[k]*invClassCounters[k]*sum;
            log(_temp, _temp);
            for( k = 0; k < n; k++ )
                P[k] = (float)temp[k];
        }
    }

#if 0
    // do the first pass over the data.
    if( compressionMethod == COMPRESSION_RANDOM_PROJ )
    {
        // In case of random projection
        // we generate a random m x n matrix consisting of -1's and 1's
        // (called Bernoulli matrix) and multiply it by each vector
        // of posterior probabilities.
        // the product is stored back into the same input vector.

        Mat_<uchar> csmatrix;
        if( m < n )
        {
            // generate random Bernoulli matrix:
            //   -1's are replaced with 0's and 1's stay 1's.
            csmatrix.create(m, n);
            rng.fill(csmatrix, RNG::UNIFORM, Scalar::all(0), Scalar::all(2));
        }
        std::vector<float> dst(m);

        for( i = 0; i < totalLeaves; i++ )
        {
            int S = sampleCounters[i];
            if( S == 0 )
                continue;

            float scale = 1.f/(S*(m < n ? std::sqrt((float)m) : 1.f));
            const int* leaf = (const int*)&posteriors[i*n];
            float* out_leaf = (float*)&posteriors[i*m];

            for( j = 0; j < m; j++ )
            {
                float val = 0;
                if( m < n )
                {
                    const uchar* csrow = csmatrix.ptr(j);
                    // Because -1's in the Bernoulli matrix are encoded as 0's,
                    // the real dot product value will be
                    // A - B, where A is the sum of leaf[j]'s for which csrow[j]==1 and
                    // B is the sum of leaf[j]'s for which csrow[j]==0.
                    // since A + B = S, then A - B = A - (S - A) = 2*A - S.
                    int A = 0;
                    for( k = 0; k < n; k++ )
                        A += leaf[k] & -(int)csrow[k];
                    val = (A*2 - S)*scale;
                }
                else
                    val = leaf[j]*scale;
                dst[j] = val;
            }

            // put the vector back (since it's shorter than the original, we can do it in-place)
            for( j = 0; j < m; j++ )
                out_leaf[j] = dst[j];
        }
    }
    else if( compressionMethod == COMPRESSION_PCA )
    {
        // In case of PCA we do 3 passes over the data:
        //   first, we compute the mean vector
        //   second, we compute the covariation matrix
        //     then we do eigen decomposition of the matrix and construct the PCA
        //     projection matrix
        //   and on the third pass we actually do PCA compression.

        int nonEmptyLeaves = 0;
        Mat_<double> _mean(1, n), _vec(1, n), _dvec(m, 1),
        _cov(n, n), _evals(n, 1), _evects(n, n);
        _mean = 0.;
        double* mean = &_mean(0,0);
        double* vec = &_vec(0,0);
        double* dvec = &_dvec(0,0);

        for( i = 0; i < totalLeaves; i++ )
        {
            int S = sampleCounters[i];
            if( S == 0 )
                continue;
            float invS = 1.f/S;
            const int* leaf = (const int*)&posteriors[0] + i*n;
            float* out_leaf = (float*)&posteriors[0] + i*n;

            for( j = 0; j < n; j++ )
            {
                float t = leaf[j]*invS;
                out_leaf[j] = t;
                mean[j] += t;
            }
            nonEmptyLeaves++;
        }

        CV_Assert( nonEmptyLeaves >= ntrees );
        _mean *= 1./nonEmptyLeaves;

        for( i = 0; i < totalLeaves; i++ )
        {
            int S = sampleCounters[i];
            if( S == 0 )
                continue;
            const float* leaf = (const float*)&posteriors[0] + i*n;
            for( j = 0; j < n; j++ )
                vec[j] = leaf[j] - mean[j];
            gemm(_vec, _vec, 1, _cov, 1, _cov, GEMM_1_T);
        }

        _cov *= 1./nonEmptyLeaves;
        eigen(_cov, _evals, _evects);
        // use the first m eigenvectors (stored as rows of the eigenvector matrix)
        // as the projection matrix in PCA
        _evects = _evects(Range(0, m), Range::all());

        for( i = 0; i < totalLeaves; i++ )
        {
            int S = sampleCounters[i];
            if( S == 0 )
                continue;
            const float* leaf = (const float*)&posteriors[0] + i*n;
            float* out_leaf = (float*)&posteriors[0] + i*m;

            for( j = 0; j < n; j++ )
                vec[j] = leaf[j] - mean[j];
            gemm(_evects, _vec, 1, Mat(), 0, _dvec, GEMM_2_T);

            for( j = 0; j < m; j++ )
                out_leaf[j] = (float)dvec[j];
        }
    }
    else
        CV_Error( CV_StsBadArg,
                 "Unknown compression method; use COMPRESSION_RANDOM_PROJ or COMPRESSION_PCA" );

    // and shrink the vector
    posteriors.resize(totalLeaves*m);
#endif
}

void FernClassifier::setVerbose(bool _verbose)
{
    verbose = _verbose;
}


/****************************************************************************************\
*                                  FernDescriptorMatcher                                 *
\****************************************************************************************/
FernDescriptorMatcher::Params::Params( int _nclasses, int _patchSize, int _signatureSize,
                                      int _nstructs, int _structSize, int _nviews, int _compressionMethod,
                                      const PatchGenerator& _patchGenerator ) :
nclasses(_nclasses), patchSize(_patchSize), signatureSize(_signatureSize),
nstructs(_nstructs), structSize(_structSize), nviews(_nviews),
compressionMethod(_compressionMethod), patchGenerator(_patchGenerator)
{}

FernDescriptorMatcher::Params::Params( const String& _filename )
{
    filename = _filename;
}

FernDescriptorMatcher::FernDescriptorMatcher( const Params& _params )
{
    prevTrainCount = 0;
    params = _params;
    if( !params.filename.empty() )
    {
        classifier = makePtr<FernClassifier>();
        FileStorage fs(params.filename, FileStorage::READ);
        if( fs.isOpened() )
            classifier->read( fs.getFirstTopLevelNode() );
    }
}

FernDescriptorMatcher::~FernDescriptorMatcher()
{}

void FernDescriptorMatcher::clear()
{
    GenericDescriptorMatcher::clear();

    classifier.release();
    prevTrainCount = 0;
}

void FernDescriptorMatcher::train()
{
    if( !classifier || prevTrainCount < (int)trainPointCollection.keypointCount() )
    {
        assert( params.filename.empty() );

        std::vector<std::vector<Point2f> > points( trainPointCollection.imageCount() );
        for( size_t imgIdx = 0; imgIdx < trainPointCollection.imageCount(); imgIdx++ )
            KeyPoint::convert( trainPointCollection.getKeypoints((int)imgIdx), points[imgIdx] );

        classifier.reset(
            new FernClassifier( points, trainPointCollection.getImages(), std::vector<std::vector<int> >(), 0, // each points is a class
                                params.patchSize, params.signatureSize, params.nstructs, params.structSize,
                                params.nviews, params.compressionMethod, params.patchGenerator ));
    }
}

bool FernDescriptorMatcher::isMaskSupported()
{
    return false;
}

void FernDescriptorMatcher::calcBestProbAndMatchIdx( const Mat& image, const Point2f& pt,
                                                    float& bestProb, int& bestMatchIdx, std::vector<float>& signature )
{
    (*classifier)( image, pt, signature);

    bestProb = -FLT_MAX;
    bestMatchIdx = -1;
    for( int ci = 0; ci < classifier->getClassCount(); ci++ )
    {
        if( signature[ci] > bestProb )
        {
            bestProb = signature[ci];
            bestMatchIdx = ci;
        }
    }
}

void FernDescriptorMatcher::knnMatchImpl( InputArray _queryImage, std::vector<KeyPoint>& queryKeypoints,
                                         std::vector<std::vector<DMatch> >& matches, int knn,
                                         const std::vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    Mat queryImage = _queryImage.getMat();

    train();

    matches.resize( queryKeypoints.size() );
    std::vector<float> signature( (size_t)classifier->getClassCount() );

    for( size_t queryIdx = 0; queryIdx < queryKeypoints.size(); queryIdx++ )
    {
        (*classifier)( queryImage, queryKeypoints[queryIdx].pt, signature);

        for( int k = 0; k < knn; k++ )
        {
            DMatch bestMatch;
            size_t best_ci = 0;
            for( size_t ci = 0; ci < signature.size(); ci++ )
            {
                if( -signature[ci] < bestMatch.distance )
                {
                    int imgIdx = -1, trainIdx = -1;
                    trainPointCollection.getLocalIdx( (int)ci , imgIdx, trainIdx );
                    bestMatch = DMatch( (int)queryIdx, trainIdx, imgIdx, -signature[ci] );
                    best_ci = ci;
                }
            }

            if( bestMatch.trainIdx == -1 )
                break;
            signature[best_ci] = -std::numeric_limits<float>::max();
            matches[queryIdx].push_back( bestMatch );
        }
    }
}

void FernDescriptorMatcher::radiusMatchImpl( InputArray _queryImage, std::vector<KeyPoint>& queryKeypoints,
                                            std::vector<std::vector<DMatch> >& matches, float maxDistance,
                                            const std::vector<Mat>& /*masks*/, bool /*compactResult*/ )
{
    Mat queryImage = _queryImage.getMat();
    train();
    matches.resize( queryKeypoints.size() );
    std::vector<float> signature( (size_t)classifier->getClassCount() );

    for( size_t i = 0; i < queryKeypoints.size(); i++ )
    {
        (*classifier)( queryImage, queryKeypoints[i].pt, signature);

        for( int ci = 0; ci < classifier->getClassCount(); ci++ )
        {
            if( -signature[ci] < maxDistance )
            {
                int imgIdx = -1, trainIdx = -1;
                trainPointCollection.getLocalIdx( ci , imgIdx, trainIdx );
                matches[i].push_back( DMatch( (int)i, trainIdx, imgIdx, -signature[ci] ) );
            }
        }
    }
}

void FernDescriptorMatcher::read( const FileNode &fn )
{
    params.nclasses = fn["nclasses"];
    params.patchSize = fn["patchSize"];
    params.signatureSize = fn["signatureSize"];
    params.nstructs = fn["nstructs"];
    params.structSize = fn["structSize"];
    params.nviews = fn["nviews"];
    params.compressionMethod = fn["compressionMethod"];

    //classifier->read(fn);
}

void FernDescriptorMatcher::write( FileStorage& fs ) const
{
    fs << "nclasses" << params.nclasses;
    fs << "patchSize" << params.patchSize;
    fs << "signatureSize" << params.signatureSize;
    fs << "nstructs" << params.nstructs;
    fs << "structSize" << params.structSize;
    fs << "nviews" << params.nviews;
    fs << "compressionMethod" << params.compressionMethod;

    //    classifier->write(fs);
}

bool FernDescriptorMatcher::empty() const
{
    return !classifier || classifier->empty();
}

Ptr<GenericDescriptorMatcher> FernDescriptorMatcher::clone( bool emptyTrainData ) const
{
    Ptr<FernDescriptorMatcher> matcher = makePtr<FernDescriptorMatcher>( params );
    if( !emptyTrainData )
    {
        CV_Error( CV_StsNotImplemented, "deep clone dunctionality is not implemented, because "
                 "FernClassifier has not copy constructor or clone method ");

        //matcher->classifier;
        matcher->params = params;
        matcher->prevTrainCount = prevTrainCount;
        matcher->trainPointCollection = trainPointCollection;
    }
    return matcher;
}

////////////////////////////////////// Planar Object Detector ////////////////////////////////////

PlanarObjectDetector::PlanarObjectDetector()
{
}

PlanarObjectDetector::PlanarObjectDetector(const FileNode& node)
{
    read(node);
}

PlanarObjectDetector::PlanarObjectDetector(const std::vector<Mat>& pyr, int npoints,
                                           int patchSize, int nstructs, int structSize,
                                           int nviews, const LDetector& detector,
                                           const PatchGenerator& patchGenerator)
{
    train(pyr, npoints, patchSize, nstructs,
          structSize, nviews, detector, patchGenerator);
}

PlanarObjectDetector::~PlanarObjectDetector()
{
}

std::vector<KeyPoint> PlanarObjectDetector::getModelPoints() const
{
    return modelPoints;
}

void PlanarObjectDetector::train(const std::vector<Mat>& pyr, int npoints,
                                 int patchSize, int nstructs, int structSize,
                                 int nviews, const LDetector& detector,
                                 const PatchGenerator& patchGenerator)
{
    modelROI = Rect(0, 0, pyr[0].cols, pyr[0].rows);
    ldetector = detector;
    ldetector.setVerbose(verbose);
    ldetector.getMostStable2D(pyr[0], modelPoints, npoints, patchGenerator);

    npoints = (int)modelPoints.size();
    fernClassifier.setVerbose(verbose);
    fernClassifier.trainFromSingleView(pyr[0], modelPoints,
                                       patchSize, (int)modelPoints.size(), nstructs, structSize, nviews,
                                       FernClassifier::COMPRESSION_NONE, patchGenerator);
}

void PlanarObjectDetector::train(const std::vector<Mat>& pyr, const std::vector<KeyPoint>& keypoints,
                                 int patchSize, int nstructs, int structSize,
                                 int nviews, const LDetector& detector,
                                 const PatchGenerator& patchGenerator)
{
    modelROI = Rect(0, 0, pyr[0].cols, pyr[0].rows);
    ldetector = detector;
    ldetector.setVerbose(verbose);
    modelPoints.resize(keypoints.size());
    std::copy(keypoints.begin(), keypoints.end(), modelPoints.begin());

    fernClassifier.setVerbose(verbose);
    fernClassifier.trainFromSingleView(pyr[0], modelPoints,
                                       patchSize, (int)modelPoints.size(), nstructs, structSize, nviews,
                                       FernClassifier::COMPRESSION_NONE, patchGenerator);
}

void PlanarObjectDetector::read(const FileNode& node)
{
    FileNodeIterator it = node["model-roi"].begin(), it_end;
    it >> modelROI.x >> modelROI.y >> modelROI.width >> modelROI.height;
    ldetector.read(node["detector"]);
    fernClassifier.read(node["fern-classifier"]);
    cv::read(node["model-points"], modelPoints);
    CV_Assert(modelPoints.size() == (size_t)fernClassifier.getClassCount());
}


void PlanarObjectDetector::write(FileStorage& fs, const String& objname) const
{
    internal::WriteStructContext ws(fs, objname, CV_NODE_MAP);

    {
        internal::WriteStructContext wsroi(fs, "model-roi", CV_NODE_SEQ + CV_NODE_FLOW);
        cv::write(fs, modelROI.x);
        cv::write(fs, modelROI.y);
        cv::write(fs, modelROI.width);
        cv::write(fs, modelROI.height);
    }
    ldetector.write(fs, "detector");
    cv::write(fs, "model-points", modelPoints);
    fernClassifier.write(fs, "fern-classifier");
}


bool PlanarObjectDetector::operator()(const Mat& image, Mat& H, std::vector<Point2f>& corners) const
{
    std::vector<Mat> pyr;
    buildPyramid(image, pyr, ldetector.nOctaves - 1);
    std::vector<KeyPoint> keypoints;
    ldetector(pyr, keypoints);

    return (*this)(pyr, keypoints, H, corners);
}

bool PlanarObjectDetector::operator()(const std::vector<Mat>& pyr, const std::vector<KeyPoint>& keypoints,
                                      Mat& matH, std::vector<Point2f>& corners, std::vector<int>* pairs) const
{
    int i, j, m = (int)modelPoints.size(), n = (int)keypoints.size();
    std::vector<int> bestMatches(m, -1);
    std::vector<float> maxLogProb(m, -FLT_MAX);
    std::vector<float> signature;
    std::vector<Point2f> fromPt, toPt;

    for( i = 0; i < n; i++ )
    {
        KeyPoint kpt = keypoints[i];
        CV_Assert(0 <= kpt.octave && kpt.octave < (int)pyr.size());
        kpt.pt.x /= (float)(1 << kpt.octave);
        kpt.pt.y /= (float)(1 << kpt.octave);
        int k = fernClassifier(pyr[kpt.octave], kpt.pt, signature);
        if( k >= 0 && (bestMatches[k] < 0 || signature[k] > maxLogProb[k]) )
        {
            maxLogProb[k] = signature[k];
            bestMatches[k] = i;
        }
    }

    if(pairs)
        pairs->resize(0);

    for( i = 0; i < m; i++ )
        if( bestMatches[i] >= 0 )
        {
            fromPt.push_back(modelPoints[i].pt);
            toPt.push_back(keypoints[bestMatches[i]].pt);
        }

    if( fromPt.size() < 4 )
        return false;

    std::vector<uchar> mask;
    matH = findHomography(fromPt, toPt, RANSAC, 10, mask);
    if( matH.data )
    {
        const Mat_<double>& H = matH;
        corners.resize(4);
        for( i = 0; i < 4; i++ )
        {
            Point2f pt((float)(modelROI.x + (i == 0 || i == 3 ? 0 : modelROI.width)),
                       (float)(modelROI.y + (i <= 1 ? 0 : modelROI.height)));
            double w = 1./(H(2,0)*pt.x + H(2,1)*pt.y + H(2,2));
            corners[i] = Point2f((float)((H(0,0)*pt.x + H(0,1)*pt.y + H(0,2))*w),
                                 (float)((H(1,0)*pt.x + H(1,1)*pt.y + H(1,2))*w));
        }
    }

    if( pairs )
    {
        for( i = j = 0; i < m; i++ )
            if( bestMatches[i] >= 0 && mask[j++] )
            {
                pairs->push_back(i);
                pairs->push_back(bestMatches[i]);
            }
    }

    return matH.data != 0;
}


void PlanarObjectDetector::setVerbose(bool _verbose)
{
    verbose = _verbose;
}

}
