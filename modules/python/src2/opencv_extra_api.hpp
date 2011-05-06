#ifndef _OPENCV_API_EXTRA_HPP_
#define _OPENCV_API_EXTRA_HPP_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/calib3d/calib3d.hpp"

namespace cv
{

template<typename _Tp>
static inline void mv2vv(const vector<Mat>& src, vector<vector<_Tp> >& dst)
{
    size_t i, n = src.size();
    dst.resize(src.size());
    for( i = 0; i < n; i++ )
        src[i].copyTo(dst[i]);
}

///////////////////////////// core /////////////////////////////

CV_WRAP_AS(getTickCount) static inline double getTickCount_()
{
    return (double)getTickCount();
}

CV_WRAP_AS(getCPUTickCount) static inline double getCPUTickCount_()
{
    return (double)getCPUTickCount();
}

CV_WRAP void randShuffle(const Mat& src, CV_OUT Mat& dst, double iterFactor=1.)
{
    src.copyTo(dst);
    randShuffle(dst, iterFactor, 0);
}

CV_WRAP static inline void SVDecomp(const Mat& src, CV_OUT Mat& w, CV_OUT Mat& u, CV_OUT Mat& vt, int flags=0 )
{
    SVD::compute(src, w, u, vt, flags);
}

CV_WRAP static inline void SVBackSubst( const Mat& w, const Mat& u, const Mat& vt,
                                        const Mat& rhs, CV_OUT Mat& dst )
{
    SVD::backSubst(w, u, vt, rhs, dst);
}

CV_WRAP static inline void mixChannels(const vector<Mat>& src, vector<Mat>& dst,
                                       const vector<int>& fromTo)
{
    if(fromTo.empty())
        return;
    CV_Assert(fromTo.size()%2 == 0);
    mixChannels(&src[0], (int)src.size(), &dst[0], (int)dst.size(), &fromTo[0], (int)(fromTo.size()/2));
}

CV_WRAP static inline bool eigen(const Mat& src, bool computeEigenvectors,
                                 CV_OUT Mat& eigenvalues, CV_OUT Mat& eigenvectors,
                                 int lowindex=-1, int highindex=-1)
{
    return computeEigenvectors ? eigen(src, eigenvalues, eigenvectors, lowindex, highindex) :
                                eigen(src, eigenvalues, lowindex, highindex);
}

CV_WRAP static inline void fillConvexPoly(Mat& img, const Mat& points,
                               const Scalar& color, int lineType=8,
                               int shift=0)
{
    CV_Assert(points.checkVector(2, CV_32S) >= 0);
    fillConvexPoly(img, (const Point*)points.data, points.rows*points.cols*points.channels()/2, color, lineType, shift);
}

CV_WRAP static inline void fillPoly(Mat& img, const vector<Mat>& pts,
                                   const Scalar& color, int lineType=8, int shift=0,
                                   Point offset=Point() )
{
    if( pts.empty() )
        return;
    AutoBuffer<Point*> _ptsptr(pts.size());
    AutoBuffer<int> _npts(pts.size());
    Point** ptsptr = _ptsptr;
    int* npts = _npts;
    
    for( size_t i = 0; i < pts.size(); i++ )
    {
        const Mat& p = pts[i];
        CV_Assert(p.checkVector(2, CV_32S) >= 0);
        ptsptr[i] = (Point*)p.data;
        npts[i] = p.rows*p.cols*p.channels()/2;
    }
    fillPoly(img, (const Point**)ptsptr, npts, (int)pts.size(), color, lineType, shift, offset);
}

CV_WRAP static inline void polylines(Mat& img, const vector<Mat>& pts,
                          bool isClosed, const Scalar& color,
                          int thickness=1, int lineType=8, int shift=0 )
{
    if( pts.empty() )
        return;
    AutoBuffer<Point*> _ptsptr(pts.size());
    AutoBuffer<int> _npts(pts.size());
    Point** ptsptr = _ptsptr;
    int* npts = _npts;
    
    for( size_t i = 0; i < pts.size(); i++ )
    {
        const Mat& p = pts[i];
        CV_Assert(p.checkVector(2, CV_32S) >= 0);
        ptsptr[i] = (Point*)p.data;
        npts[i] = p.rows*p.cols*p.channels()/2;
    }
    polylines(img, (const Point**)ptsptr, npts, (int)pts.size(), isClosed, color, thickness, lineType, shift);
}

CV_WRAP static inline void PCACompute(const Mat& data, CV_OUT Mat& mean,
                                      CV_OUT Mat& eigenvectors, int maxComponents=0)
{
    PCA pca;
    pca.mean = mean;
    pca.eigenvectors = eigenvectors;
    pca(data, Mat(), 0, maxComponents);
    pca.mean.copyTo(mean);
    pca.eigenvectors.copyTo(eigenvectors);
}
    
CV_WRAP static inline void PCAProject(const Mat& data, const Mat& mean,
                                      const Mat& eigenvectors, CV_OUT Mat& result)
{
    PCA pca;
    pca.mean = mean;
    pca.eigenvectors = eigenvectors;
    pca.project(data, result);
}

CV_WRAP static inline void PCABackProject(const Mat& data, const Mat& mean,
                                          const Mat& eigenvectors, CV_OUT Mat& result)
{
    PCA pca;
    pca.mean = mean;
    pca.eigenvectors = eigenvectors;
    pca.backProject(data, result);
}    

/////////////////////////// imgproc /////////////////////////////////

CV_WRAP static inline void HuMoments(const Moments& m, CV_OUT vector<double>& hu)
{
    hu.resize(7);
    HuMoments(m, &hu[0]);
}

CV_WRAP static inline Mat getPerspectiveTransform(const vector<Point2f>& src, const vector<Point2f>& dst)
{
    CV_Assert(src.size() == 4 && dst.size() == 4);
    return getPerspectiveTransform(&src[0], &dst[0]);
}

CV_WRAP static inline Mat getAffineTransform(const vector<Point2f>& src, const vector<Point2f>& dst)
{
    CV_Assert(src.size() == 3 && dst.size() == 3);
    return getAffineTransform(&src[0], &dst[0]);
}

CV_WRAP static inline void calcHist( const vector<Mat>& images, const vector<int>& channels,
                                     const Mat& mask, CV_OUT Mat& hist,
                                     const vector<int>& histSize,
                                     const vector<float>& ranges,
                                     bool accumulate=false)
{
    int i, dims = (int)histSize.size(), rsz = (int)ranges.size(), csz = (int)channels.size();
    CV_Assert(images.size() > 0 && dims > 0);
    CV_Assert(rsz == dims*2 || (rsz == 0 && images[0].depth() == CV_8U));
    CV_Assert(csz == 0 || csz == dims);
    float* _ranges[CV_MAX_DIM];
    if( rsz > 0 )
    {
        for( i = 0; i < rsz/2; i++ )
            _ranges[i] = (float*)&ranges[i*2];
    }
    calcHist(&images[0], (int)images.size(), csz ? &channels[0] : 0,
            mask, hist, dims, &histSize[0], rsz ? (const float**)_ranges : 0,
            true, accumulate);
}

                                     
CV_WRAP void calcBackProject( const vector<Mat>& images, const vector<int>& channels,
                              const Mat& hist, CV_OUT Mat& dst,
                              const vector<float>& ranges,
                              double scale=1 )
{
    int i, dims = hist.dims, rsz = (int)ranges.size(), csz = (int)channels.size();
    CV_Assert(images.size() > 0);
    CV_Assert(rsz == dims*2 || (rsz == 0 && images[0].depth() == CV_8U));
    CV_Assert(csz == 0 || csz == dims);
    float* _ranges[CV_MAX_DIM];
    if( rsz > 0 )
    {
        for( i = 0; i < rsz/2; i++ )
            _ranges[i] = (float*)&ranges[i*2];
    }
    calcBackProject(&images[0], (int)images.size(), csz ? &channels[0] : 0,
        hist, dst, rsz ? (const float**)_ranges : 0, scale, true);
}


static void addChildContour(const vector<Mat>& contours,
                            const Mat& hierarchy,
                            int i, vector<CvSeq>& seq,
                            vector<CvSeqBlock>& block)
{
    size_t count = contours.size();
    for( ; i >= 0; i = ((const Vec4i*)hierarchy.data)[i][0] )
    {
        const vector<Point>& ci = contours[i];
        cvMakeSeqHeaderForArray(CV_SEQ_POLYGON, sizeof(CvSeq), sizeof(Point),
                                !ci.empty() ? (void*)&ci[0] : 0, (int)ci.size(),
                                &seq[i], &block[i] );
        const Vec4i h_i = ((const Vec4i*)hierarchy.data)[i];
        int h_next = h_i[0], h_prev = h_i[1], v_next = h_i[2], v_prev = h_i[3];
        
        seq[i].h_next = (size_t)h_next < count ? &seq[h_next] : 0;
        seq[i].h_prev = (size_t)h_prev < count ? &seq[h_prev] : 0;
        seq[i].v_next = (size_t)v_next < count ? &seq[v_next] : 0;
        seq[i].v_prev = (size_t)v_prev < count ? &seq[v_prev] : 0;
        
        if( v_next >= 0 )
            addChildContour(contours, hierarchy, v_next, seq, block);
    }
}

//! draws contours in the image
CV_WRAP static inline void drawContours( Mat& image, const vector<Mat>& contours,
                                int contourIdx, const Scalar& color,
                                int thickness=1, int lineType=8,
                                const Mat& hierarchy=Mat(),
                                int maxLevel=INT_MAX, Point offset=Point() )
{
    CvMat _image = image;

    size_t i = 0, first = 0, last = contours.size();
    vector<CvSeq> seq;
    vector<CvSeqBlock> block;

    if( !last )
        return;

    seq.resize(last);
    block.resize(last);

    for( i = first; i < last; i++ )
        seq[i].first = 0;

    if( contourIdx >= 0 )
    {
        CV_Assert( 0 <= contourIdx && contourIdx < (int)last );
        first = contourIdx;
        last = contourIdx + 1;
    }

    for( i = first; i < last; i++ )
    {
        const Mat& ci = contours[i];
        int ci_size = ci.checkVector(2, CV_32S);
        CV_Assert( ci_size >= 0 );
        cvMakeSeqHeaderForArray(CV_SEQ_POLYGON, sizeof(CvSeq), sizeof(Point),
                                ci_size > 0 ? ci.data : 0, ci_size, &seq[i], &block[i] );
    }

    if( hierarchy.empty() || maxLevel == 0 )
        for( i = first; i < last; i++ )
        {
            seq[i].h_next = i < last-1 ? &seq[i+1] : 0;
            seq[i].h_prev = i > first ? &seq[i-1] : 0;
        }
    else
    {
        int hsz = hierarchy.checkVector(4, CV_32S);
        size_t count = last - first;
        CV_Assert((size_t)hsz == contours.size());
        if( count == contours.size() )
        {
            for( i = first; i < last; i++ )
            {
                const Vec4i& h_i = ((const Vec4i*)hierarchy.data)[i];
                int h_next = h_i[0], h_prev = h_i[1], v_next = h_i[2], v_prev = h_i[3];

                seq[i].h_next = (size_t)h_next < count ? &seq[h_next] : 0;
                seq[i].h_prev = (size_t)h_prev < count ? &seq[h_prev] : 0;
                seq[i].v_next = (size_t)v_next < count ? &seq[v_next] : 0;
                seq[i].v_prev = (size_t)v_prev < count ? &seq[v_prev] : 0;
            }
        }
        else
        {
            int child = ((const Vec4i*)hierarchy.data)[first][2];
            if( child >= 0 )
            {
                addChildContour(contours, hierarchy, child, seq, block);
                seq[first].v_next = &seq[child];
            }
        }
    }

    cvDrawContours( &_image, &seq[first], color, color, contourIdx >= 0 ?
                   -maxLevel : maxLevel, thickness, lineType, offset );
}


CV_WRAP static inline void approxPolyDP( const Mat& curve,
                              CV_OUT Mat& approxCurve,
                              double epsilon, bool closed )
{
    if( curve.depth() == CV_32S )
    {
        vector<Point> result;
        approxPolyDP(curve, result, epsilon, closed);
        Mat(result).copyTo(approxCurve);
    }
    else if( curve.depth() == CV_32F )
    {
        vector<Point2f> result;
        approxPolyDP(curve, result, epsilon, closed);
        Mat(result).copyTo(approxCurve);
    }
    else
        CV_Error(CV_StsUnsupportedFormat, "");
}


CV_WRAP static inline void convexHull( const Mat& points, CV_OUT Mat& hull, bool returnPoints=true, bool clockwise=false )
{
    if( !returnPoints )
    {
        vector<int> h;
        convexHull(points, h, clockwise);
        Mat(h).copyTo(hull);
    }
    else if( points.depth() == CV_32S )
    {
        vector<Point> h;
        convexHull(points, h, clockwise);
        Mat(h).copyTo(hull);
    }
    else if( points.depth() == CV_32F )
    {
        vector<Point2f> h;
        convexHull(points, h, clockwise);
        Mat(h).copyTo(hull);
    }
}

CV_WRAP static inline void fitLine( const Mat& points, CV_OUT vector<float>& line,
                      int distType, double param, double reps, double aeps )
{
    if(points.channels() == 2 || points.cols == 2)
    {
        line.resize(4);
        fitLine(points, *(Vec4f*)&line[0], distType, param, reps, aeps);
    }
    else
    {
        line.resize(6);
        fitLine(points, *(Vec6f*)&line[0], distType, param, reps, aeps);
    }
}

CV_WRAP static inline int estimateAffine3D( const Mat& from, const Mat& to,
                              CV_OUT Mat& dst, CV_OUT Mat& outliers,
                              double param1 = 3.0, double param2 = 0.99 )
{
    vector<uchar> outliers_vec;
    int res = estimateAffine3D(from, to, dst, outliers_vec, param1, param2);
    Mat(outliers_vec).copyTo(outliers);
    return res;
}


CV_WRAP static inline void cornerSubPix( const Mat& image, Mat& corners,
                                         Size winSize, Size zeroZone,
                                         TermCriteria criteria )
{
    int n = corners.checkVector(2, CV_32F);
    CV_Assert(n >= 0);
    
    if( n == 0 )
        return;
        
    CvMat _image = image;
    cvFindCornerSubPix(&_image, (CvPoint2D32f*)corners.data, n, winSize, zeroZone, criteria);
}

/////////////////////////////// calib3d ///////////////////////////////////////////

CV_WRAP static inline void convertPointsHomogeneous( const Mat& src, CV_OUT Mat& dst )
{
    int n;
    if( (n = src.checkVector(2)) >= 0 )
        dst.create(n, 2, src.depth());
    else if( (n = src.checkVector(3)) >= 0 )
        dst.create(n, 3, src.depth());
    else
        CV_Error(CV_StsBadSize, "");
    CvMat _src = src, _dst = dst;
    cvConvertPointsHomogeneous(&_src, &_dst);
}

//! finds circles' grid pattern of the specified size in the image
CV_WRAP static inline void findCirclesGridDefault( const InputArray& image, Size patternSize,
                                                   OutputArray centers, int flags=CALIB_CB_SYMMETRIC_GRID )
{
    findCirclesGrid(image, patternSize, centers, flags);
}


/*
//! initializes camera matrix from a few 3D points and the corresponding projections.
CV_WRAP static inline Mat initCameraMatrix2D( const vector<Mat>& objectPoints,
                                   const vector<Mat>& imagePoints,
                                   Size imageSize, double aspectRatio=1. )
{
    vector<vector<Point3f> > _objectPoints;
    vector<vector<Point2f> > _imagePoints;
    mv2vv(objectPoints, _objectPoints);
    mv2vv(imagePoints, _imagePoints);
    return initCameraMatrix2D(_objectPoints, _imagePoints, imageSize, aspectRatio);
}


CV_WRAP static inline double calibrateCamera( const vector<Mat>& objectPoints,
                                  const vector<Mat>& imagePoints,
                                  Size imageSize,
                                  CV_IN_OUT Mat& cameraMatrix,
                                  CV_IN_OUT Mat& distCoeffs,
                                  vector<Mat>& rvecs, vector<Mat>& tvecs,
                                  int flags=0 )
{
    vector<vector<Point3f> > _objectPoints;
    vector<vector<Point2f> > _imagePoints;
    mv2vv(objectPoints, _objectPoints);
    mv2vv(imagePoints, _imagePoints);
    return calibrateCamera(_objectPoints, _imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags);
}


CV_WRAP static inline double stereoCalibrate( const vector<Mat>& objectPoints,
                                  const vector<Mat>& imagePoints1,
                                  const vector<Mat>& imagePoints2,
                                  CV_IN_OUT Mat& cameraMatrix1, CV_IN_OUT Mat& distCoeffs1,
                                  CV_IN_OUT Mat& cameraMatrix2, CV_IN_OUT Mat& distCoeffs2,
                                  Size imageSize, CV_OUT Mat& R, CV_OUT Mat& T,
                                  CV_OUT Mat& E, CV_OUT Mat& F,
                                  TermCriteria criteria = TermCriteria(TermCriteria::COUNT+
                                                                       TermCriteria::EPS, 30, 1e-6),
                                  int flags=CALIB_FIX_INTRINSIC )
{
    vector<vector<Point3f> > _objectPoints;
    vector<vector<Point2f> > _imagePoints1;
    vector<vector<Point2f> > _imagePoints2;
    mv2vv(objectPoints, _objectPoints);
    mv2vv(imagePoints1, _imagePoints1);
    mv2vv(imagePoints2, _imagePoints2);
    return stereoCalibrate(_objectPoints, _imagePoints1, _imagePoints2, cameraMatrix1, distCoeffs1,
                            cameraMatrix2, distCoeffs2, imageSize, R, T, E, F, criteria, flags);
}

CV_WRAP static inline float rectify3Collinear( const Mat& cameraMatrix1, const Mat& distCoeffs1,
                     const Mat& cameraMatrix2, const Mat& distCoeffs2,
                     const Mat& cameraMatrix3, const Mat& distCoeffs3,
                     const vector<Mat>& imgpt1, const vector<Mat>& imgpt3,
                     Size imageSize, const Mat& R12, const Mat& T12,
                     const Mat& R13, const Mat& T13,
                     CV_OUT Mat& R1, CV_OUT Mat& R2, CV_OUT Mat& R3,
                     CV_OUT Mat& P1, CV_OUT Mat& P2, CV_OUT Mat& P3, CV_OUT Mat& Q,
                     double alpha, Size newImgSize,
                     CV_OUT Rect* roi1, CV_OUT Rect* roi2, int flags )
{
    vector<vector<Point2f> > _imagePoints1;
    vector<vector<Point2f> > _imagePoints3;
    mv2vv(imgpt1, _imagePoints1);
    mv2vv(imgpt3, _imagePoints3);
    return rectify3Collinear(cameraMatrix1, distCoeffs1,
                            cameraMatrix2, distCoeffs2,
                            cameraMatrix3, distCoeffs3,
                            _imagePoints1, _imagePoints3, imageSize,
                            R12, T12, R13, T13, R1, R2, R3, P1, P2, P3,
                            Q, alpha, newImgSize, roi1, roi2, flags);
}
*/
 
}

#endif
