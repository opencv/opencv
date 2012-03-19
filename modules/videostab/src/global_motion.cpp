#include "precomp.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videostab/global_motion.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

static Mat estimateGlobMotionLeastSquaresTranslation(
        int npoints, const Point2f *points0, const Point2f *points1, float *rmse)
{
    Mat_<float> M = Mat::eye(3, 3, CV_32F);
    for (int i = 0; i < npoints; ++i)
    {
        M(0,2) += points1[i].x - points0[i].x;
        M(1,2) += points1[i].y - points0[i].y;
    }
    M(0,2) /= npoints;
    M(1,2) /= npoints;
    if (rmse)
    {
        *rmse = 0;
        for (int i = 0; i < npoints; ++i)
            *rmse += sqr(points1[i].x - points0[i].x - M(0,2)) +
                     sqr(points1[i].y - points0[i].y - M(1,2));
        *rmse = sqrt(*rmse / npoints);
    }
    return M;
}


static Mat estimateGlobMotionLeastSquaresTranslationAndScale(
        int npoints, const Point2f *points0, const Point2f *points1, float *rmse)
{
    Mat_<float> A(2*npoints, 3), b(2*npoints, 1);
    float *a0, *a1;
    Point2f p0, p1;

    for (int i = 0; i < npoints; ++i)
    {
        a0 = A[2*i];
        a1 = A[2*i+1];
        p0 = points0[i];
        p1 = points1[i];
        a0[0] = p0.x; a0[1] = 1; a0[2] = 0;
        a1[0] = p0.y; a1[1] = 0; a1[2] = 1;
        b(2*i,0) = p1.x;
        b(2*i+1,0) = p1.y;
    }

    Mat_<float> sol;
    solve(A, b, sol, DECOMP_SVD);

    if (rmse)
        *rmse = norm(A*sol, b, NORM_L2) / sqrt(npoints);

    Mat_<float> M = Mat::eye(3, 3, CV_32F);
    M(0,0) = M(1,1) = sol(0,0);
    M(0,2) = sol(1,0);
    M(1,2) = sol(2,0);
    return M;
}


static Mat estimateGlobMotionLeastSquaresAffine(
        int npoints, const Point2f *points0, const Point2f *points1, float *rmse)
{
    Mat_<float> A(2*npoints, 6), b(2*npoints, 1);
    float *a0, *a1;
    Point2f p0, p1;

    for (int i = 0; i < npoints; ++i)
    {
        a0 = A[2*i];
        a1 = A[2*i+1];
        p0 = points0[i];
        p1 = points1[i];
        a0[0] = p0.x; a0[1] = p0.y; a0[2] = 1; a0[3] = a0[4] = a0[5] = 0;
        a1[0] = a1[1] = a1[2] = 0; a1[3] = p0.x; a1[4] = p0.y; a1[5] = 1;
        b(2*i,0) = p1.x;
        b(2*i+1,0) = p1.y;
    }

    Mat_<float> sol;
    solve(A, b, sol, DECOMP_SVD);

    if (rmse)
        *rmse = norm(A*sol, b, NORM_L2) / sqrt(npoints);

    Mat_<float> M = Mat::eye(3, 3, CV_32F);
    for (int i = 0, k = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j, ++k)
            M(i,j) = sol(k,0);

    return M;
}


Mat estimateGlobalMotionLeastSquares(
        const vector<Point2f> &points0, const vector<Point2f> &points1, int model, float *rmse)
{
    CV_Assert(points0.size() == points1.size());

    typedef Mat (*Impl)(int, const Point2f*, const Point2f*, float*);
    static Impl impls[] = { estimateGlobMotionLeastSquaresTranslation,
                            estimateGlobMotionLeastSquaresTranslationAndScale,
                            estimateGlobMotionLeastSquaresAffine };

    int npoints = static_cast<int>(points0.size());
    return impls[model](npoints, &points0[0], &points1[0], rmse);
}


Mat estimateGlobalMotionRobust(
        const vector<Point2f> &points0, const vector<Point2f> &points1, int model, const RansacParams &params,
        float *rmse, int *ninliers)
{
    CV_Assert(points0.size() == points1.size());

    typedef Mat (*Impl)(int, const Point2f*, const Point2f*, float*);
    static Impl impls[] = { estimateGlobMotionLeastSquaresTranslation,
                            estimateGlobMotionLeastSquaresTranslationAndScale,
                            estimateGlobMotionLeastSquaresAffine };

    const int npoints = static_cast<int>(points0.size());
    const int niters = static_cast<int>(ceil(log(1 - params.prob) / log(1 - pow(1 - params.eps, params.size))));

    RNG rng(0);
    vector<int> indices(params.size);
    vector<Point2f> subset0(params.size), subset1(params.size);
    vector<Point2f> subset0best(params.size), subset1best(params.size);
    Mat_<float> bestM;
    int ninliersMax = -1;
    Point2f p0, p1;
    float x, y;

    for (int iter = 0; iter < niters; ++iter)
    {
        for (int i = 0; i < params.size; ++i)
        {
            bool ok = false;
            while (!ok)
            {
                ok = true;
                indices[i] = static_cast<unsigned>(rng) % npoints;
                for (int j = 0; j < i; ++j)
                    if (indices[i] == indices[j])
                        { ok = false; break; }
            }
        }
        for (int i = 0; i < params.size; ++i)
        {
            subset0[i] = points0[indices[i]];
            subset1[i] = points1[indices[i]];
        }

        Mat_<float> M = impls[model](params.size, &subset0[0], &subset1[0], 0);

        int ninliers = 0;
        for (int i = 0; i < npoints; ++i)
        {
            p0 = points0[i]; p1 = points1[i];
            x = M(0,0)*p0.x + M(0,1)*p0.y + M(0,2);
            y = M(1,0)*p0.x + M(1,1)*p0.y + M(1,2);
            if (sqr(x - p1.x) + sqr(y - p1.y) < params.thresh * params.thresh)
                ninliers++;
        }
        if (ninliers >= ninliersMax)
        {
            bestM = M;
            ninliersMax = ninliers;
            subset0best.swap(subset0);
            subset1best.swap(subset1);
        }
    }

    if (ninliersMax < params.size)
        // compute rmse
        bestM = impls[model](params.size, &subset0best[0], &subset1best[0], rmse);
    else
    {
        subset0.resize(ninliersMax);
        subset1.resize(ninliersMax);
        for (int i = 0, j = 0; i < npoints; ++i)
        {
            p0 = points0[i]; p1 = points1[i];
            x = bestM(0,0)*p0.x + bestM(0,1)*p0.y + bestM(0,2);
            y = bestM(1,0)*p0.x + bestM(1,1)*p0.y + bestM(1,2);
            if (sqr(x - p1.x) + sqr(y - p1.y) < params.thresh * params.thresh)
            {
                subset0[j] = p0;
                subset1[j] = p1;
                j++;
            }
        }
        bestM = impls[model](ninliersMax, &subset0[0], &subset1[0], rmse);
    }

    if (ninliers)
        *ninliers = ninliersMax;

    return bestM;
}


PyrLkRobustMotionEstimator::PyrLkRobustMotionEstimator() : ransacParams_(RansacParams::affine2dMotionStd())
{
    setDetector(new GoodFeaturesToTrackDetector());
    setOptFlowEstimator(new SparsePyrLkOptFlowEstimator());
    setMotionModel(AFFINE);
    setMaxRmse(0.5f);
    setMinInlierRatio(0.1f);
}


Mat PyrLkRobustMotionEstimator::estimate(const Mat &frame0, const Mat &frame1)
{
    detector_->detect(frame0, keypointsPrev_);

    pointsPrev_.resize(keypointsPrev_.size());
    for (size_t i = 0; i < keypointsPrev_.size(); ++i)
        pointsPrev_[i] = keypointsPrev_[i].pt;

    optFlowEstimator_->run(frame0, frame1, pointsPrev_, points_, status_, noArray());

    size_t npoints = points_.size();
    pointsPrevGood_.clear();
    pointsPrevGood_.reserve(npoints);
    pointsGood_.clear();
    pointsGood_.reserve(npoints);
    for (size_t i = 0; i < npoints; ++i)
    {
        if (status_[i])
        {
            pointsPrevGood_.push_back(pointsPrev_[i]);
            pointsGood_.push_back(points_[i]);
        }
    }

    float rmse;
    int ninliers;
    Mat M = estimateGlobalMotionRobust(
            pointsPrevGood_, pointsGood_, motionModel_, ransacParams_, &rmse, &ninliers);

    if (rmse > maxRmse_ || static_cast<float>(ninliers) / pointsGood_.size() < minInlierRatio_)
        M = Mat::eye(3, 3, CV_32F);

    return M;
}


Mat getMotion(int from, int to, const vector<Mat> &motions)
{
    Mat M = Mat::eye(3, 3, CV_32F);
    if (to > from)
    {
        for (int i = from; i < to; ++i)
            M = at(i, motions) * M;
    }
    else if (from > to)
    {
        for (int i = to; i < from; ++i)
            M = at(i, motions) * M;
        M = M.inv();
    }
    return M;
}


static inline int areaSign(Point2f a, Point2f b, Point2f c)
{
    float area = (b-a).cross(c-a);
    if (area < -1e-5f) return -1;
    if (area > 1e-5f) return 1;
    return 0;
}


static inline bool segmentsIntersect(Point2f a, Point2f b, Point2f c, Point2f d)
{
    return areaSign(a,b,c) * areaSign(a,b,d) < 0 &&
           areaSign(c,d,a) * areaSign(c,d,b) < 0;
}


// Checks if rect a (with sides parallel to axis) is inside rect b (arbitrary).
// Rects must be passed in the [(0,0), (w,0), (w,h), (0,h)] order.
static inline bool isRectInside(const Point2f a[4], const Point2f b[4])
{
    for (int i = 0; i < 4; ++i)
        if (b[i].x > a[0].x && b[i].x < a[2].x && b[i].y > a[0].y && b[i].y < a[2].y)
            return false;
    for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
        if (segmentsIntersect(a[i], a[(i+1)%4], b[j], b[(j+1)%4]))
            return false;
    return true;
}


static inline bool isGoodMotion(const float M[], float w, float h, float dx, float dy)
{
    Point2f pt[4] = {Point2f(0,0), Point2f(w,0), Point2f(w,h), Point2f(0,h)};
    Point2f Mpt[4];

    for (int i = 0; i < 4; ++i)
    {
        Mpt[i].x = M[0]*pt[i].x + M[1]*pt[i].y + M[2];
        Mpt[i].x = M[3]*pt[i].x + M[4]*pt[i].y + M[5];
    }

    pt[0] = Point2f(dx, dy);
    pt[1] = Point2f(w - dx, dy);
    pt[2] = Point2f(w - dx, h - dy);
    pt[3] = Point2f(dx, h - dy);

    return isRectInside(pt, Mpt);
}


static inline void relaxMotion(const float M[], float t, float res[])
{
    res[0] = M[0]*(1.f-t) + t;
    res[1] = M[1]*(1.f-t);
    res[2] = M[2]*(1.f-t);
    res[3] = M[3]*(1.f-t);
    res[4] = M[4]*(1.f-t) + t;
    res[5] = M[5]*(1.f-t);
}


Mat ensureInclusionConstraint(const Mat &M, Size size, float trimRatio)
{
    CV_Assert(M.size() == Size(3,3) && M.type() == CV_32F);

    const float w = size.width;
    const float h = size.height;
    const float dx = floor(w * trimRatio);
    const float dy = floor(h * trimRatio);
    const float srcM[6] =
            {M.at<float>(0,0), M.at<float>(0,1), M.at<float>(0,2),
             M.at<float>(1,0), M.at<float>(1,1), M.at<float>(1,2)};

    float curM[6];
    float t = 0;
    relaxMotion(srcM, t, curM);
    if (isGoodMotion(curM, w, h, dx, dy))
        return M;

    float l = 0, r = 1;
    while (r - l > 1e-3f)
    {
        t = (l + r) * 0.5f;
        relaxMotion(srcM, t, curM);
        if (isGoodMotion(curM, w, h, dx, dy))
            r = t;
        else
            l = t;
        t = r;
        relaxMotion(srcM, r, curM);
    }

    return (1 - r) * M + r * Mat::eye(3, 3, CV_32F);
}


// TODO can be estimated for O(1) time
float estimateOptimalTrimRatio(const Mat &M, Size size)
{
    CV_Assert(M.size() == Size(3,3) && M.type() == CV_32F);

    const float w = size.width;
    const float h = size.height;
    Mat_<float> M_(M);

    Point2f pt[4] = {Point2f(0,0), Point2f(w,0), Point2f(w,h), Point2f(0,h)};
    Point2f Mpt[4];

    for (int i = 0; i < 4; ++i)
    {
        Mpt[i].x = M_(0,0)*pt[i].x + M_(0,1)*pt[i].y + M_(0,2);
        Mpt[i].y = M_(1,0)*pt[i].x + M_(1,1)*pt[i].y + M_(1,2);
    }

    float l = 0, r = 0.5f;
    while (r - l > 1e-3f)
    {
        float t = (l + r) * 0.5f;
        float dx = floor(w * t);
        float dy = floor(h * t);
        pt[0] = Point2f(dx, dy);
        pt[1] = Point2f(w - dx, dy);
        pt[2] = Point2f(w - dx, h - dy);
        pt[3] = Point2f(dx, h - dy);
        if (isRectInside(pt, Mpt))
            r = t;
        else
            l = t;
    }

    return r;
}


float alignementError(const Mat &M, const Mat &frame0, const Mat &mask0, const Mat &frame1)
{
    CV_Assert(frame0.type() == CV_8UC3 && frame1.type() == CV_8UC3);
    CV_Assert(mask0.type() == CV_8U && mask0.size() == frame0.size());
    CV_Assert(frame0.size() == frame1.size());
    CV_Assert(M.size() == Size(3,3) && M.type() == CV_32F);

    Mat_<uchar> mask0_(mask0);
    Mat_<float> M_(M);
    float err = 0;

    for (int y0 = 0; y0 < frame0.rows; ++y0)
    {
        for (int x0 = 0; x0 < frame0.cols; ++x0)
        {
            if (mask0_(y0,x0))
            {
                int x1 = cvRound(M_(0,0)*x0 + M_(0,1)*y0 + M_(0,2));
                int y1 = cvRound(M_(1,0)*x0 + M_(1,1)*y0 + M_(1,2));
                if (y1 >= 0 && y1 < frame1.rows && x1 >= 0 && x1 < frame1.cols)
                    err += std::abs(intensity(frame1.at<Point3_<uchar> >(y1,x1)) -
                                    intensity(frame0.at<Point3_<uchar> >(y0,x0)));
            }
        }
    }

    return err;
}

} // namespace videostab
} // namespace cv
