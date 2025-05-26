// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

#ifdef HAVE_EIGEN
#include <Eigen/Core>
#include <Eigen/Dense>
#include "opencv2/core/eigen.hpp"
#endif

#include "opencv2/core/cuda.hpp"

namespace opencv_test { namespace {

class Core_ReduceTest : public cvtest::BaseTest
{
public:
    Core_ReduceTest() {}
protected:
    void run( int) CV_OVERRIDE;
    int checkOp( const Mat& src, int dstType, int opType, const Mat& opRes, int dim );
    int checkCase( int srcType, int dstType, int dim, Size sz );
    int checkDim( int dim, Size sz );
    int checkSize( Size sz );
};

template<class Type>
void testReduce( const Mat& src, Mat& sum, Mat& avg, Mat& max, Mat& min, Mat& sum2, int dim )
{
    CV_Assert( src.channels() == 1 );
    if( dim == 0 ) // row
    {
        sum.create( 1, src.cols, CV_64FC1 );
        max.create( 1, src.cols, CV_64FC1 );
        min.create( 1, src.cols, CV_64FC1 );
        sum2.create( 1, src.cols, CV_64FC1 );
    }
    else
    {
        sum.create( src.rows, 1, CV_64FC1 );
        max.create( src.rows, 1, CV_64FC1 );
        min.create( src.rows, 1, CV_64FC1 );
        sum2.create( src.rows, 1, CV_64FC1 );
    }
    sum.setTo(Scalar(0));
    max.setTo(Scalar(-DBL_MAX));
    min.setTo(Scalar(DBL_MAX));
    sum2.setTo(Scalar(0));

    const Mat_<Type>& src_ = src;
    Mat_<double>& sum_ = (Mat_<double>&)sum;
    Mat_<double>& min_ = (Mat_<double>&)min;
    Mat_<double>& max_ = (Mat_<double>&)max;
    Mat_<double>& sum2_ = (Mat_<double>&)sum2;

    if( dim == 0 )
    {
        for( int ri = 0; ri < src.rows; ri++ )
        {
            for( int ci = 0; ci < src.cols; ci++ )
            {
                sum_(0, ci) += src_(ri, ci);
                max_(0, ci) = std::max( max_(0, ci), (double)src_(ri, ci) );
                min_(0, ci) = std::min( min_(0, ci), (double)src_(ri, ci) );
                sum2_(0, ci) += ((double)src_(ri, ci))*((double)src_(ri, ci));
            }
        }
    }
    else
    {
        for( int ci = 0; ci < src.cols; ci++ )
        {
            for( int ri = 0; ri < src.rows; ri++ )
            {
                sum_(ri, 0) += src_(ri, ci);
                max_(ri, 0) = std::max( max_(ri, 0), (double)src_(ri, ci) );
                min_(ri, 0) = std::min( min_(ri, 0), (double)src_(ri, ci) );
                sum2_(ri, 0) += ((double)src_(ri, ci))*((double)src_(ri, ci));
            }
        }
    }
    sum.convertTo( avg, CV_64FC1 );
    avg = avg * (1.0 / (dim==0 ? (double)src.rows : (double)src.cols));
}

void getMatTypeStr( int type, string& str)
{
    str = type == CV_8UC1 ? "CV_8UC1" :
    type == CV_8SC1 ? "CV_8SC1" :
    type == CV_16UC1 ? "CV_16UC1" :
    type == CV_16SC1 ? "CV_16SC1" :
    type == CV_32SC1 ? "CV_32SC1" :
    type == CV_32FC1 ? "CV_32FC1" :
    type == CV_64FC1 ? "CV_64FC1" : "unsupported matrix type";
}

int Core_ReduceTest::checkOp( const Mat& src, int dstType, int opType, const Mat& opRes, int dim )
{
    int srcType = src.type();
    bool support = false;
    if( opType == REDUCE_SUM || opType == REDUCE_AVG || opType == REDUCE_SUM2 )
    {
        if( srcType == CV_8U && (dstType == CV_32S || dstType == CV_32F || dstType == CV_64F) )
            support = true;
        if( srcType == CV_16U && (dstType == CV_32F || dstType == CV_64F) )
            support = true;
        if( srcType == CV_16S && (dstType == CV_32F || dstType == CV_64F) )
            support = true;
        if( srcType == CV_32F && (dstType == CV_32F || dstType == CV_64F) )
            support = true;
        if( srcType == CV_64F && dstType == CV_64F)
            support = true;
    }
    else if( opType == REDUCE_MAX )
    {
        if( srcType == CV_8U && dstType == CV_8U )
            support = true;
        if( srcType == CV_32F && dstType == CV_32F )
            support = true;
        if( srcType == CV_64F && dstType == CV_64F )
            support = true;
    }
    else if( opType == REDUCE_MIN )
    {
        if( srcType == CV_8U && dstType == CV_8U)
            support = true;
        if( srcType == CV_32F && dstType == CV_32F)
            support = true;
        if( srcType == CV_64F && dstType == CV_64F)
            support = true;
    }
    if( !support )
        return cvtest::TS::OK;

    double eps = 0.0;
    if ( opType == REDUCE_SUM || opType == REDUCE_AVG || opType == REDUCE_SUM2 )
    {
        if ( dstType == CV_32F )
            eps = 1.e-5;
        else if( dstType == CV_64F )
            eps = 1.e-8;
        else if ( dstType == CV_32S )
            eps = 0.6;
    }

    CV_Assert( opRes.type() == CV_64FC1 );
    Mat _dst, dst, diff;
    cv::reduce( src, _dst, dim, opType, dstType );
    _dst.convertTo( dst, CV_64FC1 );

    absdiff( opRes,dst,diff );
    bool check = false;
    if (dstType == CV_32F || dstType == CV_64F)
        check = countNonZero(diff>eps*dst) > 0;
    else
        check = countNonZero(diff>eps) > 0;
    if( check )
    {
        char msg[100];
        const char* opTypeStr =
          opType == REDUCE_SUM ? "REDUCE_SUM" :
          opType == REDUCE_AVG ? "REDUCE_AVG" :
          opType == REDUCE_MAX ? "REDUCE_MAX" :
          opType == REDUCE_MIN ? "REDUCE_MIN" :
          opType == REDUCE_SUM2 ? "REDUCE_SUM2" :
          "unknown operation type";
        string srcTypeStr, dstTypeStr;
        getMatTypeStr( src.type(), srcTypeStr );
        getMatTypeStr( dstType, dstTypeStr );
        const char* dimStr = dim == 0 ? "ROWS" : "COLS";

        snprintf( msg, sizeof(msg), "bad accuracy with srcType = %s, dstType = %s, opType = %s, dim = %s",
                srcTypeStr.c_str(), dstTypeStr.c_str(), opTypeStr, dimStr );
        ts->printf( cvtest::TS::LOG, msg );
        return cvtest::TS::FAIL_BAD_ACCURACY;
    }
    return cvtest::TS::OK;
}

int Core_ReduceTest::checkCase( int srcType, int dstType, int dim, Size sz )
{
    int code = cvtest::TS::OK, tempCode;
    Mat src, sum, avg, max, min, sum2;

    src.create( sz, srcType );
    randu( src, Scalar(0), Scalar(100) );

    if( srcType == CV_8UC1 )
        testReduce<uchar>( src, sum, avg, max, min, sum2, dim );
    else if( srcType == CV_8SC1 )
        testReduce<char>( src, sum, avg, max, min, sum2, dim );
    else if( srcType == CV_16UC1 )
        testReduce<unsigned short int>( src, sum, avg, max, min, sum2, dim );
    else if( srcType == CV_16SC1 )
        testReduce<short int>( src, sum, avg, max, min, sum2, dim );
    else if( srcType == CV_32SC1 )
        testReduce<int>( src, sum, avg, max, min, sum2, dim );
    else if( srcType == CV_32FC1 )
        testReduce<float>( src, sum, avg, max, min, sum2, dim );
    else if( srcType == CV_64FC1 )
        testReduce<double>( src, sum, avg, max, min, sum2, dim );
    else
        CV_Assert( 0 );

    // 1. sum
    tempCode = checkOp( src, dstType, REDUCE_SUM, sum, dim );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    // 2. avg
    tempCode = checkOp( src, dstType, REDUCE_AVG, avg, dim );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    // 3. max
    tempCode = checkOp( src, dstType, REDUCE_MAX, max, dim );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    // 4. min
    tempCode = checkOp( src, dstType, REDUCE_MIN, min, dim );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    // 5. sum2
    tempCode = checkOp( src, dstType, REDUCE_SUM2, sum2, dim );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    return code;
}

int Core_ReduceTest::checkDim( int dim, Size sz )
{
    int code = cvtest::TS::OK, tempCode;

    // CV_8UC1
    tempCode = checkCase( CV_8UC1, CV_8UC1, dim, sz );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    tempCode = checkCase( CV_8UC1, CV_32SC1, dim, sz );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    tempCode = checkCase( CV_8UC1, CV_32FC1, dim, sz );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    tempCode = checkCase( CV_8UC1, CV_64FC1, dim, sz );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    // CV_16UC1
    tempCode = checkCase( CV_16UC1, CV_32FC1, dim, sz );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    tempCode = checkCase( CV_16UC1, CV_64FC1, dim, sz );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    // CV_16SC1
    tempCode = checkCase( CV_16SC1, CV_32FC1, dim, sz );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    tempCode = checkCase( CV_16SC1, CV_64FC1, dim, sz );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    // CV_32FC1
    tempCode = checkCase( CV_32FC1, CV_32FC1, dim, sz );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    tempCode = checkCase( CV_32FC1, CV_64FC1, dim, sz );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    // CV_64FC1
    tempCode = checkCase( CV_64FC1, CV_64FC1, dim, sz );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    return code;
}

int Core_ReduceTest::checkSize( Size sz )
{
    int code = cvtest::TS::OK, tempCode;

    tempCode = checkDim( 0, sz ); // rows
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    tempCode = checkDim( 1, sz ); // cols
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    return code;
}

void Core_ReduceTest::run( int )
{
    int code = cvtest::TS::OK, tempCode;

    tempCode = checkSize( Size(1,1) );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    tempCode = checkSize( Size(1,100) );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    tempCode = checkSize( Size(100,1) );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    tempCode = checkSize( Size(1000,500) );
    code = tempCode != cvtest::TS::OK ? tempCode : code;

    ts->set_failed_test_info( code );
}


#define CHECK_C

TEST(Core_PCA, accuracy)
{
    const Size sz(200, 500);

    double diffPrjEps, diffBackPrjEps,
    prjEps, backPrjEps,
    evalEps, evecEps;
    int maxComponents = 100;
    double retainedVariance = 0.95;
    Mat rPoints(sz, CV_32FC1), rTestPoints(sz, CV_32FC1);
    RNG rng(12345);

    rng.fill( rPoints, RNG::UNIFORM, Scalar::all(0.0), Scalar::all(1.0) );
    rng.fill( rTestPoints, RNG::UNIFORM, Scalar::all(0.0), Scalar::all(1.0) );

    PCA rPCA( rPoints, Mat(), CV_PCA_DATA_AS_ROW, maxComponents ), cPCA;

    // 1. check C++ PCA & ROW
    Mat rPrjTestPoints = rPCA.project( rTestPoints );
    Mat rBackPrjTestPoints = rPCA.backProject( rPrjTestPoints );

    Mat avg(1, sz.width, CV_32FC1 );
    cv::reduce( rPoints, avg, 0, REDUCE_AVG );
    Mat Q = rPoints - repeat( avg, rPoints.rows, 1 ), Qt = Q.t(), eval, evec;
    Q = Qt * Q;
    Q = Q /(float)rPoints.rows;

    eigen( Q, eval, evec );
    /*SVD svd(Q);
     evec = svd.vt;
     eval = svd.w;*/

    Mat subEval( maxComponents, 1, eval.type(), eval.ptr() ),
    subEvec( maxComponents, evec.cols, evec.type(), evec.ptr() );

#ifdef CHECK_C
    Mat prjTestPoints, backPrjTestPoints, cPoints = rPoints.t(), cTestPoints = rTestPoints.t();
    CvMat _points, _testPoints, _avg, _eval, _evec, _prjTestPoints, _backPrjTestPoints;
#endif

    // check eigen()
    double eigenEps = 1e-4;
    double err;
    for(int i = 0; i < Q.rows; i++ )
    {
        Mat v = evec.row(i).t();
        Mat Qv = Q * v;

        Mat lv = eval.at<float>(i,0) * v;
        err = cvtest::norm(Qv, lv, NORM_L2 | NORM_RELATIVE);
        EXPECT_LE(err, eigenEps) << "bad accuracy of eigen(); i = " << i;
    }
    // check pca eigenvalues
    evalEps = 1e-5, evecEps = 5e-3;
    err = cvtest::norm(rPCA.eigenvalues, subEval, NORM_L2 | NORM_RELATIVE);
    EXPECT_LE(err , evalEps) << "pca.eigenvalues is incorrect (CV_PCA_DATA_AS_ROW)";
    // check pca eigenvectors
    for(int i = 0; i < subEvec.rows; i++)
    {
        Mat r0 = rPCA.eigenvectors.row(i);
        Mat r1 = subEvec.row(i);
        // eigenvectors have normalized length, but both directions v and -v are valid
        double err1 = cvtest::norm(r0, r1, NORM_L2 | NORM_RELATIVE);
        double err2 = cvtest::norm(r0, -r1, NORM_L2 | NORM_RELATIVE);
        err = std::min(err1, err2);
        if (err > evecEps)
        {
            Mat tmp;
            absdiff(rPCA.eigenvectors, subEvec, tmp);
            double mval = 0; Point mloc;
            minMaxLoc(tmp, 0, &mval, 0, &mloc);

            EXPECT_LE(err, evecEps) << "pca.eigenvectors is incorrect (CV_PCA_DATA_AS_ROW) at " << i << " "
                << cv::format("max diff is %g at (i=%d, j=%d) (%g vs %g)\n",
                        mval, mloc.y, mloc.x, rPCA.eigenvectors.at<float>(mloc.y, mloc.x),
                        subEvec.at<float>(mloc.y, mloc.x))
                << "r0=" << r0 << std::endl
                << "r1=" << r1 << std::endl
                << "err1=" << err1 << " err2=" << err2
            ;
        }
    }

    prjEps = 1.265, backPrjEps = 1.265;
    for( int i = 0; i < rTestPoints.rows; i++ )
    {
        // check pca project
        Mat subEvec_t = subEvec.t();
        Mat prj = rTestPoints.row(i) - avg; prj *= subEvec_t;
        err = cvtest::norm(rPrjTestPoints.row(i), prj, NORM_L2 | NORM_RELATIVE);
        if (err < prjEps)
        {
            EXPECT_LE(err, prjEps) << "bad accuracy of project() (CV_PCA_DATA_AS_ROW)";
            continue;
        }
        // check pca backProject
        Mat backPrj = rPrjTestPoints.row(i) * subEvec + avg;
        err = cvtest::norm(rBackPrjTestPoints.row(i), backPrj, NORM_L2 | NORM_RELATIVE);
        if (err > backPrjEps)
        {
            EXPECT_LE(err, backPrjEps) << "bad accuracy of backProject() (CV_PCA_DATA_AS_ROW)";
            continue;
        }
    }

    // 2. check C++ PCA & COL
    cPCA( rPoints.t(), Mat(), CV_PCA_DATA_AS_COL, maxComponents );
    diffPrjEps = 1, diffBackPrjEps = 1;
    Mat ocvPrjTestPoints = cPCA.project(rTestPoints.t());
    err = cvtest::norm(cv::abs(ocvPrjTestPoints), cv::abs(rPrjTestPoints.t()), NORM_L2 | NORM_RELATIVE);
    ASSERT_LE(err, diffPrjEps) << "bad accuracy of project() (CV_PCA_DATA_AS_COL)";
    err = cvtest::norm(cPCA.backProject(ocvPrjTestPoints), rBackPrjTestPoints.t(), NORM_L2 | NORM_RELATIVE);
    ASSERT_LE(err, diffBackPrjEps) << "bad accuracy of backProject() (CV_PCA_DATA_AS_COL)";

    // 3. check C++ PCA w/retainedVariance
    cPCA( rPoints.t(), Mat(), CV_PCA_DATA_AS_COL, retainedVariance );
    diffPrjEps = 1, diffBackPrjEps = 1;
    Mat rvPrjTestPoints = cPCA.project(rTestPoints.t());

    if( cPCA.eigenvectors.rows > maxComponents)
        err = cvtest::norm(cv::abs(rvPrjTestPoints.rowRange(0,maxComponents)), cv::abs(rPrjTestPoints.t()), NORM_L2 | NORM_RELATIVE);
    else
        err = cvtest::norm(cv::abs(rvPrjTestPoints), cv::abs(rPrjTestPoints.colRange(0,cPCA.eigenvectors.rows).t()), NORM_L2 | NORM_RELATIVE);

    ASSERT_LE(err, diffPrjEps) << "bad accuracy of project() (CV_PCA_DATA_AS_COL); retainedVariance=" << retainedVariance;
    err = cvtest::norm(cPCA.backProject(rvPrjTestPoints), rBackPrjTestPoints.t(), NORM_L2 | NORM_RELATIVE);
    ASSERT_LE(err, diffBackPrjEps) << "bad accuracy of backProject() (CV_PCA_DATA_AS_COL); retainedVariance=" << retainedVariance;

#ifdef CHECK_C
    // 4. check C PCA & ROW
    _points = cvMat(rPoints);
    _testPoints = cvMat(rTestPoints);
    _avg = cvMat(avg);
    _eval = cvMat(eval);
    _evec = cvMat(evec);
    prjTestPoints.create(rTestPoints.rows, maxComponents, rTestPoints.type() );
    backPrjTestPoints.create(rPoints.size(), rPoints.type() );
    _prjTestPoints = cvMat(prjTestPoints);
    _backPrjTestPoints = cvMat(backPrjTestPoints);

    cvCalcPCA( &_points, &_avg, &_eval, &_evec, CV_PCA_DATA_AS_ROW );
    cvProjectPCA( &_testPoints, &_avg, &_evec, &_prjTestPoints );
    cvBackProjectPCA( &_prjTestPoints, &_avg, &_evec, &_backPrjTestPoints );

    err = cvtest::norm(prjTestPoints, rPrjTestPoints, NORM_L2 | NORM_RELATIVE);
    ASSERT_LE(err, diffPrjEps) << "bad accuracy of cvProjectPCA() (CV_PCA_DATA_AS_ROW)";
    err = cvtest::norm(backPrjTestPoints, rBackPrjTestPoints, NORM_L2 | NORM_RELATIVE);
    ASSERT_LE(err, diffBackPrjEps) << "bad accuracy of cvBackProjectPCA() (CV_PCA_DATA_AS_ROW)";

    // 5. check C PCA & COL
    _points = cvMat(cPoints);
    _testPoints = cvMat(cTestPoints);
    avg = avg.t(); _avg = cvMat(avg);
    eval = eval.t(); _eval = cvMat(eval);
    evec = evec.t(); _evec = cvMat(evec);
    prjTestPoints = prjTestPoints.t(); _prjTestPoints = cvMat(prjTestPoints);
    backPrjTestPoints = backPrjTestPoints.t(); _backPrjTestPoints = cvMat(backPrjTestPoints);

    cvCalcPCA( &_points, &_avg, &_eval, &_evec, CV_PCA_DATA_AS_COL );
    cvProjectPCA( &_testPoints, &_avg, &_evec, &_prjTestPoints );
    cvBackProjectPCA( &_prjTestPoints, &_avg, &_evec, &_backPrjTestPoints );

    err = cvtest::norm(cv::abs(prjTestPoints), cv::abs(rPrjTestPoints.t()), NORM_L2 | NORM_RELATIVE);
    ASSERT_LE(err, diffPrjEps) << "bad accuracy of cvProjectPCA() (CV_PCA_DATA_AS_COL)";
    err = cvtest::norm(backPrjTestPoints, rBackPrjTestPoints.t(), NORM_L2 | NORM_RELATIVE);
    ASSERT_LE(err, diffBackPrjEps) << "bad accuracy of cvBackProjectPCA() (CV_PCA_DATA_AS_COL)";
#endif
    // Test read and write
    const std::string filename = cv::tempfile("PCA_store.yml");
    FileStorage fs( filename, FileStorage::WRITE );
    rPCA.write( fs );
    fs.release();

    PCA lPCA;
    fs.open( filename, FileStorage::READ );
    lPCA.read( fs.root() );
    err = cvtest::norm(rPCA.eigenvectors, lPCA.eigenvectors, NORM_L2 | NORM_RELATIVE);
    EXPECT_LE(err, 0) << "bad accuracy of write/load functions (YML)";
    err = cvtest::norm(rPCA.eigenvalues, lPCA.eigenvalues, NORM_L2 | NORM_RELATIVE);
    EXPECT_LE(err, 0) << "bad accuracy of write/load functions (YML)";
    err = cvtest::norm(rPCA.mean, lPCA.mean, NORM_L2 | NORM_RELATIVE);
    EXPECT_LE(err, 0) << "bad accuracy of write/load functions (YML)";
    EXPECT_EQ(0, remove(filename.c_str()));
}

class Core_ArrayOpTest : public cvtest::BaseTest
{
public:
    Core_ArrayOpTest();
    ~Core_ArrayOpTest();
protected:
    void run(int) CV_OVERRIDE;
};


Core_ArrayOpTest::Core_ArrayOpTest()
{
}
Core_ArrayOpTest::~Core_ArrayOpTest() {}

static string idx2string(const int* idx, int dims)
{
    char buf[256];
    char* ptr = buf;
    for( int k = 0; k < dims; k++ )
    {
        snprintf(ptr, sizeof(buf) - (ptr - buf), "%4d ", idx[k]);
        ptr += strlen(ptr);
    }
    ptr[-1] = '\0';
    return string(buf);
}

static const int* string2idx(const string& s, int* idx, int dims)
{
    const char* ptr = s.c_str();
    for( int k = 0; k < dims; k++ )
    {
        int n = 0;
        sscanf(ptr, "%d%n", idx + k, &n);
        ptr += n;
    }
    return idx;
}

static double getValue(SparseMat& M, const int* idx, RNG& rng)
{
    int d = M.dims();
    size_t hv = 0, *phv = 0;
    if( (unsigned)rng % 2 )
    {
        hv = d == 2 ? M.hash(idx[0], idx[1]) :
        d == 3 ? M.hash(idx[0], idx[1], idx[2]) : M.hash(idx);
        phv = &hv;
    }

    const uchar* ptr = d == 2 ? M.ptr(idx[0], idx[1], false, phv) :
    d == 3 ? M.ptr(idx[0], idx[1], idx[2], false, phv) :
    M.ptr(idx, false, phv);
    return !ptr ? 0 : M.type() == CV_32F ? *(float*)ptr : M.type() == CV_64F ? *(double*)ptr : 0;
}

static double getValue(const CvSparseMat* M, const int* idx)
{
    int type = 0;
    const uchar* ptr = cvPtrND(M, idx, &type, 0);
    return !ptr ? 0 : type == CV_32F ? *(float*)ptr : type == CV_64F ? *(double*)ptr : 0;
}

static void eraseValue(SparseMat& M, const int* idx, RNG& rng)
{
    int d = M.dims();
    size_t hv = 0, *phv = 0;
    if( (unsigned)rng % 2 )
    {
        hv = d == 2 ? M.hash(idx[0], idx[1]) :
        d == 3 ? M.hash(idx[0], idx[1], idx[2]) : M.hash(idx);
        phv = &hv;
    }

    if( d == 2 )
        M.erase(idx[0], idx[1], phv);
    else if( d == 3 )
        M.erase(idx[0], idx[1], idx[2], phv);
    else
        M.erase(idx, phv);
}

static void eraseValue(CvSparseMat* M, const int* idx)
{
    cvClearND(M, idx);
}

static void setValue(SparseMat& M, const int* idx, double value, RNG& rng)
{
    int d = M.dims();
    size_t hv = 0, *phv = 0;
    if( (unsigned)rng % 2 )
    {
        hv = d == 2 ? M.hash(idx[0], idx[1]) :
        d == 3 ? M.hash(idx[0], idx[1], idx[2]) : M.hash(idx);
        phv = &hv;
    }

    uchar* ptr = d == 2 ? M.ptr(idx[0], idx[1], true, phv) :
    d == 3 ? M.ptr(idx[0], idx[1], idx[2], true, phv) :
    M.ptr(idx, true, phv);
    if( M.type() == CV_32F )
        *(float*)ptr = (float)value;
    else if( M.type() == CV_64F )
        *(double*)ptr = value;
    else
        CV_Error(cv::Error::StsUnsupportedFormat, "");
}

#if defined(__GNUC__) && (__GNUC__ >= 11)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

template<typename Pixel>
struct InitializerFunctor{
    /// Initializer for cv::Mat::forEach test
    void operator()(Pixel & pixel, const int * idx) const {
        pixel.x = idx[0];
        pixel.y = idx[1];
        pixel.z = idx[2];
    }
};

template<typename Pixel>
struct InitializerFunctor5D{
    /// Initializer for cv::Mat::forEach test (5 dimensional case)
    void operator()(Pixel & pixel, const int * idx) const {
        pixel[0] = idx[0];
        pixel[1] = idx[1];
        pixel[2] = idx[2];
        pixel[3] = idx[3];
        pixel[4] = idx[4];
    }
};

#if defined(__GNUC__) && (__GNUC__ == 11 || __GNUC__ == 12)
#pragma GCC diagnostic pop
#endif


template<typename Pixel>
struct EmptyFunctor
{
    void operator()(const Pixel &, const int *) const {}
};


void Core_ArrayOpTest::run( int /* start_from */)
{
    int errcount = 0;

    // dense matrix operations
    {
        int sz3[] = {5, 10, 15};
        MatND A(3, sz3, CV_32F), B(3, sz3, CV_16SC4);
        CvMatND matA = cvMatND(A), matB = cvMatND(B);
        RNG rng;
        rng.fill(A, RNG::UNIFORM, Scalar::all(-10), Scalar::all(10));
        rng.fill(B, RNG::UNIFORM, Scalar::all(-10), Scalar::all(10));

        int idx0[] = {3,4,5}, idx1[] = {0, 9, 7};
        float val0 = 130;
        Scalar val1(-1000, 30, 3, 8);
        cvSetRealND(&matA, idx0, val0);
        cvSetReal3D(&matA, idx1[0], idx1[1], idx1[2], -val0);
        cvSetND(&matB, idx0, cvScalar(val1));
        cvSet3D(&matB, idx1[0], idx1[1], idx1[2], cvScalar(-val1));
        Ptr<CvMatND> matC(cvCloneMatND(&matB));

        if( A.at<float>(idx0[0], idx0[1], idx0[2]) != val0 ||
           A.at<float>(idx1[0], idx1[1], idx1[2]) != -val0 ||
           cvGetReal3D(&matA, idx0[0], idx0[1], idx0[2]) != val0 ||
           cvGetRealND(&matA, idx1) != -val0 ||

           Scalar(B.at<Vec4s>(idx0[0], idx0[1], idx0[2])) != val1 ||
           Scalar(B.at<Vec4s>(idx1[0], idx1[1], idx1[2])) != -val1 ||
           Scalar(cvGet3D(matC, idx0[0], idx0[1], idx0[2])) != val1 ||
           Scalar(cvGetND(matC, idx1)) != -val1 )
        {
            ts->printf(cvtest::TS::LOG, "one of cvSetReal3D, cvSetRealND, cvSet3D, cvSetND "
                       "or the corresponding *Get* functions is not correct\n");
            errcount++;
        }
    }
    // test cv::Mat::forEach
    {
        const int dims[3] = { 101, 107, 7 };
        typedef cv::Point3i Pixel;

        cv::Mat a = cv::Mat::zeros(3, dims, CV_32SC3);
        InitializerFunctor<Pixel> initializer;

        a.forEach<Pixel>(initializer);

        uint64 total = 0;
        bool error_reported = false;
        for (int i0 = 0; i0 < dims[0]; ++i0) {
            for (int i1 = 0; i1 < dims[1]; ++i1) {
                for (int i2 = 0; i2 < dims[2]; ++i2) {
                    Pixel& pixel = a.at<Pixel>(i0, i1, i2);
                    if (pixel.x != i0 || pixel.y != i1 || pixel.z != i2) {
                        if (!error_reported) {
                            ts->printf(cvtest::TS::LOG, "forEach is not correct.\n"
                                "First error detected at (%d, %d, %d).\n", pixel.x, pixel.y, pixel.z);
                            error_reported = true;
                        }
                        errcount++;
                    }
                    total += pixel.x;
                    total += pixel.y;
                    total += pixel.z;
                }
            }
        }
        uint64 total2 = 0;
        for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); ++i) {
            total2 += ((dims[i] - 1) * dims[i] / 2) * dims[0] * dims[1] * dims[2] / dims[i];
        }
        if (total != total2) {
            ts->printf(cvtest::TS::LOG, "forEach is not correct because total is invalid.\n");
            errcount++;
        }
    }

    // test cv::Mat::forEach
    // with a matrix that has more dimensions than columns
    // See https://github.com/opencv/opencv/issues/8447
    {
        const int dims[5] = { 2, 2, 2, 2, 2 };
        typedef cv::Vec<int, 5> Pixel;

        cv::Mat a = cv::Mat::zeros(5, dims, CV_32SC(5));
        InitializerFunctor5D<Pixel> initializer;

        a.forEach<Pixel>(initializer);

        uint64 total = 0;
        bool error_reported = false;
        for (int i0 = 0; i0 < dims[0]; ++i0) {
            for (int i1 = 0; i1 < dims[1]; ++i1) {
                for (int i2 = 0; i2 < dims[2]; ++i2) {
                    for (int i3 = 0; i3 < dims[3]; ++i3) {
                        for (int i4 = 0; i4 < dims[4]; ++i4) {
                            const int i[5] = { i0, i1, i2, i3, i4 };
                            Pixel& pixel = a.at<Pixel>(i);
                            if (pixel[0] != i0 || pixel[1] != i1 || pixel[2] != i2 || pixel[3] != i3 || pixel[4] != i4) {
                                if (!error_reported) {
                                    ts->printf(cvtest::TS::LOG, "forEach is not correct.\n"
                                        "First error detected at position (%d, %d, %d, %d, %d), got value (%d, %d, %d, %d, %d).\n",
                                        i0, i1, i2, i3, i4,
                                        pixel[0], pixel[1], pixel[2], pixel[3], pixel[4]);
                                    error_reported = true;
                                }
                                errcount++;
                            }
                            total += pixel[0];
                            total += pixel[1];
                            total += pixel[2];
                            total += pixel[3];
                            total += pixel[4];
                        }
                    }
                }
            }
        }
        uint64 total2 = 0;
        for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); ++i) {
            total2 += ((dims[i] - 1) * dims[i] / 2) * dims[0] * dims[1] * dims[2] * dims[3] * dims[4] / dims[i];
        }
        if (total != total2) {
            ts->printf(cvtest::TS::LOG, "forEach is not correct because total is invalid.\n");
            errcount++;
        }
    }

    // test const cv::Mat::forEach
    {
        const Mat a(10, 10, CV_32SC3);
        Mat b(10, 10, CV_32SC3);
        const Mat & c = b;
        a.forEach<Point3i>(EmptyFunctor<Point3i>());
        b.forEach<Point3i>(EmptyFunctor<const Point3i>());
        c.forEach<Point3i>(EmptyFunctor<Point3i>());
        // tests compilation, no runtime check is needed
    }

    RNG rng;
    const int MAX_DIM = 5, MAX_DIM_SZ = 10;
    // sparse matrix operations
    for( int si = 0; si < 10; si++ )
    {
        int depth = (unsigned)rng % 2 == 0 ? CV_32F : CV_64F;
        int dims = ((unsigned)rng % MAX_DIM) + 1;
        int i, k, size[MAX_DIM]={0}, idx[MAX_DIM]={0};
        vector<string> all_idxs;
        vector<double> all_vals;
        vector<double> all_vals2;
        string sidx, min_sidx, max_sidx;
        double min_val=0, max_val=0;

        int p = 1;
        for( k = 0; k < dims; k++ )
        {
            size[k] = ((unsigned)rng % MAX_DIM_SZ) + 1;
            p *= size[k];
        }
        SparseMat M( dims, size, depth );
        map<string, double> M0;

        int nz0 = (unsigned)rng % std::max(p/5,10);
        nz0 = std::min(std::max(nz0, 1), p);
        all_vals.resize(nz0);
        all_vals2.resize(nz0);
        Mat_<double> _all_vals(all_vals), _all_vals2(all_vals2);
        rng.fill(_all_vals, RNG::UNIFORM, Scalar(-1000), Scalar(1000));
        if( depth == CV_32F )
        {
            Mat _all_vals_f;
            _all_vals.convertTo(_all_vals_f, CV_32F);
            _all_vals_f.convertTo(_all_vals, CV_64F);
        }
        _all_vals.convertTo(_all_vals2, _all_vals2.type(), 2);
        if( depth == CV_32F )
        {
            Mat _all_vals2_f;
            _all_vals2.convertTo(_all_vals2_f, CV_32F);
            _all_vals2_f.convertTo(_all_vals2, CV_64F);
        }

        minMaxLoc(_all_vals, &min_val, &max_val);
        double _norm0 = cv/*test*/::norm(_all_vals, CV_C);
        double _norm1 = cv/*test*/::norm(_all_vals, CV_L1);
        double _norm2 = cv/*test*/::norm(_all_vals, CV_L2);

        for( i = 0; i < nz0; i++ )
        {
            for(;;)
            {
                for( k = 0; k < dims; k++ )
                    idx[k] = (unsigned)rng % size[k];
                sidx = idx2string(idx, dims);
                if( M0.count(sidx) == 0 )
                    break;
            }
            all_idxs.push_back(sidx);
            M0[sidx] = all_vals[i];
            if( all_vals[i] == min_val )
                min_sidx = sidx;
            if( all_vals[i] == max_val )
                max_sidx = sidx;
            setValue(M, idx, all_vals[i], rng);
            double v = getValue(M, idx, rng);
            if( v != all_vals[i] )
            {
                ts->printf(cvtest::TS::LOG, "%d. immediately after SparseMat[%s]=%.20g the current value is %.20g\n",
                           i, sidx.c_str(), all_vals[i], v);
                errcount++;
                break;
            }
        }

        Ptr<CvSparseMat> M2(cvCreateSparseMat(M));
        MatND Md;
        M.copyTo(Md);
        SparseMat M3; SparseMat(Md).convertTo(M3, Md.type(), 2);

        int nz1 = (int)M.nzcount(), nz2 = (int)M3.nzcount();
        double norm0 = cv/*test*/::norm(M, CV_C);
        double norm1 = cv/*test*/::norm(M, CV_L1);
        double norm2 = cv/*test*/::norm(M, CV_L2);
        double eps = depth == CV_32F ? FLT_EPSILON*100 : DBL_EPSILON*1000;

        if( nz1 != nz0 || nz2 != nz0)
        {
            errcount++;
            ts->printf(cvtest::TS::LOG, "%d: The number of non-zero elements before/after converting to/from dense matrix is not correct: %d/%d (while it should be %d)\n",
                       si, nz1, nz2, nz0 );
            break;
        }

        if( fabs(norm0 - _norm0) > fabs(_norm0)*eps ||
           fabs(norm1 - _norm1) > fabs(_norm1)*eps ||
           fabs(norm2 - _norm2) > fabs(_norm2)*eps )
        {
            errcount++;
            ts->printf(cvtest::TS::LOG, "%d: The norms are different: %.20g/%.20g/%.20g vs %.20g/%.20g/%.20g\n",
                       si, norm0, norm1, norm2, _norm0, _norm1, _norm2 );
            break;
        }

        int n = (unsigned)rng % std::max(p/5,10);
        n = std::min(std::max(n, 1), p) + nz0;

        for( i = 0; i < n; i++ )
        {
            double val1, val2, val3, val0;
            if(i < nz0)
            {
                sidx = all_idxs[i];
                string2idx(sidx, idx, dims);
                val0 = all_vals[i];
            }
            else
            {
                for( k = 0; k < dims; k++ )
                    idx[k] = (unsigned)rng % size[k];
                sidx = idx2string(idx, dims);
                val0 = M0[sidx];
            }
            val1 = getValue(M, idx, rng);
            val2 = getValue(M2, idx);
            val3 = getValue(M3, idx, rng);

            if( val1 != val0 || val2 != val0 || fabs(val3 - val0*2) > fabs(val0*2)*FLT_EPSILON )
            {
                errcount++;
                ts->printf(cvtest::TS::LOG, "SparseMat M[%s] = %g/%g/%g (while it should be %g)\n", sidx.c_str(), val1, val2, val3, val0 );
                break;
            }
        }

        for( i = 0; i < n; i++ )
        {
            double val1, val2;
            if(i < nz0)
            {
                sidx = all_idxs[i];
                string2idx(sidx, idx, dims);
            }
            else
            {
                for( k = 0; k < dims; k++ )
                    idx[k] = (unsigned)rng % size[k];
                sidx = idx2string(idx, dims);
            }
            eraseValue(M, idx, rng);
            eraseValue(M2, idx);
            val1 = getValue(M, idx, rng);
            val2 = getValue(M2, idx);
            if( val1 != 0 || val2 != 0 )
            {
                errcount++;
                ts->printf(cvtest::TS::LOG, "SparseMat: after deleting M[%s], it is =%g/%g (while it should be 0)\n", sidx.c_str(), val1, val2 );
                break;
            }
        }

        int nz = (int)M.nzcount();
        if( nz != 0 )
        {
            errcount++;
            ts->printf(cvtest::TS::LOG, "The number of non-zero elements after removing all the elements = %d (while it should be 0)\n", nz );
            break;
        }

        int idx1[MAX_DIM], idx2[MAX_DIM];
        double val1 = 0, val2 = 0;
        M3 = SparseMat(Md);
        cv::minMaxLoc(M3, &val1, &val2, idx1, idx2);
        string s1 = idx2string(idx1, dims), s2 = idx2string(idx2, dims);
        if( val1 != min_val || val2 != max_val || s1 != min_sidx || s2 != max_sidx )
        {
            errcount++;
            ts->printf(cvtest::TS::LOG, "%d. Sparse: The value and positions of minimum/maximum elements are different from the reference values and positions:\n\t"
                       "(%g, %g, %s, %s) vs (%g, %g, %s, %s)\n", si, val1, val2, s1.c_str(), s2.c_str(),
                       min_val, max_val, min_sidx.c_str(), max_sidx.c_str());
            break;
        }

        cv::minMaxIdx(Md, &val1, &val2, idx1, idx2);
        s1 = idx2string(idx1, dims), s2 = idx2string(idx2, dims);
        if( (min_val < 0 && (val1 != min_val || s1 != min_sidx)) ||
           (max_val > 0 && (val2 != max_val || s2 != max_sidx)) )
        {
            errcount++;
            ts->printf(cvtest::TS::LOG, "%d. Dense: The value and positions of minimum/maximum elements are different from the reference values and positions:\n\t"
                       "(%g, %g, %s, %s) vs (%g, %g, %s, %s)\n", si, val1, val2, s1.c_str(), s2.c_str(),
                       min_val, max_val, min_sidx.c_str(), max_sidx.c_str());
            break;
        }
    }

    ts->set_failed_test_info(errcount == 0 ? cvtest::TS::OK : cvtest::TS::FAIL_INVALID_OUTPUT);
}


template <class T>
int calcDiffElemCountImpl(const vector<Mat>& mv, const Mat& m)
{
    int diffElemCount = 0;
    const int mChannels = m.channels();
    for(int y = 0; y < m.rows; y++)
    {
        for(int x = 0; x < m.cols; x++)
        {
            const T* mElem = &m.at<T>(y, x*mChannels);
            size_t loc = 0;
            for(size_t i = 0; i < mv.size(); i++)
            {
                const size_t mvChannel = mv[i].channels();
                const T* mvElem = &mv[i].at<T>(y, x*(int)mvChannel);
                for(size_t li = 0; li < mvChannel; li++)
                    if(mElem[loc + li] != mvElem[li])
                        diffElemCount++;
                loc += mvChannel;
            }
            CV_Assert(loc == (size_t)mChannels);
        }
    }
    return diffElemCount;
}

static
int calcDiffElemCount(const vector<Mat>& mv, const Mat& m)
{
    int depth = m.depth();
    switch (depth)
    {
    case CV_8U:
        return calcDiffElemCountImpl<uchar>(mv, m);
    case CV_8S:
        return calcDiffElemCountImpl<char>(mv, m);
    case CV_16U:
        return calcDiffElemCountImpl<unsigned short>(mv, m);
    case CV_16S:
        return calcDiffElemCountImpl<short int>(mv, m);
    case CV_32S:
        return calcDiffElemCountImpl<int>(mv, m);
    case CV_32F:
        return calcDiffElemCountImpl<float>(mv, m);
    case CV_64F:
        return calcDiffElemCountImpl<double>(mv, m);
    }

    return INT_MAX;
}

class Core_MergeSplitBaseTest : public cvtest::BaseTest
{
protected:
    virtual int run_case(int depth, size_t channels, const Size& size, RNG& rng) = 0;

    virtual void run(int) CV_OVERRIDE
    {
        // m is Mat
        // mv is vector<Mat>
        const int minMSize = 1;
        const int maxMSize = 100;
        const size_t maxMvSize = 10;

        RNG& rng = theRNG();
        Size mSize(rng.uniform(minMSize, maxMSize), rng.uniform(minMSize, maxMSize));
        size_t mvSize = rng.uniform(1, maxMvSize);

        int res = cvtest::TS::OK;
        int curRes = run_case(CV_8U, mvSize, mSize, rng);
        res = curRes != cvtest::TS::OK ? curRes : res;

        curRes = run_case(CV_8S, mvSize, mSize, rng);
        res = curRes != cvtest::TS::OK ? curRes : res;

        curRes = run_case(CV_16U, mvSize, mSize, rng);
        res = curRes != cvtest::TS::OK ? curRes : res;

        curRes = run_case(CV_16S, mvSize, mSize, rng);
        res = curRes != cvtest::TS::OK ? curRes : res;

        curRes = run_case(CV_32S, mvSize, mSize, rng);
        res = curRes != cvtest::TS::OK ? curRes : res;

        curRes = run_case(CV_32F, mvSize, mSize, rng);
        res = curRes != cvtest::TS::OK ? curRes : res;

        curRes = run_case(CV_64F, mvSize, mSize, rng);
        res = curRes != cvtest::TS::OK ? curRes : res;

        ts->set_failed_test_info(res);
    }
};

class Core_MergeTest : public Core_MergeSplitBaseTest
{
public:
    Core_MergeTest() {}
    ~Core_MergeTest() {}

protected:
    virtual int run_case(int depth, size_t matCount, const Size& size, RNG& rng) CV_OVERRIDE
    {
        const int maxMatChannels = 10;

        vector<Mat> src(matCount);
        int channels = 0;
        for(size_t i = 0; i < src.size(); i++)
        {
            Mat m(size, CV_MAKETYPE(depth, rng.uniform(1,maxMatChannels)));
            rng.fill(m, RNG::UNIFORM, 0, 100, true);
            channels += m.channels();
            src[i] = m;
        }

        Mat dst;
        merge(src, dst);

        // check result
        std::stringstream commonLog;
        commonLog << "Depth " << depth << " :";
        if(dst.depth() != depth)
        {
            ts->printf(cvtest::TS::LOG, "%s incorrect depth of dst (%d instead of %d)\n",
                       commonLog.str().c_str(), dst.depth(), depth);
            return cvtest::TS::FAIL_INVALID_OUTPUT;
        }
        if(dst.size() != size)
        {
            ts->printf(cvtest::TS::LOG, "%s incorrect size of dst (%d x %d instead of %d x %d)\n",
                       commonLog.str().c_str(), dst.rows, dst.cols, size.height, size.width);
            return cvtest::TS::FAIL_INVALID_OUTPUT;
        }
        if(dst.channels() != channels)
        {
            ts->printf(cvtest::TS::LOG, "%s: incorrect channels count of dst (%d instead of %d)\n",
                       commonLog.str().c_str(), dst.channels(), channels);
            return cvtest::TS::FAIL_INVALID_OUTPUT;
        }

        int diffElemCount = calcDiffElemCount(src, dst);
        if(diffElemCount > 0)
        {
            ts->printf(cvtest::TS::LOG, "%s: there are incorrect elements in dst (part of them is %f)\n",
                       commonLog.str().c_str(), static_cast<float>(diffElemCount)/(channels*size.area()));
            return cvtest::TS::FAIL_INVALID_OUTPUT;
        }

        return cvtest::TS::OK;
    }
};

class Core_SplitTest : public Core_MergeSplitBaseTest
{
public:
    Core_SplitTest() {}
    ~Core_SplitTest() {}

protected:
    virtual int run_case(int depth, size_t channels, const Size& size, RNG& rng) CV_OVERRIDE
    {
        Mat src(size, CV_MAKETYPE(depth, (int)channels));
        rng.fill(src, RNG::UNIFORM, 0, 100, true);

        vector<Mat> dst;
        split(src, dst);

        // check result
        std::stringstream commonLog;
        commonLog << "Depth " << depth << " :";
        if(dst.size() != channels)
        {
            ts->printf(cvtest::TS::LOG, "%s incorrect count of matrices in dst (%d instead of %d)\n",
                       commonLog.str().c_str(), dst.size(), channels);
            return cvtest::TS::FAIL_INVALID_OUTPUT;
        }
        for(size_t i = 0; i < dst.size(); i++)
        {
            if(dst[i].size() != size)
            {
                ts->printf(cvtest::TS::LOG, "%s incorrect size of dst[%d] (%d x %d instead of %d x %d)\n",
                           commonLog.str().c_str(), i, dst[i].rows, dst[i].cols, size.height, size.width);
                return cvtest::TS::FAIL_INVALID_OUTPUT;
            }
            if(dst[i].depth() != depth)
            {
                ts->printf(cvtest::TS::LOG, "%s: incorrect depth of dst[%d] (%d instead of %d)\n",
                           commonLog.str().c_str(), i, dst[i].depth(), depth);
                return cvtest::TS::FAIL_INVALID_OUTPUT;
            }
            if(dst[i].channels() != 1)
            {
                ts->printf(cvtest::TS::LOG, "%s: incorrect channels count of dst[%d] (%d instead of %d)\n",
                           commonLog.str().c_str(), i, dst[i].channels(), 1);
                return cvtest::TS::FAIL_INVALID_OUTPUT;
            }
        }

        int diffElemCount = calcDiffElemCount(dst, src);
        if(diffElemCount > 0)
        {
            ts->printf(cvtest::TS::LOG, "%s: there are incorrect elements in dst (part of them is %f)\n",
                       commonLog.str().c_str(), static_cast<float>(diffElemCount)/(channels*size.area()));
            return cvtest::TS::FAIL_INVALID_OUTPUT;
        }

        return cvtest::TS::OK;
    }
};

TEST(Core_Reduce, accuracy) { Core_ReduceTest test; test.safe_run(); }
TEST(Core_Array, basic_operations) { Core_ArrayOpTest test; test.safe_run(); }

TEST(Core_Merge, shape_operations) { Core_MergeTest test; test.safe_run(); }
TEST(Core_Split, shape_operations) { Core_SplitTest test; test.safe_run(); }


TEST(Core_IOArray, submat_assignment)
{
    Mat1f A = Mat1f::zeros(2,2);
    Mat1f B = Mat1f::ones(1,3);

    EXPECT_THROW( B.colRange(0,3).copyTo(A.row(0)), cv::Exception );

    EXPECT_NO_THROW( B.colRange(0,2).copyTo(A.row(0)) );

    EXPECT_EQ( 1.0f, A(0,0) );
    EXPECT_EQ( 1.0f, A(0,1) );
}

void OutputArray_create1(OutputArray m) { m.create(1, 2, CV_32S); }
void OutputArray_create2(OutputArray m) { m.create(1, 3, CV_32F); }

TEST(Core_IOArray, submat_create)
{
    Mat1f A = Mat1f::zeros(2,2);

    EXPECT_THROW( OutputArray_create1(A.row(0)), cv::Exception );
    EXPECT_THROW( OutputArray_create2(A.row(0)), cv::Exception );
}

TEST(Core_Mat, issue4457_pass_null_ptr)
{
    ASSERT_ANY_THROW(cv::Mat mask(45, 45, CV_32F, 0));
}

TEST(Core_Mat, reshape_1942)
{
    cv::Mat A = (cv::Mat_<float>(2,3) << 3.4884074, 1.4159607, 0.78737736,  2.3456569, -0.88010466, 0.3009364);
    int cn = 0;
    ASSERT_NO_THROW(
        cv::Mat_<float> M = A.reshape(3);
        cn = M.channels();
    );
    ASSERT_EQ(1, cn);
}

static void check_ndim_shape(const cv::Mat &mat, int cn, int ndims, const int *sizes)
{
    EXPECT_EQ(mat.channels(), cn);
    EXPECT_EQ(mat.dims, ndims);

    if (mat.dims != ndims)
        return;

    for (int i = 0; i < ndims; i++)
        EXPECT_EQ(mat.size[i], sizes[i]);
}

TEST(Core_Mat, reshape_ndims_2)
{
    const cv::Mat A(8, 16, CV_8UC3);
    cv::Mat B;

    {
        int new_sizes_mask[] = { 0, 3, 4, 4 };
        int new_sizes_real[] = { 8, 3, 4, 4 };
        ASSERT_NO_THROW(B = A.reshape(1, 4, new_sizes_mask));
        check_ndim_shape(B, 1, 4, new_sizes_real);
    }
    {
        int new_sizes[] = { 16, 8 };
        ASSERT_NO_THROW(B = A.reshape(0, 2, new_sizes));
        check_ndim_shape(B, 3, 2, new_sizes);
        EXPECT_EQ(B.rows, new_sizes[0]);
        EXPECT_EQ(B.cols, new_sizes[1]);
    }
    {
        int new_sizes[] = { 2, 5, 1, 3 };
        cv::Mat A_sliced = A(cv::Range::all(), cv::Range(0, 15));
        ASSERT_ANY_THROW(A_sliced.reshape(4, 4, new_sizes));
    }
}

TEST(Core_Mat, reshape_ndims_4)
{
    const int sizes[] = { 2, 6, 4, 12 };
    const cv::Mat A(4, sizes, CV_8UC3);
    cv::Mat B;

    {
        int new_sizes_mask[] = { 0, 864 };
        int new_sizes_real[] = { 2, 864 };
        ASSERT_NO_THROW(B = A.reshape(1, 2, new_sizes_mask));
        check_ndim_shape(B, 1, 2, new_sizes_real);
        EXPECT_EQ(B.rows, new_sizes_real[0]);
        EXPECT_EQ(B.cols, new_sizes_real[1]);
    }
    {
        int new_sizes_mask[] = { 4, 0, 0, 2, 3 };
        int new_sizes_real[] = { 4, 6, 4, 2, 3 };
        ASSERT_NO_THROW(B = A.reshape(0, 5, new_sizes_mask));
        check_ndim_shape(B, 3, 5, new_sizes_real);
    }
    {
        int new_sizes_mask[] = { 1, 1 };
        ASSERT_ANY_THROW(A.reshape(0, 2, new_sizes_mask));
    }
    {
        int new_sizes_mask[] = { 4, 6, 3, 3, 0 };
        ASSERT_ANY_THROW(A.reshape(0, 5, new_sizes_mask));
    }
}

TEST(Core_Mat, reinterpret_Mat_8UC3_8SC3)
{
    cv::Mat A(8, 16, CV_8UC3, cv::Scalar(1, 2, 3));
    cv::Mat B = A.reinterpret(CV_8SC3);

    EXPECT_EQ(A.data, B.data);
    EXPECT_EQ(B.type(), CV_8SC3);
}

TEST(Core_Mat, reinterpret_Mat_8UC4_32FC1)
{
    cv::Mat A(8, 16, CV_8UC4, cv::Scalar(1, 2, 3, 4));
    cv::Mat B = A.reinterpret(CV_32FC1);

    EXPECT_EQ(A.data, B.data);
    EXPECT_EQ(B.type(), CV_32FC1);
}

TEST(Core_Mat, reinterpret_OutputArray_8UC3_8SC3) {
    cv::Mat A(8, 16, CV_8UC3, cv::Scalar(1, 2, 3));
    cv::OutputArray C(A);
    cv::Mat B = C.reinterpret(CV_8SC3);

    EXPECT_EQ(A.data, B.data);
    EXPECT_EQ(B.type(), CV_8SC3);
}

TEST(Core_Mat, reinterpret_OutputArray_8UC4_32FC1) {
    cv::Mat A(8, 16, CV_8UC4, cv::Scalar(1, 2, 3, 4));
    cv::OutputArray C(A);
    cv::Mat B = C.reinterpret(CV_32FC1);

    EXPECT_EQ(A.data, B.data);
    EXPECT_EQ(B.type(), CV_32FC1);
}

TEST(Core_Mat, push_back)
{
    Mat a = (Mat_<float>(1,2) << 3.4884074f, 1.4159607f);
    Mat b = (Mat_<float>(1,2) << 0.78737736f, 2.3456569f);

    a.push_back(b);

    ASSERT_EQ(2, a.cols);
    ASSERT_EQ(2, a.rows);

    ASSERT_FLOAT_EQ(3.4884074f, a.at<float>(0, 0));
    ASSERT_FLOAT_EQ(1.4159607f, a.at<float>(0, 1));
    ASSERT_FLOAT_EQ(0.78737736f, a.at<float>(1, 0));
    ASSERT_FLOAT_EQ(2.3456569f, a.at<float>(1, 1));

    Mat c = (Mat_<float>(2,2) << -0.88010466f, 0.3009364f, 2.22399974f, -5.45933905f);

    ASSERT_EQ(c.rows, a.cols);

    a.push_back(c.t());

    ASSERT_EQ(2, a.cols);
    ASSERT_EQ(4, a.rows);

    ASSERT_FLOAT_EQ(3.4884074f, a.at<float>(0, 0));
    ASSERT_FLOAT_EQ(1.4159607f, a.at<float>(0, 1));
    ASSERT_FLOAT_EQ(0.78737736f, a.at<float>(1, 0));
    ASSERT_FLOAT_EQ(2.3456569f, a.at<float>(1, 1));
    ASSERT_FLOAT_EQ(-0.88010466f, a.at<float>(2, 0));
    ASSERT_FLOAT_EQ(2.22399974f, a.at<float>(2, 1));
    ASSERT_FLOAT_EQ(0.3009364f, a.at<float>(3, 0));
    ASSERT_FLOAT_EQ(-5.45933905f, a.at<float>(3, 1));

    a.push_back(Mat::ones(2, 2, CV_32FC1));

    ASSERT_EQ(6, a.rows);

    for(int row=4; row<a.rows; row++) {

        for(int col=0; col<a.cols; col++) {

            ASSERT_FLOAT_EQ(1.f, a.at<float>(row, col));
        }
    }
}

TEST(Core_Mat, copyNx1ToVector)
{
    cv::Mat_<uchar> src(5, 1);
    cv::Mat_<uchar> ref_dst8;
    cv::Mat_<ushort> ref_dst16;
    std::vector<uchar> dst8;
    std::vector<ushort> dst16;

    src << 1, 2, 3, 4, 5;

    src.copyTo(ref_dst8);
    src.copyTo(dst8);

    ASSERT_PRED_FORMAT2(cvtest::MatComparator(0, 0), ref_dst8, cv::Mat_<uchar>(dst8));

    src.convertTo(ref_dst16, CV_16U);
    src.convertTo(dst16, CV_16U);

    ASSERT_PRED_FORMAT2(cvtest::MatComparator(0, 0), ref_dst16, cv::Mat_<ushort>(dst16));
}

TEST(Core_Mat, copyMakeBoderUndefinedBehavior)
{
    Mat1b src(4, 4), dst;
    randu(src, Scalar(10), Scalar(100));
    // This could trigger a (signed int)*size_t operation which is undefined behavior.
    cv::copyMakeBorder(src, dst, 1, 1, 1, 1, cv::BORDER_REFLECT_101);
    EXPECT_EQ(0, cv::norm(src.row(1), dst(Rect(1,0,4,1))));
    EXPECT_EQ(0, cv::norm(src.row(2), dst(Rect(1,5,4,1))));
    EXPECT_EQ(0, cv::norm(src.col(1), dst(Rect(0,1,1,4))));
    EXPECT_EQ(0, cv::norm(src.col(2), dst(Rect(5,1,1,4))));
}

TEST(Core_Mat, zeros)
{
  // Should not fail during linkage.
  const int dims[] = {2, 2, 4};
  cv::Mat1f mat = cv::Mat1f::zeros(3, dims);
}

TEST(Core_Matx, fromMat_)
{
    Mat_<double> a = (Mat_<double>(2,2) << 10, 11, 12, 13);
    Matx22d b(a);
    ASSERT_EQ( cvtest::norm(a, b, NORM_INF), 0.);
}

TEST(Core_Matx, from_initializer_list)
{
    Mat_<double> a = (Mat_<double>(2,2) << 10, 11, 12, 13);
    Matx22d b = {10, 11, 12, 13};
    ASSERT_EQ( cvtest::norm(a, b, NORM_INF), 0.);
}

TEST(Core_Mat, regression_9507)
{
    cv::Mat m = Mat::zeros(5, 5, CV_8UC3);
    cv::Mat m2{m};
    EXPECT_EQ(25u, m2.total());
}

TEST(Core_InputArray, empty)
{
    vector<vector<Point> > data;
    ASSERT_TRUE( _InputArray(data).empty() );
}

TEST(Core_CopyMask, bug1918)
{
    Mat_<unsigned char> tmpSrc(100,100);
    tmpSrc = 124;
    Mat_<unsigned char> tmpMask(100,100);
    tmpMask = 255;
    Mat_<unsigned char> tmpDst(100,100);
    tmpDst = 2;
    tmpSrc.copyTo(tmpDst,tmpMask);
    ASSERT_EQ(sum(tmpDst)[0], 124*100*100);
}

TEST(Core_SVD, orthogonality)
{
    for( int i = 0; i < 2; i++ )
    {
        int type = i == 0 ? CV_32F : CV_64F;
        Mat mat_D(2, 2, type);
        mat_D.setTo(88.);
        Mat mat_U, mat_W;
        SVD::compute(mat_D, mat_W, mat_U, noArray(), SVD::FULL_UV);
        mat_U *= mat_U.t();
        ASSERT_LT(cvtest::norm(mat_U, Mat::eye(2, 2, type), NORM_INF), 1e-5);
    }
}


TEST(Core_SparseMat, footprint)
{
    int n = 1000000;
    int sz[] = { n, n };
    SparseMat m(2, sz, CV_64F);

    int nodeSize0 = (int)m.hdr->nodeSize;
    double dataSize0 = ((double)m.hdr->pool.size() + (double)m.hdr->hashtab.size()*sizeof(size_t))*1e-6;
    printf("before: node size=%d bytes, data size=%.0f Mbytes\n", nodeSize0, dataSize0);

    for (int i = 0; i < n; i++)
    {
        m.ref<double>(i, i) = 1;
    }

    double dataSize1 = ((double)m.hdr->pool.size() + (double)m.hdr->hashtab.size()*sizeof(size_t))*1e-6;
    double threshold = (n*nodeSize0*1.6 + n*2.*sizeof(size_t))*1e-6;
    printf("after: data size=%.0f Mbytes, threshold=%.0f MBytes\n", dataSize1, threshold);

    ASSERT_LE((int)m.hdr->nodeSize, 32);
    ASSERT_LE(dataSize1, threshold);
}


// Can't fix without dirty hacks or broken user code (PR #4159)
TEST(Core_Mat_vector, DISABLED_OutputArray_create_getMat)
{
    cv::Mat_<uchar> src_base(5, 1);
    std::vector<uchar> dst8;

    src_base << 1, 2, 3, 4, 5;

    Mat src(src_base);
    OutputArray _dst(dst8);
    {
        _dst.create(src.rows, src.cols, src.type());
        Mat dst = _dst.getMat();
        EXPECT_EQ(src.dims, dst.dims);
        EXPECT_EQ(src.cols, dst.cols);
        EXPECT_EQ(src.rows, dst.rows);
    }
}

TEST(Core_Mat_vector, copyTo_roi_column)
{
    cv::Mat_<uchar> src_base(5, 2);
    std::vector<uchar> dst1;

    src_base << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

    Mat src_full(src_base);
    Mat src(src_full.col(0));
#if 0 // Can't fix without dirty hacks or broken user code (PR #4159)
    OutputArray _dst(dst1);
    {
        _dst.create(src.rows, src.cols, src.type());
        Mat dst = _dst.getMat();
        EXPECT_EQ(src.dims, dst.dims);
        EXPECT_EQ(src.cols, dst.cols);
        EXPECT_EQ(src.rows, dst.rows);
    }
#endif

    std::vector<uchar> dst2;
    src.copyTo(dst2);
    std::cout << "src = " << src << std::endl;
    std::cout << "dst = " << Mat(dst2) << std::endl;
    EXPECT_EQ((size_t)5, dst2.size());
    EXPECT_EQ(1, (int)dst2[0]);
    EXPECT_EQ(3, (int)dst2[1]);
    EXPECT_EQ(5, (int)dst2[2]);
    EXPECT_EQ(7, (int)dst2[3]);
    EXPECT_EQ(9, (int)dst2[4]);
}

TEST(Core_Mat_vector, copyTo_roi_row)
{
    cv::Mat_<uchar> src_base(2, 5);
    std::vector<uchar> dst1;

    src_base << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

    Mat src_full(src_base);
    Mat src(src_full.row(0));
    OutputArray _dst(dst1);
    {
        _dst.create(src.rows, src.cols, src.type());
        Mat dst = _dst.getMat();
        EXPECT_EQ(src.dims, dst.dims);
        EXPECT_EQ(src.cols, dst.cols);
        EXPECT_EQ(src.rows, dst.rows);
    }

    std::vector<uchar> dst2;
    src.copyTo(dst2);
    std::cout << "src = " << src << std::endl;
    std::cout << "dst = " << Mat(dst2) << std::endl;
    EXPECT_EQ(1, (int)dst2[0]);
    EXPECT_EQ(2, (int)dst2[1]);
    EXPECT_EQ(3, (int)dst2[2]);
    EXPECT_EQ(4, (int)dst2[3]);
    EXPECT_EQ(5, (int)dst2[4]);
}

TEST(Mat, regression_5991)
{
    int sz[] = {2,3,2};
    Mat mat(3, sz, CV_32F, Scalar(1));
    ASSERT_NO_THROW(mat.convertTo(mat, CV_8U));
    EXPECT_EQ(sz[0], mat.size[0]);
    EXPECT_EQ(sz[1], mat.size[1]);
    EXPECT_EQ(sz[2], mat.size[2]);
    EXPECT_EQ(0, cvtest::norm(mat, Mat(3, sz, CV_8U, Scalar(1)), NORM_INF));
}

TEST(Mat, regression_9720)
{
    Mat mat(1, 1, CV_32FC1);
    mat.at<float>(0) = 1.f;
    const float a = 0.1f;
    Mat me1 = (Mat)(mat.mul((a / mat)));
    Mat me2 = (Mat)(mat.mul((Mat)(a / mat)));
    Mat me3 = (Mat)(mat.mul((a * mat)));
    Mat me4 = (Mat)(mat.mul((Mat)(a * mat)));
    EXPECT_EQ(me1.at<float>(0), me2.at<float>(0));
    EXPECT_EQ(me3.at<float>(0), me4.at<float>(0));
}

#ifdef OPENCV_TEST_BIGDATA
TEST(Mat, regression_6696_BigData_8Gb)
{
    int width = 60000;
    int height = 10000;

    Mat destImageBGR = Mat(height, width, CV_8UC3, Scalar(1, 2, 3, 0));
    Mat destImageA = Mat(height, width, CV_8UC1, Scalar::all(4));

    vector<Mat> planes;
    split(destImageBGR, planes);
    planes.push_back(destImageA);
    merge(planes, destImageBGR);

    EXPECT_EQ(1, destImageBGR.at<Vec4b>(0)[0]);
    EXPECT_EQ(2, destImageBGR.at<Vec4b>(0)[1]);
    EXPECT_EQ(3, destImageBGR.at<Vec4b>(0)[2]);
    EXPECT_EQ(4, destImageBGR.at<Vec4b>(0)[3]);

    EXPECT_EQ(1, destImageBGR.at<Vec4b>(height-1, width-1)[0]);
    EXPECT_EQ(2, destImageBGR.at<Vec4b>(height-1, width-1)[1]);
    EXPECT_EQ(3, destImageBGR.at<Vec4b>(height-1, width-1)[2]);
    EXPECT_EQ(4, destImageBGR.at<Vec4b>(height-1, width-1)[3]);
}
#endif

TEST(Reduce, regression_should_fail_bug_4594)
{
    cv::Mat src = cv::Mat::eye(4, 4, CV_8U);
    std::vector<int> dst;

    EXPECT_THROW(cv::reduce(src, dst, 0, REDUCE_MIN, CV_32S), cv::Exception);
    EXPECT_THROW(cv::reduce(src, dst, 0, REDUCE_MAX, CV_32S), cv::Exception);
    EXPECT_NO_THROW(cv::reduce(src, dst, 0, REDUCE_SUM, CV_32S));
    EXPECT_NO_THROW(cv::reduce(src, dst, 0, REDUCE_AVG, CV_32S));
    EXPECT_NO_THROW(cv::reduce(src, dst, 0, REDUCE_SUM2, CV_32S));
}

TEST(Mat, push_back_vector)
{
    cv::Mat result(1, 5, CV_32FC1);

    std::vector<float> vec1(result.cols + 1);
    std::vector<int> vec2(result.cols);

    EXPECT_THROW(result.push_back(vec1), cv::Exception);
    EXPECT_THROW(result.push_back(vec2), cv::Exception);

    vec1.resize(result.cols);

    for (int i = 0; i < 5; ++i)
        result.push_back(cv::Mat(vec1).reshape(1, 1));

    ASSERT_EQ(6, result.rows);
}

TEST(Mat, regression_5917_clone_empty)
{
    Mat cloned;
    Mat_<Point2f> source(5, 0);

    ASSERT_NO_THROW(cloned = source.clone());
}

TEST(Mat, regression_7873_mat_vector_initialize)
{
    std::vector<int> dims;
    dims.push_back(12);
    dims.push_back(3);
    dims.push_back(2);
    Mat multi_mat(dims, CV_32FC1, cv::Scalar(0));

    ASSERT_EQ(3, multi_mat.dims);
    ASSERT_EQ(12, multi_mat.size[0]);
    ASSERT_EQ(3, multi_mat.size[1]);
    ASSERT_EQ(2, multi_mat.size[2]);

    std::vector<Range> ranges;
    ranges.push_back(Range(1, 2));
    ranges.push_back(Range::all());
    ranges.push_back(Range::all());
    Mat sub_mat = multi_mat(ranges);

    ASSERT_EQ(3, sub_mat.dims);
    ASSERT_EQ(1, sub_mat.size[0]);
    ASSERT_EQ(3, sub_mat.size[1]);
    ASSERT_EQ(2, sub_mat.size[2]);
}

TEST(Mat, regression_10507_mat_setTo)
{
    Size sz(6, 4);
    Mat test_mask(sz, CV_8UC1, cv::Scalar::all(255));
    test_mask.at<uchar>(1,0) = 0;
    test_mask.at<uchar>(0,1) = 0;
    for (int cn = 1; cn <= 4; cn++)
    {
        cv::Mat A(sz, CV_MAKE_TYPE(CV_32F, cn), cv::Scalar::all(5));
        A.setTo(cv::Scalar::all(std::numeric_limits<float>::quiet_NaN()), test_mask);
        int nans = 0;
        for (int y = 0; y < A.rows; y++)
        {
            for (int x = 0; x < A.cols; x++)
            {
                for (int c = 0; c < cn; c++)
                {
                    float v = A.ptr<float>(y, x)[c];
                    nans += (v == v) ? 0 : 1;
                }
            }
        }
        EXPECT_EQ(nans, cn * (sz.area() - 2)) << "A=" << A << std::endl << "mask=" << test_mask << std::endl;
    }
}

TEST(Core_Mat, empty)
{
    // Should not crash.
    uint8_t data[2] = {0, 1};
    cv::Mat mat(/*ndims=*/0, /*sizes=*/nullptr,/*type=*/CV_8UC1, /*data=*/data, /*steps=*/0);
    // No assertion, just construction test
}

TEST(Core_Mat, reverse_iterator_19967)
{
    // empty iterator (#16855)
    cv::Mat m_empty;
    EXPECT_NO_THROW(m_empty.rbegin<uchar>());
    EXPECT_NO_THROW(m_empty.rend<uchar>());
    EXPECT_TRUE(m_empty.rbegin<uchar>() == m_empty.rend<uchar>());

    // 1D test
    std::vector<uchar> data{0, 1, 2, 3};
    const std::vector<int> sizes_1d{4};

    //Base class
    cv::Mat m_1d(sizes_1d, CV_8U, data.data());
    auto mismatch_it_pair_1d = std::mismatch(data.rbegin(), data.rend(), m_1d.rbegin<uchar>());
    EXPECT_EQ(mismatch_it_pair_1d.first, data.rend());  // expect no mismatch
    EXPECT_EQ(mismatch_it_pair_1d.second, m_1d.rend<uchar>());

    //Templated derived class
    cv::Mat_<uchar> m_1d_t(static_cast<int>(sizes_1d.size()), sizes_1d.data(), data.data());
    auto mismatch_it_pair_1d_t = std::mismatch(data.rbegin(), data.rend(), m_1d_t.rbegin());
    EXPECT_EQ(mismatch_it_pair_1d_t.first, data.rend());  // expect no mismatch
    EXPECT_EQ(mismatch_it_pair_1d_t.second, m_1d_t.rend());


    // 2D test
    const std::vector<int> sizes_2d{2, 2};

    //Base class
    cv::Mat m_2d(sizes_2d, CV_8U, data.data());
    auto mismatch_it_pair_2d = std::mismatch(data.rbegin(), data.rend(), m_2d.rbegin<uchar>());
    EXPECT_EQ(mismatch_it_pair_2d.first, data.rend());
    EXPECT_EQ(mismatch_it_pair_2d.second, m_2d.rend<uchar>());

    //Templated derived class
    cv::Mat_<uchar> m_2d_t(static_cast<int>(sizes_2d.size()),sizes_2d.data(), data.data());
    auto mismatch_it_pair_2d_t = std::mismatch(data.rbegin(), data.rend(), m_2d_t.rbegin());
    EXPECT_EQ(mismatch_it_pair_2d_t.first, data.rend());
    EXPECT_EQ(mismatch_it_pair_2d_t.second, m_2d_t.rend());

    // 3D test
    std::vector<uchar> data_3d{0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<int> sizes_3d{2, 2, 2};

    //Base class
    cv::Mat m_3d(sizes_3d, CV_8U, data_3d.data());
    auto mismatch_it_pair_3d = std::mismatch(data_3d.rbegin(), data_3d.rend(), m_3d.rbegin<uchar>());
    EXPECT_EQ(mismatch_it_pair_3d.first, data_3d.rend());
    EXPECT_EQ(mismatch_it_pair_3d.second, m_3d.rend<uchar>());

    //Templated derived class
    cv::Mat_<uchar> m_3d_t(static_cast<int>(sizes_3d.size()),sizes_3d.data(), data_3d.data());
    auto mismatch_it_pair_3d_t = std::mismatch(data_3d.rbegin(), data_3d.rend(), m_3d_t.rbegin());
    EXPECT_EQ(mismatch_it_pair_3d_t.first, data_3d.rend());
    EXPECT_EQ(mismatch_it_pair_3d_t.second, m_3d_t.rend());

    // const test base class
    const cv::Mat m_1d_const(sizes_1d, CV_8U, data.data());

    auto mismatch_it_pair_1d_const = std::mismatch(data.rbegin(), data.rend(), m_1d_const.rbegin<uchar>());
    EXPECT_EQ(mismatch_it_pair_1d_const.first, data.rend());  // expect no mismatch
    EXPECT_EQ(mismatch_it_pair_1d_const.second, m_1d_const.rend<uchar>());

    EXPECT_FALSE((std::is_assignable<decltype(m_1d_const.rend<uchar>()), uchar>::value)) << "Constness of const iterator violated.";
    EXPECT_FALSE((std::is_assignable<decltype(m_1d_const.rbegin<uchar>()), uchar>::value)) << "Constness of const iterator violated.";

    // const test templated dervied class
    const cv::Mat_<uchar> m_1d_const_t(static_cast<int>(sizes_1d.size()), sizes_1d.data(), data.data());

    auto mismatch_it_pair_1d_const_t = std::mismatch(data.rbegin(), data.rend(), m_1d_const_t.rbegin());
    EXPECT_EQ(mismatch_it_pair_1d_const_t.first, data.rend());  // expect no mismatch
    EXPECT_EQ(mismatch_it_pair_1d_const_t.second, m_1d_const_t.rend());

    EXPECT_FALSE((std::is_assignable<decltype(m_1d_const_t.rend()), uchar>::value)) << "Constness of const iterator violated.";
    EXPECT_FALSE((std::is_assignable<decltype(m_1d_const_t.rbegin()), uchar>::value)) << "Constness of const iterator violated.";

}

TEST(Mat, Recreate1DMatWithSameMeta)
{
    std::vector<int> dims = {100};
    auto depth = CV_8U;
    cv::Mat m(dims, depth);

    // By default m has dims: [1, 100]
    m.dims = 1;

    EXPECT_NO_THROW(m.create(dims, depth));
}

}} // namespace
