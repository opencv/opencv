// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

const int ARITHM_NTESTS = 1000;
const int ARITHM_RNG_SEED = -1;
const int ARITHM_MAX_CHANNELS = 4;
const int ARITHM_MAX_NDIMS = 4;
const int ARITHM_MAX_SIZE_LOG = 10;

struct BaseElemWiseOp
{
    enum { FIX_ALPHA=1, FIX_BETA=2, FIX_GAMMA=4, REAL_GAMMA=8, SUPPORT_MASK=16, SCALAR_OUTPUT=32, SUPPORT_MULTICHANNELMASK=64 };
    BaseElemWiseOp(int _ninputs, int _flags, double _alpha, double _beta,
                   Scalar _gamma=Scalar::all(0), int _context=1)
    : ninputs(_ninputs), flags(_flags), alpha(_alpha), beta(_beta), gamma(_gamma), context(_context) {}
    BaseElemWiseOp() { flags = 0; alpha = beta = 0; gamma = Scalar::all(0); ninputs = 0; context = 1; }
    virtual ~BaseElemWiseOp() {}
    virtual void op(const vector<Mat>&, Mat&, const Mat&) {}
    virtual void refop(const vector<Mat>&, Mat&, const Mat&) {}
    virtual void getValueRange(int depth, double& minval, double& maxval)
    {
        minval = depth < CV_32S ? cvtest::getMinVal(depth) : depth == CV_32S ? -1000000 : -1000.;
        maxval = depth < CV_32S ? cvtest::getMaxVal(depth) : depth == CV_32S ? 1000000 : 1000.;
    }

    virtual void getRandomSize(RNG& rng, vector<int>& size)
    {
        cvtest::randomSize(rng, 2, ARITHM_MAX_NDIMS, ARITHM_MAX_SIZE_LOG, size);
    }

    virtual int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, _OutputArray::DEPTH_MASK_ALL_BUT_8S, 1,
                                  ninputs > 1 ? ARITHM_MAX_CHANNELS : 4);
    }

    virtual double getMaxErr(int depth) { return depth < CV_32F ? 1 : depth == CV_32F ? 1e-5 : 1e-12; }
    virtual void generateScalars(int depth, RNG& rng)
    {
        const double m = 3.;

        if( !(flags & FIX_ALPHA) )
        {
            alpha = exp(rng.uniform(-0.5, 0.1)*m*2*CV_LOG2);
            alpha *= rng.uniform(0, 2) ? 1 : -1;
        }
        if( !(flags & FIX_BETA) )
        {
            beta = exp(rng.uniform(-0.5, 0.1)*m*2*CV_LOG2);
            beta *= rng.uniform(0, 2) ? 1 : -1;
        }

        if( !(flags & FIX_GAMMA) )
        {
            for( int i = 0; i < 4; i++ )
            {
                gamma[i] = exp(rng.uniform(-1, 6)*m*CV_LOG2);
                gamma[i] *= rng.uniform(0, 2) ? 1 : -1;
            }
            if( flags & REAL_GAMMA )
                gamma = Scalar::all(gamma[0]);
        }

        if( depth == CV_32F )
        {
            Mat fl, db;

            db = Mat(1, 1, CV_64F, &alpha);
            db.convertTo(fl, CV_32F);
            fl.convertTo(db, CV_64F);

            db = Mat(1, 1, CV_64F, &beta);
            db.convertTo(fl, CV_32F);
            fl.convertTo(db, CV_64F);

            db = Mat(1, 4, CV_64F, &gamma[0]);
            db.convertTo(fl, CV_32F);
            fl.convertTo(db, CV_64F);
        }
    }

    int ninputs;
    int flags;
    double alpha;
    double beta;
    Scalar gamma;
    int context;
};


struct BaseAddOp : public BaseElemWiseOp
{
    BaseAddOp(int _ninputs, int _flags, double _alpha, double _beta, Scalar _gamma=Scalar::all(0))
    : BaseElemWiseOp(_ninputs, _flags, _alpha, _beta, _gamma) {}

    void refop(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        Mat temp;
        if( !mask.empty() )
        {
            cvtest::add(src[0], alpha, src.size() > 1 ? src[1] : Mat(), beta, gamma, temp, src[0].type());
            cvtest::copy(temp, dst, mask);
        }
        else
            cvtest::add(src[0], alpha, src.size() > 1 ? src[1] : Mat(), beta, gamma, dst, src[0].type());
    }
};


struct AddOp : public BaseAddOp
{
    AddOp() : BaseAddOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( mask.empty() )
            cv::add(src[0], src[1], dst);
        else
            cv::add(src[0], src[1], dst, mask);
    }
};


struct SubOp : public BaseAddOp
{
    SubOp() : BaseAddOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK, 1, -1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( mask.empty() )
            cv::subtract(src[0], src[1], dst);
        else
            cv::subtract(src[0], src[1], dst, mask);
    }
};


struct AddSOp : public BaseAddOp
{
    AddSOp() : BaseAddOp(1, FIX_ALPHA+FIX_BETA+SUPPORT_MASK, 1, 0, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( mask.empty() )
            cv::add(src[0], gamma, dst);
        else
            cv::add(src[0], gamma, dst, mask);
    }
};


struct SubRSOp : public BaseAddOp
{
    SubRSOp() : BaseAddOp(1, FIX_ALPHA+FIX_BETA+SUPPORT_MASK, -1, 0, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( mask.empty() )
            cv::subtract(gamma, src[0], dst);
        else
            cv::subtract(gamma, src[0], dst, mask);
    }
};


struct ScaleAddOp : public BaseAddOp
{
    ScaleAddOp() : BaseAddOp(2, FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::scaleAdd(src[0], alpha, src[1], dst);
    }
    double getMaxErr(int depth)
    {
        return depth <= CV_32S ? 2 : depth < CV_64F ? 1e-4 : 1e-12;
    }
};


struct AddWeightedOp : public BaseAddOp
{
    AddWeightedOp() : BaseAddOp(2, REAL_GAMMA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::addWeighted(src[0], alpha, src[1], beta, gamma[0], dst);
    }
    double getMaxErr(int depth)
    {
        return depth <= CV_32S ? 2 : depth < CV_64F ? 1e-5 : 1e-10;
    }
};

struct MulOp : public BaseElemWiseOp
{
    MulOp() : BaseElemWiseOp(2, FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {}
    void getValueRange(int depth, double& minval, double& maxval)
    {
        minval = depth < CV_32S ? cvtest::getMinVal(depth) : depth == CV_32S ? -1000000 : -1000.;
        maxval = depth < CV_32S ? cvtest::getMaxVal(depth) : depth == CV_32S ? 1000000 : 1000.;
        minval = std::max(minval, -30000.);
        maxval = std::min(maxval, 30000.);
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::multiply(src[0], src[1], dst, alpha);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::multiply(src[0], src[1], dst, alpha);
    }
    double getMaxErr(int depth)
    {
        return depth <= CV_32S ? 2 : depth < CV_64F ? 1e-5 : 1e-12;
    }
};

struct DivOp : public BaseElemWiseOp
{
    DivOp() : BaseElemWiseOp(2, FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::divide(src[0], src[1], dst, alpha);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::divide(src[0], src[1], dst, alpha);
    }
    double getMaxErr(int depth)
    {
        return depth <= CV_32S ? 2 : depth < CV_64F ? 1e-5 : 1e-12;
    }
};

struct RecipOp : public BaseElemWiseOp
{
    RecipOp() : BaseElemWiseOp(1, FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::divide(alpha, src[0], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::divide(Mat(), src[0], dst, alpha);
    }
    double getMaxErr(int depth)
    {
        return depth <= CV_32S ? 2 : depth < CV_64F ? 1e-5 : 1e-12;
    }
};

struct AbsDiffOp : public BaseAddOp
{
    AbsDiffOp() : BaseAddOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, -1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        absdiff(src[0], src[1], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::add(src[0], 1, src[1], -1, Scalar::all(0), dst, src[0].type(), true);
    }
};

struct AbsDiffSOp : public BaseAddOp
{
    AbsDiffSOp() : BaseAddOp(1, FIX_ALPHA+FIX_BETA, 1, 0, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        absdiff(src[0], gamma, dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::add(src[0], 1, Mat(), 0, -gamma, dst, src[0].type(), true);
    }
};

struct LogicOp : public BaseElemWiseOp
{
    LogicOp(char _opcode) : BaseElemWiseOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK, 1, 1, Scalar::all(0)), opcode(_opcode) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( opcode == '&' )
            cv::bitwise_and(src[0], src[1], dst, mask);
        else if( opcode == '|' )
            cv::bitwise_or(src[0], src[1], dst, mask);
        else
            cv::bitwise_xor(src[0], src[1], dst, mask);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        Mat temp;
        if( !mask.empty() )
        {
            cvtest::logicOp(src[0], src[1], temp, opcode);
            cvtest::copy(temp, dst, mask);
        }
        else
            cvtest::logicOp(src[0], src[1], dst, opcode);
    }
    double getMaxErr(int)
    {
        return 0;
    }
    char opcode;
};

struct LogicSOp : public BaseElemWiseOp
{
    LogicSOp(char _opcode)
    : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+(_opcode != '~' ? SUPPORT_MASK : 0), 1, 1, Scalar::all(0)), opcode(_opcode) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( opcode == '&' )
            cv::bitwise_and(src[0], gamma, dst, mask);
        else if( opcode == '|' )
            cv::bitwise_or(src[0], gamma, dst, mask);
        else if( opcode == '^' )
            cv::bitwise_xor(src[0], gamma, dst, mask);
        else
            cv::bitwise_not(src[0], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        Mat temp;
        if( !mask.empty() )
        {
            cvtest::logicOp(src[0], gamma, temp, opcode);
            cvtest::copy(temp, dst, mask);
        }
        else
            cvtest::logicOp(src[0], gamma, dst, opcode);
    }
    double getMaxErr(int)
    {
        return 0;
    }
    char opcode;
};

struct MinOp : public BaseElemWiseOp
{
    MinOp() : BaseElemWiseOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::min(src[0], src[1], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::min(src[0], src[1], dst);
    }
    double getMaxErr(int)
    {
        return 0;
    }
};

struct MaxOp : public BaseElemWiseOp
{
    MaxOp() : BaseElemWiseOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::max(src[0], src[1], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::max(src[0], src[1], dst);
    }
    double getMaxErr(int)
    {
        return 0;
    }
};

struct MinSOp : public BaseElemWiseOp
{
    MinSOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+REAL_GAMMA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::min(src[0], gamma[0], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::min(src[0], gamma[0], dst);
    }
    double getMaxErr(int)
    {
        return 0;
    }
};

struct MaxSOp : public BaseElemWiseOp
{
    MaxSOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+REAL_GAMMA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::max(src[0], gamma[0], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::max(src[0], gamma[0], dst);
    }
    double getMaxErr(int)
    {
        return 0;
    }
};

struct CmpOp : public BaseElemWiseOp
{
    CmpOp() : BaseElemWiseOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) { cmpop = 0; }
    void generateScalars(int depth, RNG& rng)
    {
        BaseElemWiseOp::generateScalars(depth, rng);
        cmpop = rng.uniform(0, 6);
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::compare(src[0], src[1], dst, cmpop);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::compare(src[0], src[1], dst, cmpop);
    }
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, _OutputArray::DEPTH_MASK_ALL_BUT_8S, 1, 1);
    }

    double getMaxErr(int)
    {
        return 0;
    }
    int cmpop;
};

struct CmpSOp : public BaseElemWiseOp
{
    CmpSOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+REAL_GAMMA, 1, 1, Scalar::all(0)) { cmpop = 0; }
    void generateScalars(int depth, RNG& rng)
    {
        BaseElemWiseOp::generateScalars(depth, rng);
        cmpop = rng.uniform(0, 6);
        if( depth < CV_32F )
            gamma[0] = cvRound(gamma[0]);
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::compare(src[0], gamma[0], dst, cmpop);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::compare(src[0], gamma[0], dst, cmpop);
    }
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, _OutputArray::DEPTH_MASK_ALL_BUT_8S, 1, 1);
    }
    double getMaxErr(int)
    {
        return 0;
    }
    int cmpop;
};


struct CopyOp : public BaseElemWiseOp
{
    CopyOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK+SUPPORT_MULTICHANNELMASK, 1, 1, Scalar::all(0)) {  }
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        src[0].copyTo(dst, mask);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        cvtest::copy(src[0], dst, mask);
    }
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, _OutputArray::DEPTH_MASK_ALL, 1, ARITHM_MAX_CHANNELS);
    }
    double getMaxErr(int)
    {
        return 0;
    }
};


struct SetOp : public BaseElemWiseOp
{
    SetOp() : BaseElemWiseOp(0, FIX_ALPHA+FIX_BETA+SUPPORT_MASK+SUPPORT_MULTICHANNELMASK, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>&, Mat& dst, const Mat& mask)
    {
        dst.setTo(gamma, mask);
    }
    void refop(const vector<Mat>&, Mat& dst, const Mat& mask)
    {
        cvtest::set(dst, gamma, mask);
    }
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, _OutputArray::DEPTH_MASK_ALL, 1, ARITHM_MAX_CHANNELS);
    }
    double getMaxErr(int)
    {
        return 0;
    }
};

template<typename _Tp, typename _WTp> static void
inRangeS_(const _Tp* src, const _WTp* a, const _WTp* b, uchar* dst, size_t total, int cn)
{
    size_t i;
    int c;
    for( i = 0; i < total; i++ )
    {
        _Tp val = src[i*cn];
        dst[i] = (a[0] <= val && val <= b[0]) ? uchar(255) : 0;
    }
    for( c = 1; c < cn; c++ )
    {
        for( i = 0; i < total; i++ )
        {
            _Tp val = src[i*cn + c];
            dst[i] = a[c] <= val && val <= b[c] ? dst[i] : 0;
        }
    }
}

template<typename _Tp> static void inRange_(const _Tp* src, const _Tp* a, const _Tp* b, uchar* dst, size_t total, int cn)
{
    size_t i;
    int c;
    for( i = 0; i < total; i++ )
    {
        _Tp val = src[i*cn];
        dst[i] = a[i*cn] <= val && val <= b[i*cn] ? 255 : 0;
    }
    for( c = 1; c < cn; c++ )
    {
        for( i = 0; i < total; i++ )
        {
            _Tp val = src[i*cn + c];
            dst[i] = a[i*cn + c] <= val && val <= b[i*cn + c] ? dst[i] : 0;
        }
    }
}

namespace reference {

static void inRange(const Mat& src, const Mat& lb, const Mat& rb, Mat& dst)
{
    CV_Assert( src.type() == lb.type() && src.type() == rb.type() &&
              src.size == lb.size && src.size == rb.size );
    dst.create( src.dims, &src.size[0], CV_8U );
    const Mat *arrays[]={&src, &lb, &rb, &dst, 0};
    Mat planes[4];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total();
    size_t i, nplanes = it.nplanes;
    int depth = src.depth(), cn = src.channels();

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].ptr();
        const uchar* aptr = planes[1].ptr();
        const uchar* bptr = planes[2].ptr();
        uchar* dptr = planes[3].ptr();

        switch( depth )
        {
        case CV_8U:
            inRange_((const uchar*)sptr, (const uchar*)aptr, (const uchar*)bptr, dptr, total, cn);
            break;
        case CV_8S:
            inRange_((const schar*)sptr, (const schar*)aptr, (const schar*)bptr, dptr, total, cn);
            break;
        case CV_16U:
            inRange_((const ushort*)sptr, (const ushort*)aptr, (const ushort*)bptr, dptr, total, cn);
            break;
        case CV_16S:
            inRange_((const short*)sptr, (const short*)aptr, (const short*)bptr, dptr, total, cn);
            break;
        case CV_32S:
            inRange_((const int*)sptr, (const int*)aptr, (const int*)bptr, dptr, total, cn);
            break;
        case CV_32F:
            inRange_((const float*)sptr, (const float*)aptr, (const float*)bptr, dptr, total, cn);
            break;
        case CV_64F:
            inRange_((const double*)sptr, (const double*)aptr, (const double*)bptr, dptr, total, cn);
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "");
        }
    }
}

static void inRangeS(const Mat& src, const Scalar& lb, const Scalar& rb, Mat& dst)
{
    dst.create( src.dims, &src.size[0], CV_8U );
    const Mat *arrays[]={&src, &dst, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total();
    size_t i, nplanes = it.nplanes;
    int depth = src.depth(), cn = src.channels();
    union { double d[4]; float f[4]; int i[4];} lbuf, rbuf;
    int wtype = CV_MAKETYPE(depth <= CV_32S ? CV_32S : depth, cn);
    scalarToRawData(lb, lbuf.d, wtype, cn);
    scalarToRawData(rb, rbuf.d, wtype, cn);

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].ptr();
        uchar* dptr = planes[1].ptr();

        switch( depth )
        {
        case CV_8U:
            inRangeS_((const uchar*)sptr, lbuf.i, rbuf.i, dptr, total, cn);
            break;
        case CV_8S:
            inRangeS_((const schar*)sptr, lbuf.i, rbuf.i, dptr, total, cn);
            break;
        case CV_16U:
            inRangeS_((const ushort*)sptr, lbuf.i, rbuf.i, dptr, total, cn);
            break;
        case CV_16S:
            inRangeS_((const short*)sptr, lbuf.i, rbuf.i, dptr, total, cn);
            break;
        case CV_32S:
            inRangeS_((const int*)sptr, lbuf.i, rbuf.i, dptr, total, cn);
            break;
        case CV_32F:
            inRangeS_((const float*)sptr, lbuf.f, rbuf.f, dptr, total, cn);
            break;
        case CV_64F:
            inRangeS_((const double*)sptr, lbuf.d, rbuf.d, dptr, total, cn);
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "");
        }
    }
}

} // namespace
CVTEST_GUARD_SYMBOL(inRange);

struct InRangeSOp : public BaseElemWiseOp
{
    InRangeSOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::inRange(src[0], gamma, gamma1, dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        reference::inRangeS(src[0], gamma, gamma1, dst);
    }
    double getMaxErr(int)
    {
        return 0;
    }
    void generateScalars(int depth, RNG& rng)
    {
        BaseElemWiseOp::generateScalars(depth, rng);
        Scalar temp = gamma;
        BaseElemWiseOp::generateScalars(depth, rng);
        for( int i = 0; i < 4; i++ )
        {
            gamma1[i] = std::max(gamma[i], temp[i]);
            gamma[i] = std::min(gamma[i], temp[i]);
        }
    }
    Scalar gamma1;
};


struct InRangeOp : public BaseElemWiseOp
{
    InRangeOp() : BaseElemWiseOp(3, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        Mat lb, rb;
        cvtest::min(src[1], src[2], lb);
        cvtest::max(src[1], src[2], rb);

        cv::inRange(src[0], lb, rb, dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        Mat lb, rb;
        cvtest::min(src[1], src[2], lb);
        cvtest::max(src[1], src[2], rb);

        reference::inRange(src[0], lb, rb, dst);
    }
    double getMaxErr(int)
    {
        return 0;
    }
};


struct ConvertScaleOp : public BaseElemWiseOp
{
    ConvertScaleOp() : BaseElemWiseOp(1, FIX_BETA+REAL_GAMMA, 1, 1, Scalar::all(0)), ddepth(0) { }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        src[0].convertTo(dst, ddepth, alpha, gamma[0]);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::convert(src[0], dst, CV_MAKETYPE(ddepth, src[0].channels()), alpha, gamma[0]);
    }
    int getRandomType(RNG& rng)
    {
        int srctype = cvtest::randomType(rng, _OutputArray::DEPTH_MASK_ALL, 1, ARITHM_MAX_CHANNELS);
        ddepth = cvtest::randomType(rng, _OutputArray::DEPTH_MASK_ALL, 1, 1);
        return srctype;
    }
    double getMaxErr(int)
    {
        return ddepth <= CV_32S ? 2 : ddepth < CV_64F ? 1e-3 : 1e-12;
    }
    void generateScalars(int depth, RNG& rng)
    {
        if( rng.uniform(0, 2) )
            BaseElemWiseOp::generateScalars(depth, rng);
        else
        {
            alpha = 1;
            gamma = Scalar::all(0);
        }
    }
    int ddepth;
};

struct ConvertScaleFp16Op : public BaseElemWiseOp
{
    ConvertScaleFp16Op() : BaseElemWiseOp(1, FIX_BETA+REAL_GAMMA, 1, 1, Scalar::all(0)), nextRange(0) { }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        Mat m;
        convertFp16(src[0], m);
        convertFp16(m, dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::copy(src[0], dst);
    }
    int getRandomType(RNG&)
    {
        // 0: FP32 -> FP16 -> FP32
        // 1: FP16 -> FP32 -> FP16
        int srctype = (nextRange & 1) == 0 ? CV_32F : CV_16S;
        return srctype;
    }
    void getValueRange(int, double& minval, double& maxval)
    {
        // 0: FP32 -> FP16 -> FP32
        // 1: FP16 -> FP32 -> FP16
        if( (nextRange & 1) == 0 )
        {
            // largest integer number that fp16 can express exactly
            maxval = 2048.f;
            minval = -maxval;
        }
        else
        {
            // 0: positive number range
            // 1: negative number range
            if( (nextRange & 2) == 0 )
            {
                minval = 0;      // 0x0000 +0
                maxval = 31744;  // 0x7C00 +Inf
            }
            else
            {
                minval = -32768; // 0x8000 -0
                maxval = -1024;  // 0xFC00 -Inf
            }
        }
    }
    double getMaxErr(int)
    {
        return 0.5f;
    }
    void generateScalars(int, RNG& rng)
    {
        nextRange = rng.next();
    }
    int nextRange;
};

struct ConvertScaleAbsOp : public BaseElemWiseOp
{
    ConvertScaleAbsOp() : BaseElemWiseOp(1, FIX_BETA+REAL_GAMMA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::convertScaleAbs(src[0], dst, alpha, gamma[0]);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::add(src[0], alpha, Mat(), 0, Scalar::all(gamma[0]), dst, CV_8UC(src[0].channels()), true);
    }
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, _OutputArray::DEPTH_MASK_ALL, 1,
            ninputs > 1 ? ARITHM_MAX_CHANNELS : 4);
    }
    double getMaxErr(int)
    {
        return 1;
    }
    void generateScalars(int depth, RNG& rng)
    {
        if( rng.uniform(0, 2) )
            BaseElemWiseOp::generateScalars(depth, rng);
        else
        {
            alpha = 1;
            gamma = Scalar::all(0);
        }
    }
};

namespace reference {

static void flip(const Mat& src, Mat& dst, int flipcode)
{
    CV_Assert(src.dims == 2);
    dst.create(src.size(), src.type());
    int i, j, k, esz = (int)src.elemSize(), width = src.cols*esz;

    for( i = 0; i < dst.rows; i++ )
    {
        const uchar* sptr = src.ptr(flipcode == 1 ? i : dst.rows - i - 1);
        uchar* dptr = dst.ptr(i);
        if( flipcode == 0 )
            memcpy(dptr, sptr, width);
        else
        {
            for( j = 0; j < width; j += esz )
                for( k = 0; k < esz; k++ )
                    dptr[j + k] = sptr[width - j - esz + k];
        }
    }
}


static void setIdentity(Mat& dst, const Scalar& s)
{
    CV_Assert( dst.dims == 2 && dst.channels() <= 4 );
    double buf[4];
    scalarToRawData(s, buf, dst.type(), 0);
    int i, k, esz = (int)dst.elemSize(), width = dst.cols*esz;

    for( i = 0; i < dst.rows; i++ )
    {
        uchar* dptr = dst.ptr(i);
        memset( dptr, 0, width );
        if( i < dst.cols )
            for( k = 0; k < esz; k++ )
                dptr[i*esz + k] = ((uchar*)buf)[k];
    }
}

} // namespace

struct FlipOp : public BaseElemWiseOp
{
    FlipOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) { flipcode = 0; }
    void getRandomSize(RNG& rng, vector<int>& size)
    {
        cvtest::randomSize(rng, 2, 2, ARITHM_MAX_SIZE_LOG, size);
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::flip(src[0], dst, flipcode);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        reference::flip(src[0], dst, flipcode);
    }
    void generateScalars(int, RNG& rng)
    {
        flipcode = rng.uniform(0, 3) - 1;
    }
    double getMaxErr(int)
    {
        return 0;
    }
    int flipcode;
};

struct TransposeOp : public BaseElemWiseOp
{
    TransposeOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {}
    void getRandomSize(RNG& rng, vector<int>& size)
    {
        cvtest::randomSize(rng, 2, 2, ARITHM_MAX_SIZE_LOG, size);
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::transpose(src[0], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::transpose(src[0], dst);
    }
    double getMaxErr(int)
    {
        return 0;
    }
};

struct SetIdentityOp : public BaseElemWiseOp
{
    SetIdentityOp() : BaseElemWiseOp(0, FIX_ALPHA+FIX_BETA, 1, 1, Scalar::all(0)) {}
    void getRandomSize(RNG& rng, vector<int>& size)
    {
        cvtest::randomSize(rng, 2, 2, ARITHM_MAX_SIZE_LOG, size);
    }
    void op(const vector<Mat>&, Mat& dst, const Mat&)
    {
        cv::setIdentity(dst, gamma);
    }
    void refop(const vector<Mat>&, Mat& dst, const Mat&)
    {
        reference::setIdentity(dst, gamma);
    }
    double getMaxErr(int)
    {
        return 0;
    }
};

struct SetZeroOp : public BaseElemWiseOp
{
    SetZeroOp() : BaseElemWiseOp(0, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {}
    void op(const vector<Mat>&, Mat& dst, const Mat&)
    {
        dst = Scalar::all(0);
    }
    void refop(const vector<Mat>&, Mat& dst, const Mat&)
    {
        cvtest::set(dst, Scalar::all(0));
    }
    double getMaxErr(int)
    {
        return 0;
    }
};

namespace reference {
static void exp(const Mat& src, Mat& dst)
{
    dst.create( src.dims, &src.size[0], src.type() );
    const Mat *arrays[]={&src, &dst, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t j, total = planes[0].total()*src.channels();
    size_t i, nplanes = it.nplanes;
    int depth = src.depth();

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].ptr();
        uchar* dptr = planes[1].ptr();

        if( depth == CV_32F )
        {
            for( j = 0; j < total; j++ )
                ((float*)dptr)[j] = std::exp(((const float*)sptr)[j]);
        }
        else if( depth == CV_64F )
        {
            for( j = 0; j < total; j++ )
                ((double*)dptr)[j] = std::exp(((const double*)sptr)[j]);
        }
    }
}

static void log(const Mat& src, Mat& dst)
{
    dst.create( src.dims, &src.size[0], src.type() );
    const Mat *arrays[]={&src, &dst, 0};
    Mat planes[2];

    NAryMatIterator it(arrays, planes);
    size_t j, total = planes[0].total()*src.channels();
    size_t i, nplanes = it.nplanes;
    int depth = src.depth();

    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].ptr();
        uchar* dptr = planes[1].ptr();

        if( depth == CV_32F )
        {
            for( j = 0; j < total; j++ )
                ((float*)dptr)[j] = (float)std::log(fabs(((const float*)sptr)[j]));
        }
        else if( depth == CV_64F )
        {
            for( j = 0; j < total; j++ )
                ((double*)dptr)[j] = std::log(fabs(((const double*)sptr)[j]));
        }
    }
}

} // namespace

struct ExpOp : public BaseElemWiseOp
{
    ExpOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {}
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, _OutputArray::DEPTH_MASK_FLT, 1, ARITHM_MAX_CHANNELS);
    }
    void getValueRange(int depth, double& minval, double& maxval)
    {
        maxval = depth == CV_32F ? 50 : 100;
        minval = -maxval;
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::exp(src[0], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        reference::exp(src[0], dst);
    }
    double getMaxErr(int depth)
    {
        return depth == CV_32F ? 1e-5 : 1e-12;
    }
};


struct LogOp : public BaseElemWiseOp
{
    LogOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {}
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, _OutputArray::DEPTH_MASK_FLT, 1, ARITHM_MAX_CHANNELS);
    }
    void getValueRange(int depth, double& minval, double& maxval)
    {
        maxval = depth == CV_32F ? 50 : 100;
        minval = -maxval;
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        Mat temp;
        reference::exp(src[0], temp);
        cv::log(temp, dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        Mat temp;
        reference::exp(src[0], temp);
        reference::log(temp, dst);
    }
    double getMaxErr(int depth)
    {
        return depth == CV_32F ? 1e-5 : 1e-12;
    }
};


namespace reference {
static void cartToPolar(const Mat& mx, const Mat& my, Mat& mmag, Mat& mangle, bool angleInDegrees)
{
    CV_Assert( (mx.type() == CV_32F || mx.type() == CV_64F) &&
              mx.type() == my.type() && mx.size == my.size );
    mmag.create( mx.dims, &mx.size[0], mx.type() );
    mangle.create( mx.dims, &mx.size[0], mx.type() );
    const Mat *arrays[]={&mx, &my, &mmag, &mangle, 0};
    Mat planes[4];

    NAryMatIterator it(arrays, planes);
    size_t j, total = planes[0].total();
    size_t i, nplanes = it.nplanes;
    int depth = mx.depth();
    double scale = angleInDegrees ? 180/CV_PI : 1;

    for( i = 0; i < nplanes; i++, ++it )
    {
        if( depth == CV_32F )
        {
            const float* xptr = planes[0].ptr<float>();
            const float* yptr = planes[1].ptr<float>();
            float* mptr = planes[2].ptr<float>();
            float* aptr = planes[3].ptr<float>();

            for( j = 0; j < total; j++ )
            {
                mptr[j] = std::sqrt(xptr[j]*xptr[j] + yptr[j]*yptr[j]);
                double a = atan2((double)yptr[j], (double)xptr[j]);
                if( a < 0 ) a += CV_PI*2;
                aptr[j] = (float)(a*scale);
            }
        }
        else
        {
            const double* xptr = planes[0].ptr<double>();
            const double* yptr = planes[1].ptr<double>();
            double* mptr = planes[2].ptr<double>();
            double* aptr = planes[3].ptr<double>();

            for( j = 0; j < total; j++ )
            {
                mptr[j] = std::sqrt(xptr[j]*xptr[j] + yptr[j]*yptr[j]);
                double a = atan2(yptr[j], xptr[j]);
                if( a < 0 ) a += CV_PI*2;
                aptr[j] = a*scale;
            }
        }
    }
}

} // namespace

struct CartToPolarToCartOp : public BaseElemWiseOp
{
    CartToPolarToCartOp() : BaseElemWiseOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0))
    {
        context = 3;
        angleInDegrees = true;
    }
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, _OutputArray::DEPTH_MASK_FLT, 1, 1);
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        Mat mag, angle, x, y;

        cv::cartToPolar(src[0], src[1], mag, angle, angleInDegrees);
        cv::polarToCart(mag, angle, x, y, angleInDegrees);

        Mat msrc[] = {mag, angle, x, y};
        int pairs[] = {0, 0, 1, 1, 2, 2, 3, 3};
        dst.create(src[0].dims, src[0].size, CV_MAKETYPE(src[0].depth(), 4));
        cv::mixChannels(msrc, 4, &dst, 1, pairs, 4);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        Mat mag, angle;
        reference::cartToPolar(src[0], src[1], mag, angle, angleInDegrees);
        Mat msrc[] = {mag, angle, src[0], src[1]};
        int pairs[] = {0, 0, 1, 1, 2, 2, 3, 3};
        dst.create(src[0].dims, src[0].size, CV_MAKETYPE(src[0].depth(), 4));
        cv::mixChannels(msrc, 4, &dst, 1, pairs, 4);
    }
    void generateScalars(int, RNG& rng)
    {
        angleInDegrees = rng.uniform(0, 2) != 0;
    }
    double getMaxErr(int)
    {
        return 1e-3;
    }
    bool angleInDegrees;
};


struct MeanOp : public BaseElemWiseOp
{
    MeanOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK+SCALAR_OUTPUT, 1, 1, Scalar::all(0))
    {
        context = 3;
    };
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        dst.create(1, 1, CV_64FC4);
        dst.at<Scalar>(0,0) = cv::mean(src[0], mask);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        dst.create(1, 1, CV_64FC4);
        dst.at<Scalar>(0,0) = cvtest::mean(src[0], mask);
    }
    double getMaxErr(int)
    {
        return 1e-5;
    }
};


struct SumOp : public BaseElemWiseOp
{
    SumOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SCALAR_OUTPUT, 1, 1, Scalar::all(0))
    {
        context = 3;
    };
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        dst.create(1, 1, CV_64FC4);
        dst.at<Scalar>(0,0) = cv::sum(src[0]);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        dst.create(1, 1, CV_64FC4);
        dst.at<Scalar>(0,0) = cvtest::mean(src[0])*(double)src[0].total();
    }
    double getMaxErr(int)
    {
        return 1e-5;
    }
};


struct CountNonZeroOp : public BaseElemWiseOp
{
    CountNonZeroOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SCALAR_OUTPUT+SUPPORT_MASK, 1, 1, Scalar::all(0))
    {}
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, _OutputArray::DEPTH_MASK_ALL, 1, 1);
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        Mat temp;
        src[0].copyTo(temp);
        if( !mask.empty() )
            temp.setTo(Scalar::all(0), mask);
        dst.create(1, 1, CV_32S);
        dst.at<int>(0,0) = cv::countNonZero(temp);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        Mat temp;
        cvtest::compare(src[0], 0, temp, CMP_NE);
        if( !mask.empty() )
            cvtest::set(temp, Scalar::all(0), mask);
        dst.create(1, 1, CV_32S);
        dst.at<int>(0,0) = saturate_cast<int>(cvtest::mean(temp)[0]/255*temp.total());
    }
    double getMaxErr(int)
    {
        return 0;
    }
};


struct MeanStdDevOp : public BaseElemWiseOp
{
    Scalar sqmeanRef;
    int cn;

    MeanStdDevOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK+SCALAR_OUTPUT, 1, 1, Scalar::all(0))
    {
        cn = 0;
        context = 7;
    };
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        dst.create(1, 2, CV_64FC4);
        cv::meanStdDev(src[0], dst.at<Scalar>(0,0), dst.at<Scalar>(0,1), mask);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        Mat temp;
        cvtest::convert(src[0], temp, CV_64F);
        cvtest::multiply(temp, temp, temp);
        Scalar mean = cvtest::mean(src[0], mask);
        Scalar sqmean = cvtest::mean(temp, mask);

        sqmeanRef = sqmean;
        cn = temp.channels();

        for( int c = 0; c < 4; c++ )
            sqmean[c] = std::sqrt(std::max(sqmean[c] - mean[c]*mean[c], 0.));

        dst.create(1, 2, CV_64FC4);
        dst.at<Scalar>(0,0) = mean;
        dst.at<Scalar>(0,1) = sqmean;
    }
    double getMaxErr(int)
    {
        CV_Assert(cn > 0);
        double err = sqmeanRef[0];
        for(int i = 1; i < cn; ++i)
            err = std::max(err, sqmeanRef[i]);
        return 3e-7 * err;
    }
};


struct NormOp : public BaseElemWiseOp
{
    NormOp() : BaseElemWiseOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK+SCALAR_OUTPUT, 1, 1, Scalar::all(0))
    {
        context = 1;
        normType = 0;
    };
    int getRandomType(RNG& rng)
    {
        int type = cvtest::randomType(rng, _OutputArray::DEPTH_MASK_ALL_BUT_8S, 1, 4);
        for(;;)
        {
            normType = rng.uniform(1, 8);
            if( normType == NORM_INF || normType == NORM_L1 ||
                normType == NORM_L2 || normType == NORM_L2SQR ||
                normType == NORM_HAMMING || normType == NORM_HAMMING2 )
                break;
        }
        if( normType == NORM_HAMMING || normType == NORM_HAMMING2 )
        {
            type = CV_8U;
        }
        return type;
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        dst.create(1, 2, CV_64FC1);
        dst.at<double>(0,0) = cv::norm(src[0], normType, mask);
        dst.at<double>(0,1) = cv::norm(src[0], src[1], normType, mask);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        dst.create(1, 2, CV_64FC1);
        dst.at<double>(0,0) = cvtest::norm(src[0], normType, mask);
        dst.at<double>(0,1) = cvtest::norm(src[0], src[1], normType, mask);
    }
    void generateScalars(int, RNG& /*rng*/)
    {
    }
    double getMaxErr(int)
    {
        return 1e-6;
    }
    int normType;
};


struct MinMaxLocOp : public BaseElemWiseOp
{
    MinMaxLocOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK+SCALAR_OUTPUT, 1, 1, Scalar::all(0))
    {
        context = ARITHM_MAX_NDIMS*2 + 2;
    };
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, _OutputArray::DEPTH_MASK_ALL_BUT_8S, 1, 1);
    }
    void saveOutput(const vector<int>& minidx, const vector<int>& maxidx,
                    double minval, double maxval, Mat& dst)
    {
        int i, ndims = (int)minidx.size();
        dst.create(1, ndims*2 + 2, CV_64FC1);

        for( i = 0; i < ndims; i++ )
        {
            dst.at<double>(0,i) = minidx[i];
            dst.at<double>(0,i+ndims) = maxidx[i];
        }
        dst.at<double>(0,ndims*2) = minval;
        dst.at<double>(0,ndims*2+1) = maxval;
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        int ndims = src[0].dims;
        vector<int> minidx(ndims), maxidx(ndims);
        double minval=0, maxval=0;
        cv::minMaxIdx(src[0], &minval, &maxval, &minidx[0], &maxidx[0], mask);
        saveOutput(minidx, maxidx, minval, maxval, dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        int ndims=src[0].dims;
        vector<int> minidx(ndims), maxidx(ndims);
        double minval=0, maxval=0;
        cvtest::minMaxLoc(src[0], &minval, &maxval, &minidx, &maxidx, mask);
        saveOutput(minidx, maxidx, minval, maxval, dst);
    }
    double getMaxErr(int)
    {
        return 0;
    }
};


typedef Ptr<BaseElemWiseOp> ElemWiseOpPtr;
class ElemWiseTest : public ::testing::TestWithParam<ElemWiseOpPtr> {};

TEST_P(ElemWiseTest, accuracy)
{
    ElemWiseOpPtr op = GetParam();

    int testIdx = 0;
    RNG rng((uint64)ARITHM_RNG_SEED);
    for( testIdx = 0; testIdx < ARITHM_NTESTS; testIdx++ )
    {
        vector<int> size;
        op->getRandomSize(rng, size);
        int type = op->getRandomType(rng);
        int depth = CV_MAT_DEPTH(type);
        bool haveMask = ((op->flags & BaseElemWiseOp::SUPPORT_MASK) != 0
                || (op->flags & BaseElemWiseOp::SUPPORT_MULTICHANNELMASK) != 0) && rng.uniform(0, 4) == 0;

        double minval=0, maxval=0;
        op->getValueRange(depth, minval, maxval);
        int i, ninputs = op->ninputs;
        vector<Mat> src(ninputs);
        for( i = 0; i < ninputs; i++ )
            src[i] = cvtest::randomMat(rng, size, type, minval, maxval, true);
        Mat dst0, dst, mask;
        if( haveMask ) {
            bool multiChannelMask = (op->flags & BaseElemWiseOp::SUPPORT_MULTICHANNELMASK) != 0
                    && rng.uniform(0, 2) == 0;
            int masktype = CV_8UC(multiChannelMask ? CV_MAT_CN(type) : 1);
            mask = cvtest::randomMat(rng, size, masktype, 0, 2, true);
        }

        if( (haveMask || ninputs == 0) && !(op->flags & BaseElemWiseOp::SCALAR_OUTPUT))
        {
            dst0 = cvtest::randomMat(rng, size, type, minval, maxval, false);
            dst = cvtest::randomMat(rng, size, type, minval, maxval, true);
            cvtest::copy(dst, dst0);
        }
        op->generateScalars(depth, rng);

        op->refop(src, dst0, mask);
        op->op(src, dst, mask);

        double maxErr = op->getMaxErr(depth);
        ASSERT_PRED_FORMAT2(cvtest::MatComparator(maxErr, op->context), dst0, dst) << "\nsrc[0] ~ " <<
            cvtest::MatInfo(!src.empty() ? src[0] : Mat()) << "\ntestCase #" << testIdx << "\n";
    }
}


INSTANTIATE_TEST_CASE_P(Core_Copy, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new CopyOp)));
INSTANTIATE_TEST_CASE_P(Core_Set, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new SetOp)));
INSTANTIATE_TEST_CASE_P(Core_SetZero, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new SetZeroOp)));
INSTANTIATE_TEST_CASE_P(Core_ConvertScale, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new ConvertScaleOp)));
INSTANTIATE_TEST_CASE_P(Core_ConvertScaleFp16, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new ConvertScaleFp16Op)));
INSTANTIATE_TEST_CASE_P(Core_ConvertScaleAbs, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new ConvertScaleAbsOp)));

INSTANTIATE_TEST_CASE_P(Core_Add, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new AddOp)));
INSTANTIATE_TEST_CASE_P(Core_Sub, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new SubOp)));
INSTANTIATE_TEST_CASE_P(Core_AddS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new AddSOp)));
INSTANTIATE_TEST_CASE_P(Core_SubRS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new SubRSOp)));
INSTANTIATE_TEST_CASE_P(Core_ScaleAdd, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new ScaleAddOp)));
INSTANTIATE_TEST_CASE_P(Core_AddWeighted, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new AddWeightedOp)));
INSTANTIATE_TEST_CASE_P(Core_AbsDiff, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new AbsDiffOp)));


INSTANTIATE_TEST_CASE_P(Core_AbsDiffS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new AbsDiffSOp)));

INSTANTIATE_TEST_CASE_P(Core_And, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new LogicOp('&'))));
INSTANTIATE_TEST_CASE_P(Core_AndS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new LogicSOp('&'))));
INSTANTIATE_TEST_CASE_P(Core_Or, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new LogicOp('|'))));
INSTANTIATE_TEST_CASE_P(Core_OrS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new LogicSOp('|'))));
INSTANTIATE_TEST_CASE_P(Core_Xor, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new LogicOp('^'))));
INSTANTIATE_TEST_CASE_P(Core_XorS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new LogicSOp('^'))));
INSTANTIATE_TEST_CASE_P(Core_Not, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new LogicSOp('~'))));

INSTANTIATE_TEST_CASE_P(Core_Max, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new MaxOp)));
INSTANTIATE_TEST_CASE_P(Core_MaxS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new MaxSOp)));
INSTANTIATE_TEST_CASE_P(Core_Min, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new MinOp)));
INSTANTIATE_TEST_CASE_P(Core_MinS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new MinSOp)));

INSTANTIATE_TEST_CASE_P(Core_Mul, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new MulOp)));
INSTANTIATE_TEST_CASE_P(Core_Div, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new DivOp)));
INSTANTIATE_TEST_CASE_P(Core_Recip, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new RecipOp)));

INSTANTIATE_TEST_CASE_P(Core_Cmp, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new CmpOp)));
INSTANTIATE_TEST_CASE_P(Core_CmpS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new CmpSOp)));

INSTANTIATE_TEST_CASE_P(Core_InRangeS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new InRangeSOp)));
INSTANTIATE_TEST_CASE_P(Core_InRange, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new InRangeOp)));

INSTANTIATE_TEST_CASE_P(Core_Flip, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new FlipOp)));
INSTANTIATE_TEST_CASE_P(Core_Transpose, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new TransposeOp)));
INSTANTIATE_TEST_CASE_P(Core_SetIdentity, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new SetIdentityOp)));

INSTANTIATE_TEST_CASE_P(Core_Exp, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new ExpOp)));
INSTANTIATE_TEST_CASE_P(Core_Log, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new LogOp)));

INSTANTIATE_TEST_CASE_P(Core_CountNonZero, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new CountNonZeroOp)));
INSTANTIATE_TEST_CASE_P(Core_Mean, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new MeanOp)));
INSTANTIATE_TEST_CASE_P(Core_MeanStdDev, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new MeanStdDevOp)));
INSTANTIATE_TEST_CASE_P(Core_Sum, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new SumOp)));
INSTANTIATE_TEST_CASE_P(Core_Norm, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new NormOp)));
INSTANTIATE_TEST_CASE_P(Core_MinMaxLoc, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new MinMaxLocOp)));
INSTANTIATE_TEST_CASE_P(Core_CartToPolarToCart, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new CartToPolarToCartOp)));


TEST(Core_ArithmMask, uninitialized)
{
            RNG& rng = theRNG();
            const int MAX_DIM=3;
            int sizes[MAX_DIM];
            for( int iter = 0; iter < 100; iter++ )
            {
                int dims = rng.uniform(1, MAX_DIM+1);
                int depth = rng.uniform(CV_8U, CV_64F+1);
                int cn = rng.uniform(1, 6);
                int type = CV_MAKETYPE(depth, cn);
                int op = rng.uniform(0, depth < CV_32F ? 5 : 2); // don't run binary operations between floating-point values
                int depth1 = op <= 1 ? CV_64F : depth;
                for (int k = 0; k < MAX_DIM; k++)
                {
                    sizes[k] = k < dims ? rng.uniform(1, 30) : 0;
                }
                SCOPED_TRACE(cv::format("iter=%d dims=%d depth=%d cn=%d type=%d op=%d depth1=%d dims=[%d; %d; %d]",
                                         iter,   dims,   depth,   cn,   type,   op,   depth1, sizes[0], sizes[1], sizes[2]));

                Mat a(dims, sizes, type), a1;
                Mat b(dims, sizes, type), b1;
                Mat mask(dims, sizes, CV_8U);
                Mat mask1;
                Mat c, d;

                rng.fill(a, RNG::UNIFORM, 0, 100);
                rng.fill(b, RNG::UNIFORM, 0, 100);

                // [-2,2) range means that the each generated random number
                // will be one of -2, -1, 0, 1. Saturated to [0,255], it will become
                // 0, 0, 0, 1 => the mask will be filled by ~25%.
                rng.fill(mask, RNG::UNIFORM, -2, 2);

                a.convertTo(a1, depth1);
                b.convertTo(b1, depth1);
                // invert the mask
                cv::compare(mask, 0, mask1, CMP_EQ);
                a1.setTo(0, mask1);
                b1.setTo(0, mask1);

                if( op == 0 )
                {
                    cv::add(a, b, c, mask);
                    cv::add(a1, b1, d);
                }
                else if( op == 1 )
                {
                    cv::subtract(a, b, c, mask);
                    cv::subtract(a1, b1, d);
                }
                else if( op == 2 )
                {
                    cv::bitwise_and(a, b, c, mask);
                    cv::bitwise_and(a1, b1, d);
                }
                else if( op == 3 )
                {
                    cv::bitwise_or(a, b, c, mask);
                    cv::bitwise_or(a1, b1, d);
                }
                else if( op == 4 )
                {
                    cv::bitwise_xor(a, b, c, mask);
                    cv::bitwise_xor(a1, b1, d);
                }
                Mat d1;
                d.convertTo(d1, depth);
                EXPECT_LE(cvtest::norm(c, d1, CV_C), DBL_EPSILON);
            }

            Mat_<uchar> tmpSrc(100,100);
            tmpSrc = 124;
            Mat_<uchar> tmpMask(100,100);
            tmpMask = 255;
            Mat_<uchar> tmpDst(100,100);
            tmpDst = 2;
            tmpSrc.copyTo(tmpDst,tmpMask);
}

TEST(Multiply, FloatingPointRounding)
{
    cv::Mat src(1, 1, CV_8UC1, cv::Scalar::all(110)), dst;
    cv::Scalar s(147.286359696927, 1, 1 ,1);

    cv::multiply(src, s, dst, 1, CV_16U);
    // with CV_32F this produce result 16202
    ASSERT_EQ(dst.at<ushort>(0,0), 16201);
}

TEST(Core_Add, AddToColumnWhen3Rows)
{
    cv::Mat m1 = (cv::Mat_<double>(3, 2) << 1, 2, 3, 4, 5, 6);
    m1.col(1) += 10;

    cv::Mat m2 = (cv::Mat_<double>(3, 2) << 1, 12, 3, 14, 5, 16);

    ASSERT_EQ(0, countNonZero(m1 - m2));
}

TEST(Core_Add, AddToColumnWhen4Rows)
{
    cv::Mat m1 = (cv::Mat_<double>(4, 2) << 1, 2, 3, 4, 5, 6, 7, 8);
    m1.col(1) += 10;

    cv::Mat m2 = (cv::Mat_<double>(4, 2) << 1, 12, 3, 14, 5, 16, 7, 18);

    ASSERT_EQ(0, countNonZero(m1 - m2));
}

TEST(Core_round, CvRound)
{
    ASSERT_EQ(2, cvRound(2.0));
    ASSERT_EQ(2, cvRound(2.1));
    ASSERT_EQ(-2, cvRound(-2.1));
    ASSERT_EQ(3, cvRound(2.8));
    ASSERT_EQ(-3, cvRound(-2.8));
    ASSERT_EQ(2, cvRound(2.5));
    ASSERT_EQ(4, cvRound(3.5));
    ASSERT_EQ(-2, cvRound(-2.5));
    ASSERT_EQ(-4, cvRound(-3.5));
}


typedef testing::TestWithParam<Size> Mul1;

TEST_P(Mul1, One)
{
    Size size = GetParam();
    cv::Mat src(size, CV_32FC1, cv::Scalar::all(2)), dst,
            ref_dst(size, CV_32FC1, cv::Scalar::all(6));

    cv::multiply(3, src, dst);

    ASSERT_EQ(0, cvtest::norm(dst, ref_dst, cv::NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Arithm, Mul1, testing::Values(Size(2, 2), Size(1, 1)));

class SubtractOutputMatNotEmpty : public testing::TestWithParam< tuple<cv::Size, perf::MatType, perf::MatDepth, bool> >
{
public:
    cv::Size size;
    int src_type;
    int dst_depth;
    bool fixed;

    void SetUp()
    {
        size = get<0>(GetParam());
        src_type = get<1>(GetParam());
        dst_depth = get<2>(GetParam());
        fixed = get<3>(GetParam());
    }
};

TEST_P(SubtractOutputMatNotEmpty, Mat_Mat)
{
    cv::Mat src1(size, src_type, cv::Scalar::all(16));
    cv::Mat src2(size, src_type, cv::Scalar::all(16));

    cv::Mat dst;

    if (!fixed)
    {
        cv::subtract(src1, src2, dst, cv::noArray(), dst_depth);
    }
    else
    {
        const cv::Mat fixed_dst(size, CV_MAKE_TYPE((dst_depth > 0 ? dst_depth : CV_16S), src1.channels()));
        cv::subtract(src1, src2, fixed_dst, cv::noArray(), dst_depth);
        dst = fixed_dst;
        dst_depth = fixed_dst.depth();
    }

    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(src1.size(), dst.size());
    ASSERT_EQ(dst_depth > 0 ? dst_depth : src1.depth(), dst.depth());
    ASSERT_EQ(0, cv::countNonZero(dst.reshape(1)));
}

TEST_P(SubtractOutputMatNotEmpty, Mat_Mat_WithMask)
{
    cv::Mat src1(size, src_type, cv::Scalar::all(16));
    cv::Mat src2(size, src_type, cv::Scalar::all(16));
    cv::Mat mask(size, CV_8UC1, cv::Scalar::all(255));

    cv::Mat dst;

    if (!fixed)
    {
        cv::subtract(src1, src2, dst, mask, dst_depth);
    }
    else
    {
        const cv::Mat fixed_dst(size, CV_MAKE_TYPE((dst_depth > 0 ? dst_depth : CV_16S), src1.channels()));
        cv::subtract(src1, src2, fixed_dst, mask, dst_depth);
        dst = fixed_dst;
        dst_depth = fixed_dst.depth();
    }

    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(src1.size(), dst.size());
    ASSERT_EQ(dst_depth > 0 ? dst_depth : src1.depth(), dst.depth());
    ASSERT_EQ(0, cv::countNonZero(dst.reshape(1)));
}

TEST_P(SubtractOutputMatNotEmpty, Mat_Mat_Expr)
{
    cv::Mat src1(size, src_type, cv::Scalar::all(16));
    cv::Mat src2(size, src_type, cv::Scalar::all(16));

    cv::Mat dst = src1 - src2;

    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(src1.size(), dst.size());
    ASSERT_EQ(src1.depth(), dst.depth());
    ASSERT_EQ(0, cv::countNonZero(dst.reshape(1)));
}

TEST_P(SubtractOutputMatNotEmpty, Mat_Scalar)
{
    cv::Mat src(size, src_type, cv::Scalar::all(16));

    cv::Mat dst;

    if (!fixed)
    {
        cv::subtract(src, cv::Scalar::all(16), dst, cv::noArray(), dst_depth);
    }
    else
    {
        const cv::Mat fixed_dst(size, CV_MAKE_TYPE((dst_depth > 0 ? dst_depth : CV_16S), src.channels()));
        cv::subtract(src, cv::Scalar::all(16), fixed_dst, cv::noArray(), dst_depth);
        dst = fixed_dst;
        dst_depth = fixed_dst.depth();
    }

    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(src.size(), dst.size());
    ASSERT_EQ(dst_depth > 0 ? dst_depth : src.depth(), dst.depth());
    ASSERT_EQ(0, cv::countNonZero(dst.reshape(1)));
}

TEST_P(SubtractOutputMatNotEmpty, Mat_Scalar_WithMask)
{
    cv::Mat src(size, src_type, cv::Scalar::all(16));
    cv::Mat mask(size, CV_8UC1, cv::Scalar::all(255));

    cv::Mat dst;

    if (!fixed)
    {
        cv::subtract(src, cv::Scalar::all(16), dst, mask, dst_depth);
    }
    else
    {
        const cv::Mat fixed_dst(size, CV_MAKE_TYPE((dst_depth > 0 ? dst_depth : CV_16S), src.channels()));
        cv::subtract(src, cv::Scalar::all(16), fixed_dst, mask, dst_depth);
        dst = fixed_dst;
        dst_depth = fixed_dst.depth();
    }

    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(src.size(), dst.size());
    ASSERT_EQ(dst_depth > 0 ? dst_depth : src.depth(), dst.depth());
    ASSERT_EQ(0, cv::countNonZero(dst.reshape(1)));
}

TEST_P(SubtractOutputMatNotEmpty, Scalar_Mat)
{
    cv::Mat src(size, src_type, cv::Scalar::all(16));

    cv::Mat dst;

    if (!fixed)
    {
        cv::subtract(cv::Scalar::all(16), src, dst, cv::noArray(), dst_depth);
    }
    else
    {
        const cv::Mat fixed_dst(size, CV_MAKE_TYPE((dst_depth > 0 ? dst_depth : CV_16S), src.channels()));
        cv::subtract(cv::Scalar::all(16), src, fixed_dst, cv::noArray(), dst_depth);
        dst = fixed_dst;
        dst_depth = fixed_dst.depth();
    }

    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(src.size(), dst.size());
    ASSERT_EQ(dst_depth > 0 ? dst_depth : src.depth(), dst.depth());
    ASSERT_EQ(0, cv::countNonZero(dst.reshape(1)));
}

TEST_P(SubtractOutputMatNotEmpty, Scalar_Mat_WithMask)
{
    cv::Mat src(size, src_type, cv::Scalar::all(16));
    cv::Mat mask(size, CV_8UC1, cv::Scalar::all(255));

    cv::Mat dst;

    if (!fixed)
    {
        cv::subtract(cv::Scalar::all(16), src, dst, mask, dst_depth);
    }
    else
    {
        const cv::Mat fixed_dst(size, CV_MAKE_TYPE((dst_depth > 0 ? dst_depth : CV_16S), src.channels()));
        cv::subtract(cv::Scalar::all(16), src, fixed_dst, mask, dst_depth);
        dst = fixed_dst;
        dst_depth = fixed_dst.depth();
    }

    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(src.size(), dst.size());
    ASSERT_EQ(dst_depth > 0 ? dst_depth : src.depth(), dst.depth());
    ASSERT_EQ(0, cv::countNonZero(dst.reshape(1)));
}

TEST_P(SubtractOutputMatNotEmpty, Mat_Mat_3d)
{
    int dims[] = {5, size.height, size.width};

    cv::Mat src1(3, dims, src_type, cv::Scalar::all(16));
    cv::Mat src2(3, dims, src_type, cv::Scalar::all(16));

    cv::Mat dst;

    if (!fixed)
    {
        cv::subtract(src1, src2, dst, cv::noArray(), dst_depth);
    }
    else
    {
        const cv::Mat fixed_dst(3, dims, CV_MAKE_TYPE((dst_depth > 0 ? dst_depth : CV_16S), src1.channels()));
        cv::subtract(src1, src2, fixed_dst, cv::noArray(), dst_depth);
        dst = fixed_dst;
        dst_depth = fixed_dst.depth();
    }

    ASSERT_FALSE(dst.empty());
    ASSERT_EQ(src1.dims, dst.dims);
    ASSERT_EQ(src1.size, dst.size);
    ASSERT_EQ(dst_depth > 0 ? dst_depth : src1.depth(), dst.depth());
    ASSERT_EQ(0, cv::countNonZero(dst.reshape(1)));
}

INSTANTIATE_TEST_CASE_P(Arithm, SubtractOutputMatNotEmpty, testing::Combine(
    testing::Values(cv::Size(16, 16), cv::Size(13, 13), cv::Size(16, 13), cv::Size(13, 16)),
    testing::Values(perf::MatType(CV_8UC1), CV_8UC3, CV_8UC4, CV_16SC1, CV_16SC3),
    testing::Values(-1, CV_16S, CV_32S, CV_32F),
    testing::Bool()));

TEST(Core_FindNonZero, singular)
{
    Mat img(10, 10, CV_8U, Scalar::all(0));
    vector<Point> pts, pts2(10);
    findNonZero(img, pts);
    findNonZero(img, pts2);
    ASSERT_TRUE(pts.empty() && pts2.empty());
}

TEST(Core_BoolVector, support)
{
    std::vector<bool> test;
    int i, n = 205;
    int nz = 0;
    test.resize(n);
    for( i = 0; i < n; i++ )
    {
        test[i] = theRNG().uniform(0, 2) != 0;
        nz += (int)test[i];
    }
    ASSERT_EQ( nz, countNonZero(test) );
    ASSERT_FLOAT_EQ((float)nz/n, (float)(cv::mean(test)[0]));
}

TEST(MinMaxLoc, Mat_UcharMax_Without_Loc)
{
    Mat_<uchar> mat(50, 50);
    uchar iMaxVal = std::numeric_limits<uchar>::max();
    mat.setTo(iMaxVal);

    double min, max;
    Point minLoc, maxLoc;

    minMaxLoc(mat, &min, &max, &minLoc, &maxLoc, Mat());

    ASSERT_EQ(iMaxVal, min);
    ASSERT_EQ(iMaxVal, max);

    ASSERT_EQ(Point(0, 0), minLoc);
    ASSERT_EQ(Point(0, 0), maxLoc);
}

TEST(MinMaxLoc, Mat_IntMax_Without_Mask)
{
    Mat_<int> mat(50, 50);
    int iMaxVal = std::numeric_limits<int>::max();
    mat.setTo(iMaxVal);

    double min, max;
    Point minLoc, maxLoc;

    minMaxLoc(mat, &min, &max, &minLoc, &maxLoc, Mat());

    ASSERT_EQ(iMaxVal, min);
    ASSERT_EQ(iMaxVal, max);

    ASSERT_EQ(Point(0, 0), minLoc);
    ASSERT_EQ(Point(0, 0), maxLoc);
}

TEST(Normalize, regression_5876_inplace_change_type)
{
    double initial_values[] = {1, 2, 5, 4, 3};
    float result_values[] = {0, 0.25, 1, 0.75, 0.5};
    Mat m(Size(5, 1), CV_64FC1, initial_values);
    Mat result(Size(5, 1), CV_32FC1, result_values);

    normalize(m, m, 1, 0, NORM_MINMAX, CV_32F);
    EXPECT_EQ(0, cvtest::norm(m, result, NORM_INF));
}

TEST(MinMaxLoc, regression_4955_nans)
{
    cv::Mat one_mat(2, 2, CV_32F, cv::Scalar(1));
    cv::minMaxLoc(one_mat, NULL, NULL, NULL, NULL);

    cv::Mat nan_mat(2, 2, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    cv::minMaxLoc(nan_mat, NULL, NULL, NULL, NULL);
}

TEST(Subtract, scalarc1_matc3)
{
    int scalar = 255;
    cv::Mat srcImage(5, 5, CV_8UC3, cv::Scalar::all(5)), destImage;
    cv::subtract(scalar, srcImage, destImage);

    ASSERT_EQ(0, cv::norm(cv::Mat(5, 5, CV_8UC3, cv::Scalar::all(250)), destImage, cv::NORM_INF));
}

TEST(Subtract, scalarc4_matc4)
{
    cv::Scalar sc(255, 255, 255, 255);
    cv::Mat srcImage(5, 5, CV_8UC4, cv::Scalar::all(5)), destImage;
    cv::subtract(sc, srcImage, destImage);

    ASSERT_EQ(0, cv::norm(cv::Mat(5, 5, CV_8UC4, cv::Scalar::all(250)), destImage, cv::NORM_INF));
}

TEST(Compare, empty)
{
    cv::Mat temp, dst1, dst2;
    cv::compare(temp, temp, dst1, cv::CMP_EQ);
    dst2 = temp > 5;

    EXPECT_TRUE(dst1.empty());
    EXPECT_TRUE(dst2.empty());
}

TEST(Compare, regression_8999)
{
    Mat_<double> A(4,1); A << 1, 3, 2, 4;
    Mat_<double> B(1,1); B << 2;
    Mat C;
    ASSERT_ANY_THROW({
        cv::compare(A, B, C, CMP_LT);
    });
}


TEST(Core_minMaxIdx, regression_9207_1)
{
    const int rows = 4;
    const int cols = 3;
    uchar mask_[rows*cols] = {
 255, 255, 255,
 255,   0, 255,
   0, 255, 255,
   0,   0, 255
};
    uchar src_[rows*cols] = {
   1,   1,   1,
   1,   1,   1,
   2,   1,   1,
   2,   2,   1
};
    Mat mask(Size(cols, rows), CV_8UC1, mask_);
    Mat src(Size(cols, rows), CV_8UC1, src_);
    double minVal = -0.0, maxVal = -0.0;
    int minIdx[2] = { -2, -2 }, maxIdx[2] = { -2, -2 };
    cv::minMaxIdx(src, &minVal, &maxVal, minIdx, maxIdx, mask);
    EXPECT_EQ(0, minIdx[0]);
    EXPECT_EQ(0, minIdx[1]);
    EXPECT_EQ(0, maxIdx[0]);
    EXPECT_EQ(0, maxIdx[1]);
}


TEST(Core_minMaxIdx, regression_9207_2)
{
    const int rows = 13;
    const int cols = 15;
    uchar mask_[rows*cols] = {
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,
   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,
 255,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0, 255,
 255,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255, 255,
 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0, 255, 255, 255,   0,
 255,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255,   0, 255,   0,
 255,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0, 255, 255,   0,
 255,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 255,   0,
 255,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0, 255,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};
    uchar src_[15*13] = {
   5,   5,   5,   5,   5,   6,   5,   2,   0,   4,   6,   6,   4,   1,   0,
   6,   5,   4,   4,   5,   6,   6,   5,   2,   0,   4,   6,   5,   2,   0,
   3,   2,   1,   1,   2,   4,   6,   6,   4,   2,   3,   4,   4,   2,   0,
   1,   0,   0,   0,   0,   1,   4,   5,   4,   4,   4,   4,   3,   2,   0,
   0,   0,   0,   0,   0,   0,   2,   3,   4,   4,   4,   3,   2,   1,   0,
   0,   0,   0,   0,   0,   0,   0,   2,   3,   4,   3,   2,   1,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   0,   0,   0,   1,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   0,   0,   1,
   0,   0,   0,   0,   0,   0,   0,   1,   2,   4,   3,   3,   1,   0,   1,
   0,   0,   0,   0,   0,   0,   1,   4,   5,   6,   5,   4,   3,   2,   0,
   1,   0,   0,   0,   0,   0,   3,   5,   5,   4,   3,   4,   4,   3,   0,
   2,   0,   0,   0,   0,   2,   5,   6,   5,   2,   2,   5,   4,   3,   0
};
    Mat mask(Size(cols, rows), CV_8UC1, mask_);
    Mat src(Size(cols, rows), CV_8UC1, src_);
    double minVal = -0.0, maxVal = -0.0;
    int minIdx[2] = { -2, -2 }, maxIdx[2] = { -2, -2 };
    cv::minMaxIdx(src, &minVal, &maxVal, minIdx, maxIdx, mask);
    EXPECT_EQ(0, minIdx[0]);
    EXPECT_EQ(14, minIdx[1]);
    EXPECT_EQ(0, maxIdx[0]);
    EXPECT_EQ(14, maxIdx[1]);
}

}} // namespace
