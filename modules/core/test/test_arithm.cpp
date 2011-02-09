#include "test_precomp.hpp"

using namespace cv;
using namespace std;

namespace cvtest
{

const int ARITHM_NTESTS = 1000;
const int ARITHM_RNG_SEED = -1;
const int ARITHM_MAX_CHANNELS = 4;
const int ARITHM_MAX_NDIMS = 4;
const int ARITHM_MAX_SIZE_LOG = 10;

struct BaseElemWiseOp
{
    enum { FIX_ALPHA=1, FIX_BETA=2, FIX_GAMMA=4, REAL_GAMMA=8, SUPPORT_MASK=16, SCALAR_OUTPUT=32 };
    BaseElemWiseOp(int _ninputs, int _flags, double _alpha, double _beta,
                   Scalar _gamma=Scalar::all(0), int _context=1)
    : ninputs(_ninputs), flags(_flags), alpha(_alpha), beta(_beta), gamma(_gamma), context(_context) {}
    BaseElemWiseOp() { flags = alpha = beta = 0; gamma = Scalar::all(0); }
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
        cvtest::randomSize(rng, 2, ARITHM_MAX_NDIMS, cvtest::ARITHM_MAX_SIZE_LOG, size);
    }
    
    virtual int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, cvtest::TYPE_MASK_ALL_BUT_8S, 1,
                                  ninputs > 1 ? ARITHM_MAX_CHANNELS : 4);
    }
        
    virtual int getMaxErr(int depth) { return depth < CV_32F ? 1 : 256; }    
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
    int maxErr;
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
    AddOp() : BaseAddOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( mask.empty() )
            add(src[0], src[1], dst);
        else
            add(src[0], src[1], dst, mask);
    }
};


struct SubOp : public BaseAddOp
{
    SubOp() : BaseAddOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK, 1, -1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( mask.empty() )
            subtract(src[0], src[1], dst);
        else
            subtract(src[0], src[1], dst, mask);
    }
};


struct AddSOp : public BaseAddOp
{
    AddSOp() : BaseAddOp(1, FIX_ALPHA+FIX_BETA+SUPPORT_MASK, 1, 0, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( mask.empty() )
            add(src[0], gamma, dst);
        else
            add(src[0], gamma, dst, mask);
    }
};


struct SubRSOp : public BaseAddOp
{
    SubRSOp() : BaseAddOp(1, FIX_ALPHA+FIX_BETA+SUPPORT_MASK, -1, 0, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( mask.empty() )
            subtract(gamma, src[0], dst);
        else
            subtract(gamma, src[0], dst, mask);
    }
};


struct ScaleAddOp : public BaseAddOp
{
    ScaleAddOp() : BaseAddOp(2, FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        scaleAdd(src[0], alpha, src[1], dst);
    }
};


struct AddWeightedOp : public BaseAddOp
{
    AddWeightedOp() : BaseAddOp(2, REAL_GAMMA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        addWeighted(src[0], alpha, src[1], beta, gamma[0], dst);
    }
    int getMaxErr(int depth)
    {
        return depth <= CV_32S ? 2 : depth < CV_64F ? (1 << 10) : (1 << 22);
    }
};

struct MulOp : public BaseElemWiseOp
{
    MulOp() : BaseElemWiseOp(2, FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::multiply(src[0], src[1], dst, alpha);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::multiply(src[0], src[1], dst, alpha);
    }
    int getMaxErr(int depth)
    {
        return depth < CV_32S ? 2 : depth < CV_32F ? 4 : 16;
    }
};    

struct DivOp : public BaseElemWiseOp
{
    DivOp() : BaseElemWiseOp(2, FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::divide(src[0], src[1], dst, alpha);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::divide(src[0], src[1], dst, alpha);
    }
    int getMaxErr(int depth)
    {
        return depth < CV_32S ? 2 : depth < CV_32F ? 4 : 16;
    }
};    

struct RecipOp : public BaseElemWiseOp
{
    RecipOp() : BaseElemWiseOp(1, FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::divide(alpha, src[0], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::divide(Mat(), src[0], dst, alpha);
    }
    int getMaxErr(int depth)
    {
        return depth < CV_32S ? 2 : depth < CV_32F ? 4 : 16;
    }
};        
    
struct AbsDiffOp : public BaseAddOp
{
    AbsDiffOp() : BaseAddOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, -1, Scalar::all(0)) {};
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
    AbsDiffSOp() : BaseAddOp(1, FIX_ALPHA+FIX_BETA, 1, 0, Scalar::all(0)) {};
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
    LogicOp(char _opcode) : BaseElemWiseOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK, 1, 1, Scalar::all(0)), opcode(_opcode) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( opcode == '&' )
            bitwise_and(src[0], src[1], dst, mask);
        else if( opcode == '|' )
            bitwise_or(src[0], src[1], dst, mask);
        else
            bitwise_xor(src[0], src[1], dst, mask);
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
    int getMaxErr(int depth)
    {
        return 0;
    }
    char opcode;
};

struct LogicSOp : public BaseElemWiseOp
{
    LogicSOp(char _opcode)
    : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+(_opcode != '~' ? SUPPORT_MASK : 0), 1, 1, Scalar::all(0)), opcode(_opcode) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        if( opcode == '&' )
            bitwise_and(src[0], gamma, dst, mask);
        else if( opcode == '|' )
            bitwise_or(src[0], gamma, dst, mask);
        else if( opcode == '^' )
            bitwise_xor(src[0], gamma, dst, mask);
        else
            bitwise_not(src[0], dst);
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
    int getMaxErr(int)
    {
        return 0;
    }
    char opcode;
};

struct MinOp : public BaseElemWiseOp
{
    MinOp() : BaseElemWiseOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::min(src[0], src[1], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::min(src[0], src[1], dst);
    }
    int getMaxErr(int depth)
    {
        return 0;
    }
};    

struct MaxOp : public BaseElemWiseOp
{
    MaxOp() : BaseElemWiseOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::max(src[0], src[1], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::max(src[0], src[1], dst);
    }
    int getMaxErr(int depth)
    {
        return 0;
    }
};    

struct MinSOp : public BaseElemWiseOp
{
    MinSOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+REAL_GAMMA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::min(src[0], gamma[0], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::min(src[0], gamma[0], dst);
    }
    int getMaxErr(int depth)
    {
        return 0;
    }
};    

struct MaxSOp : public BaseElemWiseOp
{
    MaxSOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+REAL_GAMMA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::max(src[0], gamma[0], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::max(src[0], gamma[0], dst);
    }
    int getMaxErr(int depth)
    {
        return 0;
    }
};
    
struct CmpOp : public BaseElemWiseOp
{
    CmpOp() : BaseElemWiseOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
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
        return cvtest::randomType(rng, cvtest::TYPE_MASK_ALL_BUT_8S, 1, 1);
    }
        
    int getMaxErr(int)
    {
        return 0;
    }
    int cmpop;
};

struct CmpSOp : public BaseElemWiseOp
{
    CmpSOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+REAL_GAMMA, 1, 1, Scalar::all(0)) {};
    void generateScalars(int depth, RNG& rng)
    {
        BaseElemWiseOp::generateScalars(depth, rng);
        cmpop = rng.uniform(0, 6);
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
        return cvtest::randomType(rng, cvtest::TYPE_MASK_ALL_BUT_8S, 1, 1);
    }
    int getMaxErr(int)
    {
        return 0;
    }
    int cmpop;
};    

    
struct CopyOp : public BaseElemWiseOp
{
    CopyOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK, 1, 1, Scalar::all(0)) {};
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
        return cvtest::randomType(rng, cvtest::TYPE_MASK_ALL, 1, ARITHM_MAX_CHANNELS);
    }
    int getMaxErr(int)
    {
        return 0;
    }
    int cmpop;
};

    
struct SetOp : public BaseElemWiseOp
{
    SetOp() : BaseElemWiseOp(0, FIX_ALPHA+FIX_BETA+SUPPORT_MASK, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>&, Mat& dst, const Mat& mask)
    {
        dst.setTo(gamma, mask);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat& mask)
    {
        cvtest::set(dst, gamma, mask);
    }
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, cvtest::TYPE_MASK_ALL, 1, ARITHM_MAX_CHANNELS);
    }
    int getMaxErr(int)
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
        dst[i] = a[0] <= val && val < b[0] ? 255 : 0;
    }
    for( c = 1; c < cn; c++ )
    {
        for( i = 0; i < total; i++ )
        {
            _Tp val = src[i*cn + c];
            dst[i] = a[c] <= val && val < b[c] ? dst[i] : 0;
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
        dst[i] = a[i*cn] <= val && val < b[i*cn] ? 255 : 0;
    }
    for( c = 1; c < cn; c++ )
    {
        for( i = 0; i < total; i++ )
        {
            _Tp val = src[i*cn + c];
            dst[i] = a[i*cn + c] <= val && val < b[i*cn + c] ? dst[i] : 0;
        }
    }
}
    

static void inRange(const Mat& src, const Mat& lb, const Mat& rb, Mat& dst)
{
    CV_Assert( src.type() == lb.type() && src.type() == rb.type() &&
              src.size == lb.size && src.size == rb.size );
    dst.create( src.dims, &src.size[0], CV_8U );
    const Mat *arrays[]={&src, &lb, &rb, &dst, 0};
    Mat planes[4];
    
    NAryMatIterator it(arrays, planes);
    size_t total = planes[0].total();
    int i, nplanes = it.nplanes, depth = src.depth(), cn = src.channels();
    
    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].data;
        const uchar* aptr = planes[1].data;
        const uchar* bptr = planes[2].data;
        uchar* dptr = planes[3].data;
        
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
    int i, nplanes = it.nplanes, depth = src.depth(), cn = src.channels();
    double lbuf[4], rbuf[4];
    int wtype = CV_MAKETYPE(depth <= CV_32S ? CV_32S : depth, cn);
    scalarToRawData(lb, lbuf, wtype, cn);
    scalarToRawData(rb, rbuf, wtype, cn);
    
    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].data;
        uchar* dptr = planes[1].data;
        
        switch( depth )
        {
        case CV_8U:
            inRangeS_((const uchar*)sptr, (const int*)lbuf, (const int*)rbuf, dptr, total, cn);
            break;
        case CV_8S:
            inRangeS_((const schar*)sptr, (const int*)lbuf, (const int*)rbuf, dptr, total, cn);
            break;
        case CV_16U:
            inRangeS_((const ushort*)sptr, (const int*)lbuf, (const int*)rbuf, dptr, total, cn);
            break;
        case CV_16S:
            inRangeS_((const short*)sptr, (const int*)lbuf, (const int*)rbuf, dptr, total, cn);
            break;
        case CV_32S:
            inRangeS_((const int*)sptr, (const int*)lbuf, (const int*)rbuf, dptr, total, cn);
            break;
        case CV_32F:
            inRangeS_((const float*)sptr, (const float*)lbuf, (const float*)rbuf, dptr, total, cn);
            break;
        case CV_64F:
            inRangeS_((const double*)sptr, (const double*)lbuf, (const double*)rbuf, dptr, total, cn);
            break;
        default:
            CV_Error(CV_StsUnsupportedFormat, "");
        }
    }
}
    
    
struct InRangeSOp : public BaseElemWiseOp
{
    InRangeSOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::inRange(src[0], gamma, gamma1, dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::inRangeS(src[0], gamma, gamma1, dst);
    }
    int getMaxErr(int)
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
    InRangeOp() : BaseElemWiseOp(3, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
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
        
        cvtest::inRange(src[0], lb, rb, dst);
    }
    int getMaxErr(int)
    {
        return 0;
    }
};    
    
    
struct ConvertScaleOp : public BaseElemWiseOp
{
    ConvertScaleOp() : BaseElemWiseOp(1, FIX_BETA+REAL_GAMMA, 1, 1, Scalar::all(0)), ddepth(0) { };
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
        int srctype = cvtest::randomType(rng, cvtest::TYPE_MASK_ALL, 1, ARITHM_MAX_CHANNELS);
        ddepth = cvtest::randomType(rng, cvtest::TYPE_MASK_ALL, 1, 1);
        return srctype;
    }
    int getMaxErr(int)
    {
        return ddepth <= CV_32S ? 2 : ddepth < CV_64F ? (1 << 14) : (1 << 18);
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

    
struct ConvertScaleAbsOp : public BaseElemWiseOp
{
    ConvertScaleAbsOp() : BaseElemWiseOp(1, FIX_BETA+REAL_GAMMA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::convertScaleAbs(src[0], dst, alpha, gamma[0]);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::add(src[0], alpha, Mat(), 0, Scalar::all(gamma[0]), dst, CV_8UC(src[0].channels()), true);
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

    
struct FlipOp : public BaseElemWiseOp
{
    FlipOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
    void getRandomSize(RNG& rng, vector<int>& size)
    {
        cvtest::randomSize(rng, 2, 2, cvtest::ARITHM_MAX_SIZE_LOG, size);
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::flip(src[0], dst, flipcode);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::flip(src[0], dst, flipcode);
    }
    void generateScalars(int depth, RNG& rng)
    {
        flipcode = rng.uniform(0, 3) - 1;
    }
    int getMaxErr(int)
    {
        return 0;
    }
    int flipcode;
};

struct TransposeOp : public BaseElemWiseOp
{
    TransposeOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
    void getRandomSize(RNG& rng, vector<int>& size)
    {
        cvtest::randomSize(rng, 2, 2, cvtest::ARITHM_MAX_SIZE_LOG, size);
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cv::transpose(src[0], dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        cvtest::transpose(src[0], dst);
    }
    int getMaxErr(int)
    {
        return 0;
    }
};    
    
struct SetIdentityOp : public BaseElemWiseOp
{
    SetIdentityOp() : BaseElemWiseOp(0, FIX_ALPHA+FIX_BETA, 1, 1, Scalar::all(0)) {};
    void getRandomSize(RNG& rng, vector<int>& size)
    {
        cvtest::randomSize(rng, 2, 2, cvtest::ARITHM_MAX_SIZE_LOG, size);
    }
    void op(const vector<Mat>&, Mat& dst, const Mat&)
    {
        cv::setIdentity(dst, gamma);
    }
    void refop(const vector<Mat>&, Mat& dst, const Mat&)
    {
        cvtest::setIdentity(dst, gamma);
    }
    int getMaxErr(int)
    {
        return 0;
    }
};    

struct SetZeroOp : public BaseElemWiseOp
{
    SetZeroOp() : BaseElemWiseOp(0, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
    void op(const vector<Mat>&, Mat& dst, const Mat&)
    {
        dst = Scalar::all(0);
    }
    void refop(const vector<Mat>&, Mat& dst, const Mat&)
    {
        cvtest::set(dst, Scalar::all(0));
    }
    int getMaxErr(int)
    {
        return 0;
    }
};

    
static void exp(const Mat& src, Mat& dst)
{
    dst.create( src.dims, &src.size[0], src.type() );
    const Mat *arrays[]={&src, &dst, 0};
    Mat planes[2];
    
    NAryMatIterator it(arrays, planes);
    size_t j, total = planes[0].total()*src.channels();
    int i, nplanes = it.nplanes, depth = src.depth();
    
    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].data;
        uchar* dptr = planes[1].data;
        
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
    int i, nplanes = it.nplanes, depth = src.depth();
    
    for( i = 0; i < nplanes; i++, ++it )
    {
        const uchar* sptr = planes[0].data;
        uchar* dptr = planes[1].data;
        
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
    
struct ExpOp : public BaseElemWiseOp
{
    ExpOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, cvtest::TYPE_MASK_FLT, 1, ARITHM_MAX_CHANNELS);
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
        cvtest::exp(src[0], dst);
    }
    int getMaxErr(int)
    {
        return (1<<10);
    }
};        


struct LogOp : public BaseElemWiseOp
{
    LogOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0)) {};
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, cvtest::TYPE_MASK_FLT, 1, ARITHM_MAX_CHANNELS);
    }
    void getValueRange(int depth, double& minval, double& maxval)
    {
        maxval = depth == CV_32F ? 50 : 100;
        minval = -maxval;
    }
    void op(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        Mat temp;
        cvtest::exp(src[0], temp);
        cv::log(temp, dst);
    }
    void refop(const vector<Mat>& src, Mat& dst, const Mat&)
    {
        Mat temp;
        cvtest::exp(src[0], temp);
        cvtest::log(temp, dst);
    }
    int getMaxErr(int)
    {
        return (1<<10);
    }
};


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
    int i, nplanes = it.nplanes, depth = mx.depth();
    double scale = angleInDegrees ? 180/CV_PI : 1;
    
    for( i = 0; i < nplanes; i++, ++it )
    {
        if( depth == CV_32F )
        {
            const float* xptr = (const float*)planes[0].data;
            const float* yptr = (const float*)planes[1].data;
            float* mptr = (float*)planes[2].data;
            float* aptr = (float*)planes[3].data;
            
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
            const double* xptr = (const double*)planes[0].data;
            const double* yptr = (const double*)planes[1].data;
            double* mptr = (double*)planes[2].data;
            double* aptr = (double*)planes[3].data;
            
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
    
    
struct CartToPolarToCartOp : public BaseElemWiseOp
{
    CartToPolarToCartOp() : BaseElemWiseOp(2, FIX_ALPHA+FIX_BETA+FIX_GAMMA, 1, 1, Scalar::all(0))
    {
        context = 3;
        angleInDegrees = true;
    }
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, cvtest::TYPE_MASK_FLT, 1, 1);
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
        cvtest::cartToPolar(src[0], src[1], mag, angle, angleInDegrees);
        Mat msrc[] = {mag, angle, src[0], src[1]};
        int pairs[] = {0, 0, 1, 1, 2, 2, 3, 3};
        dst.create(src[0].dims, src[0].size, CV_MAKETYPE(src[0].depth(), 4));
        cv::mixChannels(msrc, 4, &dst, 1, pairs, 4);
    }
    void generateScalars(int, RNG& rng)
    {
        angleInDegrees = rng.uniform(0, 2) != 0;
    }
    int getMaxErr(int)
    {
        return (1<<10);
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
    int getMaxErr(int)
    {
        return (1<<13);
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
    int getMaxErr(int)
    {
        return (1<<13);
    }
};    

    
struct CountNonZeroOp : public BaseElemWiseOp
{
    CountNonZeroOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SCALAR_OUTPUT+SUPPORT_MASK, 1, 1, Scalar::all(0))
    {}
    int getRandomType(RNG& rng)
    {
        return cvtest::randomType(rng, cvtest::TYPE_MASK_ALL, 1, 1);
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
    int getMaxErr(int)
    {
        return 0;
    }
};    

    
struct MeanStdDevOp : public BaseElemWiseOp
{
    MeanStdDevOp() : BaseElemWiseOp(1, FIX_ALPHA+FIX_BETA+FIX_GAMMA+SUPPORT_MASK+SCALAR_OUTPUT, 1, 1, Scalar::all(0))
    {
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

        for( int c = 0; c < 4; c++ )
            sqmean[c] = std::sqrt(std::max(sqmean[c] - mean[c]*mean[c], 0.)); 
        
        dst.create(1, 2, CV_64FC4);
        dst.at<Scalar>(0,0) = mean;
        dst.at<Scalar>(0,1) = sqmean;
    }
    int getMaxErr(int)
    {
        return (1<<13);
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
        return cvtest::randomType(rng, cvtest::TYPE_MASK_ALL_BUT_8S, 1, 4);
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
    void generateScalars(int, RNG& rng)
    {
        normType = 1 << rng.uniform(0, 3);
    }
    int getMaxErr(int)
    {
        return (1<<13);
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
        return cvtest::randomType(rng, cvtest::TYPE_MASK_ALL_BUT_8S, 1, 1);
    }
    void saveOutput(const vector<int>& minidx, const vector<int>& maxidx,
                    double minval, double maxval, Mat& dst)
    {
        size_t i, ndims = minidx.size();
        dst.create(1, (int)(ndims*2 + 2), CV_64FC1);
        
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
    int getMaxErr(int)
    {
        return 0;
    }
};            
    
    
}

typedef Ptr<cvtest::BaseElemWiseOp> ElemWiseOpPtr;
class ElemWiseTest : public ::testing::TestWithParam<ElemWiseOpPtr> {};

TEST_P(ElemWiseTest, accuracy)
{
    ElemWiseOpPtr op = GetParam();
    
    int testIdx = 0;
    RNG rng(cvtest::ARITHM_RNG_SEED);
    for( testIdx = 0; testIdx < cvtest::ARITHM_NTESTS; testIdx++ )
    {
        vector<int> size;
        op->getRandomSize(rng, size);
        int type = op->getRandomType(rng);
        int depth = CV_MAT_DEPTH(type);
        bool haveMask = (op->flags & cvtest::BaseElemWiseOp::SUPPORT_MASK) != 0 && rng.uniform(0, 4) == 0;
        
        double minval=0, maxval=0;
        op->getValueRange(depth, minval, maxval);
        int i, ninputs = op->ninputs;
        vector<Mat> src(ninputs);
        for( i = 0; i < ninputs; i++ )
            src[i] = cvtest::randomMat(rng, size, type, minval, maxval, true);
        Mat dst0, dst, mask;
        if( haveMask )
            mask = cvtest::randomMat(rng, size, CV_8U, 0, 2, true);
        
        if( (haveMask || ninputs == 0) && !(op->flags & cvtest::BaseElemWiseOp::SCALAR_OUTPUT))
        {
            dst0 = cvtest::randomMat(rng, size, type, minval, maxval, false);
            dst = cvtest::randomMat(rng, size, type, minval, maxval, true);
            cvtest::copy(dst, dst0);
        }
        op->generateScalars(depth, rng);
        
        op->refop(src, dst0, mask);
        op->op(src, dst, mask);
        
        double maxErr = op->getMaxErr(depth);
        vector<int> pos;
        ASSERT_PRED_FORMAT2(cvtest::MatComparator(maxErr, op->context), dst0, dst) << "\nsrc[0] ~ " << cvtest::MatInfo(!src.empty() ? src[0] : Mat()) << "\ntestCase #" << testIdx << "\n";
    }
}


INSTANTIATE_TEST_CASE_P(Core_Copy, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::CopyOp)));
INSTANTIATE_TEST_CASE_P(Core_Set, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::SetOp)));
INSTANTIATE_TEST_CASE_P(Core_SetZero, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::SetZeroOp)));
INSTANTIATE_TEST_CASE_P(Core_ConvertScale, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::ConvertScaleOp)));
INSTANTIATE_TEST_CASE_P(Core_ConvertScaleAbs, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::ConvertScaleAbsOp)));

INSTANTIATE_TEST_CASE_P(Core_Add, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::AddOp)));
INSTANTIATE_TEST_CASE_P(Core_Sub, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::SubOp)));
INSTANTIATE_TEST_CASE_P(Core_AddS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::AddSOp)));
INSTANTIATE_TEST_CASE_P(Core_SubRS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::SubRSOp)));
INSTANTIATE_TEST_CASE_P(Core_ScaleAdd, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::ScaleAddOp)));
INSTANTIATE_TEST_CASE_P(Core_AddWeighted, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::AddWeightedOp)));
INSTANTIATE_TEST_CASE_P(Core_AbsDiff, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::AbsDiffOp)));
INSTANTIATE_TEST_CASE_P(Core_AbsDiffS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::AbsDiffSOp)));

INSTANTIATE_TEST_CASE_P(Core_And, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::LogicOp('&'))));
INSTANTIATE_TEST_CASE_P(Core_AndS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::LogicSOp('&'))));
INSTANTIATE_TEST_CASE_P(Core_Or, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::LogicOp('|'))));
INSTANTIATE_TEST_CASE_P(Core_OrS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::LogicSOp('|'))));
INSTANTIATE_TEST_CASE_P(Core_Xor, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::LogicOp('^'))));
INSTANTIATE_TEST_CASE_P(Core_XorS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::LogicSOp('^'))));
INSTANTIATE_TEST_CASE_P(Core_Not, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::LogicSOp('~'))));

INSTANTIATE_TEST_CASE_P(Core_Max, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::MaxOp)));
INSTANTIATE_TEST_CASE_P(Core_MaxS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::MaxSOp)));
INSTANTIATE_TEST_CASE_P(Core_Min, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::MinOp)));
INSTANTIATE_TEST_CASE_P(Core_MinS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::MinSOp)));

INSTANTIATE_TEST_CASE_P(Core_Mul, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::MulOp)));
INSTANTIATE_TEST_CASE_P(Core_Div, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::DivOp)));
INSTANTIATE_TEST_CASE_P(Core_Recip, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::RecipOp)));

INSTANTIATE_TEST_CASE_P(Core_Cmp, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::CmpOp)));
INSTANTIATE_TEST_CASE_P(Core_CmpS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::CmpSOp)));

INSTANTIATE_TEST_CASE_P(Core_InRangeS, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::InRangeSOp)));
INSTANTIATE_TEST_CASE_P(Core_InRange, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::InRangeOp)));

INSTANTIATE_TEST_CASE_P(Core_Flip, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::FlipOp)));
INSTANTIATE_TEST_CASE_P(Core_Transpose, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::TransposeOp)));
INSTANTIATE_TEST_CASE_P(Core_SetIdentity, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::SetIdentityOp)));

INSTANTIATE_TEST_CASE_P(Core_Exp, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::ExpOp)));
INSTANTIATE_TEST_CASE_P(Core_Log, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::LogOp)));

INSTANTIATE_TEST_CASE_P(Core_CountNonZero, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::CountNonZeroOp)));
INSTANTIATE_TEST_CASE_P(Core_Mean, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::MeanOp)));
INSTANTIATE_TEST_CASE_P(Core_MeanStdDev, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::MeanStdDevOp)));
INSTANTIATE_TEST_CASE_P(Core_Sum, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::SumOp)));
INSTANTIATE_TEST_CASE_P(Core_Norm, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::NormOp)));
INSTANTIATE_TEST_CASE_P(Core_MinMaxLoc, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::MinMaxLocOp)));

INSTANTIATE_TEST_CASE_P(Core_CartToPolarToCart, ElemWiseTest, ::testing::Values(ElemWiseOpPtr(new cvtest::CartToPolarToCartOp)));
