#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include "precomp_ipp.hpp"

#ifdef HAVE_IPP_IW
#include "iw++/iw.hpp"
#endif

#define IPP_WARPAFFINE_PARALLEL 1
#define CV_TYPE(src_type) (src_type & (CV_DEPTH_MAX - 1))
#ifdef HAVE_IPP_IW

class ipp_warpAffineParallel: public cv::ParallelLoopBody
{
public:
    ipp_warpAffineParallel(::ipp::IwiImage &src, ::ipp::IwiImage &dst, IppiInterpolationType _inter, double (&_coeffs)[2][3], ::ipp::IwiBorderType _borderType, IwTransDirection _iwTransDirection, bool *_ok):m_src(src), m_dst(dst)
    {
        pOk = _ok;

        inter          = _inter;
        borderType     = _borderType;
        iwTransDirection = _iwTransDirection;

        for( int i = 0; i < 2; i++ )
            for( int j = 0; j < 3; j++ )
                coeffs[i][j] = _coeffs[i][j];

        *pOk = true;
    }
    ~ipp_warpAffineParallel() {}

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        //CV_INSTRUMENT_REGION_IPP(); //better to keep

        if(*pOk == false)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, m_dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpAffine, m_src, m_dst, coeffs, iwTransDirection, inter, ::ipp::IwiWarpAffineParams(), borderType, tile);
        }
        catch(const ::ipp::IwException &)
        {
            *pOk = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &m_src;
    ::ipp::IwiImage &m_dst;

    IppiInterpolationType inter;
    double coeffs[2][3];
    ::ipp::IwiBorderType borderType;
    IwTransDirection iwTransDirection;

    bool  *pOk;
    const ipp_warpAffineParallel& operator= (const ipp_warpAffineParallel&);
};

#if (IPP_VERSION_X100 >= 700)
int ipp_hal_warpAffine(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width,
                              int dst_height, const double M[6], int interpolation, int borderType, const double borderValue[4])
{
    //CV_INSTRUMENT_REGION_IPP(); //better to keep

    IppiInterpolationType ippInter    = ippiGetInterpolation(interpolation);
    if((int)ippInter < 0 || interpolation > 2)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

                          /* C1         C2         C3         C4 */
    char impl[7][4][3]={{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8U
                        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //8S
                        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}},   //16U
                        {{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 0}},   //16S
                        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},   //32S
                        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0}},   //32F
                        {{1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 0}}};  //64F

    if(impl[CV_TYPE(src_type)][CV_MAT_CN(src_type)-1][interpolation] == 0)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // Acquire data and begin processing
    try
    {
        ::ipp::IwiImage        iwSrc;
        iwSrc.Init({src_width, src_height}, ippiGetDataType(src_type), CV_MAT_CN(src_type), NULL, src_data, IwSize(src_step));
        ::ipp::IwiImage        iwDst({dst_width, dst_height}, ippiGetDataType(src_type), CV_MAT_CN(src_type), NULL, dst_data, dst_step);
        ::ipp::IwiBorderType   ippBorder(ippiGetBorderType(borderType), {borderValue[0], borderValue[1], borderValue[2], borderValue[3]});
        IwTransDirection       iwTransDirection = iwTransForward;
        
        if((int)ippBorder == -1)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        double coeffs[2][3];
        for( int i = 0; i < 2; i++ )
            for( int j = 0; j < 3; j++ )
                coeffs[i][j] = M[i*3 + j];

        const int threads = ippiSuggestThreadsNum(iwDst, 2);

        if(IPP_WARPAFFINE_PARALLEL && threads > 1)
        {
            bool  ok      = true;
            cv::Range range(0, (int)iwDst.m_size.height);
            ipp_warpAffineParallel invoker(iwSrc, iwDst, ippInter, coeffs, ippBorder, iwTransDirection, &ok);
            if(!ok)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;

            parallel_for_(range, invoker, threads*4);

            if(!ok)
                return CV_HAL_ERROR_NOT_IMPLEMENTED;
        } else {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiWarpAffine, iwSrc, iwDst, coeffs, iwTransDirection, ippInter, ::ipp::IwiWarpAffineParams(), ippBorder);
        }
    }
    catch (const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}
#endif
#endif
