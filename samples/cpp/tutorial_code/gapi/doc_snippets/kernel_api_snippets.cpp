// [filter2d_api]
#include <opencv2/gapi.hpp>

G_TYPED_KERNEL(GFilter2D,
               <cv::GMat(cv::GMat,int,cv::Mat,cv::Point,double,int,cv::Scalar)>,
               "org.opencv.imgproc.filters.filter2D")
{
    static cv::GMatDesc                 // outMeta's return value type
    outMeta(cv::GMatDesc    in       ,  // descriptor of input GMat
            int             ddepth   ,  // depth parameter
            cv::Mat      /* coeffs */,  // (unused)
            cv::Point    /* anchor */,  // (unused)
            double       /* scale  */,  // (unused)
            int          /* border */,  // (unused)
            cv::Scalar   /* bvalue */ ) // (unused)
    {
        return in.withDepth(ddepth);
    }
};
// [filter2d_api]

cv::GMat filter2D(cv::GMat  ,
                  int       ,
                  cv::Mat   ,
                  cv::Point ,
                  double    ,
                  int       ,
                  cv::Scalar);

// [filter2d_wrap]
cv::GMat filter2D(cv::GMat   in,
                  int        ddepth,
                  cv::Mat    k,
                  cv::Point  anchor  = cv::Point(-1,-1),
                  double     scale   = 0.,
                  int        border  = cv::BORDER_DEFAULT,
                  cv::Scalar bval    = cv::Scalar(0))
{
    return GFilter2D::on(in, ddepth, k, anchor, scale, border, bval);
}
// [filter2d_wrap]

// [compound]
#include <opencv2/gapi/gcompoundkernel.hpp>       // GAPI_COMPOUND_KERNEL()

using PointArray2f = cv::GArray<cv::Point2f>;

G_TYPED_KERNEL(HarrisCorners,
               <PointArray2f(cv::GMat,int,double,double,int,double)>,
               "org.opencv.imgproc.harris_corner")
{
    static cv::GArrayDesc outMeta(const cv::GMatDesc &,
                                  int,
                                  double,
                                  double,
                                  int,
                                  double)
    {
        // No special metadata for arrays in G-API (yet)
        return cv::empty_array_desc();
    }
};

// Define Fluid-backend-local kernels which form GoodFeatures
G_TYPED_KERNEL(HarrisResponse,
               <cv::GMat(cv::GMat,double,int,double)>,
               "org.opencv.fluid.harris_response")
{
    static cv::GMatDesc outMeta(const cv::GMatDesc &in,
                                double,
                                int,
                                double)
    {
        return in.withType(CV_32F, 1);
    }
};

G_TYPED_KERNEL(ArrayNMS,
               <PointArray2f(cv::GMat,int,double)>,
               "org.opencv.cpu.nms_array")
{
    static cv::GArrayDesc outMeta(const cv::GMatDesc &,
                                  int,
                                  double)
    {
        return cv::empty_array_desc();
    }
};

GAPI_COMPOUND_KERNEL(GFluidHarrisCorners, HarrisCorners)
{
    static PointArray2f
    expand(cv::GMat in,
           int      maxCorners,
           double   quality,
           double   minDist,
           int      blockSize,
           double   k)
    {
        cv::GMat response = HarrisResponse::on(in, quality, blockSize, k);
        return ArrayNMS::on(response, maxCorners, minDist);
    }
};

// Then implement HarrisResponse as Fluid kernel and NMSresponse
// as a generic (OpenCV) kernel
// [compound]

// [filter2d_ocv]
#include <opencv2/gapi/cpu/gcpukernel.hpp>     // GAPI_OCV_KERNEL()
#include <opencv2/imgproc.hpp>                 // cv::filter2D()

GAPI_OCV_KERNEL(GCPUFilter2D, GFilter2D)
{
    static void
    run(const cv::Mat    &in,       // in - derived from GMat
        const int         ddepth,   // opaque (passed as-is)
        const cv::Mat    &k,        // opaque (passed as-is)
        const cv::Point  &anchor,   // opaque (passed as-is)
        const double      delta,    // opaque (passed as-is)
        const int         border,   // opaque (passed as-is)
        const cv::Scalar &,         // opaque (passed as-is)
        cv::Mat          &out)      // out - derived from GMat (retval)
    {
        cv::filter2D(in, out, ddepth, k, anchor, delta, border);
    }
};
// [filter2d_ocv]

int main(int, char *[])
{
    std::cout << "This sample is non-complete. It is used as code snippents in documentation." << std::endl;

cv::Mat conv_kernel_mat;

{
// [filter2d_on]
cv::GMat in;
cv::GMat out = GFilter2D::on(/* GMat    */  in,
                             /* int     */  -1,
                             /* Mat     */  conv_kernel_mat,
                             /* Point   */  cv::Point(-1,-1),
                             /* double  */  0.,
                             /* int     */  cv::BORDER_DEFAULT,
                             /* Scalar  */  cv::Scalar(0));
// [filter2d_on]
}

{
// [filter2d_wrap_call]
cv::GMat in;
cv::GMat out = filter2D(in, -1, conv_kernel_mat);
// [filter2d_wrap_call]
}

return 0;
}
