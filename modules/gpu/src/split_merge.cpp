#include "precomp.hpp"
#include <vector>

using namespace std;

#if !defined (HAVE_CUDA)

void cv::gpu::merge(const GpuMat* /*src*/, size_t /*count*/, GpuMat& /*dst*/) { throw_nogpu(); }
void cv::gpu::merge(const vector<GpuMat>& /*src*/, GpuMat& /*dst*/) { throw_nogpu(); }
void cv::gpu::merge(const GpuMat* /*src*/, size_t /*count*/, GpuMat& /*dst*/, const Stream& /*stream*/) { throw_nogpu(); }
void cv::gpu::merge(const vector<GpuMat>& /*src*/, GpuMat& /*dst*/, const Stream& /*stream*/) { throw_nogpu(); }
void cv::gpu::split(const GpuMat& /*src*/, GpuMat* /*dst*/) { throw_nogpu(); }
void cv::gpu::split(const GpuMat& /*src*/, vector<GpuMat>& /*dst*/) { throw_nogpu(); }
void cv::gpu::split(const GpuMat& /*src*/, GpuMat* /*dst*/, const Stream& /*stream*/) { throw_nogpu(); }
void cv::gpu::split(const GpuMat& /*src*/, vector<GpuMat>& /*dst*/, const Stream& /*stream*/) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace split_merge 
{    
    extern "C" void merge_caller(const DevMem2D* src, DevMem2D& dst, 
                                 int total_channels, int elem_size, 
                                 const cudaStream_t& stream);

    extern "C" void split_caller(const DevMem2D& src, DevMem2D* dst, 
                                 int num_channels, int elem_size1, 
                                 const cudaStream_t& stream);

    void merge(const GpuMat* src, size_t n, GpuMat& dst, const cudaStream_t& stream) 
    {
        CV_Assert(src);
        CV_Assert(n > 0);

        int depth = src[0].depth();
        Size size = src[0].size();

        bool single_channel_only = true;
        int total_channels = 0;

        for (size_t i = 0; i < n; ++i)
        {
            CV_Assert(src[i].size() == size);
            CV_Assert(src[i].depth() == depth);
            single_channel_only = single_channel_only && src[i].channels() == 1;
            total_channels += src[i].channels();
        }

        CV_Assert(single_channel_only);
        CV_Assert(total_channels <= 4);

        if (total_channels == 1)  
            src[0].copyTo(dst);
        else 
        {
            dst.create(size, CV_MAKETYPE(depth, total_channels));

            DevMem2D src_as_devmem[4];
            for(size_t i = 0; i < n; ++i)
                src_as_devmem[i] = src[i];

            split_merge::merge_caller(src_as_devmem, (DevMem2D)dst, 
                                      total_channels, CV_ELEM_SIZE(depth), 
                                      stream);
        }   
    }


    void split(const GpuMat& src, GpuMat* dst, const cudaStream_t& stream) 
    {
        CV_Assert(dst);

        int depth = src.depth();
        int num_channels = src.channels();
        Size size = src.size();

        if (num_channels == 1)
        {
            src.copyTo(dst[0]);
            return;
        }

        for (int i = 0; i < num_channels; ++i)
            dst[i].create(src.size(), depth);

        CV_Assert(num_channels <= 4);

        DevMem2D dst_as_devmem[4];
        for (int i = 0; i < num_channels; ++i)
            dst_as_devmem[i] = dst[i];

        split_merge::split_caller((DevMem2D)src, dst_as_devmem, 
                                  num_channels, src.elemSize1(), 
                                  stream);
    }


}}}


void cv::gpu::merge(const GpuMat* src, size_t n, GpuMat& dst) 
{ 
    split_merge::merge(src, n, dst, 0);
}


void cv::gpu::merge(const vector<GpuMat>& src, GpuMat& dst) 
{
    split_merge::merge(&src[0], src.size(), dst, 0);
}


void cv::gpu::merge(const GpuMat* src, size_t n, GpuMat& dst, const Stream& stream) 
{ 
    split_merge::merge(src, n, dst, StreamAccessor::getStream(stream));
}


void cv::gpu::merge(const vector<GpuMat>& src, GpuMat& dst, const Stream& stream) 
{
    split_merge::merge(&src[0], src.size(), dst, StreamAccessor::getStream(stream));
}


void cv::gpu::split(const GpuMat& src, GpuMat* dst) 
{
    split_merge::split(src, dst, 0);
}


void cv::gpu::split(const GpuMat& src, vector<GpuMat>& dst) 
{
    dst.resize(src.channels());
    if(src.channels() > 0)
        split_merge::split(src, &dst[0], 0);
}


void cv::gpu::split(const GpuMat& src, GpuMat* dst, const Stream& stream) 
{
    split_merge::split(src, dst, StreamAccessor::getStream(stream));
}


void cv::gpu::split(const GpuMat& src, vector<GpuMat>& dst, const Stream& stream) 
{
    dst.resize(src.channels());
    if(src.channels() > 0)
        split_merge::split(src, &dst[0], StreamAccessor::getStream(stream));
}

#endif /* !defined (HAVE_CUDA) */