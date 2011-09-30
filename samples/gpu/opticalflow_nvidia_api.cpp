#if _MSC_VER >= 1400
#pragma warning( disable : 4201 4408 4127 4100)
#endif

#include <iostream>
#include <iomanip>
#include <memory>
#include <exception>
#include <ctime>

#include "cvconfig.h"
#include <iostream>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

#ifdef HAVE_CUDA
#include "NPP_staging/NPP_staging.hpp"
#include "NCVBroxOpticalFlow.hpp"
#endif

#if !defined(HAVE_CUDA)
int main( int argc, const char** argv )
{
    std::cout << "Please compile the library with CUDA support" << std::endl;
    return -1;
}
#else

using std::tr1::shared_ptr;

#define PARAM_INPUT "--input"
#define PARAM_SCALE "--scale"
#define PARAM_ALPHA "--alpha"
#define PARAM_GAMMA "--gamma"
#define PARAM_INNER "--inner"
#define PARAM_OUTER "--outer"
#define PARAM_SOLVER "--solver"
#define PARAM_TIME_STEP "--time-step"
#define PARAM_HELP "--help"

shared_ptr<INCVMemAllocator> g_pGPUMemAllocator;
shared_ptr<INCVMemAllocator> g_pHostMemAllocator;

class RgbToMonochrome
{
public:
    float operator ()(unsigned char b, unsigned char g, unsigned char r)
    {
        float _r = static_cast<float>(r)/255.0f;
        float _g = static_cast<float>(g)/255.0f;
        float _b = static_cast<float>(b)/255.0f;
        return (_r + _g + _b)/3.0f;
    }
};

class RgbToR
{
public:
    float operator ()(unsigned char b, unsigned char g, unsigned char r)
    {
        return static_cast<float>(r)/255.0f;
    }
};


class RgbToG
{
public:
    float operator ()(unsigned char b, unsigned char g, unsigned char r)
    {
        return static_cast<float>(g)/255.0f;
    }
};

class RgbToB
{
public:
    float operator ()(unsigned char b, unsigned char g, unsigned char r)
    {
        return static_cast<float>(b)/255.0f;
    }
};

template<class T>
NCVStatus CopyData(IplImage *image, shared_ptr<NCVMatrixAlloc<Ncv32f>> &dst)
{
    dst = shared_ptr<NCVMatrixAlloc<Ncv32f>> (new NCVMatrixAlloc<Ncv32f> (*g_pHostMemAllocator, image->width, image->height));
    ncvAssertReturn (dst->isMemAllocated (), NCV_ALLOCATOR_BAD_ALLOC);

    unsigned char *row = reinterpret_cast<unsigned char*> (image->imageData);
    T convert;
    for (int i = 0; i < image->height; ++i)
    {
        for (int j = 0; j < image->width; ++j)
        {
            if (image->nChannels < 3)
            {
                dst->ptr ()[j + i*dst->stride ()] = static_cast<float> (*(row + j*image->nChannels))/255.0f;
            }
            else
            {
                unsigned char *color = row + j * image->nChannels;
                dst->ptr ()[j +i*dst->stride ()] = convert (color[0], color[1], color[2]);
            }
        }
        row += image->widthStep;
    }
    return NCV_SUCCESS;
}

template<class T>
NCVStatus CopyData(const IplImage *image, const NCVMatrixAlloc<Ncv32f> &dst)
{
    unsigned char *row = reinterpret_cast<unsigned char*> (image->imageData);
    T convert;
    for (int i = 0; i < image->height; ++i)
    {
        for (int j = 0; j < image->width; ++j)
        {
            if (image->nChannels < 3)
            {
                dst.ptr ()[j + i*dst.stride ()] = static_cast<float>(*(row + j*image->nChannels))/255.0f;
            }
            else
            {
                unsigned char *color = row + j * image->nChannels;
                dst.ptr ()[j +i*dst.stride()] = convert (color[0], color[1], color[2]);
            }
        }
        row += image->widthStep;
    }
    return NCV_SUCCESS;
}

NCVStatus LoadImages (const char *frame0Name, 
                      const char *frame1Name, 
                      int &width, 
                      int &height, 
                      shared_ptr<NCVMatrixAlloc<Ncv32f>> &src, 
                      shared_ptr<NCVMatrixAlloc<Ncv32f>> &dst, 
                      IplImage *&firstFrame, 
                      IplImage *&lastFrame)
{
    IplImage *image;
    image = cvLoadImage (frame0Name);
    if (image == 0)
    {
        std::cout << "Could not open '" << frame0Name << "'\n";
        return NCV_FILE_ERROR;
    }
    
    firstFrame = image;
    // copy data to src
    ncvAssertReturnNcvStat (CopyData<RgbToMonochrome> (image, src));
    
    IplImage *image2;
    image2 = cvLoadImage (frame1Name);
    if (image2 == 0)
    {
        std::cout << "Could not open '" << frame1Name << "'\n";
        return NCV_FILE_ERROR;
    }
    lastFrame = image2;

    ncvAssertReturnNcvStat (CopyData<RgbToMonochrome> (image2, dst));

    width  = image->width;
    height = image->height;

    return NCV_SUCCESS;
}

template<typename T>
inline T Clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template<typename T>
inline T MapValue (T x, T a, T b, T c, T d)
{
    x = Clamp (x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}

NCVStatus ShowFlow (NCVMatrixAlloc<Ncv32f> &u, NCVMatrixAlloc<Ncv32f> &v, const char *name)
{
    IplImage *flowField;
    
    NCVMatrixAlloc<Ncv32f> host_u(*g_pHostMemAllocator, u.width(), u.height());
    ncvAssertReturn(host_u.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    NCVMatrixAlloc<Ncv32f> host_v (*g_pHostMemAllocator, u.width (), u.height ());
    ncvAssertReturn (host_v.isMemAllocated (), NCV_ALLOCATOR_BAD_ALLOC);

    ncvAssertReturnNcvStat (u.copySolid (host_u, 0));
    ncvAssertReturnNcvStat (v.copySolid (host_v, 0));

    float *ptr_u = host_u.ptr ();
    float *ptr_v = host_v.ptr ();

    float maxDisplacement = 1.0f;

    for (Ncv32u i = 0; i < u.height (); ++i)
    {
        for (Ncv32u j = 0; j < u.width (); ++j)
        {
            float d = std::max ( fabsf(*ptr_u), fabsf(*ptr_v) );
            if (d > maxDisplacement) maxDisplacement = d;
            ++ptr_u;
            ++ptr_v;
        }
        ptr_u += u.stride () - u.width ();
        ptr_v += v.stride () - v.width ();
    }

    CvSize image_size = cvSize (u.width (), u.height ());
    flowField = cvCreateImage (image_size, IPL_DEPTH_8U, 4);
    if (flowField == 0) return NCV_NULL_PTR;

    unsigned char *row = reinterpret_cast<unsigned char *> (flowField->imageData);

    ptr_u = host_u.ptr();
    ptr_v = host_v.ptr();
    for (int i = 0; i < flowField->height; ++i)
    {
        for (int j = 0; j < flowField->width; ++j)
        {
            (row + j * flowField->nChannels)[0] = 0;
            (row + j * flowField->nChannels)[1] = static_cast<unsigned char> (MapValue (-(*ptr_v), -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            (row + j * flowField->nChannels)[2] = static_cast<unsigned char> (MapValue (*ptr_u   , -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            (row + j * flowField->nChannels)[3] = 255;
            ++ptr_u;
            ++ptr_v;
        }
        row += flowField->widthStep;
        ptr_u += u.stride () - u.width ();
        ptr_v += v.stride () - v.width ();
    }
    
    cvShowImage (name, flowField);

    return NCV_SUCCESS;
}

IplImage *CreateImage (NCVMatrixAlloc<Ncv32f> &h_r, NCVMatrixAlloc<Ncv32f> &h_g, NCVMatrixAlloc<Ncv32f> &h_b)
{
    CvSize imageSize = cvSize (h_r.width (), h_r.height ());
    IplImage *image  = cvCreateImage (imageSize, IPL_DEPTH_8U, 4);
    if (image == 0) return 0;

    unsigned char *row = reinterpret_cast<unsigned char*> (image->imageData);
    
    for (int i = 0; i < image->height; ++i)
    {
        for (int j = 0; j < image->width; ++j)
        {
            int offset = j * image->nChannels;
            int pos    = i * h_r.stride () + j;
            row[offset + 0] = static_cast<unsigned char> (h_b.ptr ()[pos] * 255.0f);
            row[offset + 1] = static_cast<unsigned char> (h_g.ptr ()[pos] * 255.0f);
            row[offset + 2] = static_cast<unsigned char> (h_r.ptr ()[pos] * 255.0f);
            row[offset + 3] = 255;
        }
        row += image->widthStep;
    }
    return image;
}

void PrintHelp ()
{
    std::cout << "Usage help:\n";
    std::cout << std::setiosflags(std::ios::left);
    std::cout << "\t" << std::setw(15) << PARAM_ALPHA << " - set alpha\n";
    std::cout << "\t" << std::setw(15) << PARAM_GAMMA << " - set gamma\n";
    std::cout << "\t" << std::setw(15) << PARAM_INNER << " - set number of inner iterations\n";
    std::cout << "\t" << std::setw(15) << PARAM_INPUT << " - specify input file names (2 image files)\n";
    std::cout << "\t" << std::setw(15) << PARAM_OUTER << " - set number of outer iterations\n";
    std::cout << "\t" << std::setw(15) << PARAM_SCALE << " - set pyramid scale factor\n";
    std::cout << "\t" << std::setw(15) << PARAM_SOLVER << " - set number of basic solver iterations\n";
    std::cout << "\t" << std::setw(15) << PARAM_TIME_STEP << " - set frame interpolation time step\n";
    std::cout << "\t" << std::setw(15) << PARAM_HELP << " - display this help message\n";
}

int ProcessCommandLine(int argc, char **argv, 
                       Ncv32f &timeStep, 
                       char *&frame0Name, 
                       char *&frame1Name, 
                       NCVBroxOpticalFlowDescriptor &desc)
{
    timeStep = 0.25f;
    for (int iarg = 1; iarg < argc; ++iarg)
    {
        if (strcmp(argv[iarg], PARAM_INPUT) == 0)
        {
            if (iarg + 2 < argc)
            {
                frame0Name = argv[++iarg];
                frame1Name = argv[++iarg];
            }
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_SCALE) == 0)
        {
            if (iarg + 1 < argc)
                desc.scale_factor = static_cast<Ncv32f>(atof(argv[++iarg]));
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_ALPHA) == 0)
        {
            if (iarg + 1 < argc)
                desc.alpha = static_cast<Ncv32f>(atof(argv[++iarg]));
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_GAMMA) == 0)
        {
            if (iarg + 1 < argc)
                desc.gamma = static_cast<Ncv32f>(atof(argv[++iarg]));
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_INNER) == 0)
        {
            if (iarg + 1 < argc)
                desc.number_of_inner_iterations = static_cast<Ncv32u>(atoi(argv[++iarg]));
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_OUTER) == 0)
        {
            if (iarg + 1 < argc)
                desc.number_of_outer_iterations = static_cast<Ncv32u>(atoi(argv[++iarg]));
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_SOLVER) == 0)
        {
            if (iarg + 1 < argc)
                desc.number_of_solver_iterations = static_cast<Ncv32u>(atoi(argv[++iarg]));
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_TIME_STEP) == 0)
        {
            if (iarg + 1 < argc)
                timeStep = static_cast<Ncv32f>(atof(argv[++iarg]));
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_HELP) == 0)
        {
            PrintHelp ();
            return 0;
        }
    }
    return 0;
}


int main(int argc, char **argv)
{
    char *frame0Name = 0, *frame1Name = 0;
    Ncv32f timeStep = 0.01f;

    NCVBroxOpticalFlowDescriptor desc;

    desc.alpha = 0.197f;
    desc.gamma = 50.0f;
    desc.number_of_inner_iterations  = 10;
    desc.number_of_outer_iterations  = 77;
    desc.number_of_solver_iterations = 10;
    desc.scale_factor = 0.8f;

    int result = ProcessCommandLine (argc, argv, timeStep, frame0Name, frame1Name, desc);
    if (argc == 1 || result)
    {
        PrintHelp();
        return result;
    }

    std::cout << "OpenCV / NVIDIA Computer Vision\n";
    std::cout << "Optical Flow Demo: Frame Interpolation\n";
    std::cout << "=========================================\n";
    std::cout << "Press:\n ESC to quit\n 'a' to move to the previous frame\n 's' to move to the next frame\n";

    int devId;
    ncvAssertCUDAReturn(cudaGetDevice(&devId), -1);
    cudaDeviceProp devProp;
    ncvAssertCUDAReturn(cudaGetDeviceProperties(&devProp, devId), -1);
    std::cout << "Using GPU: " << devId << "(" << devProp.name <<
        "), arch=" << devProp.major << "." << devProp.minor << std::endl;

    g_pGPUMemAllocator  = shared_ptr<INCVMemAllocator> (new NCVMemNativeAllocator (NCVMemoryTypeDevice, devProp.textureAlignment));
    ncvAssertPrintReturn (g_pGPUMemAllocator->isInitialized (), "Device memory allocator isn't initialized", -1);

    g_pHostMemAllocator = shared_ptr<INCVMemAllocator> (new NCVMemNativeAllocator (NCVMemoryTypeHostPageable, devProp.textureAlignment));
    ncvAssertPrintReturn (g_pHostMemAllocator->isInitialized (), "Host memory allocator isn't initialized", -1);

    int width, height;

    shared_ptr<NCVMatrixAlloc<Ncv32f>> src_host;
    shared_ptr<NCVMatrixAlloc<Ncv32f>> dst_host;

    IplImage *firstFrame, *lastFrame;
    if (frame0Name != 0 && frame1Name != 0)
    {
        ncvAssertReturnNcvStat (LoadImages (frame0Name, frame1Name, width, height, src_host, dst_host, firstFrame, lastFrame));
    }
    else
    {
        ncvAssertReturnNcvStat (LoadImages ("frame10.bmp", "frame11.bmp", width, height, src_host, dst_host, firstFrame, lastFrame));
    }

    shared_ptr<NCVMatrixAlloc<Ncv32f>> src (new NCVMatrixAlloc<Ncv32f> (*g_pGPUMemAllocator, src_host->width (), src_host->height ()));
    ncvAssertReturn(src->isMemAllocated(), -1);

    shared_ptr<NCVMatrixAlloc<Ncv32f>> dst (new NCVMatrixAlloc<Ncv32f> (*g_pGPUMemAllocator, src_host->width (), src_host->height ()));
    ncvAssertReturn (dst->isMemAllocated (), -1);

    ncvAssertReturnNcvStat (src_host->copySolid ( *src, 0 ));
    ncvAssertReturnNcvStat (dst_host->copySolid ( *dst, 0 ));

#if defined SAFE_MAT_DECL
#undef SAFE_MAT_DECL
#endif
#define SAFE_MAT_DECL(name, allocator, sx, sy) \
    NCVMatrixAlloc<Ncv32f> name(*allocator, sx, sy);\
    ncvAssertReturn(name##.isMemAllocated(), -1);

    SAFE_MAT_DECL (u, g_pGPUMemAllocator, width, height);
    SAFE_MAT_DECL (v, g_pGPUMemAllocator, width, height);

    SAFE_MAT_DECL (uBck, g_pGPUMemAllocator, width, height);
    SAFE_MAT_DECL (vBck, g_pGPUMemAllocator, width, height);

    SAFE_MAT_DECL (h_r, g_pHostMemAllocator, width, height);
    SAFE_MAT_DECL (h_g, g_pHostMemAllocator, width, height);
    SAFE_MAT_DECL (h_b, g_pHostMemAllocator, width, height);

    std::cout << "Estimating optical flow\nForward...\n";

    if (NCV_SUCCESS != NCVBroxOpticalFlow (desc, *g_pGPUMemAllocator, *src, *dst, u, v, 0))
    {
        std::cout << "Failed\n";
        return -1;
    }
    
    std::cout << "Backward...\n";
    if (NCV_SUCCESS != NCVBroxOpticalFlow (desc, *g_pGPUMemAllocator, *dst, *src, uBck, vBck, 0))
    {
        std::cout << "Failed\n";
        return -1;
    }

    // matrix for temporary data
    SAFE_MAT_DECL (d_temp, g_pGPUMemAllocator, width, height);

    // first frame color components (GPU memory)
    SAFE_MAT_DECL (d_r, g_pGPUMemAllocator, width, height);
    SAFE_MAT_DECL (d_g, g_pGPUMemAllocator, width, height);
    SAFE_MAT_DECL (d_b, g_pGPUMemAllocator, width, height);

    // second frame color components (GPU memory)
    SAFE_MAT_DECL (d_rt, g_pGPUMemAllocator, width, height);
    SAFE_MAT_DECL (d_gt, g_pGPUMemAllocator, width, height);
    SAFE_MAT_DECL (d_bt, g_pGPUMemAllocator, width, height);

    // intermediate frame color components (GPU memory)
    SAFE_MAT_DECL (d_rNew, g_pGPUMemAllocator, width, height);
    SAFE_MAT_DECL (d_gNew, g_pGPUMemAllocator, width, height);
    SAFE_MAT_DECL (d_bNew, g_pGPUMemAllocator, width, height);

    // interpolated forward flow
    SAFE_MAT_DECL (ui, g_pGPUMemAllocator, width, height);
    SAFE_MAT_DECL (vi, g_pGPUMemAllocator, width, height);

    // interpolated backward flow
    SAFE_MAT_DECL (ubi, g_pGPUMemAllocator, width, height);
    SAFE_MAT_DECL (vbi, g_pGPUMemAllocator, width, height);

    // occlusion masks
    SAFE_MAT_DECL (occ0, g_pGPUMemAllocator, width, height);
    SAFE_MAT_DECL (occ1, g_pGPUMemAllocator, width, height);

    // prepare color components on host and copy them to device memory
    ncvAssertReturnNcvStat (CopyData<RgbToR> (firstFrame, h_r));
    ncvAssertReturnNcvStat (CopyData<RgbToG> (firstFrame, h_g));
    ncvAssertReturnNcvStat (CopyData<RgbToB> (firstFrame, h_b));

    ncvAssertReturnNcvStat (h_r.copySolid ( d_r, 0 ));
    ncvAssertReturnNcvStat (h_g.copySolid ( d_g, 0 ));
    ncvAssertReturnNcvStat (h_b.copySolid ( d_b, 0 ));

    ncvAssertReturnNcvStat (CopyData<RgbToR> (lastFrame, h_r));
    ncvAssertReturnNcvStat (CopyData<RgbToG> (lastFrame, h_g));
    ncvAssertReturnNcvStat (CopyData<RgbToB> (lastFrame, h_b));

    ncvAssertReturnNcvStat (h_r.copySolid ( d_rt, 0 ));
    ncvAssertReturnNcvStat (h_g.copySolid ( d_gt, 0 ));
    ncvAssertReturnNcvStat (h_b.copySolid ( d_bt, 0 ));

    std::cout << "Interpolating...\n";
    std::cout.precision (4);

    std::vector<IplImage*> frames;
    frames.push_back (firstFrame);

    // compute interpolated frames
    for (Ncv32f timePos = timeStep; timePos < 1.0f; timePos += timeStep)
    {
        ncvAssertCUDAReturn (cudaMemset (ui.ptr (), 0, ui.pitch () * ui.height ()), NCV_CUDA_ERROR);
        ncvAssertCUDAReturn (cudaMemset (vi.ptr (), 0, vi.pitch () * vi.height ()), NCV_CUDA_ERROR);

        ncvAssertCUDAReturn (cudaMemset (ubi.ptr (), 0, ubi.pitch () * ubi.height ()), NCV_CUDA_ERROR);
        ncvAssertCUDAReturn (cudaMemset (vbi.ptr (), 0, vbi.pitch () * vbi.height ()), NCV_CUDA_ERROR);

        ncvAssertCUDAReturn (cudaMemset (occ0.ptr (), 0, occ0.pitch () * occ0.height ()), NCV_CUDA_ERROR);
        ncvAssertCUDAReturn (cudaMemset (occ1.ptr (), 0, occ1.pitch () * occ1.height ()), NCV_CUDA_ERROR);

        NppStInterpolationState state;
        // interpolation state should be filled once except pSrcFrame0, pSrcFrame1, and pNewFrame
        // we will only need to reset buffers content to 0 since interpolator doesn't do this itself
        state.size  = NcvSize32u (width, height);
        state.nStep = d_r.pitch ();
        state.pSrcFrame0 = d_r.ptr ();
        state.pSrcFrame1 = d_rt.ptr ();
        state.pFU = u.ptr ();
        state.pFV = v.ptr ();
        state.pBU = uBck.ptr ();
        state.pBV = vBck.ptr ();
        state.pos = timePos;
        state.pNewFrame = d_rNew.ptr ();
        state.ppBuffers[0] = occ0.ptr ();
        state.ppBuffers[1] = occ1.ptr ();
        state.ppBuffers[2] = ui.ptr ();
        state.ppBuffers[3] = vi.ptr ();
        state.ppBuffers[4] = ubi.ptr ();
        state.ppBuffers[5] = vbi.ptr ();

        // interpolate red channel
        nppiStInterpolateFrames (&state);

        // reset buffers
        ncvAssertCUDAReturn (cudaMemset (ui.ptr (), 0, ui.pitch () * ui.height ()), NCV_CUDA_ERROR);
        ncvAssertCUDAReturn (cudaMemset (vi.ptr (), 0, vi.pitch () * vi.height ()), NCV_CUDA_ERROR);

        ncvAssertCUDAReturn (cudaMemset (ubi.ptr (), 0, ubi.pitch () * ubi.height ()), NCV_CUDA_ERROR);
        ncvAssertCUDAReturn (cudaMemset (vbi.ptr (), 0, vbi.pitch () * vbi.height ()), NCV_CUDA_ERROR);

        ncvAssertCUDAReturn (cudaMemset (occ0.ptr (), 0, occ0.pitch () * occ0.height ()), NCV_CUDA_ERROR);
        ncvAssertCUDAReturn (cudaMemset (occ1.ptr (), 0, occ1.pitch () * occ1.height ()), NCV_CUDA_ERROR);

        // interpolate green channel
        state.pSrcFrame0 = d_g.ptr ();
        state.pSrcFrame1 = d_gt.ptr ();
        state.pNewFrame  = d_gNew.ptr ();

        nppiStInterpolateFrames (&state);

        // reset buffers
        ncvAssertCUDAReturn (cudaMemset (ui.ptr (), 0, ui.pitch () * ui.height ()), NCV_CUDA_ERROR);
        ncvAssertCUDAReturn (cudaMemset (vi.ptr (), 0, vi.pitch () * vi.height ()), NCV_CUDA_ERROR);

        ncvAssertCUDAReturn (cudaMemset (ubi.ptr (), 0, ubi.pitch () * ubi.height ()), NCV_CUDA_ERROR);
        ncvAssertCUDAReturn (cudaMemset (vbi.ptr (), 0, vbi.pitch () * vbi.height ()), NCV_CUDA_ERROR);

        ncvAssertCUDAReturn (cudaMemset (occ0.ptr (), 0, occ0.pitch () * occ0.height ()), NCV_CUDA_ERROR);
        ncvAssertCUDAReturn (cudaMemset (occ1.ptr (), 0, occ1.pitch () * occ1.height ()), NCV_CUDA_ERROR);

        // interpolate blue channel
        state.pSrcFrame0 = d_b.ptr ();
        state.pSrcFrame1 = d_bt.ptr ();
        state.pNewFrame  = d_bNew.ptr ();

        nppiStInterpolateFrames (&state);

        // copy to host memory
        ncvAssertReturnNcvStat (d_rNew.copySolid (h_r, 0));
        ncvAssertReturnNcvStat (d_gNew.copySolid (h_g, 0));
        ncvAssertReturnNcvStat (d_bNew.copySolid (h_b, 0));

        // convert to IplImage
        IplImage *newFrame = CreateImage (h_r, h_g, h_b);
        if (newFrame == 0)
        {
            std::cout << "Could not create new frame in host memory\n";
            break;
        }
        frames.push_back (newFrame);
        std::cout << timePos * 100.0f << "%\r";
    }
    std::cout << std::setw (5) << "100%\n";

    frames.push_back (lastFrame);

    Ncv32u currentFrame;
    currentFrame = 0;

    ShowFlow (u, v, "Forward flow");
    ShowFlow (uBck, vBck, "Backward flow");

    cvShowImage ("Interpolated frame", frames[currentFrame]);

    bool qPressed = false;
    while ( !qPressed )
    {
        int key = toupper (cvWaitKey (10));
        switch (key)
        {
        case 27:
            qPressed = true;
            break;
        case 'A':
            if (currentFrame > 0) --currentFrame;
            cvShowImage ("Interpolated frame", frames[currentFrame]);
            break;
        case 'S':
            if (currentFrame < frames.size()-1) ++currentFrame;
            cvShowImage ("Interpolated frame", frames[currentFrame]);
            break;
        }
    }

    cvDestroyAllWindows ();

    std::vector<IplImage*>::iterator iter;
    for (iter = frames.begin (); iter != frames.end (); ++iter)
    {
        cvReleaseImage (&(*iter));
    }

    return 0;
}

#endif
