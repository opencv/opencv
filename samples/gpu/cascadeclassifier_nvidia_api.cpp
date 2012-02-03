#if _MSC_VER >= 1400
#pragma warning( disable : 4201 4408 4127 4100)
#endif

#include "cvconfig.h"
#include <iostream>
#include <iomanip>
#include <cstdio>
#include "opencv2/gpu/gpu.hpp"

#ifdef HAVE_CUDA
#include "NCVHaarObjectDetection.hpp"
#endif

using namespace std;
using namespace cv;


#if !defined(HAVE_CUDA)
int main( int argc, const char** argv )
{
    cout << "Please compile the library with CUDA support" << endl;
    return -1;
}
#else


const Size2i preferredVideoFrameSize(640, 480);
const string wndTitle = "NVIDIA Computer Vision :: Haar Classifiers Cascade";


void matPrint(Mat &img, int lineOffsY, Scalar fontColor, const string &ss)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.8;
    int fontThickness = 2;
    Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;
    putText(img, ss, org, fontFace, fontScale, CV_RGB(0,0,0), 5*fontThickness/2, 16);
    putText(img, ss, org, fontFace, fontScale, fontColor, fontThickness, 16);
}


void displayState(Mat &canvas, bool bHelp, bool bGpu, bool bLargestFace, bool bFilter, double fps)
{
    Scalar fontColorRed = CV_RGB(255,0,0);
    Scalar fontColorNV  = CV_RGB(118,185,0);

    ostringstream ss;
    ss << "FPS = " << setprecision(1) << fixed << fps;
    matPrint(canvas, 0, fontColorRed, ss.str());
    ss.str("");
    ss << "[" << canvas.cols << "x" << canvas.rows << "], " <<
        (bGpu ? "GPU, " : "CPU, ") <<
        (bLargestFace ? "OneFace, " : "MultiFace, ") <<
        (bFilter ? "Filter:ON" : "Filter:OFF");
    matPrint(canvas, 1, fontColorRed, ss.str());

    if (bHelp)
    {
        matPrint(canvas, 2, fontColorNV, "Space - switch GPU / CPU");
        matPrint(canvas, 3, fontColorNV, "M - switch OneFace / MultiFace");
        matPrint(canvas, 4, fontColorNV, "F - toggle rectangles Filter");
        matPrint(canvas, 5, fontColorNV, "H - toggle hotkeys help");
    }
    else
    {
        matPrint(canvas, 2, fontColorNV, "H - toggle hotkeys help");
    }
}


NCVStatus process(Mat *srcdst,
                  Ncv32u width, Ncv32u height,
                  NcvBool bFilterRects, NcvBool bLargestFace,
                  HaarClassifierCascadeDescriptor &haar,
                  NCVVector<HaarStage64> &d_haarStages, NCVVector<HaarClassifierNode128> &d_haarNodes,
                  NCVVector<HaarFeature64> &d_haarFeatures, NCVVector<HaarStage64> &h_haarStages,
                  INCVMemAllocator &gpuAllocator,
                  INCVMemAllocator &cpuAllocator,
                  cudaDeviceProp &devProp)
{
    ncvAssertReturn(!((srcdst == NULL) ^ gpuAllocator.isCounting()), NCV_NULL_PTR);

    NCVStatus ncvStat;

    NCV_SET_SKIP_COND(gpuAllocator.isCounting());

    NCVMatrixAlloc<Ncv8u> d_src(gpuAllocator, width, height);
    ncvAssertReturn(d_src.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
    NCVMatrixAlloc<Ncv8u> h_src(cpuAllocator, width, height);
    ncvAssertReturn(h_src.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);
    NCVVectorAlloc<NcvRect32u> d_rects(gpuAllocator, 100);
    ncvAssertReturn(d_rects.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    NCV_SKIP_COND_BEGIN

    for (Ncv32u i=0; i<(Ncv32u)srcdst->rows; i++)
    {
        memcpy(h_src.ptr() + i * h_src.stride(), srcdst->ptr(i), srcdst->cols);
    }

    ncvStat = h_src.copySolid(d_src, 0);
    ncvAssertReturnNcvStat(ncvStat);
    ncvAssertCUDAReturn(cudaStreamSynchronize(0), NCV_CUDA_ERROR);

    NCV_SKIP_COND_END

    NcvSize32u roi;
    roi.width = d_src.width();
    roi.height = d_src.height();

    Ncv32u numDetections;
    ncvStat = ncvDetectObjectsMultiScale_device(
        d_src, roi, d_rects, numDetections, haar, h_haarStages,
        d_haarStages, d_haarNodes, d_haarFeatures,
        haar.ClassifierSize,
        (bFilterRects || bLargestFace) ? 4 : 0,
        1.2f, 1,
        (bLargestFace ? NCVPipeObjDet_FindLargestObject : 0)
        | NCVPipeObjDet_VisualizeInPlace,
        gpuAllocator, cpuAllocator, devProp, 0);
    ncvAssertReturnNcvStat(ncvStat);
    ncvAssertCUDAReturn(cudaStreamSynchronize(0), NCV_CUDA_ERROR);

    NCV_SKIP_COND_BEGIN

    ncvStat = d_src.copySolid(h_src, 0);
    ncvAssertReturnNcvStat(ncvStat);
    ncvAssertCUDAReturn(cudaStreamSynchronize(0), NCV_CUDA_ERROR);

    for (Ncv32u i=0; i<(Ncv32u)srcdst->rows; i++)
    {
        memcpy(srcdst->ptr(i), h_src.ptr() + i * h_src.stride(), srcdst->cols);
    }

    NCV_SKIP_COND_END

    return NCV_SUCCESS;
}


int main(int argc, const char** argv)
{
    cout << "OpenCV / NVIDIA Computer Vision" << endl;
    cout << "Face Detection in video and live feed" << endl;
    cout << "Syntax: exename <cascade_file> <image_or_video_or_cameraid>" << endl;
    cout << "=========================================" << endl;

    ncvAssertPrintReturn(cv::gpu::getCudaEnabledDeviceCount() != 0, "No GPU found or the library is compiled without GPU support", -1);
    ncvAssertPrintReturn(argc == 3, "Invalid number of arguments", -1);

    cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

    string cascadeName = argv[1];
    string inputName = argv[2];

    NCVStatus ncvStat;
    NcvBool bQuit = false;
    VideoCapture capture;
    Size2i frameSize;

    //open content source
    Mat image = imread(inputName);
    Mat frame;
    if (!image.empty())
    {
        frameSize.width = image.cols;
        frameSize.height = image.rows;
    }
    else
    {
        if (!capture.open(inputName))
        {
            int camid = -1;

            istringstream ss(inputName);
            int x = 0;
            ss >> x;

            ncvAssertPrintReturn(capture.open(camid) != 0, "Can't open source", -1);
        }

        capture >> frame;
        ncvAssertPrintReturn(!frame.empty(), "Empty video source", -1);

        frameSize.width = frame.cols;
        frameSize.height = frame.rows;
    }

    NcvBool bUseGPU = true;
    NcvBool bLargestObject = false;
    NcvBool bFilterRects = true;
    NcvBool bHelpScreen = false;

    CascadeClassifier classifierOpenCV;
    ncvAssertPrintReturn(classifierOpenCV.load(cascadeName) != 0, "Error (in OpenCV) opening classifier", -1);

    int devId;
    ncvAssertCUDAReturn(cudaGetDevice(&devId), -1);
    cudaDeviceProp devProp;
    ncvAssertCUDAReturn(cudaGetDeviceProperties(&devProp, devId), -1);
    cout << "Using GPU: " << devId << "(" << devProp.name <<
            "), arch=" << devProp.major << "." << devProp.minor << endl;

    //==============================================================================
    //
    // Load the classifier from file (assuming its size is about 1 mb)
    // using a simple allocator
    //
    //==============================================================================

    NCVMemNativeAllocator gpuCascadeAllocator(NCVMemoryTypeDevice, static_cast<Ncv32u>(devProp.textureAlignment));
    ncvAssertPrintReturn(gpuCascadeAllocator.isInitialized(), "Error creating cascade GPU allocator", -1);
    NCVMemNativeAllocator cpuCascadeAllocator(NCVMemoryTypeHostPinned, static_cast<Ncv32u>(devProp.textureAlignment));
    ncvAssertPrintReturn(cpuCascadeAllocator.isInitialized(), "Error creating cascade CPU allocator", -1);

    Ncv32u haarNumStages, haarNumNodes, haarNumFeatures;
    ncvStat = ncvHaarGetClassifierSize(cascadeName, haarNumStages, haarNumNodes, haarNumFeatures);
    ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error reading classifier size (check the file)", -1);

    NCVVectorAlloc<HaarStage64> h_haarStages(cpuCascadeAllocator, haarNumStages);
    ncvAssertPrintReturn(h_haarStages.isMemAllocated(), "Error in cascade CPU allocator", -1);
    NCVVectorAlloc<HaarClassifierNode128> h_haarNodes(cpuCascadeAllocator, haarNumNodes);
    ncvAssertPrintReturn(h_haarNodes.isMemAllocated(), "Error in cascade CPU allocator", -1);
    NCVVectorAlloc<HaarFeature64> h_haarFeatures(cpuCascadeAllocator, haarNumFeatures);

    ncvAssertPrintReturn(h_haarFeatures.isMemAllocated(), "Error in cascade CPU allocator", -1);

    HaarClassifierCascadeDescriptor haar;
    ncvStat = ncvHaarLoadFromFile_host(cascadeName, haar, h_haarStages, h_haarNodes, h_haarFeatures);
    ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error loading classifier", -1);

    NCVVectorAlloc<HaarStage64> d_haarStages(gpuCascadeAllocator, haarNumStages);
    ncvAssertPrintReturn(d_haarStages.isMemAllocated(), "Error in cascade GPU allocator", -1);
    NCVVectorAlloc<HaarClassifierNode128> d_haarNodes(gpuCascadeAllocator, haarNumNodes);
    ncvAssertPrintReturn(d_haarNodes.isMemAllocated(), "Error in cascade GPU allocator", -1);
    NCVVectorAlloc<HaarFeature64> d_haarFeatures(gpuCascadeAllocator, haarNumFeatures);
    ncvAssertPrintReturn(d_haarFeatures.isMemAllocated(), "Error in cascade GPU allocator", -1);

    ncvStat = h_haarStages.copySolid(d_haarStages, 0);
    ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error copying cascade to GPU", -1);
    ncvStat = h_haarNodes.copySolid(d_haarNodes, 0);
    ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error copying cascade to GPU", -1);
    ncvStat = h_haarFeatures.copySolid(d_haarFeatures, 0);
    ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error copying cascade to GPU", -1);

    //==============================================================================
    //
    // Calculate memory requirements and create real allocators
    //
    //==============================================================================

    NCVMemStackAllocator gpuCounter(static_cast<Ncv32u>(devProp.textureAlignment));
    ncvAssertPrintReturn(gpuCounter.isInitialized(), "Error creating GPU memory counter", -1);
    NCVMemStackAllocator cpuCounter(static_cast<Ncv32u>(devProp.textureAlignment));
    ncvAssertPrintReturn(cpuCounter.isInitialized(), "Error creating CPU memory counter", -1);

    ncvStat = process(NULL, frameSize.width, frameSize.height,
                      false, false, haar,
                      d_haarStages, d_haarNodes,
                      d_haarFeatures, h_haarStages,
                      gpuCounter, cpuCounter, devProp);
    ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error in memory counting pass", -1);

    NCVMemStackAllocator gpuAllocator(NCVMemoryTypeDevice, gpuCounter.maxSize(), static_cast<Ncv32u>(devProp.textureAlignment));
    ncvAssertPrintReturn(gpuAllocator.isInitialized(), "Error creating GPU memory allocator", -1);
    NCVMemStackAllocator cpuAllocator(NCVMemoryTypeHostPinned, cpuCounter.maxSize(), static_cast<Ncv32u>(devProp.textureAlignment));
    ncvAssertPrintReturn(cpuAllocator.isInitialized(), "Error creating CPU memory allocator", -1);

    printf("Initialized for frame size [%dx%d]\n", frameSize.width, frameSize.height);

    //==============================================================================
    //
    // Main processing loop
    //
    //==============================================================================

    namedWindow(wndTitle, 1);
    Mat gray, frameDisp;

    do
    {
        Mat gray;
        cvtColor((image.empty() ? frame : image), gray, CV_BGR2GRAY);

        //
        // process
        //

        NcvSize32u minSize = haar.ClassifierSize;
        if (bLargestObject)
        {
            Ncv32u ratioX = preferredVideoFrameSize.width / minSize.width;
            Ncv32u ratioY = preferredVideoFrameSize.height / minSize.height;
            Ncv32u ratioSmallest = min(ratioX, ratioY);
            ratioSmallest = max((Ncv32u)(ratioSmallest / 2.5f), (Ncv32u)1);
            minSize.width *= ratioSmallest;
            minSize.height *= ratioSmallest;
        }

        Ncv32f avgTime;
        NcvTimer timer = ncvStartTimer();

        if (bUseGPU)
        {
            ncvStat = process(&gray, frameSize.width, frameSize.height,
                              bFilterRects, bLargestObject, haar,
                              d_haarStages, d_haarNodes,
                              d_haarFeatures, h_haarStages,
                              gpuAllocator, cpuAllocator, devProp);
            ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error in memory counting pass", -1);
        }
        else
        {
            vector<Rect> rectsOpenCV;

            classifierOpenCV.detectMultiScale(
                gray,
                rectsOpenCV,
                1.2f,
                bFilterRects ? 4 : 0,
                (bLargestObject ? CV_HAAR_FIND_BIGGEST_OBJECT : 0)
                | CV_HAAR_SCALE_IMAGE,
                Size(minSize.width, minSize.height));

            for (size_t rt = 0; rt < rectsOpenCV.size(); ++rt)
                rectangle(gray, rectsOpenCV[rt], Scalar(255));
        }

        avgTime = (Ncv32f)ncvEndQueryTimerMs(timer);

        cvtColor(gray, frameDisp, CV_GRAY2BGR);
        displayState(frameDisp, bHelpScreen, bUseGPU, bLargestObject, bFilterRects, 1000.0f / avgTime);
        imshow(wndTitle, frameDisp);

        //handle input
        switch (cvWaitKey(3))
        {
        case ' ':
            bUseGPU = !bUseGPU;
            break;
        case 'm':
        case 'M':
            bLargestObject = !bLargestObject;
            break;
        case 'f':
        case 'F':
            bFilterRects = !bFilterRects;
            break;
        case 'h':
        case 'H':
            bHelpScreen = !bHelpScreen;
            break;
        case 27:
            bQuit = true;
            break;
        }

        // For camera and video file, capture the next image
        if (capture.isOpened())
        {
            capture >> frame;
            if (frame.empty())
            {
                break;
            }
        }
    } while (!bQuit);

    cvDestroyWindow(wndTitle.c_str());

    return 0;
}

#endif //!defined(HAVE_CUDA)
