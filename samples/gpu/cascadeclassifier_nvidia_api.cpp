#pragma warning( disable : 4201 4408 4127 4100)
#include <cstdio>

#include "cvconfig.h"
#if !defined(HAVE_CUDA) || defined(__GNUC__)
    int main( int argc, const char** argv ) { return printf("Please compile the librarary with CUDA support."), -1; }
#else

#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include "NCVHaarObjectDetection.hpp"

using namespace cv;

const Size2i preferredVideoFrameSize(640, 480);

std::string preferredClassifier = "haarcascade_frontalface_alt.xml";
std::string wndTitle = "NVIDIA Computer Vision SDK :: Face Detection in Video Feed";


void printSyntax(void)
{
    printf("Syntax: FaceDetectionFeed.exe [-c cameranum | -v filename] classifier.xml\n");
}

void imagePrintf(Mat& img, int lineOffsY, Scalar color, const char *format, ...)
{    
    int fontFace = CV_FONT_HERSHEY_PLAIN;
    double fontScale = 1;       
    
    int baseline;
    Size textSize = cv::getTextSize("T", fontFace, fontScale, 1, &baseline);

    va_list arg_ptr;
    va_start(arg_ptr, format);

    char strBuf[4096];
    vsprintf(&strBuf[0], format, arg_ptr);

    Point org(1, 3 * textSize.height * (lineOffsY + 1) / 2);    
    putText(img, &strBuf[0], org, fontFace, fontScale, color);
    va_end(arg_ptr);    
}

NCVStatus process(Mat *srcdst,
                  Ncv32u width, Ncv32u height,
                  NcvBool bShowAllHypotheses, NcvBool bLargestFace,
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
        bShowAllHypotheses ? 0 : 4,
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

int main( int argc, const char** argv )
{
    NCVStatus ncvStat;

    printf("NVIDIA Computer Vision SDK\n");
    printf("Face Detection in video and live feed\n");
    printf("=========================================\n");
    printf("  Esc   - Quit\n");
    printf("  Space - Switch between NCV and OpenCV\n");
    printf("  L     - Switch between FullSearch and LargestFace modes\n");
    printf("  U     - Toggle unfiltered hypotheses visualization in FullSearch\n");
	
    VideoCapture capture;    
    bool bQuit = false;

    Size2i frameSize;

    if (argc != 4 && argc != 1)
    {
        printSyntax();
        return -1;
    }

   if (argc == 1 || strcmp(argv[1], "-c") == 0)
    {
        // Camera input is specified
        int camIdx = (argc == 3) ? atoi(argv[2]) : 0;
        if(!capture.open(camIdx))        
            return printf("Error opening camera\n"), -1;        
            
        capture.set(CV_CAP_PROP_FRAME_WIDTH, preferredVideoFrameSize.width);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, preferredVideoFrameSize.height);
        capture.set(CV_CAP_PROP_FPS, 25);
        frameSize = preferredVideoFrameSize;
    }
    else if (strcmp(argv[1], "-v") == 0)
    {
        // Video file input (avi)
        if(!capture.open(argv[2]))
            return printf("Error opening video file\n"), -1;

        frameSize.width  = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
        frameSize.height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    }
    else
        return printSyntax(), -1;

    NcvBool bUseOpenCV = true;
    NcvBool bLargestFace = false; //LargestFace=true is used usually during training
    NcvBool bShowAllHypotheses = false;

    CascadeClassifier classifierOpenCV;
    std::string classifierFile;
    if (argc == 1)
    {
        classifierFile = preferredClassifier;
    }
    else
    {
        classifierFile.assign(argv[3]);
    }

    if (!classifierOpenCV.load(classifierFile))
    {
        printf("Error (in OpenCV) opening classifier\n");
        printSyntax();
        return -1;
    }

    int devId;
    ncvAssertCUDAReturn(cudaGetDevice(&devId), -1);
    cudaDeviceProp devProp;
    ncvAssertCUDAReturn(cudaGetDeviceProperties(&devProp, devId), -1);
    printf("Using GPU %d %s, arch=%d.%d\n", devId, devProp.name, devProp.major, devProp.minor);

    //==============================================================================
    //
    // Load the classifier from file (assuming its size is about 1 mb)
    // using a simple allocator
    //
    //==============================================================================

    NCVMemNativeAllocator gpuCascadeAllocator(NCVMemoryTypeDevice, devProp.textureAlignment);
    ncvAssertPrintReturn(gpuCascadeAllocator.isInitialized(), "Error creating cascade GPU allocator", -1);
    NCVMemNativeAllocator cpuCascadeAllocator(NCVMemoryTypeHostPinned, devProp.textureAlignment);
    ncvAssertPrintReturn(cpuCascadeAllocator.isInitialized(), "Error creating cascade CPU allocator", -1);

    Ncv32u haarNumStages, haarNumNodes, haarNumFeatures;
    ncvStat = ncvHaarGetClassifierSize(classifierFile, haarNumStages, haarNumNodes, haarNumFeatures);
    ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error reading classifier size (check the file)", -1);

    NCVVectorAlloc<HaarStage64> h_haarStages(cpuCascadeAllocator, haarNumStages);
    ncvAssertPrintReturn(h_haarStages.isMemAllocated(), "Error in cascade CPU allocator", -1);
    NCVVectorAlloc<HaarClassifierNode128> h_haarNodes(cpuCascadeAllocator, haarNumNodes);
    ncvAssertPrintReturn(h_haarNodes.isMemAllocated(), "Error in cascade CPU allocator", -1);
    NCVVectorAlloc<HaarFeature64> h_haarFeatures(cpuCascadeAllocator, haarNumFeatures);
    ncvAssertPrintReturn(h_haarFeatures.isMemAllocated(), "Error in cascade CPU allocator", -1);

    HaarClassifierCascadeDescriptor haar;
    ncvStat = ncvHaarLoadFromFile_host(classifierFile, haar, h_haarStages, h_haarNodes, h_haarFeatures);
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

    NCVMemStackAllocator gpuCounter(devProp.textureAlignment);
    ncvAssertPrintReturn(gpuCounter.isInitialized(), "Error creating GPU memory counter", -1);
    NCVMemStackAllocator cpuCounter(devProp.textureAlignment);
    ncvAssertPrintReturn(cpuCounter.isInitialized(), "Error creating CPU memory counter", -1);

    ncvStat = process(NULL, frameSize.width, frameSize.height,
                      false, false, haar,
                      d_haarStages, d_haarNodes,
                      d_haarFeatures, h_haarStages,
                      gpuCounter, cpuCounter, devProp);
    ncvAssertPrintReturn(ncvStat == NCV_SUCCESS, "Error in memory counting pass", -1);

    NCVMemStackAllocator gpuAllocator(NCVMemoryTypeDevice, gpuCounter.maxSize(), devProp.textureAlignment);
    ncvAssertPrintReturn(gpuAllocator.isInitialized(), "Error creating GPU memory allocator", -1);
    NCVMemStackAllocator cpuAllocator(NCVMemoryTypeHostPinned, cpuCounter.maxSize(), devProp.textureAlignment);
    ncvAssertPrintReturn(cpuAllocator.isInitialized(), "Error creating CPU memory allocator", -1);

    printf("Initialized for frame size [%dx%d]\n", frameSize.width, frameSize.height);

    //==============================================================================
    //
    // Main processing loop
    //
    //==============================================================================

	namedWindow(wndTitle, 1);
    Mat frame, gray, frameDisp;

    do
    {
		// For camera and video file, capture the next image                
        capture >> frame;
        if (frame.empty())
            break;

        Mat gray;
        cvtColor(frame, gray, CV_BGR2GRAY);

        //
        // process
        //

        NcvSize32u minSize = haar.ClassifierSize;
        if (bLargestFace)
        {
            Ncv32u ratioX = preferredVideoFrameSize.width / minSize.width;
            Ncv32u ratioY = preferredVideoFrameSize.height / minSize.height;
            Ncv32u ratioSmallest = std::min(ratioX, ratioY);
            ratioSmallest = std::max((Ncv32u)(ratioSmallest / 2.5f), (Ncv32u)1);
            minSize.width *= ratioSmallest;
            minSize.height *= ratioSmallest;
        }

        Ncv32f avgTime;
        NcvTimer timer = ncvStartTimer();

        if (!bUseOpenCV)
        {
            ncvStat = process(&gray, frameSize.width, frameSize.height,
                              bShowAllHypotheses, bLargestFace, haar,
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
                bShowAllHypotheses && !bLargestFace ? 0 : 4,
                (bLargestFace ? CV_HAAR_FIND_BIGGEST_OBJECT : 0)
                | CV_HAAR_SCALE_IMAGE,
                Size(minSize.width, minSize.height));

            for (size_t rt = 0; rt < rectsOpenCV.size(); ++rt)
                rectangle(gray, rectsOpenCV[rt], Scalar(255));
        }

        avgTime = (Ncv32f)ncvEndQueryTimerMs(timer);

        cvtColor(gray, frameDisp, CV_GRAY2BGR);

        imagePrintf(frameDisp, 0, CV_RGB(255,  0,0), "Space - Switch NCV%s / OpenCV%s", bUseOpenCV?"":" (ON)", bUseOpenCV?" (ON)":"");
        imagePrintf(frameDisp, 1, CV_RGB(255,  0,0), "L - Switch FullSearch%s / LargestFace%s modes", bLargestFace?"":" (ON)", bLargestFace?" (ON)":"");
        imagePrintf(frameDisp, 2, CV_RGB(255,  0,0), "U - Toggle unfiltered hypotheses visualization in FullSearch %s", bShowAllHypotheses?"(ON)":"(OFF)");
        imagePrintf(frameDisp, 3, CV_RGB(118,185,0), "   Running at %f FPS on %s", 1000.0f / avgTime, bUseOpenCV?"CPU":"GPU");

        cv::imshow(wndTitle, frameDisp);

        switch (cvWaitKey(1))
        {
        case ' ':
            bUseOpenCV = !bUseOpenCV;
            break;
        case 'L':
        case 'l':
            bLargestFace = !bLargestFace;
            break;
        case 'U':
        case 'u':
            bShowAllHypotheses = !bShowAllHypotheses;
            break;
        case 27:
            bQuit = true;
            break;
        }

    } while (!bQuit);

    cvDestroyWindow(wndTitle.c_str());

    return 0;
}


#endif
