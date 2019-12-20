#include <iostream>
#include <stdexcept>

//OpenVX includes
#include <VX/vx.h>

//OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#ifndef VX_VERSION_1_1
const vx_enum VX_IMAGE_FORMAT = VX_IMAGE_ATTRIBUTE_FORMAT;
const vx_enum VX_IMAGE_WIDTH  = VX_IMAGE_ATTRIBUTE_WIDTH;
const vx_enum VX_IMAGE_HEIGHT = VX_IMAGE_ATTRIBUTE_HEIGHT;
const vx_enum VX_MEMORY_TYPE_HOST = VX_IMPORT_TYPE_HOST;
const vx_enum VX_MEMORY_TYPE_NONE = VX_IMPORT_TYPE_NONE;
const vx_enum VX_THRESHOLD_THRESHOLD_VALUE = VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE;
const vx_enum VX_THRESHOLD_THRESHOLD_LOWER = VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER;
const vx_enum VX_THRESHOLD_THRESHOLD_UPPER = VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER;
typedef uintptr_t vx_map_id;
#endif

enum UserMemoryMode
{
    COPY, USER_MEM
};

vx_image convertCvMatToVxImage(vx_context context, cv::Mat image, bool toCopy);
cv::Mat copyVxImageToCvMat(vx_image ovxImage);
void swapVxImage(vx_image ovxImage);
vx_status createProcessingGraph(vx_image inputImage, vx_image outputImage, vx_graph& graph);
int ovxDemo(std::string inputPath, UserMemoryMode mode);


vx_image convertCvMatToVxImage(vx_context context, cv::Mat image, bool toCopy)
{
    if (!(!image.empty() && image.dims <= 2 && image.channels() == 1))
        throw std::runtime_error("Invalid format");

    vx_uint32 width  = image.cols;
    vx_uint32 height = image.rows;

    vx_df_image color;
    switch (image.depth())
    {
    case CV_8U:
        color = VX_DF_IMAGE_U8;
        break;
    case CV_16U:
        color = VX_DF_IMAGE_U16;
        break;
    case CV_16S:
        color = VX_DF_IMAGE_S16;
        break;
    case CV_32S:
        color = VX_DF_IMAGE_S32;
        break;
    default:
        throw std::runtime_error("Invalid format");
        break;
    }

    vx_imagepatch_addressing_t addr;
    addr.dim_x = width;
    addr.dim_y = height;
    addr.stride_x = (vx_uint32)image.elemSize();
    addr.stride_y = (vx_uint32)image.step.p[0];
    vx_uint8* ovxData = image.data;

    vx_image ovxImage;
    if (toCopy)
    {
        ovxImage = vxCreateImage(context, width, height, color);
        if (vxGetStatus((vx_reference)ovxImage) != VX_SUCCESS)
            throw std::runtime_error("Failed to create image");
        vx_rectangle_t rect;

        vx_status status = vxGetValidRegionImage(ovxImage, &rect);
        if (status != VX_SUCCESS)
            throw std::runtime_error("Failed to get valid region");

#ifdef VX_VERSION_1_1
        status = vxCopyImagePatch(ovxImage, &rect, 0, &addr, ovxData, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VX_SUCCESS)
            throw std::runtime_error("Failed to copy image patch");
#else
        status = vxAccessImagePatch(ovxImage, &rect, 0, &addr, (void**)&ovxData, VX_WRITE_ONLY);
        if (status != VX_SUCCESS)
            throw std::runtime_error("Failed to access image patch");
        status = vxCommitImagePatch(ovxImage, &rect, 0, &addr, ovxData);
        if (status != VX_SUCCESS)
            throw std::runtime_error("Failed to commit image patch");
#endif
    }
    else
    {
        ovxImage = vxCreateImageFromHandle(context, color, &addr, (void**)&ovxData, VX_MEMORY_TYPE_HOST);
        if (vxGetStatus((vx_reference)ovxImage) != VX_SUCCESS)
            throw std::runtime_error("Failed to create image from handle");
    }

    return ovxImage;
}


cv::Mat copyVxImageToCvMat(vx_image ovxImage)
{
    vx_status status;
    vx_df_image df_image = 0;
    vx_uint32 width, height;
    status = vxQueryImage(ovxImage, VX_IMAGE_FORMAT, &df_image, sizeof(vx_df_image));
    if (status != VX_SUCCESS)
        throw std::runtime_error("Failed to query image");
    status = vxQueryImage(ovxImage, VX_IMAGE_WIDTH, &width, sizeof(vx_uint32));
    if (status != VX_SUCCESS)
        throw std::runtime_error("Failed to query image");
    status = vxQueryImage(ovxImage, VX_IMAGE_HEIGHT, &height, sizeof(vx_uint32));
    if (status != VX_SUCCESS)
        throw std::runtime_error("Failed to query image");

    if (!(width > 0 && height > 0)) throw std::runtime_error("Invalid format");

    int depth;
    switch (df_image)
    {
    case VX_DF_IMAGE_U8:
        depth = CV_8U;
        break;
    case VX_DF_IMAGE_U16:
        depth = CV_16U;
        break;
    case VX_DF_IMAGE_S16:
        depth = CV_16S;
        break;
    case VX_DF_IMAGE_S32:
        depth = CV_32S;
        break;
    default:
        throw std::runtime_error("Invalid format");
        break;
    }

    cv::Mat image(height, width, CV_MAKE_TYPE(depth, 1));

    vx_rectangle_t rect;
    rect.start_x = rect.start_y = 0;
    rect.end_x = width; rect.end_y = height;

    vx_imagepatch_addressing_t addr;
    addr.dim_x = width;
    addr.dim_y = height;
    addr.stride_x = (vx_uint32)image.elemSize();
    addr.stride_y = (vx_uint32)image.step.p[0];
    vx_uint8* matData = image.data;

#ifdef VX_VERSION_1_1
    status = vxCopyImagePatch(ovxImage, &rect, 0, &addr, matData, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if (status != VX_SUCCESS)
        throw std::runtime_error("Failed to copy image patch");
#else
    status = vxAccessImagePatch(ovxImage, &rect, 0, &addr, (void**)&matData, VX_READ_ONLY);
    if (status != VX_SUCCESS)
        throw std::runtime_error("Failed to access image patch");
    status = vxCommitImagePatch(ovxImage, &rect, 0, &addr, matData);
    if (status != VX_SUCCESS)
        throw std::runtime_error("Failed to commit image patch");
#endif

    return image;
}


void swapVxImage(vx_image ovxImage)
{
#ifdef VX_VERSION_1_1
    vx_status status;
    vx_memory_type_e memType;
    status = vxQueryImage(ovxImage, VX_IMAGE_MEMORY_TYPE, &memType, sizeof(vx_memory_type_e));
    if (status != VX_SUCCESS)
        throw std::runtime_error("Failed to query image");
    if (memType == VX_MEMORY_TYPE_NONE)
    {
        //was created by copying user data
        throw std::runtime_error("Image wasn't created from user handle");
    }
    else
    {
        //was created from user handle
        status = vxSwapImageHandle(ovxImage, NULL, NULL, 0);
        if (status != VX_SUCCESS)
            throw std::runtime_error("Failed to swap image handle");
    }
#else
    //not supported until OpenVX 1.1
    (void) ovxImage;
#endif
}


vx_status createProcessingGraph(vx_image inputImage, vx_image outputImage, vx_graph& graph)
{
    vx_status status;
    vx_context context = vxGetContext((vx_reference)inputImage);
    status = vxGetStatus((vx_reference)context);
    if(status != VX_SUCCESS) return status;

    graph = vxCreateGraph(context);
    status = vxGetStatus((vx_reference)graph);
    if (status != VX_SUCCESS) return status;

    vx_uint32 width, height;
    status = vxQueryImage(inputImage, VX_IMAGE_WIDTH, &width, sizeof(vx_uint32));
    if (status != VX_SUCCESS) return status;
    status = vxQueryImage(inputImage, VX_IMAGE_HEIGHT, &height, sizeof(vx_uint32));
    if (status != VX_SUCCESS) return status;

    // Intermediate images
    vx_image
        smoothed  = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT),
        cannied   = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT),
        halfImg   = vxCreateImage(context, width, height, VX_DF_IMAGE_U8),
        halfCanny = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);

    vx_image virtualImages[] = {smoothed, cannied, halfImg, halfCanny};
    for(size_t i = 0; i < sizeof(virtualImages)/sizeof(vx_image); i++)
    {
        status = vxGetStatus((vx_reference)virtualImages[i]);
        if (status != VX_SUCCESS) return status;
    }

    // Constants
    vx_uint32 threshValue = 50;
    vx_threshold thresh = vxCreateThreshold(context, VX_THRESHOLD_TYPE_BINARY, VX_TYPE_UINT8);
    vxSetThresholdAttribute(thresh, VX_THRESHOLD_THRESHOLD_VALUE,
                            &threshValue, sizeof(threshValue));

    vx_uint32 threshCannyMin = 127;
    vx_uint32 threshCannyMax = 192;
    vx_threshold threshCanny = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
    vxSetThresholdAttribute(threshCanny, VX_THRESHOLD_THRESHOLD_LOWER, &threshCannyMin,
                            sizeof(threshCannyMin));
    vxSetThresholdAttribute(threshCanny, VX_THRESHOLD_THRESHOLD_UPPER, &threshCannyMax,
                            sizeof(threshCannyMax));
    vx_float32 alphaValue = 0.5;
    vx_scalar alpha = vxCreateScalar(context, VX_TYPE_FLOAT32, &alphaValue);

    // Sequence of meaningless image operations
    vx_node nodes[] = {
        vxGaussian3x3Node(graph, inputImage, smoothed),
        vxCannyEdgeDetectorNode(graph, smoothed, threshCanny, 3, VX_NORM_L2, cannied),
        vxAccumulateWeightedImageNode(graph, inputImage, alpha, halfImg),
        vxAccumulateWeightedImageNode(graph, cannied, alpha, halfCanny),
        vxAddNode(graph, halfImg, halfCanny, VX_CONVERT_POLICY_SATURATE, outputImage)
    };

    for (size_t i = 0; i < sizeof(nodes) / sizeof(vx_node); i++)
    {
        status = vxGetStatus((vx_reference)nodes[i]);
        if (status != VX_SUCCESS) return status;
    }

    status = vxVerifyGraph(graph);
    return status;
}


int ovxDemo(std::string inputPath, UserMemoryMode mode)
{
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) return -1;

    //check image format
    if (image.depth() != CV_8U || image.channels() != 1) return -1;

    vx_status status;
    vx_context context = vxCreateContext();
    status = vxGetStatus((vx_reference)context);
    if (status != VX_SUCCESS) return status;

    //put user data from cv::Mat to vx_image
    vx_image ovxImage;
    ovxImage = convertCvMatToVxImage(context, image, mode == COPY);

    vx_uint32 width = image.cols, height = image.rows;

    vx_image ovxResult;
    cv::Mat output;
    if (mode == COPY)
    {
        //we will copy data from vx_image to cv::Mat
        ovxResult = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
        if (vxGetStatus((vx_reference)ovxResult) != VX_SUCCESS)
            throw std::runtime_error("Failed to create image");
    }
    else
    {
        //create vx_image based on user data, no copying required
        output = cv::Mat(height, width, CV_8U, cv::Scalar(0));
        ovxResult = convertCvMatToVxImage(context, output, false);
    }

    vx_graph graph;
    status = createProcessingGraph(ovxImage, ovxResult, graph);
    if (status != VX_SUCCESS) return status;

    // Graph execution
    status = vxProcessGraph(graph);
    if (status != VX_SUCCESS) return status;

    //getting resulting image in cv::Mat
    if (mode == COPY)
    {
        output = copyVxImageToCvMat(ovxResult);
    }
    else
    {
        //we should take user memory back from vx_image before using it (even before reading)
        swapVxImage(ovxResult);
    }

    //here output goes
    cv::imshow("processing result", output);
    cv::waitKey(0);

    //we need to take user memory back before releasing the image
    if (mode == USER_MEM)
        swapVxImage(ovxImage);

    cv::destroyAllWindows();

    status = vxReleaseContext(&context);
    return status;
}


int main(int argc, char *argv[])
{
    const std::string keys =
        "{help h usage ? | | }"
        "{image    | <none> | image to be processed}"
        "{mode | copy | user memory interaction mode: \n"
        "copy: create VX images and copy data to/from them\n"
        "user_mem: use handles to user-allocated memory}"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("OpenVX interoperability sample demonstrating standard OpenVX API."
                 "The application loads an image, processes it with OpenVX graph and outputs result in a window");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    std::string imgPath = parser.get<std::string>("image");
    std::string modeString = parser.get<std::string>("mode");
    UserMemoryMode mode;
    if(modeString == "copy")
    {
        mode = COPY;
    }
    else if(modeString == "user_mem")
    {
        mode = USER_MEM;
    }
    else if(modeString == "map")
    {
        std::cerr << modeString << " is not implemented in this sample" << std::endl;
        return -1;
    }
    else
    {
        std::cerr << modeString << ": unknown memory mode" << std::endl;
        return -1;
    }

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    return ovxDemo(imgPath, mode);
}
