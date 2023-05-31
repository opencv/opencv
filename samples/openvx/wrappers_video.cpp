#include <iostream>
#include <stdexcept>

//wrappers
#include "ivx.hpp"

//OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

enum UserMemoryMode
{
    COPY, USER_MEM, MAP
};

ivx::Graph createProcessingGraph(ivx::Image& inputImage, ivx::Image& outputImage);
int ovxDemo(std::string inputPath, UserMemoryMode mode);


ivx::Graph createProcessingGraph(ivx::Image& inputImage, ivx::Image& outputImage)
{
    using namespace ivx;

    Context context = inputImage.get<Context>();
    Graph graph = Graph::create(context);

    vx_uint32 width  = inputImage.width();
    vx_uint32 height = inputImage.height();

    // Intermediate images
    Image
        yuv       = Image::createVirtual(graph, 0, 0, VX_DF_IMAGE_YUV4),
        gray      = Image::createVirtual(graph),
        smoothed  = Image::createVirtual(graph),
        cannied   = Image::createVirtual(graph),
        halfImg   = Image::create(context, width, height, VX_DF_IMAGE_U8),
        halfCanny = Image::create(context, width, height, VX_DF_IMAGE_U8);

    // Constants
    vx_uint32 threshCannyMin = 127;
    vx_uint32 threshCannyMax = 192;
    Threshold threshCanny = Threshold::createRange(context, VX_TYPE_UINT8, threshCannyMin, threshCannyMax);

    ivx::Scalar alpha = ivx::Scalar::create<VX_TYPE_FLOAT32>(context, 0.5);

    // Sequence of some image operations
    Node::create(graph, VX_KERNEL_COLOR_CONVERT, inputImage, yuv);
    Node::create(graph, VX_KERNEL_CHANNEL_EXTRACT, yuv,
                 ivx::Scalar::create<VX_TYPE_ENUM>(context, VX_CHANNEL_Y), gray);
    //node can also be added in function-like style
    nodes::gaussian3x3(graph, gray, smoothed);
    Node::create(graph, VX_KERNEL_CANNY_EDGE_DETECTOR, smoothed, threshCanny,
                 ivx::Scalar::create<VX_TYPE_INT32>(context, 3),
                 ivx::Scalar::create<VX_TYPE_ENUM>(context, VX_NORM_L2), cannied);
    Node::create(graph, VX_KERNEL_ACCUMULATE_WEIGHTED, gray, alpha, halfImg);
    Node::create(graph, VX_KERNEL_ACCUMULATE_WEIGHTED, cannied, alpha, halfCanny);
    Node::create(graph, VX_KERNEL_ADD, halfImg, halfCanny,
                 ivx::Scalar::create<VX_TYPE_ENUM>(context, VX_CONVERT_POLICY_SATURATE), outputImage);

    graph.verify();

    return graph;
}


int ovxDemo(std::string inputPath, UserMemoryMode mode)
{
    using namespace cv;
    using namespace ivx;

    Mat frame;
    VideoCapture vc(inputPath);
    if (!vc.isOpened())
        return -1;

    vc >> frame;
    if (frame.empty()) return -1;

    //check frame format
    if (frame.type() != CV_8UC3) return -1;

    try
    {
        Context context = Context::create();
        //put user data from cv::Mat to vx_image
        vx_df_image color = Image::matTypeToFormat(frame.type());
        vx_uint32 width = frame.cols, height = frame.rows;
        Image ivxImage;
        if (mode == COPY)
        {
            ivxImage = Image::create(context, width, height, color);
        }
        else
        {
            ivxImage = Image::createFromHandle(context, color, Image::createAddressing(frame), frame.data);
        }

        Image ivxResult;

        Mat output;
        if (mode == COPY || mode == MAP)
        {
            //we will copy or map data from vx_image to cv::Mat
            ivxResult = ivx::Image::create(context, width, height, VX_DF_IMAGE_U8);
        }
        else // if (mode == MAP_TO_VX)
        {
            //create vx_image based on user data, no copying required
            output = cv::Mat(height, width, CV_8U, cv::Scalar(0));
            ivxResult = Image::createFromHandle(context, Image::matTypeToFormat(CV_8U),
                                                Image::createAddressing(output), output.data);
        }

        Graph graph = createProcessingGraph(ivxImage, ivxResult);

        bool stop = false;
        while (!stop)
        {
            if (mode == COPY) ivxImage.copyFrom(0, frame);

            // Graph execution
            graph.process();

            //getting resulting image in cv::Mat
            Image::Patch resultPatch;
            std::vector<void*> ptrs;
            std::vector<void*> prevPtrs(ivxResult.planes());
            if (mode == COPY)
            {
                ivxResult.copyTo(0, output);
            }
            else if (mode == MAP)
            {
                //create cv::Mat based on vx_image mapped data
                resultPatch.map(ivxResult, 0, ivxResult.getValidRegion(), VX_READ_AND_WRITE);
                //generally this is very bad idea!
                //but in our case unmap() won't happen until output is in use
                output = resultPatch.getMat();
            }
            else // if(mode == MAP_TO_VX)
            {
#ifdef VX_VERSION_1_1
                //we should take user memory back from vx_image before using it (even before reading)
                ivxResult.swapHandle(ptrs, prevPtrs);
#endif
            }

            //here output goes
            imshow("press q to quit", output);
            if ((char)waitKey(1) == 'q') stop = true;

#ifdef VX_VERSION_1_1
            //restore handle
            if (mode == USER_MEM)
            {
                ivxResult.swapHandle(prevPtrs, ptrs);
            }
#endif

            //this line is unnecessary since unmapping is done on destruction of patch
            //resultPatch.unmap();

            //grab next frame
            Mat temp = frame;
            vc >> frame;
            if (frame.empty()) stop = true;
            if (mode != COPY && frame.data != temp.data)
            {
                //frame was reallocated, pointer to data changed
                frame.copyTo(temp);
            }
        }

        destroyAllWindows();

#ifdef VX_VERSION_1_1
        if (mode != COPY)
        {
            //we should take user memory back before release
            //(it's not done automatically according to standard)
            ivxImage.swapHandle();
            if (mode == USER_MEM) ivxResult.swapHandle();
        }
#endif
    }
    catch (const ivx::RuntimeError& e)
    {
        std::cerr << "Error: code = " << e.status() << ", message = " << e.what() << std::endl;
        return e.status();
    }
    catch (const ivx::WrapperError& e)
    {
        std::cerr << "Error: message = " << e.what() << std::endl;
        return -1;
    }

    return 0;
}


int main(int argc, char *argv[])
{
    const std::string keys =
        "{help h usage ? | | }"
        "{video    | <none> | video file to be processed}"
        "{mode | copy | user memory interaction mode: \n"
        "copy: create VX images and copy data to/from them\n"
        "user_mem: use handles to user-allocated memory\n"
        "map: map resulting VX image to user memory}"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("OpenVX interoperability sample demonstrating OpenVX wrappers usage."
                 "The application opens a video and processes it with OpenVX graph while outputting result in a window");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    std::string videoPath = parser.get<std::string>("video");
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
        mode = MAP;
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

    return ovxDemo(videoPath, mode);
}
