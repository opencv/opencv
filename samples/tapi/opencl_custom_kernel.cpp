// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

static const char* opencl_kernel_src =
"__kernel void magnutude_filter_8u(\n"
"       __global const uchar* src, int src_step, int src_offset,\n"
"       __global uchar* dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,\n"
"       float scale)\n"
"{\n"
"   int x = get_global_id(0);\n"
"   int y = get_global_id(1);\n"
"   if (x < dst_cols && y < dst_rows)\n"
"   {\n"
"       int dst_idx = y * dst_step + x + dst_offset;\n"
"       if (x > 0 && x < dst_cols - 1 && y > 0 && y < dst_rows - 2)\n"
"       {\n"
"           int src_idx = y * src_step + x + src_offset;\n"
"           int dx = (int)src[src_idx]*2 - src[src_idx - 1]          - src[src_idx + 1];\n"
"           int dy = (int)src[src_idx]*2 - src[src_idx - 1*src_step] - src[src_idx + 1*src_step];\n"
"           dst[dst_idx] = convert_uchar_sat(sqrt((float)(dx*dx + dy*dy)) * scale);\n"
"       }\n"
"       else\n"
"       {\n"
"           dst[dst_idx] = 0;\n"
"       }\n"
"   }\n"
"}\n";

int main(int argc, char** argv)
{
    const char* keys =
        "{ i input    | | specify input image }"
        "{ h help     | | print help message }";

    cv::CommandLineParser args(argc, argv, keys);
    if (args.has("help"))
    {
        cout << "Usage : " << argv[0] << " [options]" << endl;
        cout << "Available options:" << endl;
        args.printMessage();
        return EXIT_SUCCESS;
    }

    cv::ocl::Context ctx = cv::ocl::Context::getDefault();
    if (!ctx.ptr())
    {
        cerr << "OpenCL is not available" << endl;
        return 1;
    }
    cv::ocl::Device device = cv::ocl::Device::getDefault();
    if (!device.compilerAvailable())
    {
        cerr << "OpenCL compiler is not available" << endl;
        return 1;
    }


    UMat src;
    {
        string image_file = args.get<string>("i");
        if (!image_file.empty())
        {
            Mat image = imread(samples::findFile(image_file));
            if (image.empty())
            {
                cout << "error read image: " << image_file << endl;
                return 1;
            }
            cvtColor(image, src, COLOR_BGR2GRAY);
        }
        else
        {
            Mat frame(cv::Size(640, 480), CV_8U, Scalar::all(128));
            Point p(frame.cols / 2, frame.rows / 2);
            line(frame, Point(0, frame.rows / 2), Point(frame.cols, frame.rows / 2), 1);
            circle(frame, p, 200, Scalar(32, 32, 32), 8, LINE_AA);
            string str = "OpenCL";
            int baseLine = 0;
            Size box = getTextSize(str, FONT_HERSHEY_COMPLEX, 2, 5, &baseLine);
            putText(frame, str, Point((frame.cols - box.width) / 2, (frame.rows - box.height) / 2 + baseLine),
                    FONT_HERSHEY_COMPLEX, 2, Scalar(255, 255, 255), 5, LINE_AA);
            frame.copyTo(src);
        }
    }


    cv::String module_name; // empty to disable OpenCL cache

    {
        cout << "OpenCL program source: " << endl;
        cout << "======================================================================================================" << endl;
        cout << opencl_kernel_src << endl;
        cout << "======================================================================================================" << endl;
        //! [Define OpenCL program source]
        cv::ocl::ProgramSource source(module_name, "simple", opencl_kernel_src, "");
        //! [Define OpenCL program source]

        //! [Compile/build OpenCL for current OpenCL device]
        cv::String errmsg;
        cv::ocl::Program program(source, "", errmsg);
        if (program.ptr() == NULL)
        {
            cerr << "Can't compile OpenCL program:" << endl << errmsg << endl;
            return 1;
        }
        //! [Compile/build OpenCL for current OpenCL device]

        if (!errmsg.empty())
        {
            cout << "OpenCL program build log:" << endl << errmsg << endl;
        }

        //! [Get OpenCL kernel by name]
        cv::ocl::Kernel k("magnutude_filter_8u", program);
        if (k.empty())
        {
            cerr << "Can't get OpenCL kernel" << endl;
            return 1;
        }
        //! [Get OpenCL kernel by name]

        UMat result(src.size(), CV_8UC1);

        //! [Define kernel parameters and run]
        size_t globalSize[2] = {(size_t)src.cols, (size_t)src.rows};
        size_t localSize[2] = {8, 8};
        bool executionResult = k
            .args(
                cv::ocl::KernelArg::ReadOnlyNoSize(src), // size is not used (similar to 'dst' size)
                cv::ocl::KernelArg::WriteOnly(result),
                (float)2.0
            )
            .run(2, globalSize, localSize, true);
        if (!executionResult)
        {
            cerr << "OpenCL kernel launch failed" << endl;
            return 1;
        }
        //! [Define kernel parameters and run]

        imshow("Source", src);
        imshow("Result", result);

        for (;;)
        {
            int key = waitKey();
            if (key == 27/*ESC*/ || key == 'q' || key == 'Q')
                break;
        }
    }
    return 0;
}
