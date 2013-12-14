// This sample shows the difference of adaptive bilateral filter and bilateral filter.
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"

using namespace cv;
using namespace std;


int main( int argc, const char** argv )
{
    const char* keys =
        "{ i | input   |          | specify input image }"
        "{ k | ksize   |     11   | specify kernel size }"
        "{ s | sSpace  |     3    | specify sigma space }"
        "{ c | sColor  |     30   | specify max color }"
        "{ h | help    | false    | print help message }";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.get<bool>("help"))
    {
        cout << "Usage : adaptive_bilateral_filter [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printParams();
        return EXIT_SUCCESS;
    }

    string src_path = cmd.get<string>("i");
    int ks = cmd.get<int>("k");
    const char * winName[] = {"input", "ABF OpenCL", "BF OpenCL"};

    Mat src = imread(src_path);
    if (src.empty())
    {
        cout << "error read image: " << src_path << endl;
        return EXIT_FAILURE;
    }

    double sigmaSpace = cmd.get<int>("s");

    // sigma for checking pixel values. This is used as is in the "normal" bilateral filter,
    // and it is used as an upper clamp on the adaptive case.
    double sigmacolor = cmd.get<int>("c");

    ocl::oclMat dsrc(src), dABFilter, dBFilter;
    Size ksize(ks, ks);

    // ksize is the total width/height of neighborhood used to calculate local variance.
    // sigmaSpace is not a priori related to ksize/2.
    ocl::adaptiveBilateralFilter(dsrc, dABFilter, ksize, sigmaSpace, sigmacolor);
    ocl::bilateralFilter(dsrc, dBFilter, ks, sigmacolor, sigmaSpace);
    Mat abFilter = dABFilter, bFilter = dBFilter;

    ocl::finish();

    imshow(winName[0], src);
    imshow(winName[1], abFilter);
    imshow(winName[2], bFilter);

    waitKey();

    return EXIT_SUCCESS;
}
