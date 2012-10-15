#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help()
{

   cout << "\nThis program demonstrates Chamfer matching -- computing a distance between an \n"
            "edge template and a query edge image.\n"
            "Usage: \n"
            "./chamfer <image edge map> <template edge map>,"
            " By default the inputs are logo_in_clutter.png logo.png\n";
}

const char* keys =
{
    "{1| |logo_in_clutter.png|image edge map    }"
    "{2| |logo.png               |template edge map}"
};

int main( int argc, const char** argv )
{

    help();
    CommandLineParser parser(argc, argv, keys);

    string image = parser.get<string>("1");
    string templ = parser.get<string>("2");
    Mat img = imread(image.c_str(), 0);
    Mat tpl = imread(templ.c_str(), 0);

    if (img.empty() || tpl.empty())
    {
        cout << "Could not read image file " << image << " or " << templ << "." << endl;
        return -1;
    }
    Mat cimg;
    cvtColor(img, cimg, CV_GRAY2BGR);

    // if the image and the template are not edge maps but normal grayscale images,
    // you might want to uncomment the lines below to produce the maps. You can also
    // run Sobel instead of Canny.

    // Canny(img, img, 5, 50, 3);
    // Canny(tpl, tpl, 5, 50, 3);

    vector<vector<Point> > results;
    vector<float> costs;
    int best = chamerMatching( img, tpl, results, costs );
    if( best < 0 )
    {
        cout << "matching not found" << endl;
        return -1;
    }

    size_t i, n = results[best].size();
    for( i = 0; i < n; i++ )
    {
        Point pt = results[best][i];
        if( pt.inside(Rect(0, 0, cimg.cols, cimg.rows)) )
           cimg.at<Vec3b>(pt) = Vec3b(0, 255, 0);
    }

    imshow("result", cimg);

    waitKey();

    return 0;
}
