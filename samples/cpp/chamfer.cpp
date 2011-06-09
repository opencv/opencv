#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <stdio.h>

using namespace cv;
using namespace std;

void help()
{
    printf("\nThis program demonstrates Chamfer matching -- computing a distance between an \n"
           "edge template and a query edge image.\n"
           "Usage:\n"
           "./chamfer [<image edge map, logo_in_clutter.png as default>\n"
           "<template edge map, logo.png as default>]\n"
           "Example: \n"
           "    ./chamfer logo_in_clutter.png logo.png\n");
}
int main( int argc, const char** argv )
{
    help();

    CommandLineParser parser(argc, argv);

    string image = parser.get<string>("0","logo_in_clutter.png");
    string tempLate = parser.get<string>("1","logo.png");
    Mat img = imread(image,0);
    Mat cimg;
    cvtColor(img, cimg, CV_GRAY2BGR);
    Mat tpl = imread(tempLate,0);

//    Mat img = imread(argc == 3 ? argv[1] : "logo_in_clutter.png", 0);
//    Mat cimg;
//    cvtColor(img, cimg, CV_GRAY2BGR);
//    Mat tpl = imread(argc == 3 ? argv[2] : "logo.png", 0);
    
//    if( argc != 1 && argc != 3 )
//    {
//        help();
//        return 0;
//    }

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
        printf("not found;\n");
        return 0;
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
