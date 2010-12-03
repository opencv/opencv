#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iostream>

using namespace cv;
using namespace std;

void help()
{
	cout <<
			"\nThis program demonstrates chamfer matching -- computing a distance between an \n"
			"edge template and a query edge image.\n"
			"Call:\n"
			"./chamfermatching [<image edge map> <template edge map>]\n"
			"By default\n"
			"the inputs are ./chamfermatching logo_in_clutter.png logo.png\n"<< endl;
}
int main( int argc, char** argv )
{
    if( argc != 1 && argc != 3 )
    {
        help();
        return 0;
    }
    Mat img = imread(argc == 3 ? argv[1] : "logo_in_clutter.png", 0);
    Mat cimg;
    cvtColor(img, cimg, CV_GRAY2BGR);
    Mat tpl = imread(argc == 3 ? argv[2] : "logo.png", 0);
    
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
        cout << "not found;\n";
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
