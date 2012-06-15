#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// static void help()
// {
//     cout << "\nThis program demonstrates kmeans clustering.\n"
//             "It generates an image with random points, then assigns a random number of cluster\n"
//             "centers and uses kmeans to move those cluster centers to their representitive location\n"
//             "Call\n"
//             "./kmeans\n" << endl;
// }

int main( int /*argc*/, char** /*argv*/ )
{
    const int MAX_CLUSTERS = 5;
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };

    Mat img(500, 500, CV_8UC3);
    RNG rng(12345);

    for(;;)
    {
        int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
        int i, sampleCount = rng.uniform(1, 1001);
        Mat points(sampleCount, 1, CV_32FC2), labels;

        clusterCount = MIN(clusterCount, sampleCount);
        Mat centers(clusterCount, 1, points.type());

        /* generate random sample from multigaussian distribution */
        for( k = 0; k < clusterCount; k++ )
        {
            Point center;
            center.x = rng.uniform(0, img.cols);
            center.y = rng.uniform(0, img.rows);
            Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                             k == clusterCount - 1 ? sampleCount :
                                             (k+1)*sampleCount/clusterCount);
            rng.fill(pointChunk, CV_RAND_NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
        }

        randShuffle(points, 1, &rng);

        kmeans(points, clusterCount, labels,
               TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
               3, KMEANS_PP_CENTERS, centers);

        img = Scalar::all(0);

        for( i = 0; i < sampleCount; i++ )
        {
            int clusterIdx = labels.at<int>(i);
            Point ipt = points.at<Point2f>(i);
            circle( img, ipt, 2, colorTab[clusterIdx], CV_FILLED, CV_AA );
        }

        imshow("clusters", img);

        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }

    return 0;
}
