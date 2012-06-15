/*Authors
* Manuela Chessa, Fabio Solari, Fabio Tatti, Silvio P. Sabatini
*
* manuela.chessa@unige.it, fabio.solari@unige.it
*
* PSPC-lab - University of Genoa
*/

#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

static void help()
{
    cout << "LogPolar Blind Spot Model sample.\nShortcuts:"
        "\n\tn for nearest pixel technique"
        "\n\tb for bilinear interpolation technique"
        "\n\to for overlapping circular receptive fields"
        "\n\ta for adjacent receptive fields"
        "\n\tq or ESC quit\n";
}

int main(int argc, char** argv)
{
    Mat img = imread(argc > 1 ? argv[1] : "lena.jpg",1); // open the image
    if(img.empty()) // check if we succeeded
    {
        cout << "can not load image\n";
        return 0;
    }
    help();

    Size s=img.size();
    int w=s.width, h=s.height;
    int ro0=3; //radius of the blind spot
    int R=120;  //number of rings

    //Creation of the four different objects that implement the four log-polar transformations
    //Off-line computation
    Point2i center(w/2,h/2);
    LogPolar_Interp nearest(w, h, center, R, ro0, INTER_NEAREST);
    LogPolar_Interp bilin(w,h, center,R,ro0);
    LogPolar_Overlapping overlap(w,h,center,R,ro0);
    LogPolar_Adjacent adj(w,h,center,R,ro0,0.25);

    namedWindow("Cartesian",1);
    namedWindow("retinal",1);
    namedWindow("cortical",1);
    int wk='n';
    Mat Cortical, Retinal;

    //On-line computation
    for(;;)
    {
        if(wk=='n'){
            Cortical=nearest.to_cortical(img);
            Retinal=nearest.to_cartesian(Cortical);
        }else if (wk=='b'){
            Cortical=bilin.to_cortical(img);
            Retinal=bilin.to_cartesian(Cortical);
        }else if (wk=='o'){
            Cortical=overlap.to_cortical(img);
            Retinal=overlap.to_cartesian(Cortical);
        }else if (wk=='a'){
            Cortical=adj.to_cortical(img);
            Retinal=adj.to_cartesian(Cortical);
        }

        imshow("Cartesian", img);
        imshow("cortical", Cortical);
        imshow("retinal", Retinal);

        int c=waitKey(15);
        if (c>0) wk=c;
        if(wk =='q' || (wk & 255) == 27) break;
    }

    return 0;
}
