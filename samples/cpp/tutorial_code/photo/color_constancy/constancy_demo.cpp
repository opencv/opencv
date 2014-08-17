#include "opencv2/photo.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

int main(int argc, char *argv[])
{
    Mat src, grey_world, max_rgb, shades_of_grey, first_order_greyEdge, second_order_greyEdge, weighted_greyEdge;

    src = imread(argv[1]);

    greyEdge(src,grey_world,0,1,0);
    greyEdge(src,max_rgb,0,-1,0);
    greyEdge(src,shades_of_grey,0,5,0);
    greyEdge(src,first_order_greyEdge,1,5,2);
    greyEdge(src,second_order_greyEdge,2,5,2);
    weightedGreyEdge(src,weighted_greyEdge);

    imshow("src",src);
    imshow("Grey-World",grey_world);
    imshow("Max-RGB",max_rgb);
    imshow("Shades-of-Grey",shades_of_grey);
    imshow("First-order-GreyEdge",first_order_greyEdge);
    imshow("Second-order-GreyEdge",second_order_greyEdge);
    imshow("Weighted-Grey-Edge",weighted_greyEdge);
    waitKey(0);
}
