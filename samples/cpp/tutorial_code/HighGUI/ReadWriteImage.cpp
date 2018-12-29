/**
 * @file ReadWriteImage.cpp
 * @brief Simple program to read and write image
 * @date Fr, December 28, 2018
 * @author Baum55
 */

#include "opencv2/imgcodecs.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/**
 * @function main
 * @brief Main function
 */
int main()
{
    Mat image = imread( "../data/butterfly.jpg" );
    if( image.empty() )
    {
        std::cout << "Error loading image 1 \n";
    }
    else
    {
        imwrite("butterfly.jpg", image);
    }

    image = imreadW( L"../data/butterfly.jpg" );
    if( image.empty() )
    {
        std::cout << "Error loading image 2 \n";
    }
    else
    {
        imwriteW( L"butterflyW.jpg", image );
    }


    std::vector< Mat > multi_image;
    imreadmulti( "../data/multipage_OSLogo.tif", multi_image );
    if( multi_image.empty() )
    {
        std::cout << "Error loading multi page image 1 \n";
    }
    else
    {
        imwrite( "OSLogo.tif", multi_image );
    }

    multi_image.clear();
    imreadmultiW( L"../data/multipage_OSLogo.tif", multi_image );
    if( multi_image.empty() )
    {
        std::cout << "Error loading multi page image 1 \n";
    }
    else
    {
        imwriteW( L"OSLogoW.tif", multi_image );
    }
}
