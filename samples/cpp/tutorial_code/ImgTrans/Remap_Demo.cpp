/**
 * @function Remap_Demo.cpp
 * @brief Demo code for Remap
 * @author Ana Huaman
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;

/// Function Headers
void update_map( int &ind, Mat &map_x, Mat &map_y );

/**
 * @function main
 */
int main(int argc, const char** argv)
{
    CommandLineParser parser(argc, argv, "{@image |../data/chicky_512.png|input image name}");
    std::string filename = parser.get<std::string>(0);
    //! [Load]
    /// Load the image
    Mat src = imread( filename, IMREAD_COLOR );
    if (src.empty())
    {
        std::cout << "Cannot read image: " << filename << std::endl;
        return -1;
    }
    //! [Load]

    //! [Create]
    /// Create dst, map_x and map_y with the same size as src:
    Mat dst(src.size(), src.type());
    Mat map_x(src.size(), CV_32FC1);
    Mat map_y(src.size(), CV_32FC1);
    //! [Create]

    //! [Window]
    /// Create window
    const char* remap_window = "Remap demo";
    namedWindow( remap_window, WINDOW_AUTOSIZE );
    //! [Window]

    //! [Loop]
    /// Index to switch between the remap modes
    int ind = 0;
    for(;;)
    {
        /// Update map_x & map_y. Then apply remap
        update_map(ind, map_x, map_y);
        remap( src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0) );

        /// Display results
        imshow( remap_window, dst );

        /// Each 1 sec. Press ESC to exit the program
        char c = (char)waitKey( 1000 );
        if( c == 27 )
        {
            break;
        }
    }
    //! [Loop]
    return 0;
}

/**
 * @function update_map
 * @brief Fill the map_x and map_y matrices with 4 types of mappings
 */
//! [Update]
void update_map( int &ind, Mat &map_x, Mat &map_y )
{
    for( int i = 0; i < map_x.rows; i++ )
    {
        for( int j = 0; j < map_x.cols; j++ )
        {
            switch( ind )
            {
            case 0:
                if( j > map_x.cols*0.25 && j < map_x.cols*0.75 && i > map_x.rows*0.25 && i < map_x.rows*0.75 )
                {
                    map_x.at<float>(i, j) = 2*( j - map_x.cols*0.25f ) + 0.5f;
                    map_y.at<float>(i, j) = 2*( i - map_x.rows*0.25f ) + 0.5f;
                }
                else
                {
                    map_x.at<float>(i, j) = 0;
                    map_y.at<float>(i, j) = 0;
                }
                break;
            case 1:
                map_x.at<float>(i, j) = (float)j;
                map_y.at<float>(i, j) = (float)(map_x.rows - i);
                break;
            case 2:
                map_x.at<float>(i, j) = (float)(map_x.cols - j);
                map_y.at<float>(i, j) = (float)i;
                break;
            case 3:
                map_x.at<float>(i, j) = (float)(map_x.cols - j);
                map_y.at<float>(i, j) = (float)(map_x.rows - i);
                break;
            default:
                break;
            } // end of switch
        }
    }
    ind = (ind+1) % 4;
}
//! [Update]
