////////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
////////////////////////////////////////////////////////////////////////////////////////

/*****************************************************************************************************
USAGE:
./opencv_annotation -images <folder location> -annotations <ouput file>

Created by: Puttemans Steven - February 2015
*****************************************************************************************************/

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>

#if defined(_WIN32)
   #include <direct.h>
#else
   #include <sys/stat.h>
#endif

using namespace std;
using namespace cv;

// Function prototypes
void on_mouse(int, int, int, int, void*);
string int2string(int);
void get_annotations(Mat, stringstream*);

// Public parameters
Mat image;
int roi_x0 = 0, roi_y0 = 0, roi_x1 = 0, roi_y1 = 0, num_of_rec = 0;
bool start_draw = false;

// Window name for visualisation purposes
const string window_name="OpenCV Based Annotation Tool";

// FUNCTION : Mouse response for selecting objects in images
// If left button is clicked, start drawing a rectangle as long as mouse moves
// Stop drawing once a new left click is detected by the on_mouse function
void on_mouse(int event, int x, int y, int , void * )
{
    // Action when left button is clicked
    if(event == EVENT_LBUTTONDOWN)
    {
        if(!start_draw)
        {
            roi_x0 = x;
            roi_y0 = y;
            start_draw = true;
        } else {
            roi_x1 = x;
            roi_y1 = y;
            start_draw = false;
        }
    }
    // Action when mouse is moving
    if((event == EVENT_MOUSEMOVE) && start_draw)
    {
        // Redraw bounding box for annotation
        Mat current_view;
        image.copyTo(current_view);
        rectangle(current_view, Point(roi_x0,roi_y0), Point(x,y), Scalar(0,0,255));
        imshow(window_name, current_view);
    }
}

// FUNCTION : snippet to convert an integer value to a string using a clean function
// instead of creating a stringstream each time inside the main code
string int2string(int num)
{
    stringstream temp_stream;
    temp_stream << num;
    return temp_stream.str();
}

// FUNCTION : given an image containing positive object instances, add all the object
// annotations to a known stringstream
void get_annotations(Mat input_image, stringstream* output_stream)
{
    // Make it possible to exit the annotation
    bool stop = false;

    // Reset the num_of_rec element at each iteration
    // Make sure the global image is set to the current image
    num_of_rec = 0;
    image = input_image;

    // Init window interface and couple mouse actions
    namedWindow(window_name, WINDOW_AUTOSIZE);
    setMouseCallback(window_name, on_mouse);

    imshow(window_name, image);
    stringstream temp_stream;
    int key_pressed = 0;

    do
    {
        // Keys for processing
        // You need to select one for confirming a selection and one to continue to the next image
        // Based on the universal ASCII code of the keystroke: http://www.asciitable.com/
        //      c = 99		    add rectangle to current image
        //	    n = 110		    save added rectangles and show next image
        //	    <ESC> = 27      exit program
        key_pressed = 0xFF & waitKey(0);
        switch( key_pressed )
        {
        case 27:
                destroyWindow(window_name);
                stop = true;
        case 99:
                // Add a rectangle to the list
                num_of_rec++;
                // Draw initiated from top left corner
                if(roi_x0<roi_x1 && roi_y0<roi_y1)
                {
                    temp_stream << " " << int2string(roi_x0) << " " << int2string(roi_y0) << " " << int2string(roi_x1-roi_x0) << " " << int2string(roi_y1-roi_y0);
                }
                // Draw initiated from bottom right corner
                if(roi_x0>roi_x1 && roi_y0>roi_y1)
                {
                    temp_stream << " " << int2string(roi_x1) << " " << int2string(roi_y1) << " " << int2string(roi_x0-roi_x1) << " " << int2string(roi_y0-roi_y1);
                }
                // Draw initiated from top right corner
                if(roi_x0>roi_x1 && roi_y0<roi_y1)
                {
                    temp_stream << " " << int2string(roi_x1) << " " << int2string(roi_y0) << " " << int2string(roi_x0-roi_x1) << " " << int2string(roi_y1-roi_y0);
                }
                // Draw initiated from bottom left corner
                if(roi_x0<roi_x1 && roi_y0>roi_y1)
                {
                    temp_stream << " " << int2string(roi_x0) << " " << int2string(roi_y1) << " " << int2string(roi_x1-roi_x0) << " " << int2string(roi_y0-roi_y1);
                }

                rectangle(input_image, Point(roi_x0,roi_y0), Point(roi_x1,roi_y1), Scalar(0,255,0), 1);

                break;
        }

        // Check if escape has been pressed
        if(stop)
        {
            break;
        }
    }
    // Continue as long as the next image key has not been pressed
    while(key_pressed != 110);

    // If there are annotations AND the next image key is pressed
    // Write the image annotations to the file
    if(num_of_rec>0 && key_pressed==110)
    {
        *output_stream << " " << num_of_rec << temp_stream.str() << endl;
    }

    // Close down the window
    destroyWindow(window_name);
}

int main( int argc, const char** argv )
{
    // If no arguments are given, then supply some information on how this tool works
    if( argc == 1 ){
        cout << "Usage: " << argv[0] << endl;
        cout << " -images <folder_location> [example - /data/testimages/]" << endl;
        cout << " -annotations <ouput_file> [example - /data/annotations.txt]" << endl;

        return -1;
    }

    // Read in the input arguments
    string image_folder;
    string annotations;
    for(int i = 1; i < argc; ++i )
    {
        if( !strcmp( argv[i], "-images" ) )
        {
            image_folder = argv[++i];
        }
        else if( !strcmp( argv[i], "-annotations" ) )
        {
            annotations = argv[++i];
        }
    }

    // Check if the folder actually exists
    // If -1 is returned then the folder actually exists, and thus you can continue
    // In all other cases there was a folder creation and thus the folder did not exist
    #if defined(_WIN32)
    if(_mkdir(image_folder.c_str()) != -1){
        // Generate an error message
        cerr << "The image folder given does not exist. Please check again!" << endl;
        // Remove the created folder again, to ensure a second run with same code fails again
        _rmdir(image_folder.c_str());
        return 0;
    }
    #else
    if(mkdir(image_folder.c_str(), 0777) != -1){
        // Generate an error message
        cerr << "The image folder given does not exist. Please check again!" << endl;
        // Remove the created folder again, to ensure a second run with same code fails again
        remove(image_folder.c_str());
        return 0;
    }
    #endif

    // Create the outputfilestream
    ofstream output(annotations.c_str());
    if ( !output.is_open() ){
        cerr << "The path for the output file contains an error and could not be opened. Please check again!" << endl;
        return 0;
    }

    // Return the image filenames inside the image folder
    vector<String> filenames;
    String folder(image_folder);
    glob(folder, filenames);

    // Loop through each image stored in the images folder
    // Create and temporarily store the annotations
    // At the end write everything to the annotations file
    for (size_t i = 0; i < filenames.size(); i++){
        // Read in an image
        Mat current_image = imread(filenames[i]);

        // Check if the image is actually read - avoid other files in the folder, because glob() takes them all
        // If not then simply skip this iteration
        if(current_image.empty()){
            continue;
        }

        // Perform annotations & generate corresponding output
        stringstream output_stream;
        get_annotations(current_image, &output_stream);

        // Store the annotations, write to the output file
        if (output_stream.str() != ""){
            output << filenames[i] << output_stream.str();
        }
    }

    return 0;
}
