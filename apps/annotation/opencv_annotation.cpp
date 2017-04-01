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
Adapted by: Puttemans Steven - April 2016 - Vectorize the process to enable better processing
                                               + early leave and store by pressing an ESC key
                                               + enable delete `d` button, to remove last annotation
*****************************************************************************************************/

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <map>

using namespace std;
using namespace cv;

// Function prototypes
void on_mouse(int, int, int, int, void*);
vector<Rect> get_annotations(Mat);

// Public parameters
Mat image;
int roi_x0 = 0, roi_y0 = 0, roi_x1 = 0, roi_y1 = 0, num_of_rec = 0;
bool start_draw = false, stop = false;

// Window name for visualisation purposes
const string window_name = "OpenCV Based Annotation Tool";

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

    // Action when mouse is moving and drawing is enabled
    if((event == EVENT_MOUSEMOVE) && start_draw)
    {
        // Redraw bounding box for annotation
        Mat current_view;
        image.copyTo(current_view);
        rectangle(current_view, Point(roi_x0,roi_y0), Point(x,y), Scalar(0,0,255));
        imshow(window_name, current_view);
    }
}

// FUNCTION : returns a vector of Rect objects given an image containing positive object instances
vector<Rect> get_annotations(Mat input_image)
{
    vector<Rect> current_annotations;

    // Make it possible to exit the annotation process
    stop = false;

    // Init window interface and couple mouse actions
    namedWindow(window_name, WINDOW_AUTOSIZE);
    setMouseCallback(window_name, on_mouse);

    image = input_image;
    imshow(window_name, image);
    int key_pressed = 0;

    do
    {
        // Get a temporary image clone
        Mat temp_image = input_image.clone();
        Rect currentRect(0, 0, 0, 0);

        // Keys for processing
        // You need to select one for confirming a selection and one to continue to the next image
        // Based on the universal ASCII code of the keystroke: http://www.asciitable.com/
        //      c = 99		    add rectangle to current image
        //	    n = 110		    save added rectangles and show next image
        //      d = 100         delete the last annotation made
        //	    <ESC> = 27      exit program
        key_pressed = 0xFF & waitKey(0);
        switch( key_pressed )
        {
        case 27:
                destroyWindow(window_name);
                stop = true;
                break;
        case 99:
                // Draw initiated from top left corner
                if(roi_x0<roi_x1 && roi_y0<roi_y1)
                {
                    currentRect.x = roi_x0;
                    currentRect.y = roi_y0;
                    currentRect.width = roi_x1-roi_x0;
                    currentRect.height = roi_y1-roi_y0;
                }
                // Draw initiated from bottom right corner
                if(roi_x0>roi_x1 && roi_y0>roi_y1)
                {
                    currentRect.x = roi_x1;
                    currentRect.y = roi_y1;
                    currentRect.width = roi_x0-roi_x1;
                    currentRect.height = roi_y0-roi_y1;
                }
                // Draw initiated from top right corner
                if(roi_x0>roi_x1 && roi_y0<roi_y1)
                {
                    currentRect.x = roi_x1;
                    currentRect.y = roi_y0;
                    currentRect.width = roi_x0-roi_x1;
                    currentRect.height = roi_y1-roi_y0;
                }
                // Draw initiated from bottom left corner
                if(roi_x0<roi_x1 && roi_y0>roi_y1)
                {
                    currentRect.x = roi_x0;
                    currentRect.y = roi_y1;
                    currentRect.width = roi_x1-roi_x0;
                    currentRect.height = roi_y0-roi_y1;
                }
                // Draw the rectangle on the canvas
                // Add the rectangle to the vector of annotations
                current_annotations.push_back(currentRect);
                break;
        case 100:
                // Remove the last annotation
                if(current_annotations.size() > 0){
                    current_annotations.pop_back();
                }
                break;
        default:
                // Default case --> do nothing at all
                // Other keystrokes can simply be ignored
                break;
        }

        // Check if escape has been pressed
        if(stop)
        {
            break;
        }

        // Draw all the current rectangles onto the top image and make sure that the global image is linked
        for(int i=0; i < (int)current_annotations.size(); i++){
            rectangle(temp_image, current_annotations[i], Scalar(0,255,0), 1);
        }
        image = temp_image;

        // Force an explicit redraw of the canvas --> necessary to visualize delete correctly
        imshow(window_name, image);
    }
    // Continue as long as the next image key has not been pressed
    while(key_pressed != 110);

    // Close down the window
    destroyWindow(window_name);

    // Return the data
    return current_annotations;
}

int main( int argc, const char** argv )
{
    // Use the cmdlineparser to process input arguments
    CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this message }"
        "{ images i       |      | (required) path to image folder [example - /data/testimages/] }"
        "{ annotations a  |      | (required) path to annotations txt file [example - /data/annotations.txt] }"
        "{ maxWindowHeight m  |  -1   | (optional) images larger in height than this value will be scaled down }"
        "{ resizeFactor r  |  2  | (optional) factor for scaling down [default = half the size] }"
    );
    // Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return 0;
    }
    string image_folder(parser.get<string>("images"));
    string annotations_file(parser.get<string>("annotations"));
    if (image_folder.empty() || annotations_file.empty()){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return -1;
    }

    int resizeFactor = parser.get<int>("resizeFactor");
    int const maxWindowHeight = parser.get<int>("maxWindowHeight") > 0 ? parser.get<int>("maxWindowHeight") : -1;

    // Start by processing the data
    // Return the image filenames inside the image folder
    map< String, vector<Rect> > annotations;
    vector<String> filenames;
    String folder(image_folder);
    glob(folder, filenames);

    // Add key tips on how to use the software when running it
    cout << "* mark rectangles with the left mouse button," << endl;
    cout << "* press 'c' to accept a selection," << endl;
    cout << "* press 'd' to delete the latest selection," << endl;
    cout << "* press 'n' to proceed with next image," << endl;
    cout << "* press 'esc' to stop." << endl;

    // Loop through each image stored in the images folder
    // Create and temporarily store the annotations
    // At the end write everything to the annotations file
    for (size_t i = 0; i < filenames.size(); i++){
        // Read in an image
        Mat current_image = imread(filenames[i]);
        bool const resize_bool = (maxWindowHeight > 0) && (current_image.rows > maxWindowHeight);

        // Check if the image is actually read - avoid other files in the folder, because glob() takes them all
        // If not then simply skip this iteration
        if(current_image.empty()){
            continue;
        }

        if(resize_bool){
            resize(current_image, current_image, Size(current_image.cols/resizeFactor, current_image.rows/resizeFactor));
        }

        // Perform annotations & store the result inside the vectorized structure
        // If the image was resized before, then resize the found annotations back to original dimensions
        vector<Rect> current_annotations = get_annotations(current_image);
        if(resize_bool){
            for(int j =0; j < (int)current_annotations.size(); j++){
                current_annotations[j].x = current_annotations[j].x * resizeFactor;
                current_annotations[j].y = current_annotations[j].y * resizeFactor;
                current_annotations[j].width = current_annotations[j].width * resizeFactor;
                current_annotations[j].height = current_annotations[j].height * resizeFactor;
            }
        }
        annotations[filenames[i]] = current_annotations;

        // Check if the ESC key was hit, then exit earlier then expected
        if(stop){
            break;
        }
    }

    // When all data is processed, store the data gathered inside the proper file
    // This now even gets called when the ESC button was hit to store preliminary results
    ofstream output(annotations_file.c_str());
    if ( !output.is_open() ){
        cerr << "The path for the output file contains an error and could not be opened. Please check again!" << endl;
        return 0;
    }

    // Store the annotations, write to the output file
    for(map<String, vector<Rect> >::iterator it = annotations.begin(); it != annotations.end(); it++){
        vector<Rect> &anno = it->second;
        output << it->first << " " << anno.size();
        for(size_t j=0; j < anno.size(); j++){
            Rect temp = anno[j];
            output << " " << temp.x << " " << temp.y << " " << temp.width << " " << temp.height;
        }
        output << endl;
    }

    return 0;
}
