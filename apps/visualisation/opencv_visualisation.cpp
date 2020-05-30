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

Software for visualising cascade classifier models trained by OpenCV and to get a better
understanding of the used features.

USAGE:
./opencv_visualisation --model=<model.xml> --image=<ref.png> --data=<video output folder>

Created by: Puttemans Steven - April 2016
*****************************************************************************************************/

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

struct rect_data{
    int x;
    int y;
    int w;
    int h;
    float weight;
};

static void printLimits(){
    cerr << "Limits of the current interface:" << endl;
    cerr << " - Only handles cascade classifier models, trained with the opencv_traincascade tool, containing stumps as decision trees [default settings]." << endl;
    cerr << " - The image provided needs to be a sample window with the original model dimensions, passed to the --image parameter." << endl;
    cerr << " - ONLY handles HAAR and LBP features." << endl;
}

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this message }"
        "{ image i        |      | (required) path to reference image }"
        "{ model m        |      | (required) path to cascade xml file }"
        "{ data d         |      | (optional) path to video output folder }"
    );
    // Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        printLimits();
        return 0;
    }
    string model(parser.get<string>("model"));
    string output_folder(parser.get<string>("data"));
    string image_ref = (parser.get<string>("image"));
    if (model.empty() || image_ref.empty()){
        parser.printMessage();
        printLimits();
        return -1;
    }

    // Value for timing
    // You can increase this to have a better visualisation during the generation
    int timing = 1;

    // Value for cols of storing elements
    int cols_prefered = 5;

    // Open the XML model
    FileStorage fs;
    bool model_ok = fs.open(model, FileStorage::READ);
    if (!model_ok){
        cerr << "the cascade file '" << model << "' could not be loaded." << endl;
        return  -1;
    }
    // Get a the required information
    // First decide which feature type we are using
    FileNode cascade = fs["cascade"];
    string feature_type = cascade["featureType"];
    bool haar = false, lbp = false;
    if (feature_type.compare("HAAR") == 0){
        haar = true;
    }
    if (feature_type.compare("LBP") == 0){
        lbp = true;
    }
    if ( feature_type.compare("HAAR") != 0 && feature_type.compare("LBP")){
        cerr << "The model is not an HAAR or LBP feature based model!" << endl;
        cerr << "Please select a model that can be visualized by the software." << endl;
        return -1;
    }

    // We make a visualisation mask - which increases the window to make it at least a bit more visible
    int resize_factor = 10;
    int resize_storage_factor = 10;
    Mat reference_image = imread(image_ref, IMREAD_GRAYSCALE );
    if (reference_image.empty()){
        cerr << "the reference image '" << image_ref << "'' could not be loaded." << endl;
        return -1;
    }
    Mat visualization;
    resize(reference_image, visualization, Size(reference_image.cols * resize_factor, reference_image.rows * resize_factor), 0, 0, INTER_LINEAR_EXACT);

    // First recover for each stage the number of weak features and their index
    // Important since it is NOT sequential when using LBP features
    vector< vector<int> > stage_features;
    FileNode stages = cascade["stages"];
    FileNodeIterator it_stages = stages.begin(), it_stages_end = stages.end();
    int idx = 0;
    for( ; it_stages != it_stages_end; it_stages++, idx++ ){
        vector<int> current_feature_indexes;
        FileNode weak_classifiers = (*it_stages)["weakClassifiers"];
        FileNodeIterator it_weak = weak_classifiers.begin(), it_weak_end = weak_classifiers.end();
        vector<int> values;
        for(int idy = 0; it_weak != it_weak_end; it_weak++, idy++ ){
            (*it_weak)["internalNodes"] >> values;
            current_feature_indexes.push_back( (int)values[2] );
        }
        stage_features.push_back(current_feature_indexes);
    }

    // If the output option has been chosen than we will store a combined image plane for
    // each stage, containing all weak classifiers for that stage.
    bool draw_planes = false;
    stringstream output_video;
    output_video << output_folder << "model_visualization.avi";
    VideoWriter result_video;
    if( output_folder.compare("") != 0 ){
        draw_planes = true;
        result_video.open(output_video.str(), VideoWriter::fourcc('X','V','I','D'), 15, Size(reference_image.cols * resize_factor, reference_image.rows * resize_factor), false);
    }

    if(haar){
        // Grab the corresponding features dimensions and weights
        FileNode features = cascade["features"];
        vector< vector< rect_data > > feature_data;
        FileNodeIterator it_features = features.begin(), it_features_end = features.end();
        for(int idf = 0; it_features != it_features_end; it_features++, idf++ ){
            vector< rect_data > current_feature_rectangles;
            FileNode rectangles = (*it_features)["rects"];
            int nrects = (int)rectangles.size();
            for(int k = 0; k < nrects; k++){
                rect_data current_data;
                FileNode single_rect = rectangles[k];
                current_data.x = (int)single_rect[0];
                current_data.y = (int)single_rect[1];
                current_data.w = (int)single_rect[2];
                current_data.h = (int)single_rect[3];
                current_data.weight = (float)single_rect[4];
                current_feature_rectangles.push_back(current_data);
            }
            feature_data.push_back(current_feature_rectangles);
        }

        // Loop over each possible feature on its index, visualise on the mask and wait a bit,
        // then continue to the next feature.
        // If visualisations should be stored then do the in between calculations
        Mat image_plane;
        Mat metadata = Mat::zeros(150, 1000, CV_8UC1);
        vector< rect_data > current_rects;
        for(int sid = 0; sid < (int)stage_features.size(); sid ++){
            if(draw_planes){
                int features_nmbr = (int)stage_features[sid].size();
                int cols = cols_prefered;
                int rows = features_nmbr / cols;
                if( (features_nmbr % cols) > 0){
                    rows++;
                }
                image_plane = Mat::zeros(reference_image.rows * resize_storage_factor * rows, reference_image.cols * resize_storage_factor * cols, CV_8UC1);
            }
            for(int fid = 0; fid < (int)stage_features[sid].size(); fid++){
                stringstream meta1, meta2;
                meta1 << "Stage " << sid << " / Feature " << fid;
                meta2 << "Rectangles: ";
                Mat temp_window = visualization.clone();
                Mat temp_metadata = metadata.clone();
                int current_feature_index = stage_features[sid][fid];
                current_rects = feature_data[current_feature_index];
                Mat single_feature = reference_image.clone();
                resize(single_feature, single_feature, Size(), resize_storage_factor, resize_storage_factor, INTER_LINEAR_EXACT);
                for(int i = 0; i < (int)current_rects.size(); i++){
                    rect_data local = current_rects[i];
                    if(draw_planes){
                        if(local.weight >= 0){
                            rectangle(single_feature, Rect(local.x * resize_storage_factor, local.y * resize_storage_factor, local.w * resize_storage_factor, local.h * resize_storage_factor), Scalar(0), FILLED);
                        }else{
                            rectangle(single_feature, Rect(local.x * resize_storage_factor, local.y * resize_storage_factor, local.w * resize_storage_factor, local.h * resize_storage_factor), Scalar(255), FILLED);
                        }
                    }
                    Rect part(local.x * resize_factor, local.y * resize_factor, local.w * resize_factor, local.h * resize_factor);
                    meta2 << part << " (w " << local.weight << ") ";
                    if(local.weight >= 0){
                        rectangle(temp_window, part, Scalar(0), FILLED);
                    }else{
                        rectangle(temp_window, part, Scalar(255), FILLED);
                    }
                }
                imshow("features", temp_window);
                putText(temp_window, meta1.str(), Point(15,15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255));
                result_video.write(temp_window);
                // Copy the feature image if needed
                if(draw_planes){
                    single_feature.copyTo(image_plane(Rect(0 + (fid%cols_prefered)*single_feature.cols, 0 + (fid/cols_prefered) * single_feature.rows, single_feature.cols, single_feature.rows)));
                }
                putText(temp_metadata, meta1.str(), Point(15,15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255));
                putText(temp_metadata, meta2.str(), Point(15,40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255));
                imshow("metadata", temp_metadata);
                waitKey(timing);
            }
            //Store the stage image if needed
            if(draw_planes){
                stringstream save_location;
                save_location << output_folder << "stage_" << sid << ".png";
                imwrite(save_location.str(), image_plane);
            }
        }
    }

    if(lbp){
        // Grab the corresponding features dimensions and weights
        FileNode features = cascade["features"];
        vector<Rect> feature_data;
        FileNodeIterator it_features = features.begin(), it_features_end = features.end();
        for(int idf = 0; it_features != it_features_end; it_features++, idf++ ){
            FileNode rectangle = (*it_features)["rect"];
            Rect current_feature ((int)rectangle[0], (int)rectangle[1], (int)rectangle[2], (int)rectangle[3]);
            feature_data.push_back(current_feature);
        }

        // Loop over each possible feature on its index, visualise on the mask and wait a bit,
        // then continue to the next feature.
        Mat image_plane;
        Mat metadata = Mat::zeros(150, 1000, CV_8UC1);
        for(int sid = 0; sid < (int)stage_features.size(); sid ++){
            if(draw_planes){
                int features_nmbr = (int)stage_features[sid].size();
                int cols = cols_prefered;
                int rows = features_nmbr / cols;
                if( (features_nmbr % cols) > 0){
                    rows++;
                }
                image_plane = Mat::zeros(reference_image.rows * resize_storage_factor * rows, reference_image.cols * resize_storage_factor * cols, CV_8UC1);
            }
            for(int fid = 0; fid < (int)stage_features[sid].size(); fid++){
                stringstream meta1, meta2;
                meta1 << "Stage " << sid << " / Feature " << fid;
                meta2 << "Rectangle: ";
                Mat temp_window = visualization.clone();
                Mat temp_metadata = metadata.clone();
                int current_feature_index = stage_features[sid][fid];
                Rect current_rect = feature_data[current_feature_index];
                Mat single_feature = reference_image.clone();
                resize(single_feature, single_feature, Size(), resize_storage_factor, resize_storage_factor, INTER_LINEAR_EXACT);

                // VISUALISATION
                // The rectangle is the top left one of a 3x3 block LBP constructor
                Rect resized(current_rect.x * resize_factor, current_rect.y * resize_factor, current_rect.width * resize_factor, current_rect.height * resize_factor);
                meta2 << resized;
                // Top left
                rectangle(temp_window, resized, Scalar(255), 1);
                // Top middle
                rectangle(temp_window, Rect(resized.x + resized.width, resized.y, resized.width, resized.height), Scalar(255), 1);
                // Top right
                rectangle(temp_window, Rect(resized.x + 2*resized.width, resized.y, resized.width, resized.height), Scalar(255), 1);
                // Middle left
                rectangle(temp_window, Rect(resized.x, resized.y + resized.height, resized.width, resized.height), Scalar(255), 1);
                // Middle middle
                rectangle(temp_window, Rect(resized.x + resized.width, resized.y + resized.height, resized.width, resized.height), Scalar(255), FILLED);
                // Middle right
                rectangle(temp_window, Rect(resized.x + 2*resized.width, resized.y + resized.height, resized.width, resized.height), Scalar(255), 1);
                // Bottom left
                rectangle(temp_window, Rect(resized.x, resized.y + 2*resized.height, resized.width, resized.height), Scalar(255), 1);
                // Bottom middle
                rectangle(temp_window, Rect(resized.x + resized.width, resized.y + 2*resized.height, resized.width, resized.height), Scalar(255), 1);
                // Bottom right
                rectangle(temp_window, Rect(resized.x + 2*resized.width, resized.y + 2*resized.height, resized.width, resized.height), Scalar(255), 1);

                if(draw_planes){
                    Rect resized_inner(current_rect.x * resize_storage_factor, current_rect.y * resize_storage_factor, current_rect.width * resize_storage_factor, current_rect.height * resize_storage_factor);
                    // Top left
                    rectangle(single_feature, resized_inner, Scalar(255), 1);
                    // Top middle
                    rectangle(single_feature, Rect(resized_inner.x + resized_inner.width, resized_inner.y, resized_inner.width, resized_inner.height), Scalar(255), 1);
                    // Top right
                    rectangle(single_feature, Rect(resized_inner.x + 2*resized_inner.width, resized_inner.y, resized_inner.width, resized_inner.height), Scalar(255), 1);
                    // Middle left
                    rectangle(single_feature, Rect(resized_inner.x, resized_inner.y + resized_inner.height, resized_inner.width, resized_inner.height), Scalar(255), 1);
                    // Middle middle
                    rectangle(single_feature, Rect(resized_inner.x + resized_inner.width, resized_inner.y + resized_inner.height, resized_inner.width, resized_inner.height), Scalar(255), FILLED);
                    // Middle right
                    rectangle(single_feature, Rect(resized_inner.x + 2*resized_inner.width, resized_inner.y + resized_inner.height, resized_inner.width, resized_inner.height), Scalar(255), 1);
                    // Bottom left
                    rectangle(single_feature, Rect(resized_inner.x, resized_inner.y + 2*resized_inner.height, resized_inner.width, resized_inner.height), Scalar(255), 1);
                    // Bottom middle
                    rectangle(single_feature, Rect(resized_inner.x + resized_inner.width, resized_inner.y + 2*resized_inner.height, resized_inner.width, resized_inner.height), Scalar(255), 1);
                    // Bottom right
                    rectangle(single_feature, Rect(resized_inner.x + 2*resized_inner.width, resized_inner.y + 2*resized_inner.height, resized_inner.width, resized_inner.height), Scalar(255), 1);

                    single_feature.copyTo(image_plane(Rect(0 + (fid%cols_prefered)*single_feature.cols, 0 + (fid/cols_prefered) * single_feature.rows, single_feature.cols, single_feature.rows)));
                }

                putText(temp_metadata, meta1.str(), Point(15,15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255));
                putText(temp_metadata, meta2.str(), Point(15,40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255));
                imshow("metadata", temp_metadata);
                imshow("features", temp_window);
                putText(temp_window, meta1.str(), Point(15,15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255));
                result_video.write(temp_window);

                waitKey(timing);
            }

            //Store the stage image if needed
            if(draw_planes){
                stringstream save_location;
                save_location << output_folder << "stage_" << sid << ".png";
                imwrite(save_location.str(), image_plane);
            }
        }
    }
    return 0;
}
