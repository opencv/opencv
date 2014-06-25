// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (c) 2011,2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_FACEREC_HPP__
#define __OPENCV_FACEREC_HPP__

#include "opencv2/core.hpp"

namespace cv
{

class CV_EXPORTS_W FaceRecognizer : public Algorithm
{
public:
    //! virtual destructor
    virtual ~FaceRecognizer() {}

    // Trains a FaceRecognizer.
    CV_WRAP virtual void train(InputArrayOfArrays src, InputArray labels) = 0;

    // Updates a FaceRecognizer.
    CV_WRAP virtual void update(InputArrayOfArrays src, InputArray labels) = 0;

    // Gets a prediction from a FaceRecognizer.
    virtual int predict(InputArray src) const = 0;

    // Predicts the label and confidence for a given sample.
    CV_WRAP virtual void predict(InputArray src, CV_OUT int &label, CV_OUT double &confidence) const = 0;

    // Serializes this object to a given filename.
    CV_WRAP virtual void save(const String& filename) const = 0;

    // Deserializes this object from a given filename.
    CV_WRAP virtual void load(const String& filename) = 0;

    // Serializes this object to a given cv::FileStorage.
    virtual void save(FileStorage& fs) const = 0;

    // Deserializes this object from a given cv::FileStorage.
    virtual void load(const FileStorage& fs) = 0;

    // Sets additional string info for the label
    virtual void setLabelInfo(int label, const String& strInfo) = 0;

    // Gets string info by label
    virtual String getLabelInfo(int label) const = 0;

    // Gets labels by string
    virtual std::vector<int> getLabelsByString(const String& str) const = 0;
};

CV_EXPORTS_W Ptr<FaceRecognizer> createEigenFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);
CV_EXPORTS_W Ptr<FaceRecognizer> createFisherFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);
CV_EXPORTS_W Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8, double threshold = DBL_MAX);

} //namespace cv

#endif //__OPENCV_FACEREC_HPP__
