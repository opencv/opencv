// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "aruco_utils.hpp"

namespace cv {
namespace aruco {
using namespace std;

void _copyVector2Output(vector<vector<Point2f> > &vec, OutputArrayOfArrays out, const float scale) {
    out.create((int)vec.size(), 1, CV_32FC2);
    if(out.isMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            Mat &m = out.getMatRef(i);
            Mat(Mat(vec[i]).t()*scale).copyTo(m);
        }
    }
    else if(out.isUMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            UMat &m = out.getUMatRef(i);
            Mat(Mat(vec[i]).t()*scale).copyTo(m);
        }
    }
    else if(out.kind() == _OutputArray::STD_VECTOR_VECTOR &&
            out.type() == CV_32FC2){
        vector<vector<Point2f>>& out_ = out.getVecVecRef<Point2f>();
        size_t i, j, nvecs = vec.size();
        out_.resize(nvecs);
        for (i = 0; i < nvecs; i++) {
            const vector<Point2f>& vec_i = vec[i];
            size_t npoints_i = vec_i.size();
            vector<Point2f>& out_i = out_[i];
            out_i.resize(npoints_i);
            for (j = 0; j < npoints_i; j++) {
                out_i[j] = vec_i[j]*scale;
            }
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented,
                 "Only Mat vector, UMat vector, and vector<vector> OutputArrays are currently supported.");
    }
}

void _convertToGrey(InputArray _in, OutputArray _out) {
    CV_Assert(_in.type() == CV_8UC1 || _in.type() == CV_8UC3);
    if(_in.type() == CV_8UC3)
        cvtColor(_in, _out, COLOR_BGR2GRAY);
    else
        _in.copyTo(_out);
}

}
}
