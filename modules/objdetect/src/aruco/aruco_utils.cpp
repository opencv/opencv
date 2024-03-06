// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "aruco_utils.hpp"

namespace cv {
namespace aruco {
using namespace std;

void _copyInput2Vector(InputArrayOfArrays inp, vector<vector<Point2f> > &vec)
{
    size_t i, nvecs = inp.size().area();
    int inpdepth = inp.depth();
    CV_Assert(inpdepth == CV_32F);
    vec.resize(nvecs);
    if(inp.isMatVector() || inp.kind() == _InputArray::STD_VECTOR_VECTOR)
    {
        for (i = 0; i < nvecs; i++)
        {
            Mat inp_i = inp.getMat((int)i);
            int j, npoints = inp_i.checkVector(2, inpdepth, true);
            CV_Assert(npoints >= 0);
            const Point2f* inpptr = inp_i.ptr<Point2f>();
            vector<Point2f>& vec_i = vec[i];
            vec_i.resize(npoints);
            for (j = 0; j < npoints; j++)
                vec_i[j] = inpptr[j];
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented,
                 "Only Mat vector, UMat vector, and vector<vector> OutputArrays are currently supported.");
    }
}

void _copyVector2Output(vector<vector<Point2f> > &vec, OutputArrayOfArrays out, const float scale) {
    size_t i, j, nvecs = vec.size();
    if(out.isMatVector()) {
        vector<Mat>& out_ = out.getMatVecRef();
        out_.resize(nvecs);
        for (i = 0; i < nvecs; i++) {
            const vector<Point2f>& vec_i = vec[i];
            Mat& out_i = out_[i];
            Mat(vec_i).reshape(2, 1).convertTo(out_i, CV_32F, scale);
        }
    }
    else if(out.isUMatVector()) {
        vector<UMat>& out_ = out.getUMatVecRef();
        out_.resize(nvecs);
        for (i = 0; i < nvecs; i++) {
            const vector<Point2f>& vec_i = vec[i];
            UMat& out_i = out_[i];
            Mat(vec_i).reshape(2, 1).convertTo(out_i, CV_32F, scale);
        }
    }
    else if(out.kind() == _OutputArray::STD_VECTOR_VECTOR &&
            out.type() == CV_32FC2){
        vector<vector<Point2f>>& out_ = out.getVecVecRef<Point2f>();
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

void _convertToGrey(InputArray _in, Mat& _out) {
    CV_Assert(_in.type() == CV_8UC1 || _in.type() == CV_8UC3 || _in.type() == CV_8UC4);
    if(_in.type() != CV_8UC1)
        cvtColor(_in, _out, COLOR_BGR2GRAY);
    else
        _out = _in.getMat();
}

}
}
