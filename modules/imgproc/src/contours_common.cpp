// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "contours_common.hpp"
#include <map>
#include <limits>
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/check.hpp"

using namespace std;
using namespace cv;

void cv::contourTreeToResults(CTree& tree,
                              int res_type,
                              OutputArrayOfArrays& _contours,
                              OutputArray& _hierarchy)
{
    // check if there are no results
    if (tree.isEmpty() || (tree.elem(0).body.isEmpty() && (tree.elem(0).first_child == -1)))
    {
        _contours.clear();
        return;
    }

    CV_Assert(tree.size() < (size_t)numeric_limits<int>::max());
    // mapping for indexes (original -> resulting)
    // -1 - based indexing
    vector<int> index_mapping(tree.size() + 1, -1);

    const int total = (int)tree.size() - 1;
    _contours.create(total, 1, 0, -1, true);
    {
        int i = 0;
        CIterator it(tree);
        while (!it.isDone())
        {
            const CNode& elem = it.getNext_s();
            CV_Assert(elem.self() != -1);
            if (elem.self() == 0)
                continue;
            index_mapping.at(elem.self() + 1) = i;
            CV_Assert(elem.body.size() < (size_t)numeric_limits<int>::max());
            const int sz = (int)elem.body.size();
            _contours.create(sz, 1, res_type, i, true);
            if (sz > 0)
            {
                Mat cmat = _contours.getMat(i);
                CV_Assert(cmat.isContinuous());
                elem.body.copyTo(cmat.data);
            }
            ++i;
        }
    }

    if (_hierarchy.needed())
    {
        _hierarchy.create(1, total, CV_32SC4, -1, true);
        Mat h_mat = _hierarchy.getMat();
        int i = 0;
        CIterator it(tree);
        while (!it.isDone())
        {
            const CNode& elem = it.getNext_s();
            if (elem.self() == 0)
                continue;
            Vec4i& h_vec = h_mat.at<Vec4i>(i);
            h_vec = Vec4i(index_mapping.at(elem.next + 1),
                          index_mapping.at(elem.prev + 1),
                          index_mapping.at(elem.first_child + 1),
                          index_mapping.at(elem.parent + 1));
            ++i;
        }
    }
}

Mat cv::buildZeroBorderedImage(InputArray _src, bool doThreshold, double foregroundValue)
{
    Mat src = _src.getMat();
    Mat padded(src.rows + 2, src.cols + 2, src.type());
    padded.row(0).setTo(Scalar::all(0));
    padded.row(padded.rows - 1).setTo(Scalar::all(0));
    padded.col(0).setTo(Scalar::all(0));
    padded.col(padded.cols - 1).setTo(Scalar::all(0));

    Mat interior = padded(Rect(1, 1, src.cols, src.rows));
    if (doThreshold)
        threshold(src, interior, 0, foregroundValue, THRESH_BINARY);
    else
        src.copyTo(interior);
    return padded;
}
