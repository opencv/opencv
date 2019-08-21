/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
//M*/

#include "precomp.hpp"

namespace cv
{

class HausdorffDistanceExtractorImpl CV_FINAL : public HausdorffDistanceExtractor
{
public:
    /* Constructor */
    HausdorffDistanceExtractorImpl(int _distanceFlag = NORM_L1, float _rankProportion=0.6)
    {
        distanceFlag = _distanceFlag;
        rankProportion = _rankProportion;
        name_ = "ShapeDistanceExtractor.HAU";
    }

    /* Destructor */
    ~HausdorffDistanceExtractorImpl()
    {
    }

    //! the main operator
    virtual float computeDistance(InputArray contour1, InputArray contour2) CV_OVERRIDE;

    //! Setters/Getters
    virtual void setDistanceFlag(int _distanceFlag) CV_OVERRIDE {distanceFlag=_distanceFlag;}
    virtual int getDistanceFlag() const CV_OVERRIDE {return distanceFlag;}

    virtual void setRankProportion(float _rankProportion) CV_OVERRIDE
    {
        CV_Assert((_rankProportion>0) && (_rankProportion<=1));
        rankProportion=_rankProportion;
    }
    virtual float getRankProportion() const CV_OVERRIDE {return rankProportion;}

    //! write/read
    virtual void write(FileStorage& fs) const CV_OVERRIDE
    {
        writeFormat(fs);
        fs << "name" << name_
           << "distance" << distanceFlag
           << "rank" << rankProportion;
    }

    virtual void read(const FileNode& fn) CV_OVERRIDE
    {
        CV_Assert( (String)fn["name"] == name_ );
        distanceFlag = (int)fn["distance"];
        rankProportion = (float)fn["rank"];
    }

private:
    int distanceFlag;
    float rankProportion;

protected:
    String name_;
};

//! Hausdorff distance for a pair of set of points
static float _apply(const Mat &set1, const Mat &set2, int distType, double propRank)
{
    // Building distance matrix //
    Mat disMat(set1.cols, set2.cols, CV_32F);
    int K = int(propRank*(disMat.rows-1));

    for (int r=0; r<disMat.rows; r++)
    {
        for (int c=0; c<disMat.cols; c++)
        {
            Point2f diff = set1.at<Point2f>(0,r)-set2.at<Point2f>(0,c);
            disMat.at<float>(r,c) = (float)norm(Mat(diff), distType);
        }
    }

    Mat shortest(disMat.rows,1,CV_32F);
    for (int ii=0; ii<disMat.rows; ii++)
    {
        Mat therow = disMat.row(ii);
        double mindis;
        minMaxIdx(therow, &mindis);
        shortest.at<float>(ii,0) = float(mindis);
    }
    Mat sorted;
    cv::sort(shortest, sorted, SORT_EVERY_ROW | SORT_DESCENDING);
    return sorted.at<float>(K,0);
}

float HausdorffDistanceExtractorImpl::computeDistance(InputArray contour1, InputArray contour2)
{
    CV_INSTRUMENT_REGION();

    Mat set1=contour1.getMat(), set2=contour2.getMat();
    if (set1.type() != CV_32F)
        set1.convertTo(set1, CV_32F);
    if (set2.type() != CV_32F)
        set2.convertTo(set2, CV_32F);
    CV_Assert((set1.channels()==2) && (set1.cols>0));
    CV_Assert((set2.channels()==2) && (set2.cols>0));

    // Force vectors column-based
    if (set1.dims > 1)
        set1 = set1.reshape(2, 1);
    if (set2.dims > 1)
        set2 = set2.reshape(2, 1);

    return std::max( _apply(set1, set2, distanceFlag, rankProportion),
                     _apply(set2, set1, distanceFlag, rankProportion) );
}

Ptr <HausdorffDistanceExtractor> createHausdorffDistanceExtractor(int distanceFlag, float rankProp)
{
    return Ptr<HausdorffDistanceExtractor>(new HausdorffDistanceExtractorImpl(distanceFlag, rankProp));
}

} // cv
