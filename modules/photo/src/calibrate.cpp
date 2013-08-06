/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//M*/

#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "hdr_common.hpp"

namespace cv
{
    
class CalibrateDebevecImpl : public CalibrateDebevec
{
public:
    CalibrateDebevecImpl(int samples, float lambda) :
        samples(samples),
        lambda(lambda),
        name("CalibrateDebevec"),
        w(tringleWeights()),
        test(false)
    {
    }
    
    void process(InputArrayOfArrays src, OutputArray dst, std::vector<float>& times)
    {
        std::vector<Mat> images;
        src.getMatVector(images);

        CV_Assert(images.size() == times.size());
        checkImageDimensions(images);
        CV_Assert(images[0].depth() == CV_8U);

        int channels = images[0].channels();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);

        dst.create(256, 1, CV_32FCC);
        Mat result = dst.getMat();
        
        std::vector<Mat> result_split(channels);
        for(int channel = 0; channel < channels; channel++) {
            Mat A = Mat::zeros(samples * images.size() + 257, 256 + samples, CV_32F);
            Mat B = Mat::zeros(A.rows, 1, CV_32F);

            int eq = 0;
            for(int i = 0; i < samples; i++) {

                int pos = 3 * (rand() % images[0].total()) + channel;
                if(test) {
                    pos = 3 * i + channel;
                }
                for(size_t j = 0; j < images.size(); j++) {

                    int val = (images[j].ptr() + pos)[0];
                    A.at<float>(eq, val) = w.at<float>(val);
                    A.at<float>(eq, 256 + i) = -w.at<float>(val);
                    B.at<float>(eq, 0) = w.at<float>(val) * log(times[j]);        
                    eq++;
                }
            }
            A.at<float>(eq, 128) = 1;
            eq++;

            for(int i = 0; i < 254; i++) {
                A.at<float>(eq, i) = lambda * w.at<float>(i + 1);
                A.at<float>(eq, i + 1) = -2 * lambda * w.at<float>(i + 1);
                A.at<float>(eq, i + 2) = lambda * w.at<float>(i + 1);
                eq++;
            }
            Mat solution;
            solve(A, B, solution, DECOMP_SVD);
            solution.rowRange(0, 256).copyTo(result_split[channel]);
        }
        merge(result_split, result);
        exp(result, result);
    }

    bool getTest() const { return test; }
    void setTest(bool val) { test = val; }

    int getSamples() const { return samples; }
    void setSamples(int val) { samples = val; }

    float getLambda() const { return lambda; }
    void setLambda(float val) { lambda = val; }

    void write(FileStorage& fs) const
    {
        fs << "name" << name
           << "samples" << samples
           << "lambda" << lambda;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        samples = fn["samples"];
        lambda = fn["lambda"];
    }

protected:
    String name;
    int samples;
    float lambda;
    bool test;
    Mat w;
};

Ptr<CalibrateDebevec> createCalibrateDebevec(int samples, float lambda)
{
    return new CalibrateDebevecImpl(samples, lambda);
}

}