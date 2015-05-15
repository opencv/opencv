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
//#include "opencv2/highgui.hpp"
#include "hdr_common.hpp"

namespace cv
{

class CalibrateDebevecImpl : public CalibrateDebevec
{
public:
    CalibrateDebevecImpl(int _samples, float _lambda, bool _random) :
        name("CalibrateDebevec"),
        samples(_samples),
        lambda(_lambda),
        random(_random),
        w(tringleWeights())
    {
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray _times)
    {
        std::vector<Mat> images;
        src.getMatVector(images);
        Mat times = _times.getMat();

        CV_Assert(images.size() == times.total());
        checkImageDimensions(images);
        CV_Assert(images[0].depth() == CV_8U);

        int channels = images[0].channels();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);

        dst.create(LDR_SIZE, 1, CV_32FCC);
        Mat result = dst.getMat();

        std::vector<Point> sample_points;
        if(random) {
            for(int i = 0; i < samples; i++) {
                sample_points.push_back(Point(rand() % images[0].cols, rand() % images[0].rows));
            }
        } else {
            int x_points = static_cast<int>(sqrt(static_cast<double>(samples) * images[0].cols / images[0].rows));
            int y_points = samples / x_points;
            int step_x = images[0].cols / x_points;
            int step_y = images[0].rows / y_points;

            for(int i = 0, x = step_x / 2; i < x_points; i++, x += step_x) {
                for(int j = 0, y = step_y / 2; j < y_points; j++, y += step_y) {
                    if( 0 <= x && x < images[0].cols &&
                        0 <= y && y < images[0].rows )
                        sample_points.push_back(Point(x, y));
                }
            }
        }

        std::vector<Mat> result_split(channels);
        for(int channel = 0; channel < channels; channel++) {
            Mat A = Mat::zeros((int)sample_points.size() * (int)images.size() + LDR_SIZE + 1, LDR_SIZE + (int)sample_points.size(), CV_32F);
            Mat B = Mat::zeros(A.rows, 1, CV_32F);

            int eq = 0;
            for(size_t i = 0; i < sample_points.size(); i++) {
                for(size_t j = 0; j < images.size(); j++) {

                    int val = images[j].ptr()[3*(sample_points[i].y * images[j].cols + sample_points[i].x) + channel];
                    A.at<float>(eq, val) = w.at<float>(val);
                    A.at<float>(eq, LDR_SIZE + (int)i) = -w.at<float>(val);
                    B.at<float>(eq, 0) = w.at<float>(val) * log(times.at<float>((int)j));
                    eq++;
                }
            }
            A.at<float>(eq, LDR_SIZE / 2) = 1;
            eq++;

            for(int i = 0; i < 254; i++) {
                A.at<float>(eq, i) = lambda * w.at<float>(i + 1);
                A.at<float>(eq, i + 1) = -2 * lambda * w.at<float>(i + 1);
                A.at<float>(eq, i + 2) = lambda * w.at<float>(i + 1);
                eq++;
            }
            Mat solution;
            solve(A, B, solution, DECOMP_SVD);
            solution.rowRange(0, LDR_SIZE).copyTo(result_split[channel]);
        }
        merge(result_split, result);
        exp(result, result);
    }

    int getSamples() const { return samples; }
    void setSamples(int val) { samples = val; }

    float getLambda() const { return lambda; }
    void setLambda(float val) { lambda = val; }

    bool getRandom() const { return random; }
    void setRandom(bool val) { random = val; }

    void write(FileStorage& fs) const
    {
        fs << "name" << name
           << "samples" << samples
           << "lambda" << lambda
           << "random" << static_cast<int>(random);
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        samples = fn["samples"];
        lambda = fn["lambda"];
        int random_val = fn["random"];
        random = (random_val != 0);
    }

protected:
    String name;
    int samples;
    float lambda;
    bool random;
    Mat w;
};

Ptr<CalibrateDebevec> createCalibrateDebevec(int samples, float lambda, bool random)
{
    return makePtr<CalibrateDebevecImpl>(samples, lambda, random);
}

class CalibrateRobertsonImpl : public CalibrateRobertson
{
public:
    CalibrateRobertsonImpl(int _max_iter, float _threshold) :
        name("CalibrateRobertson"),
        max_iter(_max_iter),
        threshold(_threshold),
        weight(RobertsonWeights())
    {
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray _times)
    {
        std::vector<Mat> images;
        src.getMatVector(images);
        Mat times = _times.getMat();

        CV_Assert(images.size() == times.total());
        checkImageDimensions(images);
        CV_Assert(images[0].depth() == CV_8U);

        int channels = images[0].channels();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);

        dst.create(LDR_SIZE, 1, CV_32FCC);
        Mat response = dst.getMat();
        response = linearResponse(3) / (LDR_SIZE / 2.0f);

        Mat card = Mat::zeros(LDR_SIZE, 1, CV_32FCC);
        for(size_t i = 0; i < images.size(); i++) {
           uchar *ptr = images[i].ptr();
           for(size_t pos = 0; pos < images[i].total(); pos++) {
               for(int c = 0; c < channels; c++, ptr++) {
                   card.at<Vec3f>(*ptr)[c] += 1;
               }
           }
        }
        card = 1.0 / card;

        Ptr<MergeRobertson> merge = createMergeRobertson();
        for(int iter = 0; iter < max_iter; iter++) {

            radiance = Mat::zeros(images[0].size(), CV_32FCC);
            merge->process(images, radiance, times, response);

            Mat new_response = Mat::zeros(LDR_SIZE, 1, CV_32FC3);
            for(size_t i = 0; i < images.size(); i++) {
                uchar *ptr = images[i].ptr();
                float* rad_ptr = radiance.ptr<float>();
                for(size_t pos = 0; pos < images[i].total(); pos++) {
                    for(int c = 0; c < channels; c++, ptr++, rad_ptr++) {
                        new_response.at<Vec3f>(*ptr)[c] += times.at<float>((int)i) * *rad_ptr;
                    }
                }
            }
            new_response = new_response.mul(card);
            for(int c = 0; c < 3; c++) {
                float middle = new_response.at<Vec3f>(LDR_SIZE / 2)[c];
                for(int i = 0; i < LDR_SIZE; i++) {
                    new_response.at<Vec3f>(i)[c] /= middle;
                }
            }
            float diff = static_cast<float>(sum(sum(abs(new_response - response)))[0] / channels);
            new_response.copyTo(response);
            if(diff < threshold) {
                break;
            }
        }
    }

    int getMaxIter() const { return max_iter; }
    void setMaxIter(int val) { max_iter = val; }

    float getThreshold() const { return threshold; }
    void setThreshold(float val) { threshold = val; }

    Mat getRadiance() const { return radiance; }

    void write(FileStorage& fs) const
    {
        fs << "name" << name
           << "max_iter" << max_iter
           << "threshold" << threshold;
    }

    void read(const FileNode& fn)
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        max_iter = fn["max_iter"];
        threshold = fn["threshold"];
    }

protected:
    String name;
    int max_iter;
    float threshold;
    Mat weight, radiance;
};

Ptr<CalibrateRobertson> createCalibrateRobertson(int max_iter, float threshold)
{
    return makePtr<CalibrateRobertsonImpl>(max_iter, threshold);
}

}
