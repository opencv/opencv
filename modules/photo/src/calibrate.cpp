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
#include <iostream>

namespace cv
{
    
class CalibrateDebevecImpl : public CalibrateDebevec
{
public:
    CalibrateDebevecImpl(int samples, float lambda, bool random) :
        samples(samples),
        lambda(lambda),
        name("CalibrateDebevec"),
        w(tringleWeights()),
        random(random)
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
        
        std::vector<Point> sample_points;
        if(random) {
            for(int i = 0; i < samples; i++) {
                sample_points.push_back(Point(rand() % images[0].cols, rand() % images[0].rows));
            }
        } else {
            int x_points = sqrt(static_cast<double>(samples) * images[0].cols / images[0].rows);
            int y_points = samples / x_points;
            int step_x = images[0].cols / x_points;
            int step_y = images[0].rows / y_points;

            for(int i = 0, x = step_x / 2; i < x_points; i++, x += step_x) {
                for(int j = 0, y = step_y; j < y_points; j++, y += step_y) {
                    sample_points.push_back(Point(x, y));
                }
            }
        }

        std::vector<Mat> result_split(channels);
        for(int channel = 0; channel < channels; channel++) {
            Mat A = Mat::zeros(sample_points.size() * images.size() + 257, 256 + sample_points.size(), CV_32F);
            Mat B = Mat::zeros(A.rows, 1, CV_32F);

            int eq = 0;
            for(size_t i = 0; i < sample_points.size(); i++) {

                for(size_t j = 0; j < images.size(); j++) {

                    int val = images[j].ptr()[3*(sample_points[i].y * images[j].cols + sample_points[j].x) + channel];
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
        random = static_cast<bool>(random_val);
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
    return new CalibrateDebevecImpl(samples, lambda, random);
}

class CalibrateRobertsonImpl : public CalibrateRobertson
{
public:
    CalibrateRobertsonImpl(int max_iter, float threshold) :
        max_iter(max_iter),
        threshold(threshold),
        name("CalibrateRobertson"),
        weight(RobertsonWeights())
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
        Mat response = dst.getMat();
        
        response = Mat::zeros(256, 1, CV_32FCC);
        for(int i = 0; i < 256; i++) {
            for(int c = 0; c < channels; c++) {
                response.at<Vec3f>(i)[c] = i / 128.0;
            }
        }

        Mat card = Mat::zeros(256, 1, CV_32FCC);
        for(int i = 0; i < images.size(); i++) {
           uchar *ptr = images[i].ptr();
           for(int pos = 0; pos < images[i].total(); pos++) {
               for(int c = 0; c < channels; c++, ptr++) {
                   card.at<Vec3f>(*ptr)[c] += 1;
               }
           }
        }
        card = 1.0 / card;

        for(int iter = 0; iter < max_iter; iter++) {

            Scalar channel_err(0, 0, 0);
            Mat radiance = Mat::zeros(images[0].size(), CV_32FCC);
            Mat wsum = Mat::zeros(images[0].size(), CV_32FCC);
            for(int i = 0; i < images.size(); i++) {
                Mat im, w;
                LUT(images[i], weight, w);
                LUT(images[i], response, im);

                Mat err_mat;
                pow(im - times[i] * radiance, 2.0f, err_mat);
                err_mat = w.mul(err_mat);
                channel_err += sum(err_mat);

                radiance += times[i] * w.mul(im);
                wsum += pow(times[i], 2) * w;
            }
            float err = (channel_err[0] + channel_err[1] + channel_err[2]) / (channels * radiance.total());
            radiance = radiance.mul(1 / wsum);

            float* rad_ptr = radiance.ptr<float>();
            response = Mat::zeros(256, 1, CV_32FC3);
            for(int i = 0; i < images.size(); i++) {
                uchar *ptr = images[i].ptr();
                for(int pos = 0; pos < images[i].total(); pos++) {
                    for(int c = 0; c < channels; c++, ptr++, rad_ptr++) {
                        response.at<Vec3f>(*ptr)[c] += times[i] * *rad_ptr;
                    }
                }
            }
            response = response.mul(card);
            for(int c = 0; c < 3; c++) {
                for(int i = 0; i < 256; i++) {
                    response.at<Vec3f>(i)[c] /= response.at<Vec3f>(128)[c];
                }
            }
        }
    }

    int getMaxIter() const { return max_iter; }
    void setMaxIter(int val) { max_iter = val; }

    float getThreshold() const { return threshold; }
    void setThreshold(float val) { threshold = val; }

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
    Mat weight;
};

Ptr<CalibrateRobertson> createCalibrateRobertson(int max_iter, float threshold)
{
    return new CalibrateRobertsonImpl(max_iter, threshold);
}

}