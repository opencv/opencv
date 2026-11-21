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

class CalibrateDebevecImpl CV_FINAL : public CalibrateDebevec
{
public:
    CalibrateDebevecImpl(int _samples, float _lambda, bool _random) :
        name("CalibrateDebevec"),
        samples(_samples),
        lambda(_lambda),
        random(_random),
        w(triangleWeights())
    {
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray _times) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        // check inputs
        std::vector<Mat> images;
        src.getMatVector(images);
        Mat times = _times.getMat();

        CV_Assert(images.size() == times.total());
        checkImageDimensions(images);
        CV_Assert(images[0].depth() == CV_8U);
        CV_Assert(times.type() == CV_32FC1);

        // create output
        int channels = images[0].channels();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);
        int rows = images[0].rows;
        int cols = images[0].cols;

        dst.create(LDR_SIZE, 1, CV_32FCC);
        Mat result = dst.getMat();

        // pick pixel locations (either random or in a rectangular grid)
        std::vector<Point> points;
        points.reserve(samples);
        if(random) {
            for(int i = 0; i < samples; i++) {
                points.push_back(Point(rand() % cols, rand() % rows));
            }
        } else {
            int x_points = static_cast<int>(sqrt(static_cast<double>(samples) * cols / rows));
            CV_Assert(0 < x_points && x_points <= cols);
            int y_points = samples / x_points;
            CV_Assert(0 < y_points && y_points <= rows);
            int step_x = cols / x_points;
            int step_y = rows / y_points;
            for(int i = 0, x = step_x / 2; i < x_points; i++, x += step_x) {
                for(int j = 0, y = step_y / 2; j < y_points; j++, y += step_y) {
                    if( 0 <= x && x < cols && 0 <= y && y < rows ) {
                        points.push_back(Point(x, y));
                    }
                }
            }
            // we can have slightly less grid points than specified
            //samples = static_cast<int>(points.size());
        }

        // we need enough equations to ensure a sufficiently overdetermined system
        // (maybe only as a warning)
        //CV_Assert(points.size() * (images.size() - 1) >= LDR_SIZE);

        // solve for imaging system response function, over each channel separately
        std::vector<Mat> result_split(channels);
        for(int ch = 0; ch < channels; ch++) {
            // initialize system of linear equations
            Mat A = Mat::zeros((int)points.size() * (int)images.size() + LDR_SIZE + 1,
                LDR_SIZE + (int)points.size(), CV_32F);
            Mat B = Mat::zeros(A.rows, 1, CV_32F);

            // include the data-fitting equations
            int k = 0;
            for(size_t i = 0; i < points.size(); i++) {
                for(size_t j = 0; j < images.size(); j++) {
                    // val = images[j].at<Vec3b>(points[i].y, points[i].x)[ch]
                    int val = images[j].ptr()[channels*(points[i].y * cols + points[i].x) + ch];
                    float wij = w.at<float>(val);
                    A.at<float>(k, val) = wij;
                    A.at<float>(k, LDR_SIZE + (int)i) = -wij;
                    B.at<float>(k, 0) = wij * log(times.at<float>((int)j));
                    k++;
                }
            }

            // fix the curve by setting its middle value to 0
            // (g is defined only up to an additive constant). Weight this equation
            // like the data equations: with weight 1 it is tiny next to the rows
            // weighted by w and lambda * w, so the offset is weakly constrained and
            // float32 rounding in the solver shifts the curve differently across builds.
            A.at<float>(k, LDR_SIZE / 2) = w.at<float>(LDR_SIZE / 2);
            k++;

            // include the smoothness equations
            for(int i = 0; i < (LDR_SIZE - 2); i++) {
                float wi = w.at<float>(i + 1);
                A.at<float>(k, i) = lambda * wi;
                A.at<float>(k, i + 1) = -2 * lambda * wi;
                A.at<float>(k, i + 2) = lambda * wi;
                k++;
            }

            // solve the overdetermined system using SVD (least-squares problem)
            Mat solution;
            solve(A, B, solution, DECOMP_SVD);
            solution.rowRange(0, LDR_SIZE).copyTo(result_split[ch]);
        }

        // combine log-exposures and take its exponent
        merge(result_split, result);
        exp(result, result);
    }

    int getSamples() const CV_OVERRIDE { return samples; }
    void setSamples(int val) CV_OVERRIDE { samples = val; }

    float getLambda() const CV_OVERRIDE { return lambda; }
    void setLambda(float val) CV_OVERRIDE { lambda = val; }

    bool getRandom() const CV_OVERRIDE { return random; }
    void setRandom(bool val) CV_OVERRIDE { random = val; }

    void write(FileStorage& fs) const CV_OVERRIDE
    {
        writeFormat(fs);
        fs << "name" << name
           << "samples" << samples
           << "lambda" << lambda
           << "random" << static_cast<int>(random);
    }

    void read(const FileNode& fn) CV_OVERRIDE
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        samples = fn["samples"];
        lambda = fn["lambda"];
        int random_val = fn["random"];
        random = (random_val != 0);
    }

protected:
    String name;  // calibration algorithm identifier
    int samples;  // number of pixel locations to sample
    float lambda; // constant that determines the amount of smoothness
    bool random;  // whether to sample locations randomly or in a grid shape
    Mat w;        // weighting function for corresponding pixel values
};

Ptr<CalibrateDebevec> createCalibrateDebevec(int samples, float lambda, bool random)
{
    return makePtr<CalibrateDebevecImpl>(samples, lambda, random);
}

class CalibrateRobertsonImpl CV_FINAL : public CalibrateRobertson
{
public:
    CalibrateRobertsonImpl(int _max_iter, float _threshold) :
        name("CalibrateRobertson"),
        max_iter(_max_iter),
        threshold(_threshold),
        weight(RobertsonWeights())
    {
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray _times) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        std::vector<Mat> images;
        src.getMatVector(images);
        Mat times = _times.getMat();

        CV_Assert(images.size() == times.total());
        checkImageDimensions(images);
        CV_Assert(images[0].depth() == CV_8U);

        int channels = images[0].channels();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);
        CV_Assert(channels >= 1 && channels <= 3);

        dst.create(LDR_SIZE, 1, CV_32FCC);
        Mat response = dst.getMat();
        response = linearResponse(channels) / (static_cast<float>(LDR_SIZE) / 2.0f);

        // Index the per-value tables by the real channel count: a fixed Vec3f
        // overruns them for one- and two-channel images.
        Mat card = Mat::zeros(LDR_SIZE, 1, CV_32FCC);
        float* card_ptr = card.ptr<float>();
        const int width = images[0].cols * channels;
        // Walk each row: an image may be a region of a larger one.
        for(size_t i = 0; i < images.size(); i++) {
            for(int y = 0; y < images[i].rows; y++) {
                const uchar* ptr = images[i].ptr<uchar>(y);
                for(int x = 0; x < width; x++) {
                    card_ptr[ptr[x] * channels + x % channels] += 1;
                }
            }
        }
        // A value that occurs in no image gives no equation for its response, and
        // 1 / 0 would turn it into 0 * inf = NaN, which also keeps the convergence
        // check below from passing. Such values get a zero response instead.
        Mat seen = card > 0;
        divide(1.0, card, card);
        card.setTo(Scalar::all(0), ~seen);
        // Unseen values are 0 in the starting curve too, so the result follows the
        // same rule for any max_iter, including 0. Merging never reads them.
        response.setTo(Scalar::all(0), ~seen);

        Ptr<MergeRobertson> merge = createMergeRobertson();
        for(int iter = 0; iter < max_iter; iter++) {

            radiance = Mat::zeros(images[0].size(), CV_32FCC);
            merge->process(images, radiance, times, response);

            Mat new_response = Mat::zeros(LDR_SIZE, 1, CV_32FCC);
            float* new_ptr = new_response.ptr<float>();
            for(size_t i = 0; i < images.size(); i++) {
                const float t = times.at<float>((int)i);
                for(int y = 0; y < images[i].rows; y++) {
                    const uchar* ptr = images[i].ptr<uchar>(y);
                    const float* rad_ptr = radiance.ptr<float>(y);
                    for(int x = 0; x < width; x++) {
                        new_ptr[ptr[x] * channels + x % channels] += t * rad_ptr[x];
                    }
                }
            }
            new_response = new_response.mul(card);
            new_ptr = new_response.ptr<float>();
            const float* old_ptr = response.ptr<float>();
            const uchar* seen_ptr = seen.ptr<uchar>();
            for(int c = 0; c < channels; c++) {
                // The curve is scaled so that the middle value is 1. When the middle
                // value has no usable estimate, scale by the nearest value that has
                // one, so that it keeps its previous estimate; with no usable value at
                // all, observed values keep their previous estimate and unseen ones 0.
                int anchor = -1;
                for(int d = 0; d < LDR_SIZE && anchor < 0; d++) {
                    for(int k : {LDR_SIZE / 2 - d, LDR_SIZE / 2 + d}) {
                        if(k >= 0 && k < LDR_SIZE && new_ptr[k * channels + c] > 0 &&
                           std::isfinite(new_ptr[k * channels + c]) && old_ptr[k * channels + c] > 0) {
                            anchor = k;
                            break;
                        }
                    }
                }
                if(anchor < 0) {
                    for(int i = 0; i < LDR_SIZE; i++) {
                        int k = i * channels + c;
                        new_ptr[k] = seen_ptr[k] ? old_ptr[k] : 0.f;
                    }
                    continue;
                }
                // Divide, as before, so that a seen middle value (old estimate 1)
                // gives the same result as dividing by it directly.
                float middle = new_ptr[anchor * channels + c] / old_ptr[anchor * channels + c];
                for(int i = 0; i < LDR_SIZE; i++) {
                    new_ptr[i * channels + c] /= middle;
                }
            }
            float diff = static_cast<float>(sum(sum(abs(new_response - response)))[0] / channels);
            new_response.copyTo(response);
            if(diff < threshold) {
                break;
            }
        }
    }

    int getMaxIter() const CV_OVERRIDE { return max_iter; }
    void setMaxIter(int val) CV_OVERRIDE { max_iter = val; }

    float getThreshold() const CV_OVERRIDE { return threshold; }
    void setThreshold(float val) CV_OVERRIDE { threshold = val; }

    Mat getRadiance() const CV_OVERRIDE { return radiance; }

    void write(FileStorage& fs) const CV_OVERRIDE
    {
        writeFormat(fs);
        fs << "name" << name
           << "max_iter" << max_iter
           << "threshold" << threshold;
    }

    void read(const FileNode& fn) CV_OVERRIDE
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
