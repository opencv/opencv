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
//M*/

#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "hdr_common.hpp"

namespace cv
{

inline void log_(const Mat& src, Mat& dst)
{
    max(src, Scalar::all(1e-4), dst);
    log(dst, dst);
}

class TonemapImpl CV_FINAL : public Tonemap
{
public:
    TonemapImpl(float _gamma) : name("Tonemap"), gamma(_gamma)
    {
    }

    void process(InputArray _src, OutputArray _dst) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        Mat src = _src.getMat();
        max(src, Scalar::all(1e-6), src);
        CV_Assert(!src.empty());
        CV_Assert(_src.dims() == 2 && _src.type() == CV_32FC3);
        _dst.create(src.size(), CV_32FC3);
        Mat dst = _dst.getMat();

        double min, max;
        minMaxLoc(src, &min, &max);
        if(max - min > DBL_EPSILON) {
            dst = (src - min) / (max - min);
        } else {
            src.copyTo(dst);
        }

        pow(dst, 1.0f / gamma, dst);
    }

    float getGamma() const CV_OVERRIDE { return gamma; }
    void setGamma(float val) CV_OVERRIDE { gamma = val; }

    void write(FileStorage& fs) const CV_OVERRIDE
    {
        writeFormat(fs);
        fs << "name" << name
           << "gamma" << gamma;
    }

    void read(const FileNode& fn) CV_OVERRIDE
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        gamma = fn["gamma"];
    }

protected:
    String name;
    float gamma;
};

Ptr<Tonemap> createTonemap(float gamma)
{
    return makePtr<TonemapImpl>(gamma);
}

class TonemapDragoImpl CV_FINAL : public TonemapDrago
{
public:
    TonemapDragoImpl(float _gamma, float _saturation, float _bias) :
        name("TonemapDrago"),
        gamma(_gamma),
        saturation(_saturation),
        bias(_bias)
    {
    }

    void process(InputArray _src, OutputArray _dst) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        Mat src = _src.getMat();
        CV_Assert(!src.empty());
        _dst.create(src.size(), CV_32FC3);
        Mat img = _dst.getMat();

        Ptr<Tonemap> linear = createTonemap(1.0f);
        linear->process(src, img);

        Mat gray_img;
        cvtColor(img, gray_img, COLOR_RGB2GRAY);
        Mat log_img;
        log_(gray_img, log_img);
        float mean = expf(static_cast<float>(sum(log_img)[0]) / log_img.total());
        gray_img /= mean;
        log_img.release();

        double max;
        minMaxLoc(gray_img, NULL, &max);
        CV_Assert(max > 0);

        Mat map;
        log(gray_img + 1.0f, map);
        Mat div;
        pow(gray_img / static_cast<float>(max), logf(bias) / logf(0.5f), div);
        log(2.0f + 8.0f * div, div);
        map = map.mul(1.0f / div);
        div.release();

        mapLuminance(img, img, gray_img, map, saturation);

        linear->setGamma(gamma);
        linear->process(img, img);
    }

    float getGamma() const CV_OVERRIDE { return gamma; }
    void setGamma(float val) CV_OVERRIDE { gamma = val; }

    float getSaturation() const CV_OVERRIDE { return saturation; }
    void setSaturation(float val) CV_OVERRIDE { saturation = val; }

    float getBias() const CV_OVERRIDE { return bias; }
    void setBias(float val) CV_OVERRIDE { bias = val; }

    void write(FileStorage& fs) const CV_OVERRIDE
    {
        writeFormat(fs);
        fs << "name" << name
           << "gamma" << gamma
           << "bias" << bias
           << "saturation" << saturation;
    }

    void read(const FileNode& fn) CV_OVERRIDE
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        gamma = fn["gamma"];
        bias = fn["bias"];
        saturation = fn["saturation"];
    }

protected:
    String name;
    float gamma, saturation, bias;
};

Ptr<TonemapDrago> createTonemapDrago(float gamma, float saturation, float bias)
{
    return makePtr<TonemapDragoImpl>(gamma, saturation, bias);
}

class TonemapReinhardImpl CV_FINAL : public TonemapReinhard
{
public:
    TonemapReinhardImpl(float _gamma, float _intensity, float _light_adapt, float _color_adapt) :
        name("TonemapReinhard"),
        gamma(_gamma),
        intensity(_intensity),
        light_adapt(_light_adapt),
        color_adapt(_color_adapt)
    {
    }

    void process(InputArray _src, OutputArray _dst) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        Mat src = _src.getMat();
        CV_Assert(!src.empty());
        _dst.create(src.size(), CV_32FC3);
        Mat img = _dst.getMat();
        Ptr<Tonemap> linear = createTonemap(1.0f);
        linear->process(src, img);

        Mat gray_img;
        cvtColor(img, gray_img, COLOR_RGB2GRAY);
        Mat log_img;
        log_(gray_img, log_img);

        float log_mean = static_cast<float>(sum(log_img)[0] / log_img.total());
        double log_min, log_max;
        minMaxLoc(log_img, &log_min, &log_max);
        log_img.release();

        double key = static_cast<float>((log_max - log_mean) / (log_max - log_min));
        float map_key = 0.3f + 0.7f * pow(static_cast<float>(key), 1.4f);
        intensity = exp(-intensity);
        Scalar chan_mean = mean(img);
        float gray_mean = static_cast<float>(mean(gray_img)[0]);

        std::vector<Mat> channels(3);
        split(img, channels);

        for(int i = 0; i < 3; i++) {
            float global = color_adapt * static_cast<float>(chan_mean[i]) + (1.0f - color_adapt) * gray_mean;
            Mat adapt = color_adapt * channels[i] + (1.0f - color_adapt) * gray_img;
            adapt = light_adapt * adapt + (1.0f - light_adapt) * global;
            pow(intensity * adapt, map_key, adapt);
            channels[i] = channels[i].mul(1.0f / (adapt + channels[i]));
        }
        gray_img.release();
        merge(channels, img);

        linear->setGamma(gamma);
        linear->process(img, img);
    }

    float getGamma() const CV_OVERRIDE { return gamma; }
    void setGamma(float val) CV_OVERRIDE { gamma = val; }

    float getIntensity() const CV_OVERRIDE { return intensity; }
    void setIntensity(float val) CV_OVERRIDE { intensity = val; }

    float getLightAdaptation() const CV_OVERRIDE { return light_adapt; }
    void setLightAdaptation(float val) CV_OVERRIDE { light_adapt = val; }

    float getColorAdaptation() const CV_OVERRIDE { return color_adapt; }
    void setColorAdaptation(float val) CV_OVERRIDE { color_adapt = val; }

    void write(FileStorage& fs) const CV_OVERRIDE
    {
        writeFormat(fs);
        fs << "name" << name
           << "gamma" << gamma
           << "intensity" << intensity
           << "light_adapt" << light_adapt
           << "color_adapt" << color_adapt;
    }

    void read(const FileNode& fn) CV_OVERRIDE
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        gamma = fn["gamma"];
        intensity = fn["intensity"];
        light_adapt = fn["light_adapt"];
        color_adapt = fn["color_adapt"];
    }

protected:
    String name;
    float gamma, intensity, light_adapt, color_adapt;
};

Ptr<TonemapReinhard> createTonemapReinhard(float gamma, float contrast, float sigma_color, float sigma_space)
{
    return makePtr<TonemapReinhardImpl>(gamma, contrast, sigma_color, sigma_space);
}

class TonemapMantiukImpl CV_FINAL : public TonemapMantiuk
{
public:
    TonemapMantiukImpl(float _gamma, float _scale, float _saturation) :
        name("TonemapMantiuk"),
        gamma(_gamma),
        scale(_scale),
        saturation(_saturation)
    {
    }

    void process(InputArray _src, OutputArray _dst) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        Mat src = _src.getMat();
        CV_Assert(!src.empty());
        _dst.create(src.size(), CV_32FC3);
        Mat img = _dst.getMat();
        Ptr<Tonemap> linear = createTonemap(1.0f);
        linear->process(src, img);

        Mat gray_img;
        cvtColor(img, gray_img, COLOR_RGB2GRAY);
        Mat log_img;
        log_(gray_img, log_img);

        std::vector<Mat> x_contrast, y_contrast;
        getContrast(log_img, x_contrast, y_contrast);

        for(size_t i = 0; i < x_contrast.size(); i++) {
            mapContrast(x_contrast[i]);
            mapContrast(y_contrast[i]);
        }

        Mat right(src.size(), CV_32F);
        calculateSum(x_contrast, y_contrast, right);

        Mat p, r, product, x = log_img;
        calculateProduct(x, r);
        r = right - r;
        r.copyTo(p);

        const float target_error = 1e-3f;
        float target_norm = static_cast<float>(right.dot(right)) * powf(target_error, 2.0f);
        int max_iterations = 100;
        float rr = static_cast<float>(r.dot(r));

        for(int i = 0; i < max_iterations; i++)
        {
            calculateProduct(p, product);
            double dprod = p.dot(product);
            CV_Assert(fabs(dprod) > 0);
            float alpha = rr / static_cast<float>(dprod);

            r -= alpha * product;
            x += alpha * p;

            float new_rr = static_cast<float>(r.dot(r));
            CV_Assert(fabs(rr) > 0);
            p = r + (new_rr / rr) * p;
            rr = new_rr;

            if(rr < target_norm) {
                break;
            }
        }
        exp(x, x);
        mapLuminance(img, img, gray_img, x, saturation);

        linear = createTonemap(gamma);
        linear->process(img, img);
    }

    float getGamma() const CV_OVERRIDE { return gamma; }
    void setGamma(float val) CV_OVERRIDE { gamma = val; }

    float getScale() const CV_OVERRIDE { return scale; }
    void setScale(float val) CV_OVERRIDE { scale = val; }

    float getSaturation() const CV_OVERRIDE { return saturation; }
    void setSaturation(float val) CV_OVERRIDE { saturation = val; }

    void write(FileStorage& fs) const CV_OVERRIDE
    {
        writeFormat(fs);
        fs << "name" << name
           << "gamma" << gamma
           << "scale" << scale
           << "saturation" << saturation;
    }

    void read(const FileNode& fn) CV_OVERRIDE
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        gamma = fn["gamma"];
        scale = fn["scale"];
        saturation = fn["saturation"];
    }

protected:
    String name;
    float gamma, scale, saturation;

    void signedPow(Mat src, float power, Mat& dst)
    {
        Mat sign = (src > 0);
        sign.convertTo(sign, CV_32F, 1.0f/255.0f);
        sign = sign * 2.0f - 1.0f;
        pow(abs(src), power, dst);
        dst = dst.mul(sign);
    }

    void mapContrast(Mat& contrast)
    {
        const float response_power = 0.4185f;
        signedPow(contrast, response_power, contrast);
        contrast *= scale;
        signedPow(contrast, 1.0f / response_power, contrast);
    }

    void getGradient(Mat src, Mat& dst, int pos)
    {
        dst = Mat::zeros(src.size(), CV_32F);
        Mat a, b;
        Mat grad = src.colRange(1, src.cols) - src.colRange(0, src.cols - 1);
        grad.copyTo(dst.colRange(pos, src.cols + pos - 1));
        if(pos == 1) {
            src.col(0).copyTo(dst.col(0));
        }
    }

    void getContrast(Mat src, std::vector<Mat>& x_contrast, std::vector<Mat>& y_contrast)
    {
        int levels = static_cast<int>(logf(static_cast<float>(min(src.rows, src.cols))) / logf(2.0f));
        x_contrast.resize(levels);
        y_contrast.resize(levels);

        Mat layer;
        src.copyTo(layer);
        for(int i = 0; i < levels; i++) {
            getGradient(layer, x_contrast[i], 0);
            getGradient(layer.t(), y_contrast[i], 0);
            resize(layer, layer, Size(layer.cols / 2, layer.rows / 2), 0, 0, INTER_LINEAR);
        }
    }

    void calculateSum(std::vector<Mat>& x_contrast, std::vector<Mat>& y_contrast, Mat& sum)
    {
        if (x_contrast.empty())
            return;
        const int last = (int)x_contrast.size() - 1;
        sum = Mat::zeros(x_contrast[last].size(), CV_32F);
        for(int i = last; i >= 0; i--)
        {
            Mat grad_x, grad_y;
            getGradient(x_contrast[i], grad_x, 1);
            getGradient(y_contrast[i], grad_y, 1);
            resize(sum, sum, x_contrast[i].size(), 0, 0, INTER_LINEAR);
            sum += grad_x + grad_y.t();
        }
    }

    void calculateProduct(Mat src, Mat& dst)
    {
        std::vector<Mat> x_contrast, y_contrast;
        getContrast(src, x_contrast, y_contrast);
        calculateSum(x_contrast, y_contrast, dst);
    }
};

Ptr<TonemapMantiuk> createTonemapMantiuk(float gamma, float scale, float saturation)
{
    return makePtr<TonemapMantiukImpl>(gamma, scale, saturation);
}

}
