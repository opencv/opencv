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

static void applyLinearNormAndGamma(const Mat& src, Mat& dst, float gamma)
{
    double min, max;
    minMaxLoc(src, &min, &max);
    const float fmin = static_cast<float>(min);
    const float fmax = static_cast<float>(max);
    if (max - min > DBL_EPSILON)
    {
        const float invRange = 1.0f / (fmax - fmin);
        const float invGamma = 1.0f / gamma;
        const int cols3 = src.cols * src.channels();
        parallel_for_(Range(0, src.rows), [&](const Range& range) {
            for (int r = range.start; r < range.end; ++r)
            {
                const float* srow = src.ptr<float>(r);
                float* drow = dst.ptr<float>(r);
                if (gamma == 1.0f)
                {
                    for (int k = 0; k < cols3; ++k)
                        drow[k] = (srow[k] - fmin) * invRange;
                }
                else
                {
                    for (int k = 0; k < cols3; ++k)
                    {
                        const float v = (srow[k] - fmin) * invRange;
                        drow[k] = std::pow(v, invGamma);
                    }
                }
            }
        });
    }
    else
    {
        if (src.data != dst.data)
            src.copyTo(dst);
        if (gamma != 1.0f)
            pow(dst, 1.0f / gamma, dst);
    }
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
        CV_Assert(!src.empty());
        CV_Assert(_src.dims() == 2 && _src.type() == CV_32FC3);
        _dst.create(src.size(), CV_32FC3);
        Mat dst = _dst.getMat();

        applyLinearNormAndGamma(src, dst, gamma);
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

        applyLinearNormAndGamma(src, img, 1.0f);

        Mat gray_img;
        cvtColor(img, gray_img, COLOR_RGB2GRAY);

        const int rows = gray_img.rows;
        const int cols = gray_img.cols;
        std::vector<double> rowLogSums(rows, 0.0);
        std::vector<float> rowMaxG(rows, -std::numeric_limits<float>::infinity());
        parallel_for_(Range(0, rows), [&](const Range& range) {
            for (int r = range.start; r < range.end; ++r)
            {
                const float* grow = gray_img.ptr<float>(r);
                double lsum = 0.0;
                float rmax = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < cols; ++c)
                {
                    const float g = grow[c];
                    lsum += std::log(std::max(g, 1e-4f));
                    if (g > rmax)
                        rmax = g;
                }
                rowLogSums[r] = lsum;
                rowMaxG[r] = rmax;
            }
        });
        double totalLogSum = 0.0;
        float globalMaxG = -std::numeric_limits<float>::infinity();
        for (int r = 0; r < rows; ++r)
        {
            totalLogSum += rowLogSums[r];
            if (rowMaxG[r] > globalMaxG)
                globalMaxG = rowMaxG[r];
        }
        const float mean = expf(static_cast<float>(totalLogSum / gray_img.total()));
        const float invMean = 1.0f / mean;
        const float maxLum = globalMaxG * invMean;
        CV_Assert(maxLum > 0);
        const float invMaxLum = 1.0f / maxLum;
        const float biasExp = logf(bias) / logf(0.5f);
        const float sat = saturation;

        parallel_for_(Range(0, rows), [&](const Range& range) {
            for (int r = range.start; r < range.end; ++r)
            {
                Vec3f* irow = img.ptr<Vec3f>(r);
                const float* grow = gray_img.ptr<float>(r);
                for (int c = 0; c < cols; ++c)
                {
                    const float lum = grow[c] * invMean;
                    const float newLum = std::log(lum + 1.0f) /
                        std::log(2.0f + 8.0f * std::pow(lum * invMaxLum, biasExp));
                    const float invLum = (lum > 0.0f) ? (1.0f / lum) : 0.0f;
                    Vec3f& pix = irow[c];
                    if (sat == 1.0f)
                    {
                        const float s = invLum * newLum;
                        pix *= s;
                    }
                    else
                    {
                        pix[0] = std::pow(pix[0] * invLum, sat) * newLum;
                        pix[1] = std::pow(pix[1] * invLum, sat) * newLum;
                        pix[2] = std::pow(pix[2] * invLum, sat) * newLum;
                    }
                }
            }
        });
        gray_img.release();

        applyLinearNormAndGamma(img, img, gamma);
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
        applyLinearNormAndGamma(src, img, 1.0f);

        Mat gray_img;
        cvtColor(img, gray_img, COLOR_RGB2GRAY);

        const int rows = gray_img.rows;
        const int cols = gray_img.cols;
        std::vector<double> rowLogSum(rows, 0.0), rowGraySum(rows, 0.0);
        std::vector<double> rowC0Sum(rows, 0.0), rowC1Sum(rows, 0.0), rowC2Sum(rows, 0.0);
        std::vector<float> rowLogMin(rows, std::numeric_limits<float>::infinity());
        std::vector<float> rowLogMax(rows, -std::numeric_limits<float>::infinity());
        parallel_for_(Range(0, rows), [&](const Range& range) {
            for (int r = range.start; r < range.end; ++r)
            {
                const Vec3f* irow = img.ptr<Vec3f>(r);
                const float* grow = gray_img.ptr<float>(r);
                double lsum = 0.0, gsum = 0.0, c0 = 0.0, c1 = 0.0, c2 = 0.0;
                float lmin = std::numeric_limits<float>::infinity();
                float lmax = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < cols; ++c)
                {
                    const float g = grow[c];
                    const float lg = std::log(std::max(g, 1e-4f));
                    lsum += lg;
                    gsum += g;
                    if (lg < lmin) lmin = lg;
                    if (lg > lmax) lmax = lg;
                    const Vec3f pix = irow[c];
                    c0 += pix[0];
                    c1 += pix[1];
                    c2 += pix[2];
                }
                rowLogSum[r] = lsum;
                rowGraySum[r] = gsum;
                rowC0Sum[r] = c0;
                rowC1Sum[r] = c1;
                rowC2Sum[r] = c2;
                rowLogMin[r] = lmin;
                rowLogMax[r] = lmax;
            }
        });
        double totalLogSum = 0.0, totalGraySum = 0.0, totalC0 = 0.0, totalC1 = 0.0, totalC2 = 0.0;
        float log_min = std::numeric_limits<float>::infinity();
        float log_max = -std::numeric_limits<float>::infinity();
        for (int r = 0; r < rows; ++r)
        {
            totalLogSum += rowLogSum[r];
            totalGraySum += rowGraySum[r];
            totalC0 += rowC0Sum[r];
            totalC1 += rowC1Sum[r];
            totalC2 += rowC2Sum[r];
            if (rowLogMin[r] < log_min) log_min = rowLogMin[r];
            if (rowLogMax[r] > log_max) log_max = rowLogMax[r];
        }

        const double totalPx = static_cast<double>(gray_img.total());
        const float log_mean = static_cast<float>(totalLogSum / totalPx);
        const float key = (log_max > log_min) ? (log_max - log_mean) / (log_max - log_min) : 1.0f;
        const float map_key = 0.3f + 0.7f * std::pow(key, 1.4f);
        const float intensity_scale = exp(-intensity);
        const float gray_mean = static_cast<float>(totalGraySum / totalPx);
        const float chan_mean[3] = {
            static_cast<float>(totalC0 / totalPx),
            static_cast<float>(totalC1 / totalPx),
            static_cast<float>(totalC2 / totalPx)
        };
        const float ca = color_adapt;
        const float invCa = 1.0f - color_adapt;
        const float la = light_adapt;
        const float globalTerm[3] = {
            (1.0f - la) * (ca * chan_mean[0] + invCa * gray_mean),
            (1.0f - la) * (ca * chan_mean[1] + invCa * gray_mean),
            (1.0f - la) * (ca * chan_mean[2] + invCa * gray_mean)
        };

        parallel_for_(Range(0, rows), [&](const Range& range) {
            for (int r = range.start; r < range.end; ++r)
            {
                Vec3f* irow = img.ptr<Vec3f>(r);
                const float* grow = gray_img.ptr<float>(r);
                for (int c = 0; c < cols; ++c)
                {
                    const float invCaG = invCa * grow[c];
                    Vec3f& pix = irow[c];
                    for (int k = 0; k < 3; ++k)
                    {
                        const float x = pix[k];
                        const float adapt = la * (ca * x + invCaG) + globalTerm[k];
                        const float aPow = std::pow(intensity_scale * adapt, map_key);
                        pix[k] = x * (1.0f / (aPow + x));
                    }
                }
            }
        });
        gray_img.release();

        applyLinearNormAndGamma(img, img, gamma);
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

Ptr<TonemapReinhard> createTonemapReinhard(float gamma, float intensity, float light_adapt, float color_adapt)
{
    return makePtr<TonemapReinhardImpl>(gamma, intensity, light_adapt, color_adapt);
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

        PyramidWorkspace ws;
        initWorkspace(src.size(), ws);

        getContrast(log_img, ws);

        Mat abs_scratch;
        for(size_t i = 0; i < ws.xc.size(); i++) {
            mapContrast(ws.xc[i], abs_scratch);
            mapContrast(ws.yc[i], abs_scratch);
        }
        abs_scratch.release();

        Mat right(src.size(), CV_32F);
        calculateSum(ws, right);

        Mat p(src.size(), CV_32F), r(src.size(), CV_32F), product(src.size(), CV_32F), x = log_img;
        calculateProduct(x, r, ws);
        subtract(right, r, r);
        r.copyTo(p);

        const float target_error = 1e-3f;
        float target_norm = static_cast<float>(right.dot(right)) * powf(target_error, 2.0f);
        right.release();
        int max_iterations = 100;
        float rr = static_cast<float>(r.dot(r));

        for(int i = 0; i < max_iterations; i++)
        {
            calculateProduct(p, product, ws);
            double dprod = p.dot(product);
            CV_Assert(fabs(dprod) > 0);
            float alpha = rr / static_cast<float>(dprod);

            updateResidualAndSolution(r, x, product, p, alpha);

            float new_rr = static_cast<float>(r.dot(r));
            CV_Assert(fabs(rr) > 0);
            updateSearchDirection(p, r, new_rr / rr);
            rr = new_rr;

            if(rr < target_norm) {
                break;
            }
        }
        ws.clear();
        p.release();
        r.release();
        product.release();
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

    struct PyramidWorkspace
    {
        std::vector<Mat> layer;
        std::vector<Mat> xc;
        std::vector<Mat> yc;
        std::vector<Mat> sum;

        void clear()
        {
            layer.clear();
            xc.clear();
            yc.clear();
            sum.clear();
        }
    };

    static void initWorkspace(Size sz, PyramidWorkspace& ws)
    {
        int levels = static_cast<int>(logf(static_cast<float>(min(sz.height, sz.width))) / logf(2.0f));
        if (levels <= 0)
        {
            ws.clear();
            return;
        }
        ws.layer.resize(levels);
        ws.xc.resize(levels);
        ws.yc.resize(levels);
        ws.sum.resize(levels);

        Size cur = sz;
        for (int i = 0; i < levels; ++i)
        {
            ws.xc[i].create(cur, CV_32F);
            ws.yc[i].create(cur, CV_32F);
            if (i > 0)
            {
                ws.layer[i].create(cur, CV_32F);
                ws.sum[i].create(cur, CV_32F);
            }
            cur = Size(cur.width / 2, cur.height / 2);
        }
    }

    static void updateResidualAndSolution(Mat& r, Mat& x, const Mat& product, const Mat& p, float alpha)
    {
        for (int y = 0; y < r.rows; ++y)
        {
            float* rrow = r.ptr<float>(y);
            float* xrow = x.ptr<float>(y);
            const float* prodrow = product.ptr<float>(y);
            const float* prow = p.ptr<float>(y);
            for (int c = 0; c < r.cols; ++c)
            {
                rrow[c] -= alpha * prodrow[c];
                xrow[c] += alpha * prow[c];
            }
        }
    }

    static void updateSearchDirection(Mat& p, const Mat& r, float beta)
    {
        for (int y = 0; y < p.rows; ++y)
        {
            float* prow = p.ptr<float>(y);
            const float* rrow = r.ptr<float>(y);
            for (int c = 0; c < p.cols; ++c)
            {
                prow[c] = rrow[c] + beta * prow[c];
            }
        }
    }

    void signedPow(Mat& contrast, float power, Mat& abs_scratch)
    {
        absdiff(contrast, Scalar::all(0), abs_scratch);
        pow(abs_scratch, power, abs_scratch);
        for (int y = 0; y < contrast.rows; ++y)
        {
            float* crow = contrast.ptr<float>(y);
            const float* arow = abs_scratch.ptr<float>(y);
            for (int x = 0; x < contrast.cols; ++x)
            {
                crow[x] = (crow[x] > 0.0f) ? arow[x] : -arow[x];
            }
        }
    }

    void mapContrast(Mat& contrast, Mat& abs_scratch)
    {
        const float response_power = 0.4185f;
        signedPow(contrast, response_power, abs_scratch);
        contrast *= scale;
        signedPow(contrast, 1.0f / response_power, abs_scratch);
    }

    static void computeForwardGradients(const Mat& layer, Mat& xc, Mat& yc)
    {
        const int rows = layer.rows;
        const int cols = layer.cols;
        xc.create(rows, cols, CV_32F);
        yc.create(rows, cols, CV_32F);
        for (int r = 0; r < rows; ++r)
        {
            const float* lrow = layer.ptr<float>(r);
            float* xrow = xc.ptr<float>(r);
            for (int c = 0; c < cols - 1; ++c)
                xrow[c] = lrow[c + 1] - lrow[c];
            if (cols > 0)
                xrow[cols - 1] = 0.0f;

            float* yrow = yc.ptr<float>(r);
            if (r < rows - 1)
            {
                const float* lnext = layer.ptr<float>(r + 1);
                for (int c = 0; c < cols; ++c)
                    yrow[c] = lnext[c] - lrow[c];
            }
            else
            {
                for (int c = 0; c < cols; ++c)
                    yrow[c] = 0.0f;
            }
        }
    }

    static void accumulateDivergence(const Mat& xc, const Mat& yc, Mat& sum, bool overwrite)
    {
        const int rows = xc.rows;
        const int cols = xc.cols;
        if (cols <= 0)
            return;
        for (int r = 0; r < rows; ++r)
        {
            const float* xrow = xc.ptr<float>(r);
            const float* yrow = yc.ptr<float>(r);
            const float* yprev = (r > 0) ? yc.ptr<float>(r - 1) : nullptr;
            float* srow = sum.ptr<float>(r);

            const float gx0 = xrow[0];
            const float gy0 = (r == 0) ? yrow[0] : (yrow[0] - yprev[0]);
            if (overwrite)
                srow[0] = gx0 + gy0;
            else
                srow[0] += gx0 + gy0;

            for (int c = 1; c < cols; ++c)
            {
                const float gx = xrow[c] - xrow[c - 1];
                const float gy = (r == 0) ? yrow[c] : (yrow[c] - yprev[c]);
                if (overwrite)
                    srow[c] = gx + gy;
                else
                    srow[c] += gx + gy;
            }
        }
    }

    void getContrast(const Mat& src, PyramidWorkspace& ws)
    {
        const int levels = static_cast<int>(ws.xc.size());
        for (int i = 0; i < levels; ++i)
        {
            const Mat& cur = (i == 0) ? src : ws.layer[i];
            computeForwardGradients(cur, ws.xc[i], ws.yc[i]);
            if (i + 1 < levels)
            {
                resize(cur, ws.layer[i + 1], ws.layer[i + 1].size(), 0, 0, INTER_LINEAR);
            }
        }
    }

    void calculateSum(PyramidWorkspace& ws, Mat& dst)
    {
        if (ws.xc.empty())
        {
            dst.setTo(Scalar::all(0));
            return;
        }
        const int last = static_cast<int>(ws.xc.size()) - 1;
        dst.create(ws.xc[0].size(), CV_32F);
        if (last == 0)
        {
            accumulateDivergence(ws.xc[0], ws.yc[0], dst, true);
            return;
        }

        accumulateDivergence(ws.xc[last], ws.yc[last], ws.sum[last], true);
        for (int i = last - 1; i >= 0; --i)
        {
            Mat& cur_sum = (i == 0) ? dst : ws.sum[i];
            resize(ws.sum[i + 1], cur_sum, ws.xc[i].size(), 0, 0, INTER_LINEAR);
            accumulateDivergence(ws.xc[i], ws.yc[i], cur_sum, false);
        }
    }

    void calculateProduct(const Mat& src, Mat& dst, PyramidWorkspace& ws)
    {
        getContrast(src, ws);
        calculateSum(ws, dst);
    }
};

Ptr<TonemapMantiuk> createTonemapMantiuk(float gamma, float scale, float saturation)
{
    return makePtr<TonemapMantiukImpl>(gamma, scale, saturation);
}

}
