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
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.
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
#include "opencv2/core/hal/intrin.hpp"
#include "hdr_common.hpp"

namespace cv
{
namespace
{

static void scaleHatWeights(const Mat& weights_lut, int channels, std::vector<float>& hat)
{
    const int n = weights_lut.rows;
    hat.resize((size_t)n);
    const float inv_cn = 1.0f / static_cast<float>(channels);
    for (int z = 0; z < n; z++)
        hat[z] = weights_lut.at<float>(z) * inv_cn;
}

static void extractLogResponseC1(const Mat& log_response, std::vector<float>& g0)
{
    const int n = log_response.rows;
    g0.resize((size_t)n);
    for (int z = 0; z < n; z++)
        g0[z] = log_response.at<float>(z);
}

static void extractLogResponseC3(const Mat& log_response,
                                 std::vector<float>& g0, std::vector<float>& g1, std::vector<float>& g2)
{
    const int n = log_response.rows;
    g0.resize((size_t)n);
    g1.resize((size_t)n);
    g2.resize((size_t)n);
    for (int z = 0; z < n; z++)
    {
        const Vec3f v = log_response.at<Vec3f>(z);
        g0[z] = v[0];
        g1[z] = v[1];
        g2[z] = v[2];
    }
}

#if CV_SIMD || CV_SIMD_SCALABLE
static inline v_float32 debevecTriangleHat(const v_int32& idx, float maxVal, float inv_cn)
{
    const v_float32 z = v_cvt_f32(idx);
    v_float32 h = v_min(z, v_sub(vx_setall_f32(maxVal), z));
    h = v_max(h, vx_setall_f32(1e-6f));
    return v_mul(h, vx_setall_f32(inv_cn));
}

static inline void debevecAccQuad(const v_uint32& ib, const v_uint32& ig, const v_uint32& ir,
                                  const float* g0, const float* g1, const float* g2,
                                  const v_float32& ln, float maxVal, float inv_cn,
                                  float* p0, float* p1, float* p2, float* pw)
{
    const v_int32 sb = v_reinterpret_as_s32(ib);
    const v_int32 sg = v_reinterpret_as_s32(ig);
    const v_int32 sr = v_reinterpret_as_s32(ir);
    const v_float32 w = v_add(v_add(debevecTriangleHat(sb, maxVal, inv_cn),
                                    debevecTriangleHat(sg, maxVal, inv_cn)),
                              debevecTriangleHat(sr, maxVal, inv_cn));
    v_store(p0, v_fma(w, v_sub(v_lut(g0, sb), ln), vx_load(p0)));
    v_store(p1, v_fma(w, v_sub(v_lut(g1, sg), ln), vx_load(p1)));
    v_store(p2, v_fma(w, v_sub(v_lut(g2, sr), ln), vx_load(p2)));
    v_store(pw, v_add(vx_load(pw), w));
}

static inline void debevecAccQuadC1(const v_uint32& iz, const float* g0,
                                    const v_float32& ln, float maxVal, float inv_cn,
                                    float* p0, float* pw)
{
    const v_int32 sz = v_reinterpret_as_s32(iz);
    const v_float32 w = debevecTriangleHat(sz, maxVal, inv_cn);
    v_store(p0, v_fma(w, v_sub(v_lut(g0, sz), ln), vx_load(p0)));
    v_store(pw, v_add(vx_load(pw), w));
}
#endif

template<typename IdxT>
static void debevecAccumulateC1(const Mat& img, const float* hat, const float* g0, float ln_t,
                                Mat& s0, Mat& wsum)
{
    const int cols = img.cols;
#if CV_SIMD || CV_SIMD_SCALABLE
    const float maxVal = (sizeof(IdxT) == 1) ? 255.f : 65535.f;
    const float inv_cn = 1.0f;
#endif
    parallel_for_(Range(0, img.rows), [&](const Range& range)
    {
        for (int y = range.start; y < range.end; y++)
        {
            const IdxT* src = img.ptr<IdxT>(y);
            float* p0 = s0.ptr<float>(y);
            float* pw = wsum.ptr<float>(y);
            int x = 0;
#if CV_SIMD || CV_SIMD_SCALABLE
            const v_float32 ln = vx_setall_f32(ln_t);
            const int n32 = VTraits<v_float32>::vlanes();
            if (sizeof(IdxT) == 1)
            {
                for (; x <= cols - n32; x += n32)
                {
                    const v_uint32 q = vx_load_expand_q(reinterpret_cast<const uchar*>(src + x));
                    debevecAccQuadC1(q, g0, ln, maxVal, inv_cn, p0 + x, pw + x);
                }
            }
            else
            {
                for (; x <= cols - n32; x += n32)
                {
                    const v_uint32 q = vx_load_expand(reinterpret_cast<const ushort*>(src + x));
                    debevecAccQuadC1(q, g0, ln, maxVal, inv_cn, p0 + x, pw + x);
                }
            }
#endif
            for (; x < cols; x++)
            {
                const IdxT z = src[x];
                const float w = hat[z];
                p0[x] += w * (g0[z] - ln_t);
                pw[x] += w;
            }
        }
    });
}

template<typename IdxT>
static void debevecAccumulateC3(const Mat& img, const float* hat,
                                const float* g0, const float* g1, const float* g2,
                                float ln_t, Mat& s0, Mat& s1, Mat& s2, Mat& wsum)
{
    const int cols = img.cols;
#if CV_SIMD || CV_SIMD_SCALABLE
    const float maxVal = (sizeof(IdxT) == 1) ? 255.f : 65535.f;
    const float inv_cn = 1.0f / 3.0f;
#endif
    parallel_for_(Range(0, img.rows), [&](const Range& range)
    {
        for (int y = range.start; y < range.end; y++)
        {
            const IdxT* src = img.ptr<IdxT>(y);
            float* p0 = s0.ptr<float>(y);
            float* p1 = s1.ptr<float>(y);
            float* p2 = s2.ptr<float>(y);
            float* pw = wsum.ptr<float>(y);
            int x = 0;
#if CV_SIMD || CV_SIMD_SCALABLE
            const v_float32 ln = vx_setall_f32(ln_t);
            if (sizeof(IdxT) == 1)
            {
                const int n8 = VTraits<v_uint8>::vlanes();
                for (; x <= cols - n8; x += n8, src += n8 * 3)
                {
                    v_uint8 vb, vg, vr;
                    v_load_deinterleave(reinterpret_cast<const uchar*>(src), vb, vg, vr);
                    v_uint16 b0, b1, g0u, g1u, r0u, r1u;
                    v_expand(vb, b0, b1);
                    v_expand(vg, g0u, g1u);
                    v_expand(vr, r0u, r1u);
                    v_uint32 qb0, qb1, qb2, qb3, qg0, qg1, qg2, qg3, qr0, qr1, qr2, qr3;
                    v_expand(b0, qb0, qb1);
                    v_expand(b1, qb2, qb3);
                    v_expand(g0u, qg0, qg1);
                    v_expand(g1u, qg2, qg3);
                    v_expand(r0u, qr0, qr1);
                    v_expand(r1u, qr2, qr3);
                    const int n32 = VTraits<v_float32>::vlanes();
                    debevecAccQuad(qb0, qg0, qr0, g0, g1, g2, ln, maxVal, inv_cn, p0 + x, p1 + x, p2 + x, pw + x);
                    debevecAccQuad(qb1, qg1, qr1, g0, g1, g2, ln, maxVal, inv_cn,
                                   p0 + x + n32, p1 + x + n32, p2 + x + n32, pw + x + n32);
                    debevecAccQuad(qb2, qg2, qr2, g0, g1, g2, ln, maxVal, inv_cn,
                                   p0 + x + 2 * n32, p1 + x + 2 * n32, p2 + x + 2 * n32, pw + x + 2 * n32);
                    debevecAccQuad(qb3, qg3, qr3, g0, g1, g2, ln, maxVal, inv_cn,
                                   p0 + x + 3 * n32, p1 + x + 3 * n32, p2 + x + 3 * n32, pw + x + 3 * n32);
                }
            }
            else
            {
                const int n16 = VTraits<v_uint16>::vlanes();
                for (; x <= cols - n16; x += n16, src += n16 * 3)
                {
                    v_uint16 vb, vg, vr;
                    v_load_deinterleave(reinterpret_cast<const ushort*>(src), vb, vg, vr);
                    v_uint32 qb0, qb1, qg0, qg1, qr0, qr1;
                    v_expand(vb, qb0, qb1);
                    v_expand(vg, qg0, qg1);
                    v_expand(vr, qr0, qr1);
                    const int n32 = VTraits<v_float32>::vlanes();
                    debevecAccQuad(qb0, qg0, qr0, g0, g1, g2, ln, maxVal, inv_cn, p0 + x, p1 + x, p2 + x, pw + x);
                    debevecAccQuad(qb1, qg1, qr1, g0, g1, g2, ln, maxVal, inv_cn,
                                   p0 + x + n32, p1 + x + n32, p2 + x + n32, pw + x + n32);
                }
            }
            src = img.ptr<IdxT>(y) + x * 3;
#endif
            for (; x < cols; x++, src += 3)
            {
                const IdxT b = src[0], g = src[1], r = src[2];
                const float w = hat[b] + hat[g] + hat[r];
                p0[x] += w * (g0[b] - ln_t);
                p1[x] += w * (g1[g] - ln_t);
                p2[x] += w * (g2[r] - ln_t);
                pw[x] += w;
            }
        }
    });
}

static void debevecFinalizeC1(const Mat& s0, const Mat& wsum, Mat& dst)
{
    const int cols = dst.cols;
    parallel_for_(Range(0, dst.rows), [&](const Range& range)
    {
        for (int y = range.start; y < range.end; y++)
        {
            const float* p0 = s0.ptr<float>(y);
            const float* pw = wsum.ptr<float>(y);
            float* d = dst.ptr<float>(y);
            int x = 0;
#if CV_SIMD || CV_SIMD_SCALABLE
            const int n32 = VTraits<v_float32>::vlanes();
            const v_float32 one = vx_setall_f32(1.0f);
            for (; x <= cols - n32; x += n32)
            {
                const v_float32 inv = v_div(one, vx_load(pw + x));
                v_store(d + x, v_exp(v_mul(vx_load(p0 + x), inv)));
            }
#endif
            for (; x < cols; x++)
                d[x] = std::exp(p0[x] * (1.0f / pw[x]));
        }
    });
}

static void debevecFinalizeC3(const Mat& s0, const Mat& s1, const Mat& s2, const Mat& wsum, Mat& dst)
{
    const int cols = dst.cols;
    parallel_for_(Range(0, dst.rows), [&](const Range& range)
    {
        for (int y = range.start; y < range.end; y++)
        {
            const float* p0 = s0.ptr<float>(y);
            const float* p1 = s1.ptr<float>(y);
            const float* p2 = s2.ptr<float>(y);
            const float* pw = wsum.ptr<float>(y);
            float* d = dst.ptr<float>(y);
            int x = 0;
#if CV_SIMD || CV_SIMD_SCALABLE
            const int n32 = VTraits<v_float32>::vlanes();
            const v_float32 one = vx_setall_f32(1.0f);
            for (; x <= cols - n32; x += n32)
            {
                const v_float32 inv = v_div(one, vx_load(pw + x));
                v_store_interleave(d + x * 3,
                                   v_exp(v_mul(vx_load(p0 + x), inv)),
                                   v_exp(v_mul(vx_load(p1 + x), inv)),
                                   v_exp(v_mul(vx_load(p2 + x), inv)));
            }
#endif
            for (; x < cols; x++)
            {
                const float inv = 1.0f / pw[x];
                d[x * 3 + 0] = std::exp(p0[x] * inv);
                d[x * 3 + 1] = std::exp(p1[x] * inv);
                d[x * 3 + 2] = std::exp(p2[x] * inv);
            }
        }
    });
}

} // namespace

class MergeDebevecImpl CV_FINAL : public MergeDebevec
{
public:
    MergeDebevecImpl() :
        name("MergeDebevec"),
        weights(triangleWeights())
    {
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray _times, InputArray input_response) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        std::vector<Mat> images;
        src.getMatVector(images);
        Mat times = _times.getMat();

        CV_Assert(images.size() == times.total());
        checkImageDimensions(images);
        int depth = images[0].depth();
        CV_Assert(depth == CV_8U || depth == CV_16U || depth == CV_32F);

        int channels = images[0].channels();
        CV_Assert(channels == 1 || channels == 3);
        Size size = images[0].size();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);

        const bool use16bitLUT = (depth == CV_16U || depth == CV_32F);
        const int lutLength = use16bitLUT ? 65536 : LDR_SIZE;

        std::vector<Mat> lutImages(images.size());
        if (depth == CV_8U || depth == CV_16U)
        {
            lutImages = images;
        }
        else
        {
            const double scale = static_cast<double>(lutLength - 1);
            for (size_t i = 0; i < images.size(); ++i)
            {
                Mat clipped;
                cv::max(images[i], 0.0, clipped);
                cv::min(clipped, 1.0, clipped);
                clipped.convertTo(lutImages[i], CV_16U, scale);
            }
        }

        dst.create(images[0].size(), CV_32FCC);
        Mat result = dst.getMat();

        Mat response = input_response.getMat();

        if(response.empty()) {
            response = linearResponse(channels, lutLength);
            if (channels == 3)
                response.at<Vec3f>(0) = response.at<Vec3f>(1);
            else if (channels == 1)
                response.at<float>(0) = response.at<float>(1);
        }

        Mat log_response;
        log(response, log_response);
        CV_Assert(log_response.rows == lutLength && log_response.cols == 1 &&
                  log_response.channels() == channels);

        Mat exp_values(times.clone());
        log(exp_values, exp_values);

        Mat weights_lut = use16bitLUT
            ? triangleWeights(lutLength)
            : weights;

        const int idxDepth = lutImages[0].depth();
        CV_Assert(idxDepth == CV_8U || idxDepth == CV_16U);

        std::vector<float> hat;
        scaleHatWeights(weights_lut, channels, hat);

        Mat weight_sum = Mat::zeros(size, CV_32F);
        if (channels == 3)
        {
            std::vector<float> g0, g1, g2;
            extractLogResponseC3(log_response, g0, g1, g2);
            Mat s0 = Mat::zeros(size, CV_32F);
            Mat s1 = Mat::zeros(size, CV_32F);
            Mat s2 = Mat::zeros(size, CV_32F);
            for (size_t i = 0; i < lutImages.size(); i++)
            {
                const float ln_t = exp_values.at<float>((int)i);
                if (idxDepth == CV_8U)
                    debevecAccumulateC3<uchar>(lutImages[i], hat.data(), g0.data(), g1.data(), g2.data(),
                                               ln_t, s0, s1, s2, weight_sum);
                else
                    debevecAccumulateC3<ushort>(lutImages[i], hat.data(), g0.data(), g1.data(), g2.data(),
                                                ln_t, s0, s1, s2, weight_sum);
            }
            debevecFinalizeC3(s0, s1, s2, weight_sum, result);
        }
        else
        {
            std::vector<float> g0;
            extractLogResponseC1(log_response, g0);
            Mat s0 = Mat::zeros(size, CV_32F);
            for (size_t i = 0; i < lutImages.size(); i++)
            {
                const float ln_t = exp_values.at<float>((int)i);
                if (idxDepth == CV_8U)
                    debevecAccumulateC1<uchar>(lutImages[i], hat.data(), g0.data(), ln_t, s0, weight_sum);
                else
                    debevecAccumulateC1<ushort>(lutImages[i], hat.data(), g0.data(), ln_t, s0, weight_sum);
            }
            debevecFinalizeC1(s0, weight_sum, result);
        }
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray times) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        process(src, dst, times, Mat());
    }

protected:
    String name;
    Mat weights;
};

Ptr<MergeDebevec> createMergeDebevec()
{
    return makePtr<MergeDebevecImpl>();
}

class MergeMertensImpl CV_FINAL : public MergeMertens
{
public:
    MergeMertensImpl(float _wcon, float _wsat, float _wexp) :
        name("MergeMertens"),
        wcon(_wcon),
        wsat(_wsat),
        wexp(_wexp)
    {
    }

    void process(InputArrayOfArrays src, OutputArrayOfArrays dst, InputArray, InputArray) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        process(src, dst);
    }

    void process(InputArrayOfArrays src, OutputArray dst) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        std::vector<Mat> images;
        src.getMatVector(images);
        checkImageDimensions(images);

        int channels = images[0].channels();
        CV_Assert(channels == 1 || channels == 3);
        Size size = images[0].size();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);

        std::vector<Mat> weights(images.size());
        Mat weight_sum = Mat::zeros(size, CV_32F);
        Mutex weight_sum_mutex;

        parallel_for_(Range(0, static_cast<int>(images.size())), [&](const Range& range) {
            for(int i = range.start; i < range.end; i++) {
                Mat img, gray, contrast, saturation, wellexp;
                std::vector<Mat> splitted(channels);

                images[i].convertTo(img, CV_32F, 1.0f/255.0f);
                if(channels == 3) {
                    cvtColor(img, gray, COLOR_RGB2GRAY);
                } else {
                    img.copyTo(gray);
                }
                images[i] = img;
                split(img, splitted);

                Laplacian(gray, contrast, CV_32F);
                contrast = abs(contrast);

                Mat mean = Mat::zeros(size, CV_32F);
                for(int c = 0; c < channels; c++) {
                    mean += splitted[c];
                }
                mean /= channels;

                saturation = Mat::zeros(size, CV_32F);
                for(int c = 0; c < channels;  c++) {
                    Mat deviation = splitted[c] - mean;
                    pow(deviation, 2.0f, deviation);
                    saturation += deviation;
                }
                sqrt(saturation, saturation);

                wellexp = Mat::ones(size, CV_32F);
                for(int c = 0; c < channels; c++) {
                    Mat expo = splitted[c] - 0.5f;
                    pow(expo, 2.0f, expo);
                    expo = -expo / 0.08f;
                    exp(expo, expo);
                    wellexp = wellexp.mul(expo);
                }

                pow(contrast, wcon, contrast);
                pow(saturation, wsat, saturation);
                pow(wellexp, wexp, wellexp);

                weights[i] = contrast;
                if(channels == 3) {
                    weights[i] = weights[i].mul(saturation);
                }
                weights[i] = weights[i].mul(wellexp) + 1e-12f;

                AutoLock lock(weight_sum_mutex);
                weight_sum += weights[i];
            }
        });

        int maxlevel = static_cast<int>(logf(static_cast<float>(min(size.width, size.height))) / logf(2.0f));
        std::vector<Mat> res_pyr(maxlevel + 1);
        std::vector<Mutex> res_pyr_mutexes(maxlevel + 1);

        parallel_for_(Range(0, static_cast<int>(images.size())), [&](const Range& range) {
            for(int i = range.start; i < range.end; i++) {
                weights[i] /= weight_sum;

                std::vector<Mat> img_pyr, weight_pyr;
                buildPyramid(images[i], img_pyr, maxlevel);
                buildPyramid(weights[i], weight_pyr, maxlevel);

                for(int lvl = 0; lvl < maxlevel; lvl++) {
                    Mat up;
                    pyrUp(img_pyr[lvl + 1], up, img_pyr[lvl].size());
                    img_pyr[lvl] -= up;
                }
                for(int lvl = 0; lvl <= maxlevel; lvl++) {
                    std::vector<Mat> splitted(channels);
                    split(img_pyr[lvl], splitted);
                    for(int c = 0; c < channels; c++) {
                        splitted[c] = splitted[c].mul(weight_pyr[lvl]);
                    }
                    merge(splitted, img_pyr[lvl]);

                    AutoLock lock(res_pyr_mutexes[lvl]);
                    if(res_pyr[lvl].empty()) {
                        res_pyr[lvl] = img_pyr[lvl];
                    } else {
                        res_pyr[lvl] += img_pyr[lvl];
                    }
                }
            }
        });
        for(int lvl = maxlevel; lvl > 0; lvl--) {
            Mat up;
            pyrUp(res_pyr[lvl], up, res_pyr[lvl - 1].size());
            res_pyr[lvl - 1] += up;
        }
        dst.create(size, CV_32FCC);
        res_pyr[0].copyTo(dst);
    }

    float getContrastWeight() const CV_OVERRIDE { return wcon; }
    void setContrastWeight(float val) CV_OVERRIDE { wcon = val; }

    float getSaturationWeight() const CV_OVERRIDE { return wsat; }
    void setSaturationWeight(float val) CV_OVERRIDE { wsat = val; }

    float getExposureWeight() const CV_OVERRIDE { return wexp; }
    void setExposureWeight(float val) CV_OVERRIDE { wexp = val; }

    void write(FileStorage& fs) const CV_OVERRIDE
    {
        writeFormat(fs);
        fs << "name" << name
           << "contrast_weight" << wcon
           << "saturation_weight" << wsat
           << "exposure_weight" << wexp;
    }

    void read(const FileNode& fn) CV_OVERRIDE
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        wcon = fn["contrast_weight"];
        wsat = fn["saturation_weight"];
        wexp = fn["exposure_weight"];
    }

protected:
    String name;
    float wcon, wsat, wexp;
};

Ptr<MergeMertens> createMergeMertens(float wcon, float wsat, float wexp)
{
    return makePtr<MergeMertensImpl>(wcon, wsat, wexp);
}

class MergeRobertsonImpl CV_FINAL : public MergeRobertson
{
public:
    MergeRobertsonImpl() :
        name("MergeRobertson"),
        weight(RobertsonWeights())
    {
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray _times, InputArray input_response) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        std::vector<Mat> images;
        src.getMatVector(images);
        Mat times = _times.getMat();

        CV_Assert(images.size() == times.total());
        checkImageDimensions(images);
        int depth = images[0].depth();
        CV_Assert(depth == CV_8U || depth == CV_16U || depth == CV_32F);

        int channels = images[0].channels();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);

        const bool use16bitLUT = (depth == CV_16U || depth == CV_32F);
        const int lutLength = use16bitLUT ? 65536 : LDR_SIZE;

        // Build LUT index images (see MergeDebevecImpl for details).
        std::vector<Mat> lutImages(images.size());
        if (depth == CV_8U || depth == CV_16U)
        {
            lutImages = images;
        }
        else // CV_32F
        {
            const double scale = static_cast<double>(lutLength - 1);
            for (size_t i = 0; i < images.size(); ++i)
            {
                images[i].convertTo(lutImages[i], CV_16U, scale);
            }
        }

        dst.create(images[0].size(), CV_32FCC);
        Mat result = dst.getMat();

        Mat response = input_response.getMat();
        if(response.empty()) {
            float middle = static_cast<float>(lutLength) / 2.0f;
            response = linearResponse(channels, lutLength) / middle;
        }
        CV_Assert(response.rows == lutLength && response.cols == 1 &&
                  response.channels() == channels);

        result = Mat::zeros(images[0].size(), CV_32FCC);
        Mat wsum = Mat::zeros(images[0].size(), CV_32FCC);

        Mat weight_lut = use16bitLUT
            ? RobertsonWeights(lutLength)
            : weight;

        for(size_t i = 0; i < images.size(); i++) {
            Mat im, w;
            LUT(lutImages[i], weight_lut, w);
            LUT(lutImages[i], response, im);

            result += times.at<float>((int)i) * w.mul(im);
            wsum += times.at<float>((int)i) * times.at<float>((int)i) * w;
        }
        result = result.mul(1 / (wsum + Scalar::all(DBL_EPSILON)));
    }

    void process(InputArrayOfArrays src, OutputArray dst, InputArray times) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        process(src, dst, times, Mat());
    }

protected:
    String name;
    Mat weight;
};

Ptr<MergeRobertson> createMergeRobertson()
{
    return makePtr<MergeRobertsonImpl>();
}

}
