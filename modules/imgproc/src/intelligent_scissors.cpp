// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "precomp.hpp"
//#include "opencv2/imgproc/segmentation.hpp"

#include <opencv2/core/utils/logger.hpp>

#include <queue>  // std::priority_queue

namespace cv {
namespace segmentation {

namespace {

// 0 1 2
// 3 x 4
// 5 6 7
static const int neighbors[8][2] = {
    { -1, -1 },
    {  0, -1 },
    {  1, -1 },
    { -1,  0 },
    {  1,  0 },
    { -1,  1 },
    {  0,  1 },
    {  1,  1 },
};

// encoded reverse direction
static const int neighbors_encode[8] = {
    7+1, 6+1, 5+1,
    4+1,      3+1,
    2+1, 1+1, 0+1
};

#define ACOS_TABLE_SIZE 64
// acos_table[x + ACOS_TABLE_SIZE] = acos(x / ACOS_TABLE_SIZE) / CV_PI (see local_cost)
//    x = [ -ACOS_TABLE_SIZE .. ACOS_TABLE_SIZE ]
float* getAcosTable()
{
    constexpr int N = ACOS_TABLE_SIZE;
    static bool initialized = false;
    static float acos_table[2*N + 1] = { 0 };
    if (!initialized)
    {
        const float CV_PI_inv = static_cast<float>(1.0 / CV_PI);
        for (int i = -N; i <= N; i++)
        {
           acos_table[i + N] = acosf(i / (float)N) * CV_PI_inv;
        }
        initialized = true;
    }
    return acos_table;
}

} // namespace anon

struct IntelligentScissorsMB::Impl
{
    // proposed weights from the article (sum = 1.0)
    float weight_non_edge = 0.43f;
    float weight_gradient_direction = 0.43f;
    float weight_gradient_magnitude = 0.14f;

    enum EdgeFeatureMode {
        FEATURE_ZERO_CROSSING = 0,
        FEATURE_CANNY
    };
    EdgeFeatureMode edge_mode = FEATURE_ZERO_CROSSING;

    // FEATURE_ZERO_CROSSING
    float edge_gradient_magnitude_min_value = 0.0f;

    // FEATURE_CANNY
    double edge_canny_threshold1 = 10;
    double edge_canny_threshold2 = 100;
    int edge_canny_apertureSize = 3;
    bool edge_canny_L2gradient = false;


    float gradient_magnitude_threshold_max = 0.0f;  // disabled thresholding

    int sobelKernelSize = 3;  // 1 or 3
    int laplacianKernelSize = 3;  // 1 or 3

    // image features
    Mat_<Point2f> gradient_direction;  //< I: normalized laplacian x/y components
    Mat_<float> gradient_magnitude;  //< Fg: gradient cost function
    Mat_<uchar> non_edge_feature;  //< Fz: zero-crossing function

    float weight_non_edge_compute = 0.0f;

    // encoded paths map (produced by `buildMap()`)
    Mat_<uchar> optimalPathsMap;

    void resetFeatures_()
    {
        CV_TRACE_FUNCTION();

        gradient_direction.release();
        gradient_magnitude.release();
        non_edge_feature.release();

        weight_non_edge_compute = weight_non_edge;

        optimalPathsMap.release();
    }

    Size src_size;
    Mat image_;
    Mat grayscale_;
    void initImage_(InputArray image)
    {
        CV_TRACE_FUNCTION();
        if (!image_.empty())
            return;
        CV_CheckType(image.type(), image.type() == CV_8UC1 || image.type() == CV_8UC3 || image.type() == CV_8UC4, "");
        src_size = image.size();
        image_ = image.getMat();
    }
    void initGrayscale_(InputArray image)
    {
        CV_TRACE_FUNCTION();
        if (!grayscale_.empty())
            return;
        CV_Assert(!image.empty());
        CV_CheckType(image.type(), image.type() == CV_8UC1 || image.type() == CV_8UC3 || image.type() == CV_8UC4, "");
        src_size = image.size();
        if (image.channels() > 1)
            cvtColor(image, grayscale_, COLOR_BGR2GRAY);
        else
            grayscale_ = image.getMat();
    }
    Mat Ix_, Iy_;
    void initImageDerives_(InputArray image)
    {
        CV_TRACE_FUNCTION();
        if (!Ix_.empty())
            return;
        initGrayscale_(image);
        Sobel(grayscale_, Ix_, CV_32FC1, 1, 0, sobelKernelSize);
        Sobel(grayscale_, Iy_, CV_32FC1, 0, 1, sobelKernelSize);
    }
    Mat image_magnitude_;
    void initImageMagnitude_(InputArray image)
    {
        CV_TRACE_FUNCTION();
        if (!image_magnitude_.empty())
            return;
        initImageDerives_(image);
        magnitude(Ix_, Iy_, image_magnitude_);
    }

    void cleanupFeaturesTemporaryArrays_()
    {
        CV_TRACE_FUNCTION();
        image_.release();
        grayscale_.release();
        Ix_.release();
        Iy_.release();
        image_magnitude_.release();
    }

    Impl()
    {
        // nothing
        CV_TRACE_FUNCTION();
    }

    void setWeights(float weight_non_edge_, float weight_gradient_direction_, float weight_gradient_magnitude_)
    {
        CV_TRACE_FUNCTION();

        CV_CheckGE(weight_non_edge_, 0.0f, "");
        CV_CheckGE(weight_gradient_direction_, 0.0f, "");
        CV_CheckGE(weight_gradient_magnitude_, 0.0f, "");
        CV_CheckGE(weight_non_edge_ + weight_gradient_direction_ + weight_gradient_magnitude_, FLT_EPSILON, "Sum of weights must be greater than zero");
        weight_non_edge = weight_non_edge_;
        weight_gradient_direction = weight_gradient_direction_;
        weight_gradient_magnitude = weight_gradient_magnitude_;
        resetFeatures_();
    }

    void setGradientMagnitudeMaxLimit(float gradient_magnitude_threshold_max_)
    {
        CV_TRACE_FUNCTION();

        CV_CheckGE(gradient_magnitude_threshold_max_, 0.0f, "");
        gradient_magnitude_threshold_max = gradient_magnitude_threshold_max_;
        resetFeatures_();
    }

    void setEdgeFeatureZeroCrossingParameters(float gradient_magnitude_min_value_)
    {
        CV_TRACE_FUNCTION();

        CV_CheckGE(gradient_magnitude_min_value_, 0.0f, "");
        edge_mode = FEATURE_ZERO_CROSSING;
        edge_gradient_magnitude_min_value = gradient_magnitude_min_value_;
        resetFeatures_();
    }

    void setEdgeFeatureCannyParameters(
            double threshold1, double threshold2,
            int apertureSize = 3, bool L2gradient = false
    )
    {
        CV_TRACE_FUNCTION();

        CV_CheckGE(threshold1, 0.0, "");
        CV_CheckGE(threshold2, 0.0, "");
        edge_mode = FEATURE_CANNY;
        edge_canny_threshold1 = threshold1;
        edge_canny_threshold2 = threshold2;
        edge_canny_apertureSize = apertureSize;
        edge_canny_L2gradient = L2gradient;
        resetFeatures_();
    }

    void applyImageFeatures(
            InputArray non_edge, InputArray gradient_direction_, InputArray gradient_magnitude_,
            InputArray image
    )
    {
        CV_TRACE_FUNCTION();

        resetFeatures_();
        cleanupFeaturesTemporaryArrays_();

        src_size = Size(0, 0);
        if (!non_edge.empty())
            src_size = non_edge.size();
        if (!gradient_direction_.empty())
        {
            Size gradient_direction_size = gradient_direction_.size();
            if (!src_size.empty())
                CV_CheckEQ(src_size, gradient_direction_size, "");
            else
                src_size = gradient_direction_size;
        }
        if (!gradient_magnitude_.empty())
        {
            Size gradient_magnitude_size = gradient_magnitude_.size();
            if (!src_size.empty())
                CV_CheckEQ(src_size, gradient_magnitude_size, "");
            else
                src_size = gradient_magnitude_size;
        }
        if (!image.empty())
        {
            Size image_size = image.size();
            if (!src_size.empty())
                CV_CheckEQ(src_size, image_size, "");
            else
                src_size = image_size;
        }
        // src_size must be filled
        CV_Assert(!src_size.empty());

        if (!non_edge.empty())
        {
            CV_CheckTypeEQ(non_edge.type(), CV_8UC1, "");
            non_edge_feature = non_edge.getMat();
        }
        else
        {
            if (weight_non_edge == 0.0f)
            {
                non_edge_feature.create(src_size);
                non_edge_feature.setTo(0);
            }
            else
            {
                if (image.empty())
                    CV_Error(Error::StsBadArg, "Non-edge feature parameter is missing. Input image parameter is required to extract this feature");
                extractEdgeFeature_(image);
            }
        }

        if (!gradient_direction_.empty())
        {
            CV_CheckTypeEQ(gradient_direction_.type(), CV_32FC2, "");
            gradient_direction = gradient_direction_.getMat();
        }
        else
        {
            if (weight_gradient_direction == 0.0f)
            {
                gradient_direction.create(src_size);
                gradient_direction.setTo(Scalar::all(0));
            }
            else
            {
                if (image.empty())
                    CV_Error(Error::StsBadArg, "Gradient direction feature parameter is missing. Input image parameter is required to extract this feature");
                extractGradientDirection_(image);
            }
        }

        if (!gradient_magnitude_.empty())
        {
            CV_CheckTypeEQ(gradient_magnitude_.type(), CV_32FC1, "");
            gradient_magnitude = gradient_magnitude_.getMat();
        }
        else
        {
            if (weight_gradient_magnitude == 0.0f)
            {
                gradient_magnitude.create(src_size);
                gradient_magnitude.setTo(Scalar::all(0));
            }
            else
            {
                if (image.empty())
                    CV_Error(Error::StsBadArg, "Gradient magnitude feature parameter is missing. Input image parameter is required to extract this feature");
                extractGradientMagnitude_(image);
            }
        }

        cleanupFeaturesTemporaryArrays_();
    }


    void extractEdgeFeature_(InputArray image)
    {
        CV_TRACE_FUNCTION();

        if (edge_mode == FEATURE_CANNY)
        {
            CV_LOG_DEBUG(NULL, "Canny(" << edge_canny_threshold1 << ", " << edge_canny_threshold2 << ")");
            Mat img_canny;
            Canny(image, img_canny, edge_canny_threshold1, edge_canny_threshold2, edge_canny_apertureSize, edge_canny_L2gradient);
#if 0
            threshold(img_canny, non_edge_feature, 254, 1, THRESH_BINARY_INV);
#else
            // Canny result values are 0 or 255
            bitwise_not(img_canny, non_edge_feature);
            weight_non_edge_compute = weight_non_edge * (1.0f / 255.0f);
#endif
        }
        else // if (edge_mode == FEATURE_ZERO_CROSSING)
        {
            initGrayscale_(image);
            Mat_<short> laplacian;
            Laplacian(grayscale_, laplacian, CV_16S, laplacianKernelSize);
            Mat_<uchar> zero_crossing(src_size, 1);

            const size_t zstep = zero_crossing.step[0];
            for (int y = 0; y < src_size.height - 1; y++)
            {
                const short* row0 = laplacian.ptr<short>(y);
                const short* row1 = laplacian.ptr<short>(y + 1);
                uchar* zrow0 = zero_crossing.ptr<uchar>(y);
                //uchar* zrow1 = zero_crossing.ptr<uchar>(y + 1);
                for (int x = 0; x < src_size.width - 1; x++)
                {
                    const int v = row0[x];
                    const int neg_v = -v;
                    //  - * 1
                    //  2 3 4
                    const int v1 = row0[x + 1];
                    const int v2 = (x > 0) ? row1[x - 1] : v;
                    const int v3 = row1[x + 0];
                    const int v4 = row1[x + 1];
                    if (v < 0)
                    {
                        if (v1 > 0)
                        {
                            zrow0[x + ((v1 < neg_v) ? 1 : 0)] = 0;
                        }
                        if (v2 > 0)
                        {
                            zrow0[x + ((v2 < neg_v) ? (zstep - 1) : 0)] = 0;
                        }
                        if (v3 > 0)
                        {
                            zrow0[x + ((v3 < neg_v) ? (zstep + 0) : 0)] = 0;
                        }
                        if (v4 > 0)
                        {
                            zrow0[x + ((v4 < neg_v) ? (zstep + 1) : 0)] = 0;
                        }
                    }
                    else
                    {
                        if (v1 < 0)
                        {
                            zrow0[x + ((v1 > neg_v) ? 1 : 0)] = 0;
                        }
                        if (v2 < 0)
                        {
                            zrow0[x + ((v2 > neg_v) ? (zstep - 1) : 0)] = 0;
                        }
                        if (v3 < 0)
                        {
                            zrow0[x + ((v3 > neg_v) ? (zstep + 0) : 0)] = 0;
                        }
                        if (v4 < 0)
                        {
                            zrow0[x + ((v4 > neg_v) ? (zstep + 1) : 0)] = 0;
                        }
                    }
                }
            }

            if (edge_gradient_magnitude_min_value > 0)
            {
                initImageMagnitude_(image);
                Mat mask = image_magnitude_ < edge_gradient_magnitude_min_value;
                zero_crossing.setTo(1, mask);  // reset low-amplitude noise
            }

            non_edge_feature = zero_crossing;
        }
    }


    void extractGradientDirection_(InputArray image)
    {
        CV_TRACE_FUNCTION();

        initImageMagnitude_(image);  // calls internally: initImageDerives_(image);
        gradient_direction.create(src_size);
        for (int y = 0; y < src_size.height; y++)
        {
            const float* magnitude_row = image_magnitude_.ptr<float>(y);
            const float* Ix_row = Ix_.ptr<float>(y);
            const float* Iy_row = Iy_.ptr<float>(y);
            Point2f* gradient_direction_row = gradient_direction.ptr<Point2f>(y);
            for (int x = 0; x < src_size.width; x++)
            {
                const float m = magnitude_row[x];
                if (m > FLT_EPSILON)
                {
                    float m_inv = 1.0f / m;
                    gradient_direction_row[x] = Point2f(Ix_row[x] * m_inv, Iy_row[x] * m_inv);
                }
                else
                {
                    gradient_direction_row[x] = Point2f(0, 0);
                }
            }
        }
    }

    void extractGradientMagnitude_(InputArray image)
    {
        CV_TRACE_FUNCTION();

        initImageMagnitude_(image);  // calls internally: initImageDerives_(image);
        Mat m;
        double max_m = 0;
        if (gradient_magnitude_threshold_max > 0)
        {
            threshold(image_magnitude_, m, gradient_magnitude_threshold_max, 0, THRESH_TRUNC);
            max_m = gradient_magnitude_threshold_max;
        }
        else
        {
            m = image_magnitude_;
            minMaxLoc(m, 0, &max_m);
        }
        if (max_m <= FLT_EPSILON)
        {
            CV_LOG_INFO(NULL, "IntelligentScissorsMB: input image gradient is almost zero")
            gradient_magnitude.create(src_size);
            gradient_magnitude.setTo(0);
        }
        else
        {
            m.convertTo(gradient_magnitude, CV_32F, -1.0 / max_m, 1.0);  // normalize and inverse to range 0..1
        }
    }

    void applyImage(InputArray image)
    {
        CV_TRACE_FUNCTION();

        CV_CheckType(image.type(), image.type() == CV_8UC1 || image.type() == CV_8UC3 || image.type() == CV_8UC4, "");

        resetFeatures_();
        cleanupFeaturesTemporaryArrays_();
        extractEdgeFeature_(image);
        extractGradientDirection_(image);
        extractGradientMagnitude_(image);
        cleanupFeaturesTemporaryArrays_();
    }


    // details: see section 3.1 of the article
    const float* acos_table = getAcosTable();
    float local_cost(const Point& p, const Point& q) const
    {
        const bool isDiag = (p.x != q.x) && (p.y != q.y);

        float fG = gradient_magnitude.at<float>(q);

        const Point2f diff((float)(q.x - p.x), (float)(q.y - p.y));

        const Point2f Ip = gradient_direction(p);
        const Point2f Iq = gradient_direction(q);

        const Point2f Dp(Ip.y, -Ip.x);  // D(p) - 90 degrees clockwise
        const Point2f Dq(Iq.y, -Iq.x);  // D(q) - 90 degrees clockwise

        float dp = Dp.dot(diff);  // dp(p, q)
        float dq = Dq.dot(diff);  // dq(p, q)
        if (dp < 0)
        {
            dp = -dp;  // ensure dp >= 0
            dq = -dq;
        }

        const float sqrt2_inv = 0.7071067811865475f; // 1.0 / sqrt(2)
        if (isDiag)
        {
            dp *= sqrt2_inv;  // normalize length of (q - p)
            dq *= sqrt2_inv;  // normalize length of (q - p)
        }
        else
        {
            fG *= sqrt2_inv;
        }

#if 1
        int dp_i = cvFloor(dp * ACOS_TABLE_SIZE);  // dp is in range 0..1
        dp_i = std::min(ACOS_TABLE_SIZE, std::max(0, dp_i));
        int dq_i = cvFloor(dq * ACOS_TABLE_SIZE);  // dq is in range -1..1
        dq_i = std::min(ACOS_TABLE_SIZE, std::max(-ACOS_TABLE_SIZE, dq_i));
        const float fD = acos_table[dp_i + ACOS_TABLE_SIZE] + acos_table[dq_i + ACOS_TABLE_SIZE];
#else
        const float CV_PI_inv = static_cast<float>(1.0 / CV_PI);
        const float fD = (acosf(dp) + acosf(dq)) * CV_PI_inv;  // TODO optimize acos calls (through tables)
#endif

        float cost =
            weight_non_edge_compute * non_edge_feature.at<uchar>(q) +
            weight_gradient_direction * fD +
            weight_gradient_magnitude * fG;
        return cost;
    }

    struct Pix
    {
        Point pt;
        float cost;  // NOTE: do not remove cost from here through replacing by cost(pt) map access

        inline bool operator > (const Pix &b) const
        {
            return cost > b.cost;
        }
    };

    void buildMap(const Point& start_point)
    {
        CV_TRACE_FUNCTION();

        CV_Assert(!src_size.empty());
        CV_Assert(!gradient_magnitude.empty() && "Features are missing. applyImage() must be called first");

        CV_CheckGE(weight_non_edge + weight_gradient_direction + weight_gradient_magnitude, FLT_EPSILON, "");

#if 0  // debug
        Rect wholeImage(0, 0, src_size.width, src_size.height);
        Rect roi = Rect(start_point.x - 5, start_point.y - 5, 11, 11) & wholeImage;
        std::cout << roi << std::endl;
        std::cout << gradient_magnitude(roi) << std::endl;
        std::cout << gradient_direction(roi) << std::endl;
        std::cout << non_edge_feature(roi) << std::endl;
#endif

        optimalPathsMap.release();
        optimalPathsMap.create(src_size);
        optimalPathsMap.setTo(0);  // optimalPathsMap(start_point) = 0;

        //
        // Section 3.2
        // Live-Wire 2-D DP graph search.
        //

        Mat_<float> cost_map(src_size, FLT_MAX);  // g(q)
        Mat_<uchar> processed(src_size, (uchar)0);  // e(q)

        // Note: std::vector is faster than std::deque
        // TODO check std::set
        std::priority_queue< Pix, std::vector<Pix>, std::greater<Pix> > L;

        cost_map(start_point) = 0;
        L.emplace(Pix{ start_point, 0/*cost*/ });

        while (!L.empty())
        {
            Pix pix = L.top(); L.pop();
            Point q = pix.pt;  // 'q' from the article
            if (processed(q))
                continue;  // already processed (with lower cost, see note below)
            processed(q) = 1;
#if 1
            const float cost_q = pix.cost;
#else
            const float cost_q = cost_map(q);
            CV_Assert(cost_q == pix.cost);
#endif
            for (int n = 0; n < 8; n++)  // scan neighbours
            {
                Point r(q.x + neighbors[n][0], q.y + neighbors[n][1]);  // 'r' from the article
                if (r.x < 0 || r.x >= src_size.width || r.y < 0 || r.y >= src_size.height)
                    continue;  // out of range

#if !defined(__EMSCRIPTEN__)  // slower in JS
                float& cost_r = cost_map(r);
                if (cost_r < cost_q)
                    continue;  // already processed
#else
                if (processed(r))
                    continue;  // already processed

                float& cost_r = cost_map(r);
                CV_DbgCheckLE(cost_q, cost_r, "INTERNAL ERROR: sorted queue is corrupted");
#endif

                float cost = cost_q + local_cost(q, r);  // TODO(opt): compute partially until cost < cost_r
                if (cost < cost_r)
                {
#if 0  // avoid compiler warning
                    if (cost_r != FLT_MAX)
                    {
                        // In article the point 'r' is removed from the queue L
                        // to be re-inserted again with sorting against new optimized cost.
                        // We can do nothing, because "new point" will be placed before in the sorted queue.
                        // Old point will be skipped through "if (processed(q))" check above after processing of new optimal candidate.
                        //
                        // This approach leads to some performance impact, however it is much smaller than element removal from the sorted queue.
                        // So, do nothing.
                    }
#endif
                    cost_r = cost;
                    L.emplace(Pix{ r, cost });
                    optimalPathsMap(r) = (uchar)neighbors_encode[n];
                }
            }
        }
    }

    void getContour(const Point& target, OutputArray contour_, bool backward)
    {
        CV_TRACE_FUNCTION();

        CV_Assert(!optimalPathsMap.empty() && "buildMap() must be called before getContour()");

        const int cols = optimalPathsMap.cols;
        const int rows = optimalPathsMap.rows;

        std::vector<Point> result; result.reserve(512);

        size_t loop_check = 4096;
        Point pt = target;
        for (size_t i = 0; i < (size_t)rows * cols; i++)  // don't hang on invalid maps
        {
            CV_CheckLT(pt.x, cols, "");
            CV_CheckLT(pt.y, rows, "");
            result.push_back(pt);
            int direction = (int)optimalPathsMap(pt);
            if (direction == 0)
                break;  // stop, start point is reached
            CV_CheckLT(direction, 9, "Map is invalid");
            Point next(pt.x + neighbors[direction - 1][0], pt.y + neighbors[direction - 1][1]);
            pt = next;

            if (result.size() == loop_check)  // optional sanity check of invalid maps with loops (don't eat huge amount of memory)
            {
                loop_check *= 4;  // next limit for loop check
                for (const auto& pt_check : result)
                {
                    CV_CheckNE(pt_check, pt, "Map is invalid. Contour loop is detected");
                }
            }
        }

        if (backward)
        {
            _InputArray(result).copyTo(contour_);
        }
        else
        {
            const int N = (int)result.size();
            const int sz[1] = { N };
            contour_.create(1, sz, CV_32SC2);
            Mat_<Point> contour = contour_.getMat();
            for (int i = 0; i < N; i++)
            {
                contour.at<Point>(i) = result[N - (i + 1)];
            }
        }
    }
};



IntelligentScissorsMB::IntelligentScissorsMB()
    : impl(std::make_shared<Impl>())
{
    // nothing
}

IntelligentScissorsMB& IntelligentScissorsMB::setWeights(float weight_non_edge, float weight_gradient_direction, float weight_gradient_magnitude)
{
    CV_DbgAssert(impl);
    impl->setWeights(weight_non_edge, weight_gradient_direction, weight_gradient_magnitude);
    return *this;
}

IntelligentScissorsMB& IntelligentScissorsMB::setGradientMagnitudeMaxLimit(float gradient_magnitude_threshold_max)
{
    CV_DbgAssert(impl);
    impl->setGradientMagnitudeMaxLimit(gradient_magnitude_threshold_max);
    return *this;
}

IntelligentScissorsMB& IntelligentScissorsMB::setEdgeFeatureZeroCrossingParameters(float gradient_magnitude_min_value)
{
    CV_DbgAssert(impl);
    impl->setEdgeFeatureZeroCrossingParameters(gradient_magnitude_min_value);
    return *this;
}

IntelligentScissorsMB& IntelligentScissorsMB::setEdgeFeatureCannyParameters(
        double threshold1, double threshold2,
        int apertureSize, bool L2gradient
)
{
    CV_DbgAssert(impl);
    impl->setEdgeFeatureCannyParameters(threshold1, threshold2, apertureSize, L2gradient);
    return *this;
}

IntelligentScissorsMB& IntelligentScissorsMB::applyImage(InputArray image)
{
    CV_DbgAssert(impl);
    impl->applyImage(image);
    return *this;
}

IntelligentScissorsMB& IntelligentScissorsMB::applyImageFeatures(
        InputArray non_edge, InputArray gradient_direction, InputArray gradient_magnitude,
        InputArray image
)
{
    CV_DbgAssert(impl);
    impl->applyImageFeatures(non_edge, gradient_direction, gradient_magnitude, image);
    return *this;
}

void IntelligentScissorsMB::buildMap(const Point& pt)
{
    CV_DbgAssert(impl);
    impl->buildMap(pt);
}

void IntelligentScissorsMB::getContour(const Point& target, OutputArray contour, bool backward) const
{
    CV_DbgAssert(impl);
    impl->getContour(target, contour, backward);
}

}}  // namespace
