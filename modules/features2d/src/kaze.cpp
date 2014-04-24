#include "precomp.hpp"
#include "kaze/KAZE.h"

namespace cv
{
    KAZE::KAZE(bool _extended /* = false */)
        : extended(_extended)
    {
    }

    KAZE::~KAZE()
    {

    }

    // returns the descriptor size in bytes
    int KAZE::descriptorSize() const
    {
        return extended ? 128 : 64;
    }

    // returns the descriptor type
    int KAZE::descriptorType() const
    {
        return CV_32F;
    }

    // returns the default norm type
    int KAZE::defaultNorm() const
    {
        return NORM_L2;
    }

    void KAZE::operator()(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints) const
    {
        detectImpl(image, keypoints, mask);
    }

    void KAZE::operator()(InputArray image, InputArray mask,
        std::vector<KeyPoint>& keypoints,
        OutputArray descriptors,
        bool useProvidedKeypoints) const
    {
        cv::Mat img = image.getMat();
        if (img.type() != CV_8UC1)
            cvtColor(image, img, COLOR_BGR2GRAY);

        Mat img1_32;
        img.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);

        cv::Mat& desc = descriptors.getMatRef();

        KAZEOptions options;
        options.img_width = img.cols;
        options.img_height = img.rows;
        options.extended = extended;

        KAZEFeatures impl(options);
        impl.Create_Nonlinear_Scale_Space(img1_32);

        if (!useProvidedKeypoints)
        {
            impl.Feature_Detection(keypoints);
        }

        if (!mask.empty())
        {
            cv::KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
        }

        impl.Feature_Description(keypoints, desc);

        CV_Assert(!desc.rows || desc.cols == descriptorSize() && "Descriptor size does not match expected");
        CV_Assert(!desc.rows || (desc.type() & descriptorType()) && "Descriptor type does not match expected");
    }

    void KAZE::detectImpl(InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask) const
    {
        Mat img = image.getMat();
        if (img.type() != CV_8UC1)
            cvtColor(image, img, COLOR_BGR2GRAY);

        Mat img1_32;
        img.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);

        KAZEOptions options;
        options.img_width = img.cols;
        options.img_height = img.rows;
        options.extended = extended;

        KAZEFeatures impl(options);
        impl.Create_Nonlinear_Scale_Space(img1_32);
        impl.Feature_Detection(keypoints);

        if (!mask.empty())
        {
            cv::KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
        }
    }

    void KAZE::computeImpl(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) const
    {
        cv::Mat img = image.getMat();
        if (img.type() != CV_8UC1)
            cvtColor(image, img, COLOR_BGR2GRAY);

        Mat img1_32;
        img.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);

        cv::Mat& desc = descriptors.getMatRef();

        KAZEOptions options;
        options.img_width = img.cols;
        options.img_height = img.rows;
        options.extended = extended;

        KAZEFeatures impl(options);
        impl.Create_Nonlinear_Scale_Space(img1_32);
        impl.Feature_Description(keypoints, desc);

        CV_Assert(!desc.rows || desc.cols == descriptorSize() && "Descriptor size does not match expected");
        CV_Assert(!desc.rows || (desc.type() & descriptorType()) && "Descriptor type does not match expected");
    }
}