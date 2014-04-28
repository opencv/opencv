#include "precomp.hpp"
#include "akaze/AKAZE.h"

namespace cv
{

    AKAZE::AKAZE(int _descriptor, int _descriptor_size, int _descriptor_channels)
        : descriptor_channels(_descriptor_channels)
        , descriptor(_descriptor)
        , descriptor_size(_descriptor_size)
    {

    }

    AKAZE::~AKAZE()
    {

    }

    // returns the descriptor size in bytes
    int AKAZE::descriptorSize() const
    {
        if (descriptor < MLDB_UPRIGHT)
        {
            return 64;
        }
        else
        {
            // We use the full length binary descriptor -> 486 bits
            if (descriptor_size == 0)
            {
                int t = (6 + 36 + 120) * descriptor_channels;
                return (int)ceil(t / 8.);
            }
            else
            {
                // We use the random bit selection length binary descriptor
                return (int)ceil(descriptor_size / 8.);
            }
        }
    }

    // returns the descriptor type
    int AKAZE::descriptorType() const
    {
        if (descriptor < MLDB_UPRIGHT)
        {
            return CV_32F;
        }
        else
        {
            return CV_8U;
        }
    }

    // returns the default norm type
    int AKAZE::defaultNorm() const
    {
        if (descriptor < MLDB_UPRIGHT)
        {
            return NORM_L2;
        }
        else
        {
            return NORM_HAMMING;
        }
    }


    void AKAZE::operator()(InputArray image, InputArray mask,
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

        AKAZEOptions options;
        options.img_width = img.cols;
        options.img_height = img.rows;

        AKAZEFeatures impl(options);
        impl.Create_Nonlinear_Scale_Space(img1_32);

        if (!useProvidedKeypoints)
        {
            impl.Feature_Detection(keypoints);
        }

        if (!mask.empty())
        {
            cv::KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
        }

        impl.Compute_Descriptors(keypoints, desc);

        CV_Assert((!desc.rows || desc.cols == descriptorSize())    && "Descriptor size does not match expected");
        CV_Assert((!desc.rows || (desc.type() & descriptorType())) && "Descriptor type does not match expected");
    }

    void AKAZE::detectImpl(InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask) const
    {
        cv::Mat img = image.getMat();
        if (img.type() != CV_8UC1)
            cvtColor(image, img, COLOR_BGR2GRAY);

        Mat img1_32;
        img.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);

        AKAZEOptions options;
        options.img_width = img.cols;
        options.img_height = img.rows;

        AKAZEFeatures impl(options);
        impl.Create_Nonlinear_Scale_Space(img1_32);
        impl.Feature_Detection(keypoints);

        if (!mask.empty())
        {
            cv::KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
        }
    }

    void AKAZE::computeImpl(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) const
    {
        cv::Mat img = image.getMat();
        if (img.type() != CV_8UC1)
            cvtColor(image, img, COLOR_BGR2GRAY);

        Mat img1_32;
        img.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);

        cv::Mat& desc = descriptors.getMatRef();

        AKAZEOptions options;
        options.img_width = img.cols;
        options.img_height = img.rows;

        AKAZEFeatures impl(options);
        impl.Create_Nonlinear_Scale_Space(img1_32);
        impl.Compute_Descriptors(keypoints, desc);

        CV_Assert((!desc.rows || desc.cols == descriptorSize())    && "Descriptor size does not match expected");
        CV_Assert((!desc.rows || (desc.type() & descriptorType())) && "Descriptor type does not match expected");
    }
}