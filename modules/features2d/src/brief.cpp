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
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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
#include <algorithm>
#include <vector>

using namespace cv;

inline int smoothedSum(const Mat& sum, const KeyPoint& pt, int y, int x)
{
    static const int HALF_KERNEL = BriefDescriptorExtractor::KERNEL_SIZE / 2;

    int img_y = (int)(pt.pt.y + 0.5) + y;
    int img_x = (int)(pt.pt.x + 0.5) + x;
    return   sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
           - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
           - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
           + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL);
}

void pixelTests16(const Mat& sum, const std::vector<KeyPoint>& keypoints, Mat& descriptors)
{
    for (int i = 0; i < (int)keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(i);
        const KeyPoint& pt = keypoints[i];
#include "generated_16.i"
    }
}

void pixelTests32(const Mat& sum, const std::vector<KeyPoint>& keypoints, Mat& descriptors)
{
    for (int i = 0; i < (int)keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(i);
        const KeyPoint& pt = keypoints[i];

#include "generated_32.i"
    }
}

void pixelTests64(const Mat& sum, const std::vector<KeyPoint>& keypoints, Mat& descriptors)
{
    for (int i = 0; i < (int)keypoints.size(); ++i)
    {
        uchar* desc = descriptors.ptr(i);
        const KeyPoint& pt = keypoints[i];

#include "generated_64.i"
    }
}

namespace cv
{

HammingLUT::ResultType HammingLUT::operator()( const unsigned char* a, const unsigned char* b, int size ) const
{
    ResultType result = 0;
    for (int i = 0; i < size; i++)
    {
        result += byteBitsLookUp(a[i] ^ b[i]);
    }
    return result;
}

Hamming::ResultType Hamming::operator()(const unsigned char* a, const unsigned char* b, int size) const
{
#if __GNUC__
    ResultType result = 0;
    for (int i = 0; i < size; i += sizeof(unsigned long))
    {
        unsigned long a2 = *reinterpret_cast<const unsigned long*> (a + i);
        unsigned long b2 = *reinterpret_cast<const unsigned long*> (b + i);
        result += __builtin_popcountl(a2 ^ b2);
    }
    return result;
#else
    return HammingLUT()(a,b,size);
#endif
}

BriefDescriptorExtractor::BriefDescriptorExtractor(int bytes) :
    bytes_(bytes), test_fn_(NULL)
{
    switch (bytes)
    {
        case 16:
            test_fn_ = pixelTests16;
            break;
        case 32:
            test_fn_ = pixelTests32;
            break;
        case 64:
            test_fn_ = pixelTests64;
            break;
        default:
            CV_Error(CV_StsBadArg, "bytes must be 16, 32, or 64");
    }
}

int BriefDescriptorExtractor::descriptorSize() const
{
    return bytes_;
}

int BriefDescriptorExtractor::descriptorType() const
{
    return CV_8UC1;
}

void BriefDescriptorExtractor::computeImpl(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const
{
    // Construct integral image for fast smoothing (box filter)
    Mat sum;

    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    ///TODO allow the user to pass in a precomputed integral image
    //if(image.type() == CV_32S)
    //  sum = image;
    //else

    integral( grayImage, sum, CV_32S);

    //Remove keypoints very close to the border
    removeBorderKeypoints(keypoints, image.size(), PATCH_SIZE/2 + KERNEL_SIZE/2);

    descriptors = Mat::zeros(keypoints.size(), bytes_, CV_8U);
    test_fn_(sum, keypoints, descriptors);
}

/**
 *  \brief template meta programming struct that gives number of bits in a byte
 *  @TODO Maybe unintuitive and should just use python to generate the entries in the LUT
 */
template<unsigned char b>
struct ByteBits
{
    /**
     * number of bits in the byte given by the template constant
     */
    enum
    {
        COUNT = ((b >> 0) & 1) +
                ((b >> 1) & 1) +
                ((b >> 2) & 1) +
                ((b >> 3) & 1) +
                ((b >> 4) & 1) +
                ((b >> 5) & 1) +
                ((b >> 6) & 1) +
                ((b >> 7) & 1)
    };
};

unsigned char HammingLUT::byteBitsLookUp(unsigned char b)
{
    static const unsigned char table[256] =
    {
        ByteBits<0>::COUNT,
        ByteBits<1>::COUNT,
        ByteBits<2>::COUNT,
        ByteBits<3>::COUNT,
        ByteBits<4>::COUNT,
        ByteBits<5>::COUNT,
        ByteBits<6>::COUNT,
        ByteBits<7>::COUNT,
        ByteBits<8>::COUNT,
        ByteBits<9>::COUNT,
        ByteBits<10>::COUNT,
        ByteBits<11>::COUNT,
        ByteBits<12>::COUNT,
        ByteBits<13>::COUNT,
        ByteBits<14>::COUNT,
        ByteBits<15>::COUNT,
        ByteBits<16>::COUNT,
        ByteBits<17>::COUNT,
        ByteBits<18>::COUNT,
        ByteBits<19>::COUNT,
        ByteBits<20>::COUNT,
        ByteBits<21>::COUNT,
        ByteBits<22>::COUNT,
        ByteBits<23>::COUNT,
        ByteBits<24>::COUNT,
        ByteBits<25>::COUNT,
        ByteBits<26>::COUNT,
        ByteBits<27>::COUNT,
        ByteBits<28>::COUNT,
        ByteBits<29>::COUNT,
        ByteBits<30>::COUNT,
        ByteBits<31>::COUNT,
        ByteBits<32>::COUNT,
        ByteBits<33>::COUNT,
        ByteBits<34>::COUNT,
        ByteBits<35>::COUNT,
        ByteBits<36>::COUNT,
        ByteBits<37>::COUNT,
        ByteBits<38>::COUNT,
        ByteBits<39>::COUNT,
        ByteBits<40>::COUNT,
        ByteBits<41>::COUNT,
        ByteBits<42>::COUNT,
        ByteBits<43>::COUNT,
        ByteBits<44>::COUNT,
        ByteBits<45>::COUNT,
        ByteBits<46>::COUNT,
        ByteBits<47>::COUNT,
        ByteBits<48>::COUNT,
        ByteBits<49>::COUNT,
        ByteBits<50>::COUNT,
        ByteBits<51>::COUNT,
        ByteBits<52>::COUNT,
        ByteBits<53>::COUNT,
        ByteBits<54>::COUNT,
        ByteBits<55>::COUNT,
        ByteBits<56>::COUNT,
        ByteBits<57>::COUNT,
        ByteBits<58>::COUNT,
        ByteBits<59>::COUNT,
        ByteBits<60>::COUNT,
        ByteBits<61>::COUNT,
        ByteBits<62>::COUNT,
        ByteBits<63>::COUNT,
        ByteBits<64>::COUNT,
        ByteBits<65>::COUNT,
        ByteBits<66>::COUNT,
        ByteBits<67>::COUNT,
        ByteBits<68>::COUNT,
        ByteBits<69>::COUNT,
        ByteBits<70>::COUNT,
        ByteBits<71>::COUNT,
        ByteBits<72>::COUNT,
        ByteBits<73>::COUNT,
        ByteBits<74>::COUNT,
        ByteBits<75>::COUNT,
        ByteBits<76>::COUNT,
        ByteBits<77>::COUNT,
        ByteBits<78>::COUNT,
        ByteBits<79>::COUNT,
        ByteBits<80>::COUNT,
        ByteBits<81>::COUNT,
        ByteBits<82>::COUNT,
        ByteBits<83>::COUNT,
        ByteBits<84>::COUNT,
        ByteBits<85>::COUNT,
        ByteBits<86>::COUNT,
        ByteBits<87>::COUNT,
        ByteBits<88>::COUNT,
        ByteBits<89>::COUNT,
        ByteBits<90>::COUNT,
        ByteBits<91>::COUNT,
        ByteBits<92>::COUNT,
        ByteBits<93>::COUNT,
        ByteBits<94>::COUNT,
        ByteBits<95>::COUNT,
        ByteBits<96>::COUNT,
        ByteBits<97>::COUNT,
        ByteBits<98>::COUNT,
        ByteBits<99>::COUNT,
        ByteBits<100>::COUNT,
        ByteBits<101>::COUNT,
        ByteBits<102>::COUNT,
        ByteBits<103>::COUNT,
        ByteBits<104>::COUNT,
        ByteBits<105>::COUNT,
        ByteBits<106>::COUNT,
        ByteBits<107>::COUNT,
        ByteBits<108>::COUNT,
        ByteBits<109>::COUNT,
        ByteBits<110>::COUNT,
        ByteBits<111>::COUNT,
        ByteBits<112>::COUNT,
        ByteBits<113>::COUNT,
        ByteBits<114>::COUNT,
        ByteBits<115>::COUNT,
        ByteBits<116>::COUNT,
        ByteBits<117>::COUNT,
        ByteBits<118>::COUNT,
        ByteBits<119>::COUNT,
        ByteBits<120>::COUNT,
        ByteBits<121>::COUNT,
        ByteBits<122>::COUNT,
        ByteBits<123>::COUNT,
        ByteBits<124>::COUNT,
        ByteBits<125>::COUNT,
        ByteBits<126>::COUNT,
        ByteBits<127>::COUNT,
        ByteBits<128>::COUNT,
        ByteBits<129>::COUNT,
        ByteBits<130>::COUNT,
        ByteBits<131>::COUNT,
        ByteBits<132>::COUNT,
        ByteBits<133>::COUNT,
        ByteBits<134>::COUNT,
        ByteBits<135>::COUNT,
        ByteBits<136>::COUNT,
        ByteBits<137>::COUNT,
        ByteBits<138>::COUNT,
        ByteBits<139>::COUNT,
        ByteBits<140>::COUNT,
        ByteBits<141>::COUNT,
        ByteBits<142>::COUNT,
        ByteBits<143>::COUNT,
        ByteBits<144>::COUNT,
        ByteBits<145>::COUNT,
        ByteBits<146>::COUNT,
        ByteBits<147>::COUNT,
        ByteBits<148>::COUNT,
        ByteBits<149>::COUNT,
        ByteBits<150>::COUNT,
        ByteBits<151>::COUNT,
        ByteBits<152>::COUNT,
        ByteBits<153>::COUNT,
        ByteBits<154>::COUNT,
        ByteBits<155>::COUNT,
        ByteBits<156>::COUNT,
        ByteBits<157>::COUNT,
        ByteBits<158>::COUNT,
        ByteBits<159>::COUNT,
        ByteBits<160>::COUNT,
        ByteBits<161>::COUNT,
        ByteBits<162>::COUNT,
        ByteBits<163>::COUNT,
        ByteBits<164>::COUNT,
        ByteBits<165>::COUNT,
        ByteBits<166>::COUNT,
        ByteBits<167>::COUNT,
        ByteBits<168>::COUNT,
        ByteBits<169>::COUNT,
        ByteBits<170>::COUNT,
        ByteBits<171>::COUNT,
        ByteBits<172>::COUNT,
        ByteBits<173>::COUNT,
        ByteBits<174>::COUNT,
        ByteBits<175>::COUNT,
        ByteBits<176>::COUNT,
        ByteBits<177>::COUNT,
        ByteBits<178>::COUNT,
        ByteBits<179>::COUNT,
        ByteBits<180>::COUNT,
        ByteBits<181>::COUNT,
        ByteBits<182>::COUNT,
        ByteBits<183>::COUNT,
        ByteBits<184>::COUNT,
        ByteBits<185>::COUNT,
        ByteBits<186>::COUNT,
        ByteBits<187>::COUNT,
        ByteBits<188>::COUNT,
        ByteBits<189>::COUNT,
        ByteBits<190>::COUNT,
        ByteBits<191>::COUNT,
        ByteBits<192>::COUNT,
        ByteBits<193>::COUNT,
        ByteBits<194>::COUNT,
        ByteBits<195>::COUNT,
        ByteBits<196>::COUNT,
        ByteBits<197>::COUNT,
        ByteBits<198>::COUNT,
        ByteBits<199>::COUNT,
        ByteBits<200>::COUNT,
        ByteBits<201>::COUNT,
        ByteBits<202>::COUNT,
        ByteBits<203>::COUNT,
        ByteBits<204>::COUNT,
        ByteBits<205>::COUNT,
        ByteBits<206>::COUNT,
        ByteBits<207>::COUNT,
        ByteBits<208>::COUNT,
        ByteBits<209>::COUNT,
        ByteBits<210>::COUNT,
        ByteBits<211>::COUNT,
        ByteBits<212>::COUNT,
        ByteBits<213>::COUNT,
        ByteBits<214>::COUNT,
        ByteBits<215>::COUNT,
        ByteBits<216>::COUNT,
        ByteBits<217>::COUNT,
        ByteBits<218>::COUNT,
        ByteBits<219>::COUNT,
        ByteBits<220>::COUNT,
        ByteBits<221>::COUNT,
        ByteBits<222>::COUNT,
        ByteBits<223>::COUNT,
        ByteBits<224>::COUNT,
        ByteBits<225>::COUNT,
        ByteBits<226>::COUNT,
        ByteBits<227>::COUNT,
        ByteBits<228>::COUNT,
        ByteBits<229>::COUNT,
        ByteBits<230>::COUNT,
        ByteBits<231>::COUNT,
        ByteBits<232>::COUNT,
        ByteBits<233>::COUNT,
        ByteBits<234>::COUNT,
        ByteBits<235>::COUNT,
        ByteBits<236>::COUNT,
        ByteBits<237>::COUNT,
        ByteBits<238>::COUNT,
        ByteBits<239>::COUNT,
        ByteBits<240>::COUNT,
        ByteBits<241>::COUNT,
        ByteBits<242>::COUNT,
        ByteBits<243>::COUNT,
        ByteBits<244>::COUNT,
        ByteBits<245>::COUNT,
        ByteBits<246>::COUNT,
        ByteBits<247>::COUNT,
        ByteBits<248>::COUNT,
        ByteBits<249>::COUNT,
        ByteBits<250>::COUNT,
        ByteBits<251>::COUNT,
        ByteBits<252>::COUNT,
        ByteBits<253>::COUNT,
        ByteBits<254>::COUNT,
        ByteBits<255>::COUNT
    };

    return table[b];
}

} // namespace cv
