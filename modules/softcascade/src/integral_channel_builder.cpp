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
// Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
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
//     and / or other materials provided with the distribution.
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

namespace {

class ICFBuilder : public cv::ChannelFeatureBuilder
{
    virtual ~ICFBuilder() {}
    virtual cv::AlgorithmInfo* info() const;
    virtual void operator()(cv::InputArray _frame, CV_OUT cv::OutputArray _integrals) const
    {
        CV_Assert(_frame.type() == CV_8UC3);

        cv::Mat frame      = _frame.getMat();
        int h = frame.rows;
        int w = frame.cols;
        _integrals.create(h / 4 * 10 + 1, w / 4 + 1, CV_32SC1);
        cv::Mat& integrals = _integrals.getMatRef();

        cv::Mat channels, gray;

        channels.create(h * 10, w, CV_8UC1);
        channels.setTo(0);

        cvtColor(frame, gray, CV_BGR2GRAY);

        cv::Mat df_dx, df_dy, mag, angle;
        cv::Sobel(gray, df_dx, CV_32F, 1, 0);
        cv::Sobel(gray, df_dy, CV_32F, 0, 1);

        cv::cartToPolar(df_dx, df_dy, mag, angle, true);
        mag *= (1.f / (8 * sqrt(2.f)));

        cv::Mat nmag;
        mag.convertTo(nmag, CV_8UC1);

        angle *=  6 / 360.f;

        for (int y = 0; y < h; ++y)
        {
            uchar* magnitude = nmag.ptr<uchar>(y);
            float* ang = angle.ptr<float>(y);

            for (int x = 0; x < w; ++x)
            {
                channels.ptr<uchar>(y + (h * (int)ang[x]))[x] = magnitude[x];
            }
        }

        cv::Mat luv, shrunk;
        cv::cvtColor(frame, luv, CV_BGR2Luv);

        std::vector<cv::Mat> splited;
        for (int i = 0; i < 3; ++i)
            splited.push_back(channels(cv::Rect(0, h * (7 + i), w, h)));
        split(luv, splited);

        float shrinkage = static_cast<float>(integrals.cols - 1) / channels.cols;

        CV_Assert(shrinkage == 0.25);

        cv::resize(channels, shrunk, cv::Size(), shrinkage, shrinkage, CV_INTER_AREA);
        cv::integral(shrunk, integrals, cv::noArray(), CV_32S);
    }
};

}

CV_INIT_ALGORITHM(ICFBuilder, "ChannelFeatureBuilder.ICFBuilder", );

cv::ChannelFeatureBuilder::~ChannelFeatureBuilder() {}

cv::Ptr<cv::ChannelFeatureBuilder> cv::ChannelFeatureBuilder::create()
{
    cv::Ptr<cv::ChannelFeatureBuilder> builder(new ICFBuilder());
    return builder;
}

cv::ChannelFeature::ChannelFeature(int x, int y, int w, int h, int ch)
: bb(cv::Rect(x, y, w, h)), channel(ch) {}

bool cv::ChannelFeature::operator ==(cv::ChannelFeature b)
{
    return bb == b.bb && channel == b.channel;
}

bool cv::ChannelFeature::operator !=(cv::ChannelFeature b)
{
    return bb != b.bb || channel != b.channel;
}


float cv::ChannelFeature::operator() (const cv::Mat& integrals, const cv::Size& model) const
{
    int step = model.width + 1;

    const int* ptr = integrals.ptr<int>(0) + (model.height * channel + bb.y) * step + bb.x;

    int a = ptr[0];
    int b = ptr[bb.width];

    ptr += bb.height * step;

    int c = ptr[bb.width];
    int d = ptr[0];

    return (float)(a - b + c - d);
}

void cv::write(cv::FileStorage& fs, const string&, const cv::ChannelFeature& f)
{
    fs << "{" << "channel" << f.channel << "rect" << f.bb << "}";
}

std::ostream& cv::operator<<(std::ostream& out, const cv::ChannelFeature& m)
{
    out << m.channel << " " << m.bb;
    return out;
}

cv::ChannelFeature::~ChannelFeature(){}
