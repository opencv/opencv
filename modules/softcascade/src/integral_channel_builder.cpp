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

using namespace cv::softcascade;

class HOG6MagLuv : public ChannelFeatureBuilder
{
    enum {N_CHANNELS = 10};
public:
    virtual ~HOG6MagLuv() {}
    virtual cv::AlgorithmInfo* info() const;

    virtual int totalChannels() const {return N_CHANNELS; }

    virtual void operator()(cv::InputArray _frame, cv::OutputArray _integrals, cv::Size channelsSize) const
    {
        CV_Assert(_frame.type() == CV_8UC3);

        cv::Mat frame      = _frame.getMat();
        int h = frame.rows;
        int w = frame.cols;

        if (channelsSize != cv::Size())
            _integrals.create(channelsSize.height * N_CHANNELS + 1, channelsSize.width + 1, CV_32SC1);

        if(_integrals.empty())
            _integrals.create(frame.rows * N_CHANNELS + 1, frame.cols + 1, CV_32SC1);

        cv::Mat& integrals = _integrals.getMatRef();

        cv::Mat channels, gray;

        channels.create(h * N_CHANNELS, w, CV_8UC1);
        channels.setTo(0);

        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::Mat df_dx, df_dy, mag, angle;
        cv::Sobel(gray, df_dx, CV_32F, 1, 0);
        cv::Sobel(gray, df_dy, CV_32F, 0, 1);

        cv::cartToPolar(df_dx, df_dy, mag, angle, true);
        mag *= (1.f / (8 * sqrt(2.f)));

        cv::Mat nmag = channels(cv::Rect(0, h * (N_CHANNELS - 4), w, h));
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
        cv::cvtColor(frame, luv, cv::COLOR_BGR2Luv);

        std::vector<cv::Mat> splited;
        for (int i = 0; i < 3; ++i)
            splited.push_back(channels(cv::Rect(0, h * (7 + i), w, h)));
        split(luv, splited);
        cv::resize(channels, shrunk, cv::Size(integrals.cols - 1, integrals.rows - 1), -1 , -1, cv::INTER_AREA);
        cv::integral(shrunk, integrals, cv::noArray(), CV_32S);
    }
};

}

using cv::softcascade::ChannelFeatureBuilder;
using cv::softcascade::ChannelFeature;

CV_INIT_ALGORITHM(HOG6MagLuv,  "ChannelFeatureBuilder.HOG6MagLuv", )

ChannelFeatureBuilder::~ChannelFeatureBuilder() {}

cv::Ptr<ChannelFeatureBuilder> ChannelFeatureBuilder::create(const cv::String& featureType)
{
    return Algorithm::create<ChannelFeatureBuilder>("ChannelFeatureBuilder." + featureType);
}

ChannelFeature::ChannelFeature(int x, int y, int w, int h, int ch)
: bb(cv::Rect(x, y, w, h)), channel(ch) {}

bool ChannelFeature::operator ==(ChannelFeature b)
{
    return bb == b.bb && channel == b.channel;
}

bool ChannelFeature::operator !=(ChannelFeature b)
{
    return bb != b.bb || channel != b.channel;
}


float ChannelFeature::operator() (const cv::Mat& integrals, const cv::Size& model) const
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

void cv::softcascade::write(cv::FileStorage& fs, const cv::String&, const ChannelFeature& f)
{
    fs << "{" << "channel" << f.channel << "rect" << f.bb << "}";
}

std::ostream& cv::softcascade::operator<<(std::ostream& out, const ChannelFeature& m)
{
    return out << m.channel << " " << "[" << m.bb.width << " x " << m.bb.height << " from (" << m.bb.x << ", " << m.bb.y << ")]";
}

ChannelFeature::~ChannelFeature(){}

namespace {

using namespace cv::softcascade;

class ChannelFeaturePool : public FeaturePool
{
public:
    ChannelFeaturePool(cv::Size m, int n, int ch) : FeaturePool(), model(m), N_CHANNELS(ch)
    {
        CV_Assert(m != cv::Size() && n > 0 && (ch == 10 || ch == 8));
        fill(n);
    }

    virtual int size() const { return (int)pool.size(); }
    virtual float apply(int fi, int si, const cv::Mat& integrals) const;
    virtual void write( cv::FileStorage& fs, int index) const;

    virtual ~ChannelFeaturePool() {}

private:

    void fill(int desired);

    cv::Size model;
    std::vector<ChannelFeature> pool;
    int N_CHANNELS;
};

float ChannelFeaturePool::apply(int fi, int si, const cv::Mat& integrals) const
{
    return pool[fi](integrals.row(si), model);
}

void ChannelFeaturePool::write( cv::FileStorage& fs, int index) const
{

    CV_Assert((index >= 0) && (index < (int)pool.size()));
    fs << pool[index];
}

void ChannelFeaturePool::fill(int desired)
{
    using namespace cv::softcascade::internal;
    int mw = model.width;
    int mh = model.height;

    int maxPoolSize = (mw -1) * mw / 2 * (mh - 1) * mh / 2 * N_CHANNELS;

    int nfeatures = std::min(desired, maxPoolSize);
    pool.reserve(nfeatures);

    Random::engine eng((Random::seed_type)FEATURE_RECT_SEED);
    Random::engine eng_ch(DCHANNELS_SEED);

    Random::uniform chRand(0, N_CHANNELS - 1);

    Random::uniform xRand(0, model.width  - 2);
    Random::uniform yRand(0, model.height - 2);

    Random::uniform wRand(1, model.width  - 1);
    Random::uniform hRand(1, model.height - 1);

    while (pool.size() < size_t(nfeatures))
    {
        int x = xRand(eng);
        int y = yRand(eng);

        int w = 1 + wRand(eng, model.width  - x - 1);
        int h = 1 + hRand(eng, model.height - y - 1);

        CV_Assert(w > 0);
        CV_Assert(h > 0);

        CV_Assert(w + x < model.width);
        CV_Assert(h + y < model.height);

        int ch = chRand(eng_ch);

        ChannelFeature f(x, y, w, h, ch);

        if (std::find(pool.begin(), pool.end(),f) == pool.end())
        {
            pool.push_back(f);
        }
    }
}

}

cv::Ptr<FeaturePool> FeaturePool::create(const cv::Size& model, int nfeatures, int nchannels )
{
    cv::Ptr<FeaturePool> pool(new ChannelFeaturePool(model, nfeatures, nchannels));
    return pool;
}
