// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_FLUID_BUFFER_HPP
#define OPENCV_GAPI_FLUID_BUFFER_HPP

#include <list>
#include <numeric> // accumulate
#include <ostream> // ostream
#include <cstdint> // uint8_t

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/own/mat.hpp>
#include <opencv2/gapi/gmat.hpp>

#include "opencv2/gapi/util/optional.hpp"
#include "opencv2/gapi/own/scalar.hpp"
#include "opencv2/gapi/own/mat.hpp"

namespace cv {
namespace gapi {
namespace fluid {

struct Border
{
#if !defined(GAPI_STANDALONE)
    // This constructor is required to support existing kernels which are part of G-API
    Border(int _type, cv::Scalar _val) : type(_type), value(to_own(_val)) {};
#endif // !defined(GAPI_STANDALONE)
    Border(int _type, cv::gapi::own::Scalar _val) : type(_type), value(_val) {};
    int type;
    cv::gapi::own::Scalar value;
};

using BorderOpt = util::optional<Border>;

bool operator == (const Border& b1, const Border& b2);

class GAPI_EXPORTS Buffer;

class GAPI_EXPORTS View
{
public:
    View() = default;

    const uint8_t* InLineB(int index) const; // -(w-1)/2...0...+(w-1)/2 for Filters
    template<typename T> const inline T* InLine(int i) const
    {
        const uint8_t* ptr = this->InLineB(i);
        return reinterpret_cast<const T*>(ptr);
    }

    operator bool() const;
    bool ready() const;
    int length() const;
    int y() const;

    const GMatDesc& meta() const;

    class GAPI_EXPORTS Priv;      // internal use only
    Priv& priv();               // internal use only
    const Priv& priv() const;   // internal use only

    View(Priv* p);

private:
    std::shared_ptr<Priv> m_priv;
};

class GAPI_EXPORTS Buffer
{
public:
    // Default constructor (executable creation stage,
    // all following initialization performed in Priv::init())
    Buffer();
    // Scratch constructor (user kernels)
    Buffer(const cv::GMatDesc &desc);

    // Constructor for intermediate buffers (for tests)
    Buffer(const cv::GMatDesc &desc,
           int max_line_consumption, int border_size,
           int skew,
           int wlpi,
           BorderOpt border);
    // Constructor for in/out buffers (for tests)
    Buffer(const cv::gapi::own::Mat &data, bool is_input);

    uint8_t* OutLineB(int index = 0);
    template<typename T> inline T* OutLine(int index = 0)
    {
        uint8_t* ptr = this->OutLineB(index);
        return reinterpret_cast<T*>(ptr);
    }

    int y() const;

    int linesReady() const;
    void debug(std::ostream &os) const;
    int length() const;
    int lpi() const;  // LPI for WRITER

    const GMatDesc& meta() const;

    View mkView(int borderSize, bool ownStorage);

    class GAPI_EXPORTS Priv;      // internal use only
    Priv& priv();               // internal use only
    const Priv& priv() const;   // internal use only

private:
    std::shared_ptr<Priv> m_priv;

};

} // namespace cv::gapi::fluid
} // namespace cv::gapi
} // namespace cv

#endif // OPENCV_GAPI_FLUID_BUFFER_HPP
