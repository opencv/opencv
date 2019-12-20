// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GCALL_HPP
#define OPENCV_GAPI_GCALL_HPP

#include <opencv2/gapi/garg.hpp>      // GArg
#include <opencv2/gapi/gmat.hpp>      // GMat
#include <opencv2/gapi/gscalar.hpp>   // GScalar
#include <opencv2/gapi/garray.hpp>    // GArray<T>

namespace cv {

struct GKernel;

// The whole idea of this class is to represent an operation
// which is applied to arguments. This is part of public API,
// since it is what users should use to define kernel interfaces.

class GAPI_EXPORTS GCall final
{
public:
    class Priv;

    explicit GCall(const GKernel &k);
    ~GCall();

    template<typename... Ts>
    GCall& pass(Ts&&... args)
    {
        setArgs({cv::GArg(std::move(args))...});
        return *this;
    }

    // A generic yield method - obtain a link to operator's particular GMat output
    GMat    yield      (int output = 0);
    GMatP   yieldP     (int output = 0);
    GScalar yieldScalar(int output = 0);

    template<class T> GArray<T> yieldArray(int output = 0)
    {
        return GArray<T>(yieldArray(output));
    }

    // Internal use only
    Priv& priv();
    const Priv& priv() const;

protected:
    std::shared_ptr<Priv> m_priv;

    void setArgs(std::vector<GArg> &&args);

    // Public version returns a typed array, this one is implementation detail
    detail::GArrayU yieldArray(int output = 0);
};

} // namespace cv

#endif // OPENCV_GAPI_GCALL_HPP
