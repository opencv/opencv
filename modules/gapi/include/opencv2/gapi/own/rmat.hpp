// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_OWN_RMAT_HPP
#define OPENCV_GAPI_OWN_RMAT_HPP

#include <opencv2/gapi/gmat.hpp>

namespace cv { namespace gapi { namespace own {

class RMat
{
public:
    class Adapter
    {
    public:
        virtual ~Adapter() = default;
        virtual GMatDesc desc() const = 0;
        virtual cv::Mat access() const = 0;
        virtual void flush() const = 0;
    };
    using AdapterP = std::shared_ptr<Adapter>;

    RMat() = default;

    // FIXME? make private?
    RMat(AdapterP&& a) : m_adapter(std::move(a)) {}

    GMatDesc desc() const { return m_adapter->desc(); }
    cv::Mat access() const { return m_adapter->access(); }
    void flush() const { return m_adapter->flush(); }

    // TODO:
    // think on better names
    template<typename T> bool is() const
    {
        static_assert(std::is_base_of<Adapter, T>::value, "T is not derived from Adapter!");
        return dynamic_cast<T*>(m_adapter.get()) != nullptr;
    }

    template<typename T> T& as() const
    {
        static_assert(std::is_base_of<Adapter, T>::value, "T is not derived from Adapter!");
        return dynamic_cast<T&>(*m_adapter.get());
    }

private:
    AdapterP m_adapter;
};

template<typename T, typename... Ts>
RMat make_rmat(Ts&&... args) { return { std::make_shared<T>(std::forward<Ts...>(args)...) }; }

} //namespace own
} //namespace gapi
} //namespace cv

#endif /* OPENCV_GAPI_OWN_RMAT_HPP */
