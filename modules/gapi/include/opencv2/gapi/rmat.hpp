// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_RMAT_HPP
#define OPENCV_GAPI_RMAT_HPP

#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/own/exports.hpp>

// Forward declaration
namespace cv {
namespace gapi {
namespace s11n {
    struct IOStream;
    struct IIStream;
} // namespace s11n
} // namespace gapi
} // namespace cv

namespace cv {

// "Remote Mat", a general class which provides an abstraction layer over the data
// storage and placement (host, remote device etc) and allows to access this data.
//
// The device specific implementation is hidden in the RMat::Adapter class
//
// The basic flow is the following:
// * Backend which is aware of the remote device:
//   - Implements own AdapterT class which is derived from RMat::Adapter
//   - Wraps device memory into RMat via make_rmat utility function:
//         cv::RMat rmat = cv::make_rmat<AdapterT>(args);
//
// * End user:
//   - Writes the code which works with RMats without any knowledge of the remote device:
//     void func(const cv::RMat& in_rmat, cv::RMat& out_rmat) {
//         // Fetch input data from the device, get mapped memory for output
//         cv::RMat::View  in_view =  in_rmat.access(Access::R);
//         cv::RMat::View out_view = out_rmat.access(Access::W);
//         performCalculations(in_view, out_view);
//         // data from out_view is transferred to the device when out_view is destroyed
//     }
/** \addtogroup gapi_data_structures
 * @{
 */
class GAPI_EXPORTS RMat
{
public:
    // A lightweight wrapper on image data:
    // - Doesn't own the memory;
    // - Doesn't implement copy semantics (it's assumed that a view is created each time
    // wrapped data is being accessed);
    // - Has an optional callback which is called when the view is destroyed.
    class GAPI_EXPORTS View
    {
    public:
        using DestroyCallback = std::function<void()>;
        using stepsT = std::vector<size_t>;

        View() = default;
        View(const GMatDesc& desc, uchar* data, const stepsT& steps = {}, DestroyCallback&& cb = nullptr);
        View(const GMatDesc& desc, uchar* data, size_t step, DestroyCallback&& cb = nullptr);

        View(const View&) = delete;
        View& operator=(const View&) = delete;
        View(View&&) = default;
        View& operator=(View&& v);
        ~View() { if (m_cb) m_cb(); }

        cv::Size size() const { return m_desc.size; }
        const std::vector<int>& dims() const { return m_desc.dims; }
        int cols() const { return m_desc.size.width; }
        int rows() const { return m_desc.size.height; }
        int type() const;
        int depth() const { return m_desc.depth; }
        int chan() const { return m_desc.chan; }
        size_t elemSize() const { return CV_ELEM_SIZE(type()); }

        template<typename T = uchar> T* ptr(int y = 0) {
            return reinterpret_cast<T*>(m_data + step()*y);
        }
        template<typename T = uchar> const T* ptr(int y = 0) const {
            return reinterpret_cast<T*>(m_data + step()*y);
        }
        template<typename T = uchar> T* ptr(int y, int x) {
            return reinterpret_cast<T*>(m_data + step()*y + step(1)*x);
        }
        template<typename T = uchar> const T* ptr(int y, int x) const {
            return reinterpret_cast<const T*>(m_data + step()*y + step(1)*x);
        }
        size_t step(size_t i = 0) const { GAPI_DbgAssert(i<m_steps.size()); return m_steps[i]; }
        const stepsT& steps() const { return m_steps; }

    private:
        GMatDesc m_desc;
        uchar* m_data = nullptr;
        stepsT m_steps = {0u};
        DestroyCallback m_cb = nullptr;
    };

    enum class Access { R, W };
    class Adapter
    {
    public:
        virtual ~Adapter() = default;
        virtual GMatDesc desc() const = 0;
        // Implementation is responsible for setting the appropriate callback to
        // the view when accessed for writing, to ensure that the data from the view
        // is transferred to the device when the view is destroyed
        virtual View access(Access) = 0;
        virtual void serialize(cv::gapi::s11n::IOStream&) {
            GAPI_Assert(false && "Generic serialize method should never be called for RMat adapter");
        }
        virtual void deserialize(cv::gapi::s11n::IIStream&) {
            GAPI_Assert(false && "Generic deserialize method should never be called for RMat adapter");
        }
    };
    using AdapterP = std::shared_ptr<Adapter>;

    RMat() = default;
    RMat(AdapterP&& a) : m_adapter(std::move(a)) {}
    GMatDesc desc() const { return m_adapter->desc(); }

    // Note: When accessed for write there is no guarantee that returned view
    // will contain actual snapshot of the mapped device memory
    // (no guarantee that fetch from a device is performed). The only
    // guaranty is that when the view is destroyed, its data will be
    // transferred to the device
    View access(Access a) const { return m_adapter->access(a); }

    // Cast underlying RMat adapter to the particular adapter type,
    // return nullptr if underlying type is different
    template<typename T> T* get() const
    {
        static_assert(std::is_base_of<Adapter, T>::value, "T is not derived from Adapter!");
        GAPI_Assert(m_adapter != nullptr);
        return dynamic_cast<T*>(m_adapter.get());
    }

    void serialize(cv::gapi::s11n::IOStream& os) const {
        m_adapter->serialize(os);
    }

private:
    AdapterP m_adapter = nullptr;
};

template<typename T, typename... Ts>
RMat make_rmat(Ts&&... args) { return { std::make_shared<T>(std::forward<Ts>(args)...) }; }
/** @} */

} //namespace cv

#endif /* OPENCV_GAPI_RMAT_HPP */
