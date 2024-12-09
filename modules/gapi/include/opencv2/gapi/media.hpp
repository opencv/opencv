// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_MEDIA_HPP
#define OPENCV_GAPI_MEDIA_HPP

#include <memory>     // unique_ptr<>, shared_ptr<>
#include <array>      // array<>
#include <functional> // function<>
#include <utility>    // forward<>()

#include <opencv2/gapi/gframe.hpp>
#include <opencv2/gapi/util/any.hpp>

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

/** \addtogroup gapi_data_structures
 * @{
 *
 * @brief Extra G-API data structures used to pass input/output data
 * to the graph for processing.
 */

/**
 * @brief cv::MediaFrame class represents an image/media frame
 * obtained from an external source.
 *
 * cv::MediaFrame represents image data as specified in
 * cv::MediaFormat. cv::MediaFrame is designed to be a thin wrapper over some
 * external memory of buffer; the class itself provides an uniform
 * interface over such types of memory. cv::MediaFrame wraps data from
 * a camera driver or from a media codec and provides an abstraction
 * layer over this memory to G-API. MediaFrame defines a compact interface
 * to access and manage the underlying data; the implementation is
 * fully defined by the associated Adapter (which is usually
 * user-defined).
 *
 * @sa cv::RMat
 */
class GAPI_EXPORTS MediaFrame {
public:
    /// This enum defines different types of cv::MediaFrame provided
    /// access to the underlying data. Note that different flags can't
    /// be combined in this version.
    enum class Access {
        R, ///< Access data for reading
        W, ///< Access data for writing
    };
    class IAdapter;
    class View;
    using AdapterPtr = std::unique_ptr<IAdapter>;

    /**
     * @brief Constructs an empty MediaFrame
     *
     * The constructed object has no any data associated with it.
     */
    MediaFrame();

    /**
     * @brief Constructs a MediaFrame with the given
     * Adapter. MediaFrame takes ownership over the passed adapter.
     *
     * @param p an unique pointer to instance of IAdapter derived class.
     */
    explicit MediaFrame(AdapterPtr &&p);

    /**
     * @overload
     * @brief Constructs a MediaFrame with the given parameters for
     * the Adapter. The adapter of type `T` is costructed on the fly.
     *
     * @param args list of arguments to construct an adapter of type
     * `T`.
     */
    template<class T, class... Args> static cv::MediaFrame Create(Args&&... args);

    /**
     * @brief Obtain access to the underlying data with the given
     * mode.
     *
     * Depending on the associated Adapter and the data wrapped, this
     * method may be cheap (e.g., the underlying memory is local) or
     * costly (if the underlying memory is external or device
     * memory).
     *
     * @param mode an access mode flag
     * @return a MediaFrame::View object. The views should be handled
     * carefully, refer to the MediaFrame::View documentation for details.
     */
    View access(Access mode) const;

    /**
     * @brief Returns a media frame descriptor -- the information
     * about the media format, dimensions, etc.
     * @return a cv::GFrameDesc
     */
    cv::GFrameDesc desc() const;

    // FIXME: design a better solution
    // Should be used only if the actual adapter provides implementation
    /// @private -- exclude from the OpenCV documentation for now.
    cv::util::any blobParams() const;

    /**
     * @brief Casts and returns the associated MediaFrame adapter to
     * the particular adapter type `T`, returns nullptr if the type is
     * different.
     *
     * This method may be useful if the adapter type is known by the
     * caller, and some lower level access to the memory is required.
     * Depending on the memory type, it may be more efficient than
     * access().
     *
     * @return a pointer to the adapter object, nullptr if the adapter
     * type is different.
     */
    template<typename T> T* get() const {
        static_assert(std::is_base_of<IAdapter, T>::value,
                      "T is not derived from cv::MediaFrame::IAdapter!");
        auto* adapter = getAdapter();
        GAPI_Assert(adapter != nullptr);
        return dynamic_cast<T*>(adapter);
    }

    /**
     * @brief Serialize MediaFrame's data to a byte array.
     *
     * @note The actual logic is implemented by frame's adapter class.
     * Does nothing by default.
     *
     * @param os Bytestream to store serialized MediaFrame data in.
     */
    void serialize(cv::gapi::s11n::IOStream& os) const;

private:
    struct Priv;
    std::shared_ptr<Priv> m;
    IAdapter* getAdapter() const;
};

template<class T, class... Args>
inline cv::MediaFrame cv::MediaFrame::Create(Args&&... args) {
    std::unique_ptr<T> ptr(new T(std::forward<Args>(args)...));
    return cv::MediaFrame(std::move(ptr));
}

/**
 * @brief Provides access to the MediaFrame's underlying data.
 *
 * This object contains the necessary information to access the pixel
 * data of the associated MediaFrame: arrays of pointers and strides
 * (distance between every plane row, in bytes) for every image
 * plane, as defined in cv::MediaFormat.
 * There may be up to four image planes in MediaFrame.
 *
 * Depending on the MediaFrame::Access flag passed in
 * MediaFrame::access(), a MediaFrame::View may be read- or
 * write-only.
 *
 * Depending on the MediaFrame::IAdapter implementation associated
 * with the parent MediaFrame, writing to memory with
 * MediaFrame::Access::R flag may have no effect or lead to
 * undefined behavior. Same applies to reading the memory with
 * MediaFrame::Access::W flag -- again, depending on the IAdapter
 * implementation, the host-side buffer the view provides access to
 * may have no current data stored in (so in-place editing of the
 * buffer contents may not be possible).
 *
 * MediaFrame::View objects must be handled carefully, as an external
 * resource associated with MediaFrame may be locked for the time the
 * MediaFrame::View object exists. Obtaining MediaFrame::View should
 * be seen as "map" and destroying it as "unmap" in the "map/unmap"
 * idiom (applicable to OpenCL, device memory, remote
 * memory).
 *
 * When a MediaFrame buffer is accessed for writing, and the memory
 * under MediaFrame::View::Ptrs is altered, the data synchronization
 * of a host-side and device/remote buffer is not guaranteed until the
 * MediaFrame::View is destroyed. In other words, the real data on the
 * device or in a remote target may be updated at the MediaFrame::View
 * destruction only -- but it depends on the associated
 * MediaFrame::IAdapter implementation.
 */
class GAPI_EXPORTS MediaFrame::View final {
public:
    static constexpr const size_t MAX_PLANES = 4;
    using Ptrs     = std::array<void*, MAX_PLANES>;
    using Strides  = std::array<std::size_t, MAX_PLANES>; // in bytes
    using Callback = std::function<void()>;

    /// @private
    View(Ptrs&& ptrs, Strides&& strs, Callback &&cb = [](){});

    /// @private
    View(const View&) = delete;

    /// @private
    View(View&&) = default;

    /// @private
    View& operator = (const View&) = delete;

    ~View();

    Ptrs    ptr; ///< Array of image plane pointers
    Strides stride; ///< Array of image plane strides, in bytes.

private:
    Callback m_cb;
};

/**
 * @brief An interface class for MediaFrame data adapters.
 *
 * Implement this interface to wrap media data in the MediaFrame. It
 * makes sense to implement this class if there is a custom
 * cv::gapi::wip::IStreamSource defined -- in this case, a stream
 * source can produce MediaFrame objects with this adapter and the
 * media data may be passed to graph without any copy. For example, a
 * GStreamer-based stream source can implement an adapter over
 * `GstBuffer` and G-API will transparently use it in the graph.
 */
class GAPI_EXPORTS MediaFrame::IAdapter {
public:
    virtual ~IAdapter() = 0;
    virtual cv::GFrameDesc meta() const = 0;
    virtual MediaFrame::View access(MediaFrame::Access) = 0;
    // FIXME: design a better solution
    // The default implementation does nothing
    virtual cv::util::any blobParams() const;
    virtual void serialize(cv::gapi::s11n::IOStream&) {
        GAPI_Error("Generic serialize method of MediaFrame::IAdapter does nothing by default. "
                             "Please, implement it in derived class to properly serialize the object.");
    }
    virtual void deserialize(cv::gapi::s11n::IIStream&) {
        GAPI_Error("Generic deserialize method of MediaFrame::IAdapter does nothing by default. "
                             "Please, implement it in derived class to properly deserialize the object.");
    }
};
/** @} */

} //namespace cv

#endif // OPENCV_GAPI_MEDIA_HPP
