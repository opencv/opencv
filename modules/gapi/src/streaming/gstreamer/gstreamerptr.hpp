// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERPTR_HPP
#define OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERPTR_HPP

#include <opencv2/gapi.hpp>

#include <utility>

#ifdef HAVE_GSTREAMER
#include <gst/gst.h>
#include <gst/video/video-frame.h>

namespace cv {
namespace gapi {
namespace wip {
namespace gst {

template<typename T> static inline void GStreamerPtrUnrefObject(T* ptr)
{
    if (ptr)
    {
        gst_object_unref(G_OBJECT(ptr));
    }
}

template<typename T> static inline void GStreamerPtrRelease(T* ptr);

template<> inline void GStreamerPtrRelease<GError>(GError* ptr)
{
    if (ptr)
    {
        g_error_free(ptr);
    }
}

template<> inline void GStreamerPtrRelease<GstElement>(GstElement* ptr)
{
    GStreamerPtrUnrefObject<GstElement>(ptr);
}

template<> inline void GStreamerPtrRelease<GstElementFactory>(GstElementFactory* ptr)
{
    GStreamerPtrUnrefObject<GstElementFactory>(ptr);
}

template<> inline void GStreamerPtrRelease<GstPad>(GstPad* ptr)
{
    GStreamerPtrUnrefObject<GstPad>(ptr);
}

template<> inline void GStreamerPtrRelease<GstBus>(GstBus* ptr)
{
    GStreamerPtrUnrefObject<GstBus>(ptr);
}

template<> inline void GStreamerPtrRelease<GstAllocator>(GstAllocator* ptr)
{
    GStreamerPtrUnrefObject<GstAllocator>(ptr);
}

template<> inline void GStreamerPtrRelease<GstVideoInfo>(GstVideoInfo* ptr)
{
    if (ptr)
    {
        gst_video_info_free(ptr);
    }
}

template<> inline void GStreamerPtrRelease<GstCaps>(GstCaps* ptr)
{
    if (ptr)
    {
        gst_caps_unref(ptr);
    }
}

template<> inline void GStreamerPtrRelease<GstMemory>(GstMemory* ptr)
{
    if (ptr)
    {
        gst_memory_unref(ptr);
    }
}

template<> inline void GStreamerPtrRelease<GstBuffer>(GstBuffer* ptr)
{
    if (ptr)
    {
        gst_buffer_unref(ptr);
    }
}

template<> inline void GStreamerPtrRelease<GstSample>(GstSample* ptr)
{
    if (ptr)
    {
        gst_sample_unref(ptr);
    }
}

template<> inline void GStreamerPtrRelease<GstMessage>(GstMessage* ptr)
{
    if (ptr)
    {
        gst_message_unref(ptr);
    }
}

template<> inline void GStreamerPtrRelease<GstIterator>(GstIterator* ptr)
{
    if (ptr)
    {
        gst_iterator_free(ptr);
    }
}

template<> inline void GStreamerPtrRelease<GstQuery>(GstQuery* ptr)
{
    if (ptr)
    {
        gst_query_unref(ptr);
    }
}

template<> inline void GStreamerPtrRelease<char>(char* ptr)
{
    if (ptr)
    {
        g_free(ptr);
    }
}

// NOTE: The main concept of this class is to be owner of some passed to it piece of memory.
//       (be owner = free this memory or reduce reference count to it after use).
//       More specifically, GStreamerPtr is designed to own memory returned from GStreamer/GLib
//       functions, which are marked as [transfer full] in documentation.
//       [transfer full] means that function fully transfers ownership of returned memory to the
//       receiving piece of code.
//
//       Memory ownership and ownership transfer concept:
// https://developer.gnome.org/programming-guidelines/stable/memory-management.html.en#g-clear-object

// NOTE: GStreamerPtr can only own strong references, not floating ones.
//       For floating references please call g_object_ref_sink(reference) before wrapping
//       it with GStreamerPtr.
//       See https://developer.gnome.org/gobject/stable/gobject-The-Base-Object-Type.html#floating-ref
//       for floating references.
// NOTE: GStreamerPtr doesn't support pointers to arrays, only pointers to single objects.
template<typename T> class GStreamerPtr :
    public std::unique_ptr<T, decltype(&GStreamerPtrRelease<T>)>
{
    using BaseClass = std::unique_ptr<T, decltype(&GStreamerPtrRelease<T>)>;

public:
    constexpr GStreamerPtr() noexcept : BaseClass(nullptr, GStreamerPtrRelease<T>) { }
    constexpr GStreamerPtr(std::nullptr_t) noexcept : BaseClass(nullptr, GStreamerPtrRelease<T>) { }
    explicit GStreamerPtr(typename BaseClass::pointer p) noexcept :
        BaseClass(p, GStreamerPtrRelease<T>) { }

    GStreamerPtr& operator=(T* p) noexcept { *this = std::move(GStreamerPtr<T>(p)); return *this; }

    inline operator T*() noexcept { return this->get(); }
    // There is no const correctness in GStreamer C API
    inline operator /*const*/ T*() const noexcept { return (T*)this->get(); }
};

} // namespace gst
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_GSTREAMER
#endif // OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMERPTR_HPP
