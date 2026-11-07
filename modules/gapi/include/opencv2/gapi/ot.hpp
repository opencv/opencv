// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#ifndef OPENCV_GAPI_OT_HPP
#define OPENCV_GAPI_OT_HPP

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/s11n.hpp>
#include <opencv2/gapi/gkernel.hpp>

namespace cv {
namespace gapi {
/**
 * @brief This namespace contains G-API Operation Types for
 * VAS Object Tracking module functionality.
 */
namespace ot {

/**
 * @enum TrackingStatus
 *
 * Tracking status twin for vas::ot::TrackingStatus
 */
enum TrackingStatus
{
    NEW = 0,     /**< The object is newly added. */
    TRACKED,     /**< The object is being tracked. */
    LOST         /**< The object gets lost now. The object can be tracked again
                      by specifying detected object manually. */
};

struct GAPI_EXPORTS_W_SIMPLE ObjectTrackerParams
{
    /**
     * Maximum number of trackable objects in a frame.
     * Valid range: 1 <= max_num_objects. Or it can be -1 if there is no limitation
     * of maximum number in X86. KMB/TBH has limitation up to 1024.
     * Default value is -1 which means there is no limitation in X86. KMB/TBH is -1 means 200.
     */
    GAPI_PROP_RW int32_t max_num_objects = -1;

    /**
     * Input color format. Supports 0(BGR), 1(NV12), 2(BGRX) and 4(I420)
     */
    GAPI_PROP_RW int32_t input_image_format = 0;

    /**
     * Specifies whether tracker to use detection class for keeping id of an object.
     * If it is true, new detection will be associated from previous tracking only when
     * those two have same class.
     * class id of an object is fixed across video frames.
     * If it is false, new detection can be associated across different-class objects.
     * In this case, the class id of an object may change across video frames depending on the tracker input.
     * It is recommended to turn this option off when it is likely that detector confuses the class of object.
     * For example, when detector confuses bicycle and motorbike. Turning this option off will increase
     * the tracking reliability as tracker will ignore the class label of detector.
     * @n
     * Default value is true.
     */
    GAPI_PROP_RW bool tracking_per_class = true;

    bool operator==(const ObjectTrackerParams& other) const
    {
        return max_num_objects == other.max_num_objects
            && input_image_format == other.input_image_format
            && tracking_per_class == other.tracking_per_class;
    }
};

using GTrackedInfo = std::tuple<cv::GArray<cv::Rect>, cv::GArray<int32_t>, cv::GArray<uint64_t>, cv::GArray<int>>;

G_API_OP(GTrackFromMat, <GTrackedInfo(cv::GMat, cv::GArray<cv::Rect>, cv::GArray<int32_t>, float)>, "com.intel.track_from_mat")
{
    static std::tuple<cv::GArrayDesc, cv::GArrayDesc,
                      cv::GArrayDesc, cv::GArrayDesc> outMeta(cv::GMatDesc, cv::GArrayDesc, cv::GArrayDesc, float)
    {
        return std::make_tuple(cv::empty_array_desc(), cv::empty_array_desc(),
                               cv::empty_array_desc(), cv::empty_array_desc());
    }
};

G_API_OP(GTrackFromFrame, <GTrackedInfo(cv::GFrame, cv::GArray<cv::Rect>, cv::GArray<int32_t>, float)>, "com.intel.track_from_frame")
{
    static std::tuple<cv::GArrayDesc, cv::GArrayDesc,
                      cv::GArrayDesc, cv::GArrayDesc> outMeta(cv::GFrameDesc, cv::GArrayDesc, cv::GArrayDesc, float)
    {
       return std::make_tuple(cv::empty_array_desc(), cv::empty_array_desc(),
                              cv::empty_array_desc(), cv::empty_array_desc());
    }
};

/**
 * @brief   Tracks objects with video frames.
 *          If a detected object is overlapped enough with one of tracked object, the tracked object's
 *          informationis updated with the input detected object.
 *          On the other hand, if a detected object is overlapped with none of tracked objects,
 *          the detected object is newly added and ObjectTracker starts to track the object.
 *          In zero term tracking type, ObjectTracker clears tracked objects in case that empty
 *          list of detected objects is passed in.
 *
 * @param mat                       Input frame.
 * @param detected_rects            Detected objects rectangles in the input frame.
 * @param detected_class_labels     Detected objects class labels in the input frame.
 * @param delta                     Frame_delta_t Delta time between two consecutive tracking in seconds.
 *                                  The valid range is [0.005 ~ 0.5].
 * @return                          Tracking results of target objects.
 *                                  cv::GArray<cv::Rect>  Array of rectangles for tracked objects.
 *                                  cv::GArray<int32_t>   Array of detected objects labels.
 *                                  cv::GArray<uint64_t>  Array of tracking IDs for objects.
 *                                                        Numbering sequence starts from 1.
 *                                                        The value 0 means the tracking ID of this object has
 *                                                        not been assigned.
 *                                  cv::GArray<int>       Array of tracking statuses for objects.
 */
GAPI_EXPORTS_W std::tuple<cv::GArray<cv::Rect>,
                          cv::GArray<int>,
                          cv::GArray<uint64_t>,
                          cv::GArray<int>>
    track(const cv::GMat& mat,
          const cv::GArray<cv::Rect>& detected_rects,
          const cv::GArray<int>& detected_class_labels,
          float delta);


/**
   @overload
 * @brief   Tracks objects with video frames. Overload of track(...) for frame as GFrame.
 *
 * @param frame                     Input frame.
 * @param detected_rects            Detected objects rectangles in the input frame.
 * @param detected_class_labels     Detected objects class labels in the input frame.
 * @param delta                     Frame_delta_t Delta time between two consecutive tracking in seconds.
 *                                  The valid range is [0.005 ~ 0.5].
 * @return                          Tracking results of target objects.
 * @return                          Tracking results of target objects.
 *                                  cv::GArray<cv::Rect>          Array of rectangles for tracked objects.
 *                                  cv::GArray<int32_t>           Array of detected objects labels.
 *                                  cv::GArray<uint64_t>          Array of tracking IDs for objects.
 *                                                                Numbering sequence starts from 1.
 *                                                                The value 0 means the tracking ID of this object has
 *                                                                not been assigned.
 *                                  cv::GArray<int>    Array of tracking statuses for objects.
 */
GAPI_EXPORTS_W std::tuple<cv::GArray<cv::Rect>,
                         cv::GArray<int>,
                         cv::GArray<uint64_t>,
                         cv::GArray<int>>
    track(const cv::GFrame& frame,
          const cv::GArray<cv::Rect>& detected_rects,
          const cv::GArray<int>& detected_class_labels,
          float delta);
} // namespace ot
} // namespace gapi
} // namespace cv

// FIXME: move to a separate file?
namespace cv
{
namespace detail
{
template<> struct CompileArgTag<cv::gapi::ot::ObjectTrackerParams>
{
    static const char* tag()
    {
        return "cv.gapi.ot.object_tracker_params";
    }
};
} // namespace detail

namespace gapi
{
namespace s11n
{
namespace detail
{
template<> struct S11N<cv::gapi::ot::ObjectTrackerParams> {
    static void serialize(IOStream &os, const cv::gapi::ot::ObjectTrackerParams &p) {
        os << p. max_num_objects << p.input_image_format << p.tracking_per_class;
    }
    static cv::gapi::ot::ObjectTrackerParams deserialize(IIStream &is) {
        cv::gapi::ot::ObjectTrackerParams p;
        is >> p. max_num_objects >> p.input_image_format >> p.tracking_per_class;
        return p;
    }
};
} // namespace detail
} // namespace s11n
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_OT_HPP
