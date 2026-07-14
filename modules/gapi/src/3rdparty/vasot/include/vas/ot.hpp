/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef VAS_OT_HPP
#define VAS_OT_HPP

#include <vas/common.hpp>

#include <opencv2/core.hpp>

#include <iostream>
#include <map>
#include <memory>
#include <vector>

namespace vas {

/**
 * @namespace vas::ot
 * @brief %vas::ot namespace.
 *
 * The ot namespace has classes, functions, and definitions for object tracker.
 * It is a general tracker, and an object is represented as a rectangular box.
 * Thus, you can use any kind of detector if it generates a rectangular box as output.
 * Once an object is added to object tracker, the object is started to be tracked.
 */
namespace ot {

/**
 * Returns current version.
 */
VAS_EXPORT vas::Version GetVersion() noexcept;

/**
 * @enum TrackingType
 *
 * Tracking type.
 */
enum class TrackingType {
    LONG_TERM,
    SHORT_TERM,
    ZERO_TERM,
    SHORT_TERM_KCFVAR,
    SHORT_TERM_IMAGELESS,
    ZERO_TERM_IMAGELESS,
    ZERO_TERM_COLOR_HISTOGRAM
};

/**
 * @enum TrackingStatus
 *
 * Tracking status.
 */
enum class TrackingStatus {
    NEW = 0, /**< The object is newly added. */
    TRACKED, /**< The object is being tracked. */
    LOST     /**< The object gets lost now. The object can be tracked again automatically(long term tracking) or by
                specifying detected object manually(short term and zero term tracking). */
};

/**
 * @class DetectedObject
 * @brief Represents an input object.
 *
 * In order to track an object, detected object should be added one or more times to ObjectTracker.
 * When an object is required to be added to ObjectTracker, you can create an instance of this class and fill its
 * values.
 */
class DetectedObject {
  public:
    /**
     * Default constructor.
     */
    DetectedObject() : rect(), class_label() {
    }

    /**
     * Constructor with specific values.
     *
     * @param[in] input_rect Rectangle of input object.
     * @param[in] input_class_label Class label of input object.
     */
    DetectedObject(const cv::Rect &input_rect, int32_t input_class_label)
        : rect(input_rect), class_label(input_class_label) {
    }

  public:
    /**
     * Object rectangle.
     */
    cv::Rect rect;

    /**
     * Input class label.
     * It is an arbitrary value that is specified by user.
     * You can utilize this value to categorize input objects.
     * Same value will be assigned to the class_label in Object class.
     */
    int32_t class_label;
};

/**
 * @class Object
 * @brief Represents tracking result of a target object.
 *
 * It contains tracking information of a target object.
 * ObjectTracker generates an instance of this class per tracked object when Track method is called.
 */
class Object {
  public:
    /**
     * Object rectangle.
     */
    cv::Rect rect;

    /**
     * Tracking ID.
     * Numbering sequence starts from 1.
     * The value 0 means the tracking ID of this object has not been assigned.
     */
    uint64_t tracking_id;

    /**
     * Class label.
     * This is specified by DetectedObject.
     */
    int32_t class_label;

    /**
     * Tracking status.
     */
    TrackingStatus status;

    /**
     * Index in the DetectedObject vector.
     * If the Object was not in detection input at this frame, then it will be -1.
     */
    int32_t association_idx;
};

VAS_EXPORT std::ostream &operator<<(std::ostream &os, TrackingStatus ts);
VAS_EXPORT std::ostream &operator<<(std::ostream &os, const Object &object);

/**
 * @class ObjectTracker
 * @brief Tracks objects from video frames.
 *
 * This class tracks objects from the input frames.
 * In order to create an instance of this class, you need to use ObjectTracker::Builder class.
 * @n
 * ObjectTracker can run in three different ways as TrackingType defines.
 * @n
 * In short term tracking, an object is added at the beginning, and the object is tracked with consecutive input frames.
 * It is recommended to update the tracked object's information for every 10-20 frames.
 * @n
 * Zero term tracking can be thought as association between a detected object and tracked object.
 * Detected objects should always be added when Track method is invoked.
 * For each frame, detected objects are mapped to tracked objects with this tracking type, which enables ID tracking of
 detected objects.
 * @n
 * Long term tracking is deprecated.
 * In long term tracking, an object is added at the beginning, and the object is tracked with consecutive input frames.
 * User doesn't need to update manually the object's information.
 * Long term tracking takes relatively long time to track objects.
 * @n
 * You can specify tracking type by setting attributes of Builder class when you create instances of this class.
 * It is not possible to run ObjectTracker with two or more different tracking types in one instance.
 * You can also limit the number of tracked objects by setting attributes of Builder class.
 * @n
 * Currently, ObjectTracker does not support HW offloading.
 * It is possible to run ObjectTracker only on CPU.
 * @n@n
 * Following sample code shows how to use short term tracking type.
 * Objects are added to ObjectTracker at the beginnning of tracking and in the middle of tracking periodically as well.
 * @code
    cv::VideoCapture video("/path/to/video/source");
    cv::Mat frame;
    cv::Mat first_frame;
    video >> first_frame;

    vas::ot::ObjectTracker::Builder ot_builder;
    auto ot = ot_builder.Build(vas::ot::TrackingType::SHORT_TERM);

    vas::pvd::PersonVehicleDetector::Builder pvd_builder;
    auto pvd = pvd_builder.Build("/path/to/directory/of/fd/model/files");

    std::vector<vas::pvd::PersonVehicle> person_vehicles;
    std::vector<vas::ot::DetectedObject> detected_objects;

    // Assume that there're objects in the first frame
    person_vehicles = pvd->Detect(first_frame);
    for (const auto& pv : person_vehicles)
        detected_objects.emplace_back(pv.rect, static_cast<int32_t>(pv.type));

    ot->Track(first_frame, detected_objects);

    // Assume that now pvd is running in another thread
    StartThread(pvd);

    while (video.read(frame))
    {
        detected_objects.clear();

        // Assume that frames are forwarded to the thread on which pvd is running
        EnqueueFrame(frame);

        // Assumes that pvd is adding its result into a queue in another thread.
        // Assumes also that latency from the last pvd frame to current frame is ignorable.
        person_vehicles = DequeuePersonVehicles();
        if (!person_vehicles.empty())
        {
            detected_objects.clear();
            for (const auto& pv : person_vehicles)
                detected_objects.emplace_back(pv.rect, static_cast<int32_t>(pv.type));
        }

        auto objects = ot->Track(frame, detected_objects);
        for (const auto& object : objects)
        {
            // Handle tracked object
        }
    }
 * @endcode
 * @n
 * Following sample code shows how to use zero term tracking type.
 * In this sample, pvd runs for each input frame.
 * After pvd generates results, ot runs with the results and object IDs are preserved.
 * @code
    cv::VideoCapture video("/path/to/video/source");
    cv::Mat frame;

    vas::ot::ObjectTracker::Builder ot_builder;
    auto ot = ot_builder.Build(vas::ot::TrackingType::ZERO_TERM);

    vas::pvd::PersonVehicleDetector::Builder pvd_builder;
    auto pvd = pvd_builder.Build("/path/to/directory/of/fd/model/files");

    std::vector<vas::ot::DetectedObject> detected_objects;

    ot->SetFrameDeltaTime(0.033f);
    while (video.read(frame))
    {
        detected_objects.clear();

        auto person_vehicles = pvd->Detect(first_frame);
        for (const auto& pv : person_vehicles)
            detected_objects.emplace_back(pv.rect, static_cast<int32_t>(pv.type));

        auto objects = ot->Track(frame, detected_objects);
        for (const auto& object : objects)
        {
            // Handle tracked object
        }
    }
 * @endcode
 */
class ObjectTracker {
  public:
    class Builder;

  public:
    ObjectTracker() = delete;
    ObjectTracker(const ObjectTracker &) = delete;
    ObjectTracker(ObjectTracker &&) = delete;

    /**
     * Destructor.
     */
    VAS_EXPORT ~ObjectTracker();

  public:
    ObjectTracker &operator=(const ObjectTracker &) = delete;
    ObjectTracker &operator=(ObjectTracker &&) = delete;

  public:
    /**
     * Tracks objects with video frames.
     * Also, this method is used to add detected objects.
     * If a detected object is overlapped enough with one of tracked object, the tracked object's information is updated
     * with the input detected object. On the other hand, if a detected object is overlapped with none of tracked
     * objects, the detected object is newly added and ObjectTracker starts to track the object. In long term and short
     * term tracking type, ObjectTracker continues to track objects in case that empty list of detected objects is
     * passed in. In zero term tracking type, however, ObjectTracker clears tracked objects in case that empty list of
     * detected objects is passed in.
     * @n
     * The supported color formats are BGR, NV12, BGRx and I420.
     *
     * @param[in] frame Input frame.
     * @param[in] detected_objects Detected objects in the input frame. Default value is an empty vector.
     * @return Information of tracked objects.
     * @exception std::invalid_argument Input frame is invalid.
     */
    VAS_EXPORT std::vector<Object>
    Track(const cv::Mat &frame, const std::vector<DetectedObject> &detected_objects = std::vector<DetectedObject>());

    /**
     * This function is to set a parameter indicating 'delta time' between now and last call to Track() in seconds.
     * The default value of the delta time is 0.033f which is tuned for 30 fps video frame rate.
     * It is to achieve improved tracking quality for other frame rates or inconstant frame rate by frame drops.
     * If input frames come from a video stream of constant frame rate, then a user needs to set this value as 1.0/fps
     * just after video open. For example, 60 fps video stream should set 0.0167f. If input frames have inconstant frame
     * rate, then a user needs to call this function before the Track() function.
     *
     * @param[in] frame_delta_t Delta time between two consecutive tracking in seconds. The valid range is [0.005 ~
     * 0.5].
     */
    VAS_EXPORT void SetFrameDeltaTime(float frame_delta_t);

    /**
     * Returns the tracking type of current instance.
     */
    VAS_EXPORT TrackingType GetTrackingType() const noexcept;

    /**
     * Returns the currently set maximum number of trackable objects.
     */
    VAS_EXPORT int32_t GetMaxNumObjects() const noexcept;

    /**
     * Returns the currently set frame delta time.
     */
    VAS_EXPORT float GetFrameDeltaTime() const noexcept;

    /**
     * Returns the currently set color format.
     */
    VAS_EXPORT vas::ColorFormat GetInputColorFormat() const noexcept;

    /**
     * Returns the backend type of current instance.
     */
    VAS_EXPORT vas::BackendType GetBackendType() const noexcept;

    /**
     * Returns the current set tracking per class.
     */
    VAS_EXPORT bool GetTrackingPerClass() const noexcept;

  private:
    class Impl;

  private:
    explicit ObjectTracker(Impl *impl);

  private:
    std::unique_ptr<Impl> impl_;
    friend class Builder;
};

/**
 * @class ObjectTracker::Builder
 * @brief Creates ObjectTracker instances.
 *
 * This class is used to build ObjectTracker instances.
 * All the attributes of this class affects how ObjectTracker is initialized.
 */
class ObjectTracker::Builder {
  public:
    /**
     * Default constructor.
     */
    VAS_EXPORT Builder();

    /**
     * Destructor.
     */
    VAS_EXPORT ~Builder();

  public:
    /**
     * Creates an instance of ObjectTracker based on tracking type and attributes you set.
     * In case that you set valid values for all attributes, an instance of ObjectTracker is created successfully.
     *
     * @param[in] tracking_type Tracking type for newly created ObjectTracker instance.
     * @exception std::invalid_argument One or more attributes you set are invalid.
     * @return ObjectTracker instance.
     */
    VAS_EXPORT std::unique_ptr<ObjectTracker> Build(TrackingType tracking_type) const;

  public:
    /**
     * Specifies HW backend on which object tracker runs.
     * @n
     * Default value is vas::BackendType::CPU.
     */
    vas::BackendType backend_type;

    /**
     * Maximum number of trackable objects in a frame.
     * @n
     * Valid range: 1 <= max_num_objects. Or it can be -1 if there is no limitation of maximum number in X86.
     * @n
     * Default value is -1 which means there is no limitation in X86.
     */
    int32_t max_num_objects;

    /**
     * Input color format vas::ColorFormat. Supports BGR, BGRX, NV12 and I420
     * @n
     * Default value is BGR.
     */
    vas::ColorFormat input_image_format;

    /**
     * Specifies whether tracker to use detection class for keeping id of an object.
     * If it is true, new detection will be associated from previous tracking only when those two have same class.
     * class id of an object is fixed across video frames.
     * If it is false, new detection can be associated across different-class objects.
     * In this case, the class id of an object may change across video frames depending on the tracker input.
     * It is recommended to turn this option off when it is likely that detector confuses the class of object.
     * For example, when detector confuses bicycle and motorbike. Turning this option off will increase the tracking
     * reliability as tracker will ignore the class label of detector.
     * @n
     * Default value is true.
     */
    bool tracking_per_class;

    /**
     * Platform configuration
     * You can set various configuraions for each platform using predefined configurations
     * @n
     * For Parallelization in KCFVAR mode, use key "max_num_threads" to set the maximum number of threads. Consult the
     * following format
     * @code platform_config["max_num_threads"] = "2"; // set maximum number of threads(concurrency level) to 2 @endcode
     * @n
     * Default value is 1
     * if value >=1, set value as the number of threads to process OT in parallel mode
     * if value >= Number of available cores OR value is -1, limit concurrency level to maximum available logical cores
     * otherwise: @exception Invalid input
     */
    std::map<std::string, std::string> platform_config;
};

}; // namespace ot
}; // namespace vas

#endif // VAS_OT_HPP
