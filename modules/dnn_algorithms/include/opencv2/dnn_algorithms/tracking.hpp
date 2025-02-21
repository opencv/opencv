// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ALGORITHMS_TRACKING_HPP
#define OPENCV_DNN_ALGORITHMS_TRACKING_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/dnn.hpp"

namespace cv
{

/** @brief the GOTURN (Generic Object Tracking Using Regression Networks) tracker
 *
 *  GOTURN (@cite GOTURN) is kind of trackers based on Convolutional Neural Networks (CNN). While taking all advantages of CNN trackers,
 *  GOTURN is much faster due to offline training without online fine-tuning nature.
 *  GOTURN tracker addresses the problem of single target tracking: given a bounding box label of an object in the first frame of the video,
 *  we track that object through the rest of the video. NOTE: Current method of GOTURN does not handle occlusions; however, it is fairly
 *  robust to viewpoint changes, lighting changes, and deformations.
 *  Inputs of GOTURN are two RGB patches representing Target and Search patches resized to 227x227.
 *  Outputs of GOTURN are predicted bounding box coordinates, relative to Search patch coordinate system, in format X1,Y1,X2,Y2.
 *  Original paper is here: <http://davheld.github.io/GOTURN/GOTURN.pdf>
 *  As long as original authors implementation: <https://github.com/davheld/GOTURN#train-the-tracker>
 *  Implementation of training algorithm is placed in separately here due to 3d-party dependencies:
 *  <https://github.com/Auron-X/GOTURN_Training_Toolkit>
 *  GOTURN architecture goturn.prototxt and trained model goturn.caffemodel are accessible on opencv_extra GitHub repository.
 */
class CV_EXPORTS_W TrackerGOTURN : public Tracker
{
protected:
    TrackerGOTURN();  // use ::create()
public:
    virtual ~TrackerGOTURN() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        CV_PROP_RW std::string modelTxt;
        CV_PROP_RW std::string modelBin;
    };

    /** @brief Constructor
    @param parameters GOTURN parameters TrackerGOTURN::Params
    */
    static CV_WRAP
    Ptr<TrackerGOTURN> create(const TrackerGOTURN::Params& parameters = TrackerGOTURN::Params());

    /** @brief Constructor
    @param model pre-loaded GOTURN model
    */
    static CV_WRAP Ptr<TrackerGOTURN> create(const dnn::Net& model);

    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};

class CV_EXPORTS_W TrackerDaSiamRPN : public Tracker
{
protected:
    TrackerDaSiamRPN();  // use ::create()
public:
    virtual ~TrackerDaSiamRPN() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        CV_PROP_RW std::string model;
        CV_PROP_RW std::string kernel_cls1;
        CV_PROP_RW std::string kernel_r1;
        CV_PROP_RW int backend;
        CV_PROP_RW int target;
    };

    /** @brief Constructor
    @param parameters DaSiamRPN parameters TrackerDaSiamRPN::Params
    */
    static CV_WRAP
    Ptr<TrackerDaSiamRPN> create(const TrackerDaSiamRPN::Params& parameters = TrackerDaSiamRPN::Params());

    /** @brief Constructor
     *  @param siam_rpn pre-loaded SiamRPN model
     *  @param kernel_cls1 pre-loaded CLS model
     *  @param kernel_r1 pre-loaded R1 model
     */
    static CV_WRAP
    Ptr<TrackerDaSiamRPN> create(const dnn::Net& siam_rpn, const dnn::Net& kernel_cls1, const dnn::Net& kernel_r1);

    /** @brief Return tracking score
    */
    CV_WRAP virtual float getTrackingScore() = 0;

    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};

/** @brief the Nano tracker is a super lightweight dnn-based general object tracking.
 *
 *  Nano tracker is much faster and extremely lightweight due to special model structure, the whole model size is about 1.9 MB.
 *  Nano tracker needs two models: one for feature extraction (backbone) and the another for localization (neckhead).
 *  Model download link: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2
 *  Original repo is here: https://github.com/HonglinChu/NanoTrack
 *  Author: HongLinChu, 1628464345@qq.com
 */
class CV_EXPORTS_W TrackerNano : public Tracker
{
protected:
    TrackerNano();  // use ::create()
public:
    virtual ~TrackerNano() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        CV_PROP_RW std::string backbone;
        CV_PROP_RW std::string neckhead;
        CV_PROP_RW int backend;
        CV_PROP_RW int target;
    };

    /** @brief Constructor
    @param parameters NanoTrack parameters TrackerNano::Params
    */
    static CV_WRAP
    Ptr<TrackerNano> create(const TrackerNano::Params& parameters = TrackerNano::Params());

    /** @brief Constructor
     *  @param backbone pre-loaded backbone model
     *  @param neckhead pre-loaded neckhead model
     */
    static CV_WRAP
    Ptr<TrackerNano> create(const dnn::Net& backbone, const dnn::Net& neckhead);

    /** @brief Return tracking score
    */
    CV_WRAP virtual float getTrackingScore() = 0;

    //void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    //bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};

/** @brief the VIT tracker is a super lightweight dnn-based general object tracking.
 *
 *  VIT tracker is much faster and extremely lightweight due to special model structure, the model file is about 767KB.
 *  Model download link: https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack
 *  Author: PengyuLiu, 1872918507@qq.com
 */
class CV_EXPORTS_W TrackerVit : public Tracker
{
protected:
    TrackerVit();  // use ::create()
public:
    virtual ~TrackerVit() CV_OVERRIDE;

    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_WRAP Params();
        CV_PROP_RW std::string net;
        CV_PROP_RW int backend;
        CV_PROP_RW int target;
        CV_PROP_RW Scalar meanvalue;
        CV_PROP_RW Scalar stdvalue;
        CV_PROP_RW float tracking_score_threshold;
    };

    /** @brief Constructor
    @param parameters vit tracker parameters TrackerVit::Params
    */
    static CV_WRAP
    Ptr<TrackerVit> create(const TrackerVit::Params& parameters = TrackerVit::Params());

    /** @brief Constructor
     *  @param model pre-loaded DNN model
     *  @param meanvalue mean value for image preprocessing
     *  @param stdvalue std value for image preprocessing
     *  @param tracking_score_threshold threshold for tracking score
     */
    static CV_WRAP
    Ptr<TrackerVit> create(const dnn::Net& model, Scalar meanvalue = Scalar(0.485, 0.456, 0.406),
                           Scalar stdvalue = Scalar(0.229, 0.224, 0.225), float tracking_score_threshold = 0.20f);

    /** @brief Return tracking score
    */
    CV_WRAP virtual float getTrackingScore() = 0;

    // void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    // bool update(InputArray image, CV_OUT Rect& boundingBox) CV_OVERRIDE;
};

} // cv

#endif
