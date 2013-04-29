Soft Cascade Classifier
=======================

.. highlight:: cpp

Soft Cascade Classifier for Object Detection
--------------------------------------------

Cascade detectors have been shown to operate extremely rapidly, with high accuracy, and have important applications in different spheres. The initial goal for this cascade implementation was the fast and accurate pedestrian detector but it also useful in general. Soft cascade is trained with AdaBoost. But instead of training sequence of stages, the soft cascade is trained as a one long stage of T weak classifiers. Soft cascade is formulated as follows:

.. math::
    \texttt{H}(x) = \sum _{\texttt{t}=1..\texttt{T}} {\texttt{s}_t(x)}

where :math:`\texttt{s}_t(x) = \alpha_t\texttt{h}_t(x)` are the set of thresholded weak classifiers selected during AdaBoost training scaled by the associated weights. Let

.. math::
    \texttt{H}_t(x) = \sum _{\texttt{i}=1..\texttt{t}} {\texttt{s}_i(x)}

be the partial sum of sample responses before :math:`t`-the weak classifier will be applied. The function :math:`\texttt{H}_t(x)` of :math:`t` for sample :math:`x` named *sample trace*.
After each weak classifier evaluation, the sample trace at the point :math:`t` is compared with the rejection threshold :math:`r_t`. The sequence of :math:`r_t` named *rejection trace*.

The sample has been rejected if it fall rejection threshold. So stageless cascade allows to reject not-object sample as soon as possible. Another meaning of the sample trace is a confidence with that sample recognized as desired object. At each :math:`t` that confidence depend on all previous weak classifier. This feature of soft cascade is resulted in more accurate detection. The original formulation of soft cascade can be found in [BJ05]_.

.. [BJ05] Lubomir Bourdev and Jonathan Brandt. tRobust Object Detection Via Soft Cascade. IEEE CVPR, 2005.
.. [BMTG12] Rodrigo Benenson, Markus Mathias, Radu Timofte and Luc Van Gool. Pedestrian detection at 100 frames per second. IEEE CVPR, 2012.


softcascade::Detector
---------------------
.. ocv:class:: softcascade::Detector : public Algorithm

Implementation of soft (stageless) cascaded detector. ::

    class Detector : public Algorithm
    {
    public:

        enum { NO_REJECT = 1, DOLLAR = 2, /*PASCAL = 4,*/ DEFAULT = NO_REJECT};

        Detector(double minScale = 0.4, double maxScale = 5., int scales = 55, int rejCriteria = 1);
        virtual ~Detector();
        cv::AlgorithmInfo* info() const;
        virtual bool load(const FileNode& fileNode);
        virtual void read(const FileNode& fileNode);
        virtual void detect(InputArray image, InputArray rois, std::vector<Detection>& objects) const;
        virtual void detect(InputArray image, InputArray rois, OutputArray rects, OutputArray confs) const;

    }



softcascade::Detector::Detector
----------------------------------------
An empty cascade will be created.

.. ocv:function:: softcascade::Detector::Detector( double minScale=0.4, double maxScale=5., int scales=55, int rejCriteria=1 )

.. ocv:pyfunction:: cv2.softcascade_Detector([minScale[, maxScale[, scales[, rejCriteria]]]]) -> <softcascade_Detector object>

    :param minScale: a minimum scale relative to the original size of the image on which cascade will be applied.

    :param maxScale: a maximum scale relative to the original size of the image on which cascade will be applied.

    :param scales: a number of scales from minScale to maxScale.

    :param rejCriteria: algorithm used for non maximum suppression.



softcascade::Detector::~Detector
-----------------------------------------
Destructor for Detector.

.. ocv:function:: softcascade::Detector::~Detector()



softcascade::Detector::load
---------------------------
Load cascade from FileNode.

.. ocv:function:: bool softcascade::Detector::load(const FileNode& fileNode)

.. ocv:pyfunction:: cv2.softcascade_Detector.load(fileNode) -> retval

    :param fileNode: File node from which the soft cascade are read.



softcascade::Detector::detect
-----------------------------
Apply cascade to an input frame and return the vector of Detection objects.

.. ocv:function:: void softcascade::Detector::detect(InputArray image, InputArray rois, std::vector<Detection>& objects) const

.. ocv:function:: void softcascade::Detector::detect(InputArray image, InputArray rois, OutputArray rects, OutputArray confs) const

.. ocv:pyfunction:: cv2.softcascade_Detector.detect(image, rois[, rects[, confs]]) -> rects, confs

    :param image: a frame on which detector will be applied.

    :param rois: a vector of regions of interest. Only the objects that fall into one of the regions will be returned.

    :param objects: an output array of Detections.

    :param rects: an output array of bounding rectangles for detected objects.

    :param confs: an output array of confidence for detected objects. i-th bounding rectangle corresponds i-th confidence.


softcascade::ChannelFeatureBuilder
----------------------------------
.. ocv:class:: softcascade::ChannelFeatureBuilder : public Algorithm

Public interface for of soft (stageless) cascaded detector. ::

    class ChannelFeatureBuilder : public Algorithm
    {
    public:
        virtual ~ChannelFeatureBuilder();

        virtual void operator()(InputArray src, OutputArray channels) const = 0;

        static cv::Ptr<ChannelFeatureBuilder> create();
    };


softcascade::ChannelFeatureBuilder:~ChannelFeatureBuilder
---------------------------------------------------------
Destructor for ChannelFeatureBuilder.

.. ocv:function:: softcascade::ChannelFeatureBuilder::~ChannelFeatureBuilder()

.. ocv:pyfunction:: cv2.softcascade_ChannelFeatureBuilder_create(featureType) -> retval


softcascade::ChannelFeatureBuilder::operator()
----------------------------------------------
Create channel feature integrals for input image.

.. ocv:function:: void softcascade::ChannelFeatureBuilder::operator()(InputArray src, OutputArray channels) const

.. ocv:pyfunction:: cv2.softcascade_ChannelFeatureBuilder.compute(src, channels) -> None

    :param src source frame

    :param channels in OutputArray of computed channels
