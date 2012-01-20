High Level Functionality
========================

.. highlight:: cpp

Stitcher
--------
.. ocv:class:: Stitcher

High level image stitcher. It's possible to use this class without being aware of the entire stitching pipeline. However, to be able to achieve higher stitching stability and quality of the final images at least being familiar with the theory is recommended (see :ref:`stitching-pipeline`). ::

    class CV_EXPORTS Stitcher
    {
    public:
        enum { ORIG_RESOL = -1 };
        enum Status { OK, ERR_NEED_MORE_IMGS };

        // Creates stitcher with default parameters
        static Stitcher createDefault(bool try_use_gpu = false);

        Status estimateTransform(InputArray images);
        Status estimateTransform(InputArray images, const std::vector<std::vector<Rect> > &rois);

        Status composePanorama(OutputArray pano);
        Status composePanorama(InputArray images, OutputArray pano);

        Status stitch(InputArray images, OutputArray pano);
        Status stitch(InputArray images, const std::vector<std::vector<Rect> > &rois, OutputArray pano);

        double registrationResol() const { return registr_resol_; }
        void setRegistrationResol(double resol_mpx) { registr_resol_ = resol_mpx; }

        double seamEstimationResol() const { return seam_est_resol_; }
        void setSeamEstimationResol(double resol_mpx) { seam_est_resol_ = resol_mpx; }

        double compositingResol() const { return compose_resol_; }
        void setCompositingResol(double resol_mpx) { compose_resol_ = resol_mpx; }

        double panoConfidenceThresh() const { return conf_thresh_; }
        void setPanoConfidenceThresh(double conf_thresh) { conf_thresh_ = conf_thresh; }

        bool waveCorrection() const { return do_wave_correct_; }
        void setWaveCorrection(bool flag) { do_wave_correct_ = flag; }

        detail::WaveCorrectKind waveCorrectKind() const { return wave_correct_kind_; }
        void setWaveCorrectKind(detail::WaveCorrectKind kind) { wave_correct_kind_ = kind; }

        Ptr<detail::FeaturesFinder> featuresFinder() { return features_finder_; }
        const Ptr<detail::FeaturesFinder> featuresFinder() const { return features_finder_; }
        void setFeaturesFinder(Ptr<detail::FeaturesFinder> features_finder)
            { features_finder_ = features_finder; }

        Ptr<detail::FeaturesMatcher> featuresMatcher() { return features_matcher_; }
        const Ptr<detail::FeaturesMatcher> featuresMatcher() const { return features_matcher_; }
        void setFeaturesMatcher(Ptr<detail::FeaturesMatcher> features_matcher)
            { features_matcher_ = features_matcher; }

        const cv::Mat& matchingMask() const { return matching_mask_; }
        void setMatchingMask(const cv::Mat &mask)
        { 
            CV_Assert(mask.type() == CV_8U && mask.cols == mask.rows);
            matching_mask_ = mask.clone(); 
        }

        Ptr<detail::BundleAdjusterBase> bundleAdjuster() { return bundle_adjuster_; }
        const Ptr<detail::BundleAdjusterBase> bundleAdjuster() const { return bundle_adjuster_; }
        void setBundleAdjuster(Ptr<detail::BundleAdjusterBase> bundle_adjuster)
            { bundle_adjuster_ = bundle_adjuster; }

        Ptr<WarperCreator> warper() { return warper_; }
        const Ptr<WarperCreator> warper() const { return warper_; }
        void setWarper(Ptr<WarperCreator> warper) { warper_ = warper; }

        Ptr<detail::ExposureCompensator> exposureCompensator() { return exposure_comp_; }
        const Ptr<detail::ExposureCompensator> exposureCompensator() const { return exposure_comp_; }
        void setExposureCompensator(Ptr<detail::ExposureCompensator> exposure_comp)
            { exposure_comp_ = exposure_comp; }

        Ptr<detail::SeamFinder> seamFinder() { return seam_finder_; }
        const Ptr<detail::SeamFinder> seamFinder() const { return seam_finder_; }
        void setSeamFinder(Ptr<detail::SeamFinder> seam_finder) { seam_finder_ = seam_finder; }

        Ptr<detail::Blender> blender() { return blender_; }
        const Ptr<detail::Blender> blender() const { return blender_; }
        void setBlender(Ptr<detail::Blender> blender) { blender_ = blender; }

    private: 
        /* hidden */
    };

Stitcher::createDefault
-----------------------
Creates a stitcher with the default parameters.

.. ocv:function:: Stitcher Stitcher::createDefault(bool try_use_gpu = false)

    :param try_use_gpu: Flag indicating whether GPU should be used whenever it's possible.

    :return: Stitcher class instance.

Stitcher::estimateTransform
---------------------------

These functions try to match the given images and to estimate rotations of each camera.

.. note:: Use the functions only if you're aware of the stitching pipeline, otherwise use :ocv:func:`Stitcher::stitch`.

.. ocv:function:: Status Stitcher::estimateTransform(InputArray images)

.. ocv:function:: Status Stitcher::estimateTransform(InputArray images, const std::vector<std::vector<Rect> > &rois)

    :param images: Input images.

    :param rois: Region of interest rectangles.
    
    :return: Status code.

Stitcher::composePanorama
-------------------------

These functions try to compose the given images (or images stored internally from the other function calls) into the final pano under the assumption that the image transformations were estimated before.

.. note:: Use the functions only if you're aware of the stitching pipeline, otherwise use :ocv:func:`Stitcher::stitch`.

.. ocv:function:: Status Stitcher::composePanorama(OutputArray pano)

.. ocv:function:: Status Stitcher::composePanorama(InputArray images, OutputArray pano)

    :param images: Input images.
    
    :param pano: Final pano.

    :return: Status code.

Stitcher::stitch
----------------

These functions try to stitch the given images.

.. ocv:function:: Status Stitcher::stitch(InputArray images, OutputArray pano)

.. ocv:function:: Status Stitcher::stitch(InputArray images, const std::vector<std::vector<Rect> > &rois, OutputArray pano)

    :param images: Input images.
    
    :param rois: Region of interest rectangles.

    :param pano: Final pano.

    :return: Status code.

WarperCreator
-------------
.. ocv:class:: WarperCreator

Image warper factories base class. ::

    class WarperCreator
    {
    public:
        virtual ~WarperCreator() {}
        virtual Ptr<detail::RotationWarper> create(float scale) const = 0;
    };

PlaneWarper
-----------
.. ocv:class:: PlaneWarper

Plane warper factory class. ::

    class PlaneWarper : public WarperCreator
    {
    public:
        Ptr<detail::RotationWarper> create(float scale) const { return new detail::PlaneWarper(scale); }
    };

.. seealso:: :ocv:class:`detail::PlaneWarper`

CylindricalWarper
-----------------
.. ocv:class:: CylindricalWarper

Cylindrical warper factory class. ::

    class CylindricalWarper: public WarperCreator
    {
    public:
        Ptr<detail::RotationWarper> create(float scale) const { return new detail::CylindricalWarper(scale); }
    };

.. seealso:: :ocv:class:`detail::CylindricalWarper`

SphericalWarper
---------------
.. ocv:class:: SphericalWarper

Spherical warper factory class. ::

    class SphericalWarper: public WarperCreator
    {
    public:
        Ptr<detail::RotationWarper> create(float scale) const { return new detail::SphericalWarper(scale); }
    };

.. seealso:: :ocv:class:`detail::SphericalWarper`

