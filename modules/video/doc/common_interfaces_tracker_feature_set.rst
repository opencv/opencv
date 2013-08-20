Common Interfaces of TrackerFeatureSet
======================================

.. highlight:: cpp


TrackerFeatureSet
-----------------

Class that manages the extraction and selection of features

[AAM]_ Feature Extraction and Feature Set Refinement (Feature Processing and Feature Selection). See table I and section III C
[AMVOT]_ Appearance modelling -> Visual representation (Table II, section 3.1 - 3.2)

.. ocv:class:: TrackerFeatureSet

TrackerFeatureSet class::

   class CV_EXPORTS_W TrackerFeatureSet
   {
    public:

     TrackerFeatureSet();
     ~TrackerFeatureSet();

     void extraction( const std::vector<Mat>& images );
     void selection();
     void removeOutliers();

     bool addTrackerFeature( String trackerFeatureType );
     bool addTrackerFeature( Ptr<TrackerFeature>& feature );

     const std::vector<std::pair<String, Ptr<TrackerFeature> > >& getTrackerFeature() const;
     const std::vector<Mat>& getResponses() const;

   };


TrackerFeatureSet is an aggregation of :ocv:class:`TrackerFeature`

.. seealso::

   :ocv:class:`TrackerFeature`

TrackerFeatureSet::extraction
-----------------------------

Extract features from the images collection

.. ocv:function:: void TrackerFeatureSet::extraction( const std::vector<Mat>& images )

    :param images: The input images

TrackerFeatureSet::selection
----------------------------

Identify most effective features for all feature types (optional)

.. ocv:function:: void TrackerFeatureSet::selection()

TrackerFeatureSet::removeOutliers
---------------------------------

Remove outliers for all feature types (optional)

.. ocv:function:: void TrackerFeatureSet::removeOutliers()

TrackerFeatureSet::addTrackerFeature
------------------------------------

Add TrackerFeature in the collection. Return true if TrackerFeature is added, false otherwise

.. ocv:function:: bool TrackerFeatureSet::addTrackerFeature( String trackerFeatureType )

   :param trackerFeatureType: The TrackerFeature name

.. ocv:function:: bool TrackerFeatureSet::addTrackerFeature( Ptr<TrackerFeature>& feature )

   :param feature: The TrackerFeature class


The modes available now:

* ``"HAAR"`` -- Haar Feature-based

The modes available soon:

* ``"HOG"`` -- Histogram of Oriented Gradients features

* ``"LBP"`` -- Local Binary Pattern features

* ``"FEATURE2D"`` -- All types of Feature2D

Example ``TrackerFeatureSet::addTrackerFeature`` : ::

   //sample usage:

   Ptr<TrackerFeature> trackerFeature = new TrackerFeatureHAAR( HAARparameters );
   featureSet->addTrackerFeature( trackerFeature );

   //or add CSC sampler with default parameters
   //featureSet->addTrackerFeature( "HAAR" );


.. note:: If you use the second method, you must initialize the TrackerFeature

TrackerFeatureSet::getTrackerFeature
------------------------------------

Get the TrackerFeature collection (TrackerFeature name, TrackerFeature pointer)

.. ocv:function:: const std::vector<std::pair<String, Ptr<TrackerFeature> > >& TrackerFeatureSet::getTrackerFeature() const

TrackerFeatureSet::getResponses
-------------------------------

Get the responses

.. ocv:function:: const std::vector<Mat>& TrackerFeatureSet::getResponses() const

.. note:: Be sure to call extraction before getResponses

Example ``TrackerFeatureSet::getResponses`` : ::

   //get the patches from sampler
   std::vector<Mat> detectSamples = sampler->getSamples();

   if( detectSamples.empty() )
      return false;

   //features extraction
   featureSet->extraction( detectSamples );

   //get responses
   std::vector<Mat> response = featureSet->getResponses();

TrackerFeature
--------------

Abstract base class for TrackerFeature that represents the feature.

.. ocv:class:: TrackerFeature

TrackerFeature class::

   class CV_EXPORTS_W TrackerFeature
   {
    public:
     virtual ~TrackerFeature();

     static Ptr<TrackerFeature> create( const String& trackerFeatureType );

     void compute( const std::vector<Mat>& images, Mat& response );

     virtual void selection( Mat& response, int npoints ) = 0;

     String getClassName() const;
   };

TrackerFeature::create
----------------------

Create TrackerFeature by tracker feature type

.. ocv:function:: static Ptr<TrackerFeature> TrackerFeature::create( const String& trackerFeatureType )

   :param trackerFeatureType: The TrackerFeature name

The modes available now:

* ``"HAAR"`` -- Haar Feature-based

The modes available soon:

* ``"HOG"`` -- Histogram of Oriented Gradients features

* ``"LBP"`` -- Local Binary Pattern features

* ``"FEATURE2D"`` -- All types of Feature2D

TrackerFeature::compute
-----------------------

Compute the features in the images collection

.. ocv:function:: void TrackerFeature::compute( const std::vector<Mat>& images, Mat& response )

   :param images: The images

   :param response: The output response

TrackerFeature::selection
-------------------------

Identify most effective features

.. ocv:function:: void TrackerFeature::selection( Mat& response, int npoints )

   :param response:  Collection of response for the specific TrackerFeature

   :param npoints: Max number of features

.. note:: This method modifies the response parameter

TrackerFeature::getClassName
----------------------------

Get the name of the specific TrackerFeature

.. ocv:function::  String TrackerFeature::getClassName() const

Specialized TrackerFeature
==========================

In [AAM]_ table I and section III C are described the most known features type. At moment only :ocv:class:`TrackerFeatureHAAR` is implemented.

TrackerFeatureHAAR : TrackerFeature
-----------------------------------

TrackerFeature based on HAAR features, used by TrackerMIL and many others algorithms

.. ocv:class:: TrackerFeatureHAAR

TrackerFeatureHAAR class::

   class CV_EXPORTS_W TrackerFeatureHAAR : TrackerFeature
   {
    public:

     TrackerFeatureHAAR( const TrackerFeatureHAAR::Params &parameters = TrackerFeatureHAAR::Params() );
     ~TrackerFeatureHAAR();

     void selection( Mat& response, int npoints );
   };

.. note:: HAAR features implementation is copied from apps/traincascade and modified according to MIL implementation

TrackerFeatureHAAR::Params
--------------------------

.. ocv:struct:: TrackerFeatureHAAR::Params

List of TrackerFeatureHAAR parameters::

   struct CV_EXPORTS Params
   {
    Params();
    int numFeatures; // # of rects
    Size rectSize;   // rect size
   };

TrackerFeatureHAAR::TrackerFeatureHAAR
--------------------------------------

Constructor

.. ocv:function:: TrackerFeatureHAAR::TrackerFeatureHAAR( const TrackerFeatureHAAR::Params &parameters = TrackerFeatureHAAR::Params() )

    :param parameters: TrackerFeatureHAAR parameters :ocv:struct:`TrackerFeatureHAAR::Params`


TrackerFeatureHAAR::selection
-----------------------------

Identify most effective features

.. ocv:function:: void TrackerFeatureHAAR::selection( Mat& response, int npoints )

   :param response:  Collection of response for the specific TrackerFeature

   :param npoints: Max number of features

.. note:: This method modifies the response parameter

TrackerFeatureHOG
-----------------

TODO To be implemented

TrackerFeatureLBP
-----------------

TODO To be implemented

TrackerFeatureFeature2d
-----------------------

TODO To be implemented
