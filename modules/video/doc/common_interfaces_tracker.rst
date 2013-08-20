Common Interfaces of Tracker
============================

.. highlight:: cpp


Tracker : Algorithm
-------------------

.. ocv:class:: Tracker

Base abstract class for the long-term tracker::

   class CV_EXPORTS_W Tracker : public virtual Algorithm
   {
     virtual ~Tracker();

     bool init( const Mat& image, const Rect& boundingBox );

     bool update( const Mat& image, Rect& boundingBox );

     static Ptr<Tracker> create( const String& trackerType );

   };

Tracker::init
-------------

Initialize the tracker with a know bounding box that surrounding the target

.. ocv:function:: bool Tracker::init( const Mat& image, const Rect& boundingBox )

    :param image: The initial frame

    :param boundingBox: The initial boundig box


Tracker::update
---------------

Update the tracker, find the new most likely bounding box for the target

.. ocv:function:: bool Tracker::update( const Mat& image, Rect& boundingBox )

    :param image: The current frame

    :param boundingBox: The boundig box that represent the new target location


Tracker::create
---------------

Creates a tracker by its name.

.. ocv:function::  static Ptr<Tracker> Tracker::create( const String& trackerType )

   :param trackerType: Tracker type


Creating Own Tracker
--------------------

If you want create a new tracker, you should follow some simple rules.

First, your tracker should be inherit from :ocv:class:`Tracker`, so you must implement two method:

* Tracker: initImpl, it should be called once in the first frame, here you should initialize all structures. The second argument is the initial bounding box of the target.

* Tracker:updateImpl, it should be called at the begin of in loop through video frames. Here you should overwrite the bounding box with new location.

Example of creating specialized Tracker ``TrackerMIL`` : ::

   class CV_EXPORTS_W TrackerMIL : public Tracker
   {
    public:
     TrackerMIL( const TrackerMIL::Params &parameters = TrackerMIL::Params() );
     virtual ~TrackerMIL();
     ...

    protected:
     bool initImpl( const Mat& image, const Rect& boundingBox );
     bool updateImpl( const Mat& image, Rect& boundingBox );
     ...
   };


Every tracker has three component :ocv:class:`TrackerSampler`, :ocv:class:`TrackerFeatureSet` and :ocv:class:`TrackerModel`.
The first two are instantiated from Tracker base class, instead the last component is abstract, so you must implement your TrackerModel.

Finally add your tracker in the file video_init.cpp

TrackerSampler
..............

TrackerSampler is already instantiated, but you should define the sampling algorithm and add the classes (or single class) to TrackerSampler.
You can choose one of the ready implementation as TrackerSamplerCSC or you can implement your sampling method, in this case
the class must inherit  :ocv:class:`TrackerSamplerAlgorithm`. Fill the samplingImpl method that writes the result in "sample" output argument.

Example of creating specialized TrackerSamplerAlgorithm ``TrackerSamplerCSC`` : ::

   class CV_EXPORTS_W TrackerSamplerCSC : public TrackerSamplerAlgorithm
   {
    public:
     TrackerSamplerCSC( const TrackerSamplerCSC::Params &parameters = TrackerSamplerCSC::Params() );
     ~TrackerSamplerCSC();
     ...

    protected:
     bool samplingImpl( const Mat& image, Rect boundingBox, std::vector<Mat>& sample );
     ...

   };

Example of adding TrackerSamplerAlgorithm to TrackerSampler : ::

   //sampler is the TrackerSampler
   Ptr<TrackerSamplerAlgorithm> CSCSampler = new TrackerSamplerCSC( CSCparameters );
   if( !sampler->addTrackerSamplerAlgorithm( CSCSampler ) )
    return false;

   //or add CSC sampler with default parameters
   //sampler->addTrackerSamplerAlgorithm( "CSC" );

.. seealso::

   :ocv:class:`TrackerSamplerCSC`, :ocv:class:`TrackerSamplerAlgorithm`


TrackerFeatureSet
.................

TrackerFeatureSet is already instantiated (as first) , but you should define what kinds of features you'll use in your tracker.
You can use multiple feature types, so you can add a ready implementation as :ocv:class:`TrackerFeatureHAAR` in your TrackerFeatureSet or develop your own implementation.
In this case, in the computeImpl method put the code that extract the features and
in the selection method optionally put the code for the refinement and selection of the features.

Example of creating specialized TrackerFeature ``TrackerFeatureHAAR`` : ::

   class CV_EXPORTS_W TrackerFeatureHAAR : public TrackerFeature
   {
    public:
     TrackerFeatureHAAR( const TrackerFeatureHAAR::Params &parameters = TrackerFeatureHAAR::Params() );
     ~TrackerFeatureHAAR();
     void selection( Mat& response, int npoints );
     ...

    protected:
     bool computeImpl( const std::vector<Mat>& images, Mat& response );
     ...

   };

Example of adding TrackerFeature to TrackerFeatureSet : ::

   //featureSet is the TrackerFeatureSet
   Ptr<TrackerFeature> trackerFeature = new TrackerFeatureHAAR( HAARparameters );
   featureSet->addTrackerFeature( trackerFeature );

.. seealso::

   :ocv:class:`TrackerFeatureHAAR`, :ocv:class:`TrackerFeatureSet`

TrackerModel
............

TrackerModel is abstract, so in your implementation you must develop your TrackerModel that inherit from :ocv:class:`TrackerModel`.
Fill the method for the estimation of the state "modelEstimationImpl", that estimates the most likely target location,
see [AAM]_ table I (ME) for further information. Fill "modelUpdateImpl" in order to update the model, see [AAM]_ table I (MU).
In this class you can use the :c:type:`ConfidenceMap` and :c:type:`Trajectory` to storing the model. The first represents the model on the all
possible candidate states and the second represents the list of all estimated states.

Example of creating specialized TrackerModel ``TrackerMILModel`` : ::

   class TrackerMILModel : public TrackerModel
   {
    public:
     TrackerMILModel( const Rect& boundingBox );
     ~TrackerMILModel();
     ...

    protected:
     void modelEstimationImpl( const std::vector<Mat>& responses );
     void modelUpdateImpl();
     ...

   };

And add it in your Tracker : ::

   bool TrackerMIL::initImpl( const Mat& image, const Rect& boundingBox )
   {
     ...
     //model is the general TrackerModel field od the general Tracker
     model = new TrackerMILModel( boundingBox );
     ...
   }


In the last step you should define the TrackerStateEstimator based on your implementation or you can use one of ready class as :ocv:class:`TrackerStateEstimatorMILBoosting`.
It represent the statistical part of the model that estimates the most likely target state.

Example of creating specialized TrackerStateEstimator ``TrackerStateEstimatorMILBoosting`` : ::

   class CV_EXPORTS_W TrackerStateEstimatorMILBoosting : public TrackerStateEstimator
   {
    class TrackerMILTargetState : public TrackerTargetState
    {
    ...
    };

    public:
     TrackerStateEstimatorMILBoosting( int nFeatures = 250 );
     ~TrackerStateEstimatorMILBoosting();
     ...

    protected:
     Ptr<TrackerTargetState> estimateImpl( const std::vector<ConfidenceMap>& confidenceMaps );
     void updateImpl( std::vector<ConfidenceMap>& confidenceMaps );
     ...

   };

And add it in your TrackerModel : ::

   //model is the TrackerModel of your Tracker
   Ptr<TrackerStateEstimatorMILBoosting> stateEstimator = new TrackerStateEstimatorMILBoosting( params.featureSetNumFeatures );
   model->setTrackerStateEstimator( stateEstimator );

.. seealso::

   :ocv:class:`TrackerModel`, :ocv:class:`TrackerStateEstimatorMILBoosting`, :ocv:class:`TrackerTargetState`


During this step, you should define your TrackerTargetState based on your implementation. :ocv:class:`TrackerTargetState` base class has only the bounding box (upper-left position, width and height), you can
enrich it adding scale factor, target rotation, etc.

Example of creating specialized TrackerTargetState ``TrackerMILTargetState`` : ::

   class TrackerMILTargetState : public TrackerTargetState
   {
    public:
     TrackerMILTargetState( const Point2f& position, int targetWidth, int targetHeight, bool foreground, const Mat& features );
     ~TrackerMILTargetState();
     ...

    private:
     bool isTarget;
     Mat targetFeatures;
     ...

   };


Try it
......

To try your tracker you can use the demo at https://github.com/lenlen/opencv/blob/tracking_api/samples/cpp/tracker.cpp.

The first argument is the name of the tracker and the second is a video source.
