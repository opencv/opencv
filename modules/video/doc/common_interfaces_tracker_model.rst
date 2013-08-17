Common Interfaces of TrackerModel
=================================

.. highlight:: cpp

ConfidenceMap
-------------

Represents the model of the target at frame :math:`k` (all states and scores)
    
[AAM]_ The set of the pair  :math:`\langle \hat{x}^{i}_{k}, C^{i}_{k} \rangle`

.. c:type:: ConfidenceMap

ConfidenceMap::

   typedef std::vector<std::pair<Ptr<TrackerTargetState>, float> > ConfidenceMap;

.. seealso::

   :ocv:class:`TrackerTargetState`

Trajectory
----------

Represents the estimate states for all frames

[AAM]_ :math:`x_{k}` is the trajectory of the target up to time :math:`k`
 
.. c:type:: Trajectory

Trajectory::

   typedef std::vector<Ptr<TrackerTargetState> > Trajectory;

.. seealso::

   :ocv:class:`TrackerTargetState`
   
TrackerTargetState
------------------

Abstract base class for TrackerTargetState that represents a possible state of the target.

[AAM]_ :math:`\hat{x}^{i}_{k}` all the states candidates.

Inherits this class with your Target state

.. ocv:class:: TrackerTargetState

TrackerTargetState class::

   class CV_EXPORTS_W TrackerTargetState
   {
    public:
     virtual ~TrackerTargetState(){};
     
     Point2f getTargetPosition() const;
     void setTargetPosition( const Point2f& position );
     
     int getTargetWidth() const;
     void setTargetWidth( int width );
     
     int getTargetHeight() const;
     void setTargetHeight( int height );
   
   };

In own implementation you can add scale variation, width, height, orientation, etc.


TrackerStateEstimator
---------------------

Abstract base class for TrackerStateEstimator that estimates the most likely target state.
 
[AAM]_ State estimator
 
[AMVOT]_ Statistical modeling (Fig. 3), Table III (generative) - IV (discriminative) - V (hybrid)

.. ocv:class:: TrackerStateEstimator

TrackerStateEstimator class::

   class CV_EXPORTS_W TrackerStateEstimator
   {
    public:
     virtual ~TrackerStateEstimator();
   
     static Ptr<TrackerStateEstimator> create( const String& trackeStateEstimatorType );
   
     Ptr<TrackerTargetState> estimate( const std::vector<ConfidenceMap>& confidenceMaps );
     void update( std::vector<ConfidenceMap>& confidenceMaps );
   
     String getClassName() const;
   
   };

TrackerStateEstimator::create
-----------------------------

Create TrackerStateEstimator by tracker state estimator type

.. ocv:function::  static Ptr<TrackerStateEstimator> TrackerStateEstimator::create( const String& trackeStateEstimatorType )
 
   :param trackeStateEstimatorType: The TrackerStateEstimator name
   
The modes available now:

* ``"BOOSTING"`` -- Boosting-based discriminative appearance models. See [AMVOT]_ section 4.4 
   
The modes available soon:

* ``"SVM"`` -- SVM-based discriminative appearance models. See [AMVOT]_ section 4.5

TrackerStateEstimator::estimate
-------------------------------

Estimate the most likely target state, return the estimated state

.. ocv:function::  Ptr<TrackerTargetState> TrackerStateEstimator::estimate( const std::vector<ConfidenceMap>& confidenceMaps )

   :param confidenceMaps: The overall appearance model as a list of :c:type:`ConfidenceMap`

TrackerStateEstimator::update
-----------------------------

Update the ConfidenceMap with the scores

.. ocv:function::  void TrackerStateEstimator::update( std::vector<ConfidenceMap>& confidenceMaps )

   :param confidenceMaps: The overall appearance model as a list of :c:type:`ConfidenceMap`

TrackerStateEstimator::getClassName
-----------------------------------

Get the name of the specific TrackerStateEstimator

.. ocv:function::  String TrackerStateEstimator::getClassName() const
  
TrackerModel
------------

Abstract class that represents the model of the target. It must be instantiated by specialized tracker
 
[AAM]_ Ak

Inherits this with your TrackerModel

.. ocv:class:: TrackerModel

TrackerModel class::
   
   class CV_EXPORTS_W TrackerModel
   {
    public:
   
     TrackerModel();
     virtual ~TrackerModel();
   
     void modelEstimation( const std::vector<Mat>& responses );
     void modelUpdate();
     bool runStateEstimator();
   
     bool setTrackerStateEstimator( Ptr<TrackerStateEstimator> trackerStateEstimator );
     void setLastTargetState( const Ptr<TrackerTargetState>& lastTargetState );
   
     Ptr<TrackerTargetState> getLastTargetState() const;
     const std::vector<ConfidenceMap>& getConfidenceMaps() const;
     const ConfidenceMap& getLastConfidenceMap() const;
     Ptr<TrackerStateEstimator> getTrackerStateEstimator() const;
   };
   
TrackerModel::modelEstimation
-----------------------------

Estimate the most likely target location

[AAM]_ ME, Model Estimation table I

.. ocv:function::  void TrackerModel::modelEstimation( const std::vector<Mat>& responses )
   
   :param responses: Features extracted from :ocv:class:`TrackerFeatureSet`

   
TrackerModel::modelUpdate
-------------------------

Update the model
   
[AAM]_ MU, Model Update table I

.. ocv:function::  void TrackerModel::modelUpdate()
   

TrackerModel::runStateEstimator
-------------------------------

Run the TrackerStateEstimator, return true if is possible to estimate a new state, false otherwise

.. ocv:function::  void TrackerModel::runStateEstimator()

TrackerModel::setTrackerStateEstimator
--------------------------------------

Set TrackerEstimator, return true if the tracker state estimator is added, false otherwise

.. ocv:function::  bool TrackerModel::setTrackerStateEstimator( Ptr<TrackerStateEstimator> trackerStateEstimator )
   
   :param trackerStateEstimator: The :ocv:class:`TrackerStateEstimator`
   
.. note:: You can add only one  :ocv:class:`TrackerStateEstimator`

TrackerModel::setLastTargetState
--------------------------------

Set the current :ocv:class:`TrackerTargetState` in the :c:type:`Trajectory`

.. ocv:function::  void TrackerModel::setLastTargetState( const Ptr<TrackerTargetState>& lastTargetState )
   
   :param lastTargetState: The current :ocv:class:`TrackerTargetState`


TrackerModel::getLastTargetState
--------------------------------

Get the last :ocv:class:`TrackerTargetState` from :c:type:`Trajectory`

.. ocv:function:: Ptr<TrackerTargetState> TrackerModel::getLastTargetState() const
   

TrackerModel::getConfidenceMaps
-------------------------------

Get the list of the :c:type:`ConfidenceMap`

.. ocv:function:: const std::vector<ConfidenceMap>& TrackerModel::getConfidenceMaps() const

TrackerModel::getLastConfidenceMap
----------------------------------

Get the last :c:type:`ConfidenceMap` for the current frame

.. ocv:function:: const ConfidenceMap& TrackerModel::getLastConfidenceMap() const

TrackerModel::getTrackerStateEstimator
--------------------------------------

Get the :ocv:class:`TrackerStateEstimator`

.. ocv:function:: Ptr<TrackerStateEstimator> TrackerModel::getTrackerStateEstimator() const

Specialized TrackerStateEstimator
=================================

In [AMVOT]_  Statistical modeling (Fig. 3), Table III (generative) - IV (discriminative) - V (hybrid) are described the most known statistical model.

At moment only :ocv:class:`TrackerStateEstimatorMILBoosting` is implemented.

TrackerStateEstimatorMILBoosting
--------------------------------

TrackerStateEstimator based on Boosting

.. ocv:class:: TrackerStateEstimatorMILBoosting

TrackerStateEstimatorMILBoosting class::

	class CV_EXPORTS_W TrackerStateEstimatorMILBoosting : public TrackerStateEstimator
	{
	 public:
	  class TrackerMILTargetState : public TrackerTargetState
	  {
	   ...
	  };
	  TrackerStateEstimatorMILBoosting( int numFeatures = 250 );
	  ~TrackerStateEstimatorMILBoosting();

	  void setCurrentConfidenceMap( ConfidenceMap& confidenceMap );
	};

TrackerStateEstimatorMILBoosting::TrackerMILTargetState
-------------------------------------------------------

Implementation of the target state for TrackerStateEstimatorMILBoosting

.. ocv:class:: TrackerStateEstimatorMILBoosting::TrackerMILTargetState

TrackerMILTargetState class::

     class TrackerMILTargetState : public TrackerTargetState
     {
      public:
      TrackerMILTargetState( const Point2f& position, int targetWidth, int targetHeight, bool foreground, const Mat& features );
      ~TrackerMILTargetState(){};

      void setTargetFg( bool foreground );
      void setFeatures( const Mat& features );
      bool isTargetFg() const;
      Mat getFeatures() const;
     };

TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting::setTargetFg
-------------------------------------------------------------------------------

Set label: true for target foreground, false for background

.. ocv:function::  TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting::setTargetFg( bool foreground )

    :param foreground: Label for background/foreground
    
TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting::setFeatures
-------------------------------------------------------------------------------

Set the features extracted from :ocv:class:`TrackerFeatureSet`

.. ocv:function::  TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting::setFeatures( const Mat& features )

    :param features: The features extracted
    
TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting::isTargetFg
------------------------------------------------------------------------------

Get the label. Return true for target foreground, false for background

.. ocv:function:: bool TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting::isTargetFg() const
    
TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting::getFeatures
-------------------------------------------------------------------------------

Get the features extracted

.. ocv:function:: Mat TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting::setFeatures() const
    
TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting
------------------------------------------------------------------

Constructor

.. ocv:function::  TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting( int numFeatures )

    :param numFeatures: Number of features for each sample
   
TrackerStateEstimatorMILBoosting::setCurrentConfidenceMap
---------------------------------------------------------

Set the current confidenceMap

.. ocv:function::  void TrackerStateEstimatorMILBoosting::setCurrentConfidenceMap( ConfidenceMap& confidenceMap )

    :param confidenceMap: The current :c:type:`ConfidenceMap`
