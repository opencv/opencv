Tracker Algorithms
==================

.. highlight:: cpp

Two algorithms will be implemented soon, the first is MIL (Multiple Instance Learning) [MIL]_ and second is Online Boosting [OLB]_.

.. [MIL] B Babenko, M-H Yang, and S Belongie, Visual Tracking with Online Multiple Instance Learning, In CVPR, 2009

.. [OLB] H Grabner, M Grabner, and H Bischof, Real-time tracking via on-line boosting, In Proc. BMVC, volume 1, pages 47– 56, 2006

TrackerMIL
----------

The MIL algorithm trains a classifier in an online manner to separate the object from the background. Multiple Instance Learning avoids the drift problem for a robust tracking.

Original code can be found here http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml

.. ocv:class:: TrackerMIL

Implementation of TrackerMIL from :ocv:class:`Tracker`::

   class CV_EXPORTS_W TrackerMIL : public Tracker
   {
    public:

     TrackerMIL( const TrackerMIL::Params &parameters = TrackerMIL::Params() );
   
     virtual ~TrackerMIL();
   
     void read( const FileNode& fn );
     void write( FileStorage& fs ) const;
   
   };

TrackerMIL::Params
------------------

.. ocv:struct:: TrackerMIL::Params

List of MIL parameters::

   struct CV_EXPORTS Params
   {
    Params();
    //parameters for sampler
    float samplerInitInRadius;   // radius for gathering positive instances during init
    int samplerInitMaxNegNum;    // # negative samples to use during init
    float samplerSearchWinSize;  // size of search window
    float samplerTrackInRadius;  // radius for gathering positive instances during tracking
    int samplerTrackMaxPosNum;   // # positive samples to use during tracking
    int samplerTrackMaxNegNum;   // # negative samples to use during tracking

    int featureSetNumFeatures;   // # features

    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
   };
   
TrackerMIL::TrackerMIL
----------------------

Constructor

.. ocv:function:: bool TrackerMIL::TrackerMIL( const TrackerMIL::Params &parameters = TrackerMIL::Params() )

    :param parameters: MIL parameters :ocv:struct:`TrackerMIL::Params`


TrackerBoosting
---------------

This is a real-time object tracking based on a novel on-line version of the AdaBoost algorithm.
The classifier uses the surrounding background as negative examples in update step to avoid the drifting problem. 

.. ocv:class:: TrackerBoosting

Implementation of TrackerBoosting from :ocv:class:`Tracker`::

   class CV_EXPORTS_W TrackerBoosting : public Tracker
   {
    public:

     TrackerBoosting( const TrackerBoosting::Params &parameters = TrackerBoosting::Params() );
   
     virtual ~TrackerBoosting();
   
     void read( const FileNode& fn );
     void write( FileStorage& fs ) const;

   
   };
   
   
TODO
----

* TrackerBoosting
* porting of boosting method from original MIL
