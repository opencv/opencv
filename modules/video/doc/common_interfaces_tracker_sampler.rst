Common Interfaces of TrackerSampler
===================================

.. highlight:: cpp


TrackerSampler
--------------

Class that manages the sampler in order to select regions for the update the model of the tracker

[AAM]_ Sampling e Labeling. See table I and section III B

.. ocv:class:: TrackerSampler

TrackerSampler class::

   class CV_EXPORTS_W TrackerSampler
   {
    public:
   
     TrackerSampler();
     ~TrackerSampler();
   
     void sampling( const Mat& image, Rect boundingBox );
     
     const std::vector<std::pair<String, Ptr<TrackerSamplerAlgorithm> > >& getSamplers() const;
     const std::vector<Mat>& getSamples() const;
   
     bool addTrackerSamplerAlgorithm( String trackerSamplerAlgorithmType );
     bool addTrackerSamplerAlgorithm( Ptr<TrackerSamplerAlgorithm>& sampler );
   
   };


TrackerSampler is an aggregation of :ocv:class:`TrackerSamplerAlgorithm`

.. seealso::

   :ocv:class:`TrackerSamplerAlgorithm`
   
TrackerSampler::sampling
------------------------

Computes the regions starting from a position in an image

.. ocv:function::  void TrackerSampler::sampling( const Mat& image, Rect boundingBox ) 

   :param image: The current frame
   
   :param boundingBox: The bounding box from which regions can be calculated
   

TrackerSampler::getSamplers
---------------------------

Return the collection of the :ocv:class:`TrackerSamplerAlgorithm`

.. ocv:function::  std::vector<std::pair<String, Ptr<TrackerSamplerAlgorithm> > >& TrackerSampler::getSamplers() const 

   
TrackerSampler::getSamples
--------------------------

Return the samples from all :ocv:class:`TrackerSamplerAlgorithm`, [AAM]_ Fig. 1 variable Sk

.. ocv:function:: const std::vector<Mat>& TrackerSampler::getSamples() const

TrackerSampler::addTrackerSamplerAlgorithm
------------------------------------------

Add TrackerSamplerAlgorithm in the collection.
Return true if sampler is added, false otherwise

.. ocv:function:: bool TrackerSampler::addTrackerSamplerAlgorithm( String trackerSamplerAlgorithmType )
   
   :param trackerSamplerAlgorithmType: The TrackerSamplerAlgorithm name

.. ocv:function:: bool TrackerSampler::addTrackerSamplerAlgorithm( Ptr<TrackerSamplerAlgorithm>& sampler )

   :param sampler: The TrackerSamplerAlgorithm class
   

The modes available now:

* ``"CSC"`` -- Current State Center
    
The modes available soon:

* ``"CS"`` -- Current State
    
Example ``TrackerSamplerAlgorithm::addTrackerSamplerAlgorithm`` : ::

    //sample usage:
    
     TrackerSamplerCSC::Params CSCparameters;
     Ptr<TrackerSamplerAlgorithm> CSCSampler = new TrackerSamplerCSC( CSCparameters );
     
     if( !sampler->addTrackerSamplerAlgorithm( CSCSampler ) )
       return false;
   
     //or add CSC sampler with default parameters
     //sampler->addTrackerSamplerAlgorithm( "CSC" );
     
   
.. note:: If you use the second method, you must initialize the TrackerSamplerAlgorithm


TrackerSamplerAlgorithm
-----------------------

Abstract base class for TrackerSamplerAlgorithm that represents the algorithm for the specific sampler.

.. ocv:class:: TrackerSamplerAlgorithm

TrackerSamplerAlgorithm class::

   class CV_EXPORTS_W TrackerSamplerAlgorithm
   {
    public:
   
     virtual ~TrackerSamplerAlgorithm();
   
     static Ptr<TrackerSamplerAlgorithm> create( const String& trackerSamplerType );
   
     bool sampling( const Mat& image, Rect boundingBox, std::vector<Mat>& sample );
     String getClassName() const;
   };

TrackerSamplerAlgorithm::create
-------------------------------

Create TrackerSamplerAlgorithm by tracker sampler type.

.. ocv:function:: static Ptr<TrackerSamplerAlgorithm> TrackerSamplerAlgorithm::create( const String& trackerSamplerType )
   
   :param trackerSamplerType: The trackerSamplerType name
   
The modes available now:

* ``"CSC"`` -- Current State Center


TrackerSamplerAlgorithm::sampling
---------------------------------

Computes the regions starting from a position in an image. Return true if samples are computed, false otherwise

.. ocv:function:: bool TrackerSamplerAlgorithm::sampling( const Mat& image, Rect boundingBox, std::vector<Mat>& sample )
   
   :param image: The current frame
   
   :param boundingBox: The bounding box from which regions can be calculated
   
   :sample: The computed samples [AAM]_ Fig. 1 variable Sk

TrackerSamplerAlgorithm::getClassName
-------------------------------------

Get the name of the specific TrackerSamplerAlgorithm

.. ocv:function::  String TrackerSamplerAlgorithm::getClassName() const

Specialized TrackerSamplerAlgorithm
===================================

In [AAM]_ table I are described the most known sampling strategies. At moment only :ocv:class:`TrackerSamplerCSC` is implemented.

TrackerSamplerCSC
-----------------

TrackerSampler based on CSC (current state centered), used by MIL [MIL]_ algorithm :ocv:class:`TrackerMIL`

.. ocv:class:: TrackerSamplerCSC

TrackerSamplerCSC class::

 
   class CV_EXPORTS_W TrackerSamplerCSC
   {
    public:
     
     TrackerSamplerCSC( const TrackerSamplerCSC::Params &parameters = TrackerSamplerCSC::Params() );
     void setMode( int samplingMode );
   
     ~TrackerSamplerCSC();
   };
   

TrackerSamplerCSC::Params
-------------------------

.. ocv:struct:: TrackerSamplerCSC::Params

List of TrackerSamplerCSC parameters::

   struct CV_EXPORTS Params
   {
    Params();
    float initInRad;        // radius for gathering positive instances during init
    float trackInPosRad;    // radius for gathering positive instances during tracking
    float searchWinSize;    // size of search window
    int initMaxNegNum;      // # negative samples to use during init
    int trackMaxPosNum;     // # positive samples to use during training
    int trackMaxNegNum;     // # negative samples to use during training
   }; 
   
   
TrackerSamplerCSC::TrackerSamplerCSC
------------------------------------

Constructor

.. ocv:function:: TrackerSamplerCSC::TrackerSamplerCSC( const TrackerSamplerCSC::Params &parameters = TrackerSamplerCSC::Params() )

    :param parameters: TrackerSamplerCSC parameters :ocv:struct:`TrackerSamplerCSC::Params`

TrackerSamplerCSC::setMode
--------------------------

Set the sampling mode of TrackerSamplerCSC

.. ocv:function:: void TrackerSamplerCSC::setMode( int samplingMode )

    :param samplingMode: The sampling mode
    
The modes are:

* ``"MODE_INIT_POS = 1"`` -- for the positive sampling in initialization step
* ``"MODE_INIT_NEG = 2"`` -- for the negative sampling in initialization step
* ``"MODE_TRACK_POS = 3"`` -- for the positive sampling in update step
* ``"MODE_TRACK_NEG = 4"`` -- for the negative sampling in update step
* ``"MODE_DETECT = 5"`` -- for the sampling in detection step
    
TrackerSamplerCS
----------------

TrackerSampler based on CS (current state)

.. ocv:class:: TrackerSamplerCS : public TrackerSamplerAlgorithm

TODO
