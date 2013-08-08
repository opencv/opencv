Common Interfaces of Tracker
============================

.. highlight:: cpp


Tracker
-------

.. ocv:class:: Tracker : public virtual Algorithm

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
   
The following detector types are supported:

* ``"MIL"`` -- :ocv:class:`TrackerMIL`
* ``"BOOSTING"`` -- :ocv:class:`TrackerBoosting`

Creating Own Tracker
--------------------

TODO...

