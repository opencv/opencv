Common Interfaces of Tracker
============================

.. highlight:: cpp


Tracker
-------

.. ocv:class:: Tracker

Base class for the long-term tracker::

   class CV_EXPORTS_W Tracker : public virtual Algorithm
   {
     virtual ~Tracker();

     bool init( const Mat& image, const Rect& boundingBox );

     bool update( const Mat& image, Rect& boundingBox );

     static Ptr<Tracker> create( const String& trackerType );
   
   };

Creating Own Tracker
--------------------

