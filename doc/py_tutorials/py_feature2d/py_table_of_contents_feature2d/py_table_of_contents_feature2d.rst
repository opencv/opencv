.. _PY_Table-Of-Content-Feature2D:

Feature Detection and Description
------------------------------------------

*  :ref:`Features_Meaning`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |f2d_1|     What are the main features in an image? How can finding those features be useful to us?

  =========== ======================================================

  .. |f2d_1|  image:: images/features_icon.jpg
                 :height: 90pt
                 :width:  90pt


*  :ref:`Harris_Corners`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |f2d_2|     Okay, Corners are good features? But how do we find them?

  =========== ======================================================

  .. |f2d_2|  image:: images/harris_icon.jpg
                 :height: 90pt
                 :width:  90pt


*  :ref:`shi_tomasi`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |f2d_3|     We will look into Shi-Tomasi corner detection

  =========== ======================================================

  .. |f2d_3|  image:: images/shi_icon.jpg
                 :height: 90pt
                 :width:  90pt


*  :ref:`sift_intro`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |f2d_4|     Harris corner detector is not good enough when scale of image changes. Lowe developed a breakthrough method to find scale-invariant features and it is called SIFT

  =========== ======================================================

  .. |f2d_4|  image:: images/sift_icon.jpg
                 :height: 90pt
                 :width:  90pt


*  :ref:`SURF`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |f2d_5|     SIFT is really good, but not fast enough, so people came up with a speeded-up version called SURF.

  =========== ======================================================

  .. |f2d_5|  image:: images/surf_icon.jpg
                 :height: 90pt
                 :width:  90pt


*  :ref:`FAST`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |f2d_06|    All the above feature detection methods are good in some way. But they are not fast enough to work in real-time applications like SLAM. There comes the FAST algorithm, which is really "FAST".

  =========== ======================================================

  .. |f2d_06|  image:: images/fast_icon.jpg
                 :height: 90pt
                 :width:  90pt


*  :ref:`BRIEF`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |f2d_07|    SIFT uses a feature descriptor with 128 floating point numbers. Consider thousands of such features. It takes lots of memory and more time for matching. We can compress it to make it faster. But still we have to calculate it first. There comes BRIEF which gives the shortcut to find binary descriptors with less memory, faster matching, still higher recognition rate.

  =========== ======================================================

  .. |f2d_07|  image:: images/brief.jpg
                 :height: 90pt
                 :width:  90pt


*  :ref:`ORB`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |f2d_08|    SIFT and SURF are good in what they do, but what if you have to pay a few dollars every year to use them in your applications? Yeah, they are patented!!! To solve that problem, OpenCV devs came up with a new "FREE" alternative to SIFT & SURF, and that is ORB.
  =========== ======================================================

  .. |f2d_08|  image:: images/orb.jpg
                 :height: 90pt
                 :width:  90pt


*  :ref:`Matcher`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |f2d_09|    We know a great deal about feature detectors and descriptors. It is time to learn how to match different descriptors. OpenCV provides two techniques, Brute-Force matcher and FLANN based matcher.
  =========== ======================================================

  .. |f2d_09|  image:: images/matching.jpg
                 :height: 90pt
                 :width:  90pt


*  :ref:`PY_feature_homography`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |f2d_10|    Now we know about feature matching. Let's mix it up with `calib3d` module to find objects in a complex image.
  =========== ======================================================

  .. |f2d_10|  image:: images/homography_icon.jpg
                 :height: 90pt
                 :width:  90pt


.. raw:: latex

   \pagebreak

.. We use a custom table of content format and as the table of content only informs Sphinx about the hierarchy of the files, no need to show it.
.. toctree::
   :hidden:

   ../py_features_meaning/py_features_meaning
   ../py_features_harris/py_features_harris
   ../py_shi_tomasi/py_shi_tomasi
   ../py_sift_intro/py_sift_intro
   ../py_surf_intro/py_surf_intro
   ../py_fast/py_fast
   ../py_brief/py_brief
   ../py_orb/py_orb
   ../py_matcher/py_matcher
   ../py_feature_homography/py_feature_homography
