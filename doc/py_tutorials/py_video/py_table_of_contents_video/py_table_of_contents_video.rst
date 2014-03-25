.. _PY_Table-Of-Content-Video:

Video Analysis
------------------------------------------

*  :ref:`meanshift`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |vdo_1|     We have already seen an example of color-based tracking. It is simpler. This time, we see much more better algorithms like "Meanshift", and its upgraded version, "Camshift" to find and track them.

  =========== ======================================================

  .. |vdo_1|  image:: images/camshift.jpg
                 :height: 90pt
                 :width:  90pt


*  :ref:`Lucas_Kanade`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |vdo_2|     Now let's discuss an important concept, "Optical Flow", which is related to videos and has many applications.
  =========== ======================================================

  .. |vdo_2|  image:: images/opticalflow.jpeg
                 :height: 90pt
                 :width:  90pt


*  :ref:`py_background_subtraction`

  .. tabularcolumns:: m{100pt} m{300pt}
  .. cssclass:: toctableopencv

  =========== ======================================================
  |vdo_b|     In several applications, we need to extract foreground for further operations like object tracking. Background Subtraction is a well-known method in those cases.
  =========== ======================================================

  .. |vdo_b|  image:: images/background.jpg
                 :height: 90pt
                 :width:  90pt



.. raw:: latex

   \pagebreak

.. We use a custom table of content format and as the table of content only informs Sphinx about the hierarchy of the files, no need to show it.
.. toctree::
   :hidden:

   ../py_meanshift/py_meanshift
   ../py_lucas_kanade/py_lucas_kanade
   ../py_bg_subtraction/py_bg_subtraction
