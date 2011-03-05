Dynamic Structures
==================

.. highlight:: python



.. index:: CvMemStorage

.. _CvMemStorage:

CvMemStorage
------------

`id=0.11586833925 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/core/CvMemStorage>`__

.. class:: CvMemStorage



Growing memory storage.

Many OpenCV functions use a given storage area for their results
and working storage.  These storage areas can be created using
:ref:`CreateMemStorage`
.  OpenCV Python tracks the objects occupying a
CvMemStorage, and automatically releases the CvMemStorage when there are
no objects referring to it.  For this reason, there is explicit function
to release a CvMemStorage.




.. doctest::


    
    >>> import cv
    >>> image = cv.LoadImageM("building.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
    >>> seq = cv.FindContours(image, cv.CreateMemStorage(), cv.CV_RETR_TREE, cv.CV_CHAIN_APPROX_SIMPLE)
    >>> del seq   # associated storage is also released
    

..


.. index:: CvSeq

.. _CvSeq:

CvSeq
-----

`id=0.0938210237552 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/core/CvSeq>`__

.. class:: CvSeq



Growable sequence of elements.

Many OpenCV functions return a CvSeq object.  The CvSeq obect is a sequence, so these are all legal:



::


    
    seq = cv.FindContours(scribble, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
    # seq is a sequence of point pairs
    print len(seq)
    # FindContours returns a sequence of (x,y) points, so to print them out:
    for (x,y) in seq:
       print (x,y)
    print seq[10]            # tenth entry in the seqeuence
    print seq[::-1]          # reversed sequence
    print sorted(list(seq))  # sorted sequence
    

..

Also, a CvSeq object has methods
``h_next()``
,
``h_prev()``
,
``v_next()``
and
``v_prev()``
.
Some OpenCV functions (for example 
:ref:`FindContours`
) can return multiple CvSeq objects, connected by these relations.
In this case the methods return the other sequences.  If no relation between sequences exists, then the methods return 
``None``
.


.. index:: CvSet

.. _CvSet:

CvSet
-----

`id=0.165386903844 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/core/CvSet>`__

.. class:: CvSet



Collection of nodes.

Some OpenCV functions return a CvSet object. The CvSet obect is iterable, for example:




::


    
    for i in s:
      print i
    print set(s)
    print list(s)
    

..


.. index:: CloneSeq

.. _CloneSeq:

CloneSeq
--------

`id=0.893022984961 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/core/CloneSeq>`__


.. function:: CloneSeq(seq,storage)-> None

    Creates a copy of a sequence.





    
    :param seq: Sequence 
    
    :type seq: :class:`CvSeq`
    
    
    :param storage: The destination storage block to hold the new sequence header and the copied data, if any. If it is NULL, the function uses the storage block containing the input sequence. 
    
    :type storage: :class:`CvMemStorage`
    
    
    
The function makes a complete copy of the input sequence and returns it.


.. index:: CreateMemStorage

.. _CreateMemStorage:

CreateMemStorage
----------------

`id=0.141261875659 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/core/CreateMemStorage>`__


.. function:: CreateMemStorage(blockSize = 0) -> memstorage

    Creates memory storage.





    
    :param blockSize: Size of the storage blocks in bytes. If it is 0, the block size is set to a default value - currently it is  about 64K. 
    
    :type blockSize: int
    
    
    
The function creates an empty memory storage. See 
:ref:`CvMemStorage`
description.


.. index:: SeqInvert

.. _SeqInvert:

SeqInvert
---------

`id=0.420185773758 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/core/SeqInvert>`__


.. function:: SeqInvert(seq)-> None

    Reverses the order of sequence elements.





    
    :param seq: Sequence 
    
    :type seq: :class:`CvSeq`
    
    
    
The function reverses the sequence in-place - makes the first element go last, the last element go first and so forth.


.. index:: SeqRemove

.. _SeqRemove:

SeqRemove
---------

`id=0.405976799419 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/core/SeqRemove>`__


.. function:: SeqRemove(seq,index)-> None

    Removes an element from the middle of a sequence.





    
    :param seq: Sequence 
    
    :type seq: :class:`CvSeq`
    
    
    :param index: Index of removed element 
    
    :type index: int
    
    
    
The function removes elements with the given
index. If the index is out of range the function reports an error. An
attempt to remove an element from an empty sequence is a special
case of this situation. The function removes an element by shifting
the sequence elements between the nearest end of the sequence and the
``index``
-th position, not counting the latter.



.. index:: SeqRemoveSlice

.. _SeqRemoveSlice:

SeqRemoveSlice
--------------

`id=0.589674828285 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/core/SeqRemoveSlice>`__


.. function:: SeqRemoveSlice(seq,slice)-> None

    Removes a sequence slice.





    
    :param seq: Sequence 
    
    :type seq: :class:`CvSeq`
    
    
    :param slice: The part of the sequence to remove 
    
    :type slice: :class:`CvSlice`
    
    
    
The function removes a slice from the sequence.

