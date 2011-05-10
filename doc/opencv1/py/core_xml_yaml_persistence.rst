XML/YAML Persistence
====================

.. highlight:: python



.. index:: Load

.. _Load:

Load
----




.. function:: Load(filename,storage=NULL,name=NULL)-> generic

    Loads an object from a file.





    
    :param filename: File name 
    
    :type filename: str
    
    
    :param storage: Memory storage for dynamic structures, such as  :ref:`CvSeq`  or  :ref:`CvGraph`  . It is not used for matrices or images. 
    
    :type storage: :class:`CvMemStorage`
    
    
    :param name: Optional object name. If it is NULL, the first top-level object in the storage will be loaded. 
    
    :type name: str
    
    
    
The function loads an object from a file. It provides a
simple interface to 
:ref:`Read`
. After the object is loaded, the file
storage is closed and all the temporary buffers are deleted. Thus,
to load a dynamic structure, such as a sequence, contour, or graph, one
should pass a valid memory storage destination to the function.


.. index:: Save

.. _Save:

Save
----




.. function:: Save(filename,structPtr,name=NULL,comment=NULL)-> None

    Saves an object to a file.





    
    :param filename: File name 
    
    :type filename: str
    
    
    :param structPtr: Object to save 
    
    :type structPtr: :class:`generic`
    
    
    :param name: Optional object name. If it is NULL, the name will be formed from  ``filename`` . 
    
    :type name: str
    
    
    :param comment: Optional comment to put in the beginning of the file 
    
    :type comment: str
    
    
    
The function saves an object to a file. It provides a simple interface to 
:ref:`Write`
.

