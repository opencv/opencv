XML/YAML Persistence (C API)
==============================

The section describes the OpenCV 1.x API for reading and writing data structures to/from XML or YAML files. It is now recommended to use the new C++ interface for reading and writing data.

.. highlight:: c

CvFileStorage
-------------

.. ocv:struct:: CvFileStorage

The structure ``CvFileStorage`` is a "black box" representation of the file storage associated with a file on disk. Several functions that are described below take ``CvFileStorage*`` as inputs and allow the user to save or to load hierarchical collections that consist of scalar values, standard CXCore objects (such as matrices, sequences, graphs), and user-defined objects.

OpenCV can read and write data in XML (http://www.w3c.org/XML) or YAML
(http://www.yaml.org) formats. Below is an example of 3x3 floating-point identity matrix ``A``, stored in XML and YAML files using CXCore functions:

XML: ::

  <?xml version="1.0">
  <opencv_storage>
  <A type_id="opencv-matrix">
    <rows>3</rows>
    <cols>3</cols>
    <dt>f</dt>
    <data>1. 0. 0. 0. 1. 0. 0. 0. 1.</data>
  </A>
  </opencv_storage>

YAML: ::

  %YAML:1.0
  A: !!opencv-matrix
    rows: 3
    cols: 3
    dt: f
    data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1.]

As it can be seen from the examples, XML uses nested tags to represent
hierarchy, while YAML uses indentation for that purpose (similar
to the Python programming language).

The same functions can read and write data in both formats;
the particular format is determined by the extension of the opened file, ".xml" for XML files and ".yml" or ".yaml" for YAML.

CvFileNode
----------

.. ocv:struct:: CvFileNode

File storage node. When XML/YAML file is read, it is first parsed and stored in the memory as a hierarchical collection of nodes. Each node can be a "leaf", that is, contain a single number or a string, or be a collection of other nodes. Collections are also referenced to as "structures" in the data writing functions. There can be named collections (mappings), where each element has a name and is accessed by a name, and ordered collections (sequences), where elements do not have names, but rather accessed by index.

    .. ocv:member:: int tag
    
        type of the file node:
        
            * CV_NODE_NONE - empty node
            * CV_NODE_INT - an integer
            * CV_NODE_REAL - a floating-point number
            * CV_NODE_STR - text string
            * CV_NODE_SEQ - a sequence
            * CV_NODE_MAP - a mapping

        type of the node can be retrieved using ``CV_NODE_TYPE(node->tag)`` macro.

    .. ocv:member:: CvTypeInfo* info
    
        optional pointer to the user type information. If you look at the matrix representation in XML and YAML, shown above, you may notice ``type_id="opencv-matrix"`` or ``!!opencv-matrix`` strings. They are used to specify that the certain element of a file is a representation of a data structure of certain type  ("opencv-matrix" corresponds to :ocv:struct:`CvMat`). When a file is parsed, such type identifiers are passed to :ocv:cfunc:`FindType` to find type information and the pointer to it is stored in the file node. See :ocv:struct:`CvTypeInfo` for more details.
        
    .. ocv:member:: union data
    
        the node data, declared as: ::
        
            union
            {
                double f; /* scalar floating-point number */
                int i;    /* scalar integer number */
                CvString str; /* text string */
                CvSeq* seq; /* sequence (ordered collection of file nodes) */
                struct CvMap* map; /* map (collection of named file nodes) */
            } data;

        ..
        
        Primitive nodes are read using :ocv:cfunc:`ReadInt`, :ocv:cfunc:`ReadReal` and :ocv:cfunc:`ReadString`. Sequences are read by iterating through ``node->data.seq`` (see "Dynamic Data Structures" section). Mappings are read using :ocv:cfunc:`GetFileNodeByName`. Nodes with the specified type (so that ``node->info != NULL``) can be read using :ocv:cfunc:`Read`.

CvAttrList
----------

.. ocv:struct:: CvAttrList

List of attributes. ::

    typedef struct CvAttrList
    {
        const char** attr; /* NULL-terminated array of (attribute_name,attribute_value) pairs */
        struct CvAttrList* next; /* pointer to next chunk of the attributes list */
    }
    CvAttrList;
    
    /* initializes CvAttrList structure */
    inline CvAttrList cvAttrList( const char** attr=NULL, CvAttrList* next=NULL );
    
    /* returns attribute value or 0 (NULL) if there is no such attribute */
    const char* cvAttrValue( const CvAttrList* attr, const char* attr_name );

..

In the current implementation, attributes are used to pass extra parameters when writing user objects (see 
:ocv:cfunc:`Write`). XML attributes inside tags are not supported, aside from the object type specification (``type_id`` attribute).

CvTypeInfo
----------

.. ocv:struct:: CvTypeInfo

Type information. ::

    typedef int (CV_CDECL *CvIsInstanceFunc)( const void* structPtr );
    typedef void (CV_CDECL *CvReleaseFunc)( void** structDblPtr );
    typedef void* (CV_CDECL *CvReadFunc)( CvFileStorage* storage, CvFileNode* node );
    typedef void (CV_CDECL *CvWriteFunc)( CvFileStorage* storage,
                                          const char* name,
                                          const void* structPtr,
                                          CvAttrList attributes );
    typedef void* (CV_CDECL *CvCloneFunc)( const void* structPtr );
    
    typedef struct CvTypeInfo
    {
        int flags; /* not used */
        int header_size; /* sizeof(CvTypeInfo) */
        struct CvTypeInfo* prev; /* previous registered type in the list */
        struct CvTypeInfo* next; /* next registered type in the list */
        const char* type_name; /* type name, written to file storage */
    
        /* methods */
        CvIsInstanceFunc is_instance; /* checks if the passed object belongs to the type */
        CvReleaseFunc release; /* releases object (memory etc.) */
        CvReadFunc read; /* reads object from file storage */
        CvWriteFunc write; /* writes object to file storage */
        CvCloneFunc clone; /* creates a copy of the object */
    }
    CvTypeInfo;

..

The structure contains information about one of the standard or user-defined types. Instances of the type may or may not contain a pointer to the corresponding  :ocv:struct:`CvTypeInfo` structure. In any case, there is a way to find the type info structure for a given object using the  :ocv:cfunc:`TypeOf` function. Alternatively, type info can be found by type name using :ocv:cfunc:`FindType`, which is used when an object is read from file storage. The user can register a new type with :ocv:cfunc:`RegisterType`
that adds the type information structure into the beginning of the type list. Thus, it is possible to create specialized types from generic standard types and override the basic methods.

Clone
-----
Makes a clone of an object.

.. ocv:cfunction:: void* cvClone( const void* structPtr )
    
    :param structPtr: The object to clone 

The function finds the type of a given object and calls ``clone`` with the passed object. Of course, if you know the object type, for example, ``structPtr`` is ``CvMat*``, it is faster to call the specific function, like :ocv:cfunc:`CloneMat`.

EndWriteStruct
--------------
Finishes writing to a file node collection.

.. ocv:cfunction:: void  cvEndWriteStruct(CvFileStorage* fs)

    :param fs: File storage 

.. seealso:: :ocv:cfunc:`StartWriteStruct`.

FindType
--------
Finds a type by its name.

.. ocv:cfunction:: CvTypeInfo* cvFindType(const char* typeName)
    
    :param typeName: Type name 

The function finds a registered type by its name. It returns NULL if there is no type with the specified name.

FirstType
---------
Returns the beginning of a type list.

.. ocv:cfunction:: CvTypeInfo* cvFirstType(void)

The function returns the first type in the list of registered types. Navigation through the list can be done via the ``prev`` and  ``next`` fields of the  :ocv:struct:`CvTypeInfo` structure.

GetFileNode
-----------
Finds a node in a map or file storage.

.. ocv:cfunction:: CvFileNode* cvGetFileNode( CvFileStorage* fs, CvFileNode* map, const CvStringHashNode* key, int createMissing=0 )
    
    :param fs: File storage 

    :param map: The parent map. If it is NULL, the function searches a top-level node. If both  ``map``  and  ``key``  are NULLs, the function returns the root file node - a map that contains top-level nodes. 

    :param key: Unique pointer to the node name, retrieved with  :ocv:cfunc:`GetHashedKey` 

    :param createMissing: Flag that specifies whether an absent node should be added to the map 

The function finds a file node. It is a faster version of  :ocv:cfunc:`GetFileNodeByName`
(see :ocv:cfunc:`GetHashedKey` discussion). Also, the function can insert a new node, if it is not in the map yet.

GetFileNodeByName
-----------------
Finds a node in a map or file storage.

.. ocv:cfunction:: CvFileNode* cvGetFileNodeByName( const CvFileStorage* fs, const CvFileNode* map, const char* name)
    
    :param fs: File storage 

    :param map: The parent map. If it is NULL, the function searches in all the top-level nodes (streams), starting with the first one. 

    :param name: The file node name 

The function finds a file node by ``name``. The node is searched either in ``map`` or, if the pointer is NULL, among the top-level file storage nodes. Using this function for maps and  :ocv:cfunc:`GetSeqElem`
(or sequence reader) for sequences, it is possible to navigate through the file storage. To speed up multiple queries for a certain key (e.g., in the case of an array of structures) one may use a combination of  :ocv:cfunc:`GetHashedKey` and :ocv:cfunc:`GetFileNode`.

GetFileNodeName
---------------
Returns the name of a file node.

.. ocv:cfunction:: const char* cvGetFileNodeName( const CvFileNode* node )

    :param node: File node 

The function returns the name of a file node or NULL, if the file node does not have a name or if  ``node`` is  ``NULL``.

GetHashedKey
------------
Returns a unique pointer for a given name.

.. ocv:cfunction:: CvStringHashNode* cvGetHashedKey( CvFileStorage* fs, const char* name, int len=-1, int createMissing=0 )

    :param fs: File storage 

    :param name: Literal node name 

    :param len: Length of the name (if it is known apriori), or -1 if it needs to be calculated 

    :param createMissing: Flag that specifies, whether an absent key should be added into the hash table 

The function returns a unique pointer for each particular file node name. This pointer can be then passed to the :ocv:cfunc:`GetFileNode` function that is faster than  :ocv:cfunc:`GetFileNodeByName`
because it compares text strings by comparing pointers rather than the strings' content.

Consider the following example where an array of points is encoded as a sequence of 2-entry maps: ::
    
    points:
      - { x: 10, y: 10 }
      - { x: 20, y: 20 }
      - { x: 30, y: 30 }
      # ...

..

Then, it is possible to get hashed "x" and "y" pointers to speed up decoding of the points. ::

    #include "cxcore.h"
    
    int main( int argc, char** argv )
    {
        CvFileStorage* fs = cvOpenFileStorage( "points.yml", 0, CV_STORAGE_READ );
        CvStringHashNode* x_key = cvGetHashedNode( fs, "x", -1, 1 );
        CvStringHashNode* y_key = cvGetHashedNode( fs, "y", -1, 1 );
        CvFileNode* points = cvGetFileNodeByName( fs, 0, "points" );
    
        if( CV_NODE_IS_SEQ(points->tag) )
        {
            CvSeq* seq = points->data.seq;
            int i, total = seq->total;
            CvSeqReader reader;
            cvStartReadSeq( seq, &reader, 0 );
            for( i = 0; i < total; i++ )
            {
                CvFileNode* pt = (CvFileNode*)reader.ptr;
    #if 1 /* faster variant */
                CvFileNode* xnode = cvGetFileNode( fs, pt, x_key, 0 );
                CvFileNode* ynode = cvGetFileNode( fs, pt, y_key, 0 );
                assert( xnode && CV_NODE_IS_INT(xnode->tag) &&
                        ynode && CV_NODE_IS_INT(ynode->tag));
                int x = xnode->data.i; // or x = cvReadInt( xnode, 0 );
                int y = ynode->data.i; // or y = cvReadInt( ynode, 0 );
    #elif 1 /* slower variant; does not use x_key & y_key */
                CvFileNode* xnode = cvGetFileNodeByName( fs, pt, "x" );
                CvFileNode* ynode = cvGetFileNodeByName( fs, pt, "y" );
                assert( xnode && CV_NODE_IS_INT(xnode->tag) &&
                        ynode && CV_NODE_IS_INT(ynode->tag));
                int x = xnode->data.i; // or x = cvReadInt( xnode, 0 );
                int y = ynode->data.i; // or y = cvReadInt( ynode, 0 );
    #else /* the slowest yet the easiest to use variant */
                int x = cvReadIntByName( fs, pt, "x", 0 /* default value */ );
                int y = cvReadIntByName( fs, pt, "y", 0 /* default value */ );
    #endif
                CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
                printf("
            }
        }
        cvReleaseFileStorage( &fs );
        return 0;
    }

..

Please note that whatever method of accessing a map you are using, it is
still much slower than using plain sequences; for example, in the above
example, it is more efficient to encode the points as pairs of integers
in a single numeric sequence.

GetRootFileNode
---------------
Retrieves one of the top-level nodes of the file storage.

.. ocv:cfunction:: CvFileNode* cvGetRootFileNode( const CvFileStorage* fs, int stream_index=0 )
    
    :param fs: File storage 

    :param stream_index: Zero-based index of the stream. See  :ocv:cfunc:`StartNextStream` . In most cases, there is only one stream in the file; however, there can be several. 

The function returns one of the top-level file nodes. The top-level nodes do not have a name, they correspond to the streams that are stored one after another in the file storage. If the index is out of range, the function returns a NULL pointer, so all the top-level nodes can be iterated by subsequent calls to the function with ``stream_index=0,1,...``, until the NULL pointer is returned. This function
can be used as a base for recursive traversal of the file storage.


Load
----
Loads an object from a file.

.. ocv:cfunction:: void* cvLoad( const char* filename, CvMemStorage* storage=NULL, const char* name=NULL, const char** realName=NULL )
.. ocv:pyoldfunction:: cv.Load(filename, storage=None, name=None)-> generic
    
    :param filename: File name 

    :param storage: Memory storage for dynamic structures, such as  :ocv:struct:`CvSeq`  or  :ocv:struct:`CvGraph`  . It is not used for matrices or images. 

    :param name: Optional object name. If it is NULL, the first top-level object in the storage will be loaded. 

    :param realName: Optional output parameter that will contain the name of the loaded object (useful if  ``name=NULL`` ) 

The function loads an object from a file. It basically reads the specified file, find the first top-level node and calls :ocv:cfunc:`Read` for that node. If the file node does not have type information or the type information can not be found by the type name, the function returns NULL. After the object is loaded, the file storage is closed and all the temporary buffers are deleted. Thus, to load a dynamic structure, such as a sequence, contour, or graph, one should pass a valid memory storage destination to the function.

OpenFileStorage
---------------
Opens file storage for reading or writing data.

.. ocv:cfunction:: CvFileStorage* cvOpenFileStorage( const char* filename, CvMemStorage* memstorage, int flags)

    :param filename: Name of the file associated with the storage 

    :param memstorage: Memory storage used for temporary data and for
        storing dynamic structures, such as  :ocv:struct:`CvSeq`  or  :ocv:struct:`CvGraph` .
        If it is NULL, a temporary memory storage is created and used. 

    :param flags: Can be one of the following:

            * **CV_STORAGE_READ** the storage is open for reading 

            * **CV_STORAGE_WRITE** the storage is open for writing 

The function opens file storage for reading or writing data. In the latter case, a new file is created or an existing file is rewritten. The type of the read or written file is determined by the filename extension:  ``.xml`` for  ``XML`` and  ``.yml`` or  ``.yaml`` for  ``YAML``. The function returns a pointer to the :ocv:struct:`CvFileStorage` structure. If the file cannot be opened then the function returns ``NULL``.

Read
----
Decodes an object and returns a pointer to it.

.. ocv:cfunction:: void* cvRead( CvFileStorage* fs, CvFileNode* node, CvAttrList* attributes=NULL )
    
    :param fs: File storage 

    :param node: The root object node 

    :param attributes: Unused parameter 

The function decodes a user object (creates an object in a native representation from the file storage subtree) and returns it. The object to be decoded must be an instance of a registered type that supports the ``read`` method (see :ocv:struct:`CvTypeInfo`). The type of the object is determined by the type name that is encoded in the file. If the object is a dynamic structure, it is created either in memory storage and passed to :ocv:cfunc:`OpenFileStorage` or, if a NULL pointer was passed, in temporary
memory storage, which is released when :ocv:cfunc:`ReleaseFileStorage` is called. Otherwise, if the object is not a dynamic structure, it is created in a heap and should be released with a specialized function or by using the generic :ocv:cfunc:`Release`.

ReadByName
----------
Finds an object by name and decodes it.

.. ocv:cfunction:: void* cvReadByName( CvFileStorage* fs, const CvFileNode* map, const char* name, CvAttrList* attributes=NULL )
    
    :param fs: File storage 

    :param map: The parent map. If it is NULL, the function searches a top-level node. 

    :param name: The node name 

    :param attributes: Unused parameter 

The function is a simple superposition of :ocv:cfunc:`GetFileNodeByName` and  :ocv:cfunc:`Read`.

ReadInt
-------
Retrieves an integer value from a file node.

.. ocv:cfunction:: int cvReadInt( const CvFileNode* node, int defaultValue=0 )

    :param node: File node 

    :param defaultValue: The value that is returned if  ``node``  is NULL 

The function returns an integer that is represented by the file node. If the file node is NULL, the 
``defaultValue`` is returned (thus, it is convenient to call the function right after :ocv:cfunc:`GetFileNode` without checking for a NULL pointer). If the file node has type  ``CV_NODE_INT``, then  ``node->data.i`` is returned. If the file node has type  ``CV_NODE_REAL``, then  ``node->data.f``
is converted to an integer and returned. Otherwise the error is reported.

ReadIntByName
-------------
Finds a file node and returns its value.

.. ocv:cfunction:: int cvReadIntByName( const CvFileStorage* fs, const CvFileNode* map, const char* name, int defaultValue=0 )

    :param fs: File storage 

    :param map: The parent map. If it is NULL, the function searches a top-level node. 

    :param name: The node name 

    :param defaultValue: The value that is returned if the file node is not found 

The function is a simple superposition of  :ocv:cfunc:`GetFileNodeByName` and  :ocv:cfunc:`ReadInt`.

ReadRawData
-----------
Reads multiple numbers.

.. ocv:cfunction:: void cvReadRawData( const CvFileStorage* fs, const CvFileNode* src, void* dst, const char* dt)

    :param fs: File storage 

    :param src: The file node (a sequence) to read numbers from 

    :param dst: Pointer to the destination array 

    :param dt: Specification of each array element. It has the same format as in  :ocv:cfunc:`WriteRawData` . 

The function reads elements from a file node that represents a sequence of scalars.


ReadRawDataSlice
----------------
Initializes file node sequence reader.

.. ocv:cfunction:: void cvReadRawDataSlice( const CvFileStorage* fs, CvSeqReader* reader, int count, void* dst, const char* dt )
    
    :param fs: File storage 

    :param reader: The sequence reader. Initialize it with  :ocv:cfunc:`StartReadRawData` . 

    :param count: The number of elements to read 

    :param dst: Pointer to the destination array 

    :param dt: Specification of each array element. It has the same format as in  :ocv:cfunc:`WriteRawData` . 

The function reads one or more elements from the file node, representing a sequence, to a user-specified array. The total number of read sequence elements is a product of ``total``
and the number of components in each array element. For example, if ``dt=2if``, the function will read ``total*3`` sequence elements. As with any sequence, some parts of the file node sequence can be skipped or read repeatedly by repositioning the reader using :ocv:cfunc:`SetSeqReaderPos`.

ReadReal
--------
Retrieves a floating-point value from a file node.

.. ocv:cfunction:: double cvReadReal( const CvFileNode* node, double defaultValue=0. )
    
    :param node: File node 

    :param defaultValue: The value that is returned if  ``node``  is NULL 

The function returns a floating-point value
that is represented by the file node. If the file node is NULL, the
``defaultValue``
is returned (thus, it is convenient to call
the function right after 
:ocv:cfunc:`GetFileNode`
without checking for a NULL
pointer). If the file node has type 
``CV_NODE_REAL``
,
then 
``node->data.f``
is returned. If the file node has type
``CV_NODE_INT``
, then 
``node-:math:`>`data.f``
is converted to floating-point
and returned. Otherwise the result is not determined.


ReadRealByName
--------------
Finds a file node and returns its value.

.. ocv:cfunction:: double  cvReadRealByName( const CvFileStorage* fs, const CvFileNode* map, const char* name, double defaultValue=0.)
    
    :param fs: File storage 

    :param map: The parent map. If it is NULL, the function searches a top-level node. 

    :param name: The node name 

    :param defaultValue: The value that is returned if the file node is not found 

The function is a simple superposition of 
:ocv:cfunc:`GetFileNodeByName`
and 
:ocv:cfunc:`ReadReal`
.


ReadString
----------
Retrieves a text string from a file node.

.. ocv:cfunction:: const char* cvReadString( const CvFileNode* node, const char* defaultValue=NULL )
    
    :param node: File node 

    :param defaultValue: The value that is returned if  ``node``  is NULL 

The function returns a text string that is represented
by the file node. If the file node is NULL, the 
``defaultValue``
is returned (thus, it is convenient to call the function right after
:ocv:cfunc:`GetFileNode`
without checking for a NULL pointer). If
the file node has type 
``CV_NODE_STR``
, then 
``node-:math:`>`data.str.ptr``
is returned. Otherwise the result is not determined.


ReadStringByName
----------------
Finds a file node by its name and returns its value.

.. ocv:cfunction:: const char* cvReadStringByName( const CvFileStorage* fs, const CvFileNode* map, const char* name, const char* defaultValue=NULL )
    
    :param fs: File storage 

    :param map: The parent map. If it is NULL, the function searches a top-level node. 

    :param name: The node name 

    :param defaultValue: The value that is returned if the file node is not found 

The function is a simple superposition of 
:ocv:cfunc:`GetFileNodeByName`
and 
:ocv:cfunc:`ReadString`
.


RegisterType
------------
Registers a new type.

.. ocv:cfunction:: void cvRegisterType(const CvTypeInfo* info)

    :param info: Type info structure 

The function registers a new type, which is
described by 
``info``
. The function creates a copy of the structure,
so the user should delete it after calling the function.


Release
-------
Releases an object.

.. ocv:cfunction:: void cvRelease( void** structPtr )

    :param structPtr: Double pointer to the object 

The function finds the type of a given object and calls 
``release``
with the double pointer.


ReleaseFileStorage
------------------
Releases file storage.

.. ocv:cfunction:: void  cvReleaseFileStorage(CvFileStorage** fs)
    
    :param fs: Double pointer to the released file storage 

The function closes the file associated with the storage and releases all the temporary structures. It must be called after all I/O operations with the storage are finished.


Save
----
Saves an object to a file.

.. ocv:cfunction:: void cvSave( const char* filename, const void* structPtr, const char* name=NULL, const char* comment=NULL, CvAttrList attributes=cvAttrList())
.. ocv:pyoldfunction:: cv.Save(filename, structPtr, name=None, comment=None)-> None
    
    :param filename: File name 

    :param structPtr: Object to save 

    :param name: Optional object name. If it is NULL, the name will be formed from  ``filename`` . 

    :param comment: Optional comment to put in the beginning of the file 

    :param attributes: Optional attributes passed to  :ocv:cfunc:`Write` 

The function saves an object to a file. It provides a simple interface to 
:ocv:cfunc:`Write`
.


StartNextStream
---------------
Starts the next stream.

.. ocv:cfunction:: void cvStartNextStream(CvFileStorage* fs)

    :param fs: File storage 

The function finishes the currently written stream and starts the next stream. In the case of XML the file with multiple streams looks like this: ::

    <opencv_storage>
    <!-- stream #1 data -->
    </opencv_storage>
    <opencv_storage>
    <!-- stream #2 data -->
    </opencv_storage>
    ...

The YAML file will look like this: ::

    %YAML:1.0
    # stream #1 data
    ...
    ---
    # stream #2 data

This is useful for concatenating files or for resuming the writing process.


StartReadRawData
----------------
Initializes the file node sequence reader.

.. ocv:cfunction:: void cvStartReadRawData( const CvFileStorage* fs, const CvFileNode* src, CvSeqReader* reader)

    :param fs: File storage 

    :param src: The file node (a sequence) to read numbers from 

    :param reader: Pointer to the sequence reader 

The function initializes the sequence reader to read data from a file node. The initialized reader can be then passed to :ocv:cfunc:`ReadRawDataSlice`.


StartWriteStruct
----------------
Starts writing a new structure.

.. ocv:cfunction:: void  cvStartWriteStruct( CvFileStorage* fs, const char* name, int struct_flags, const char* typeName=NULL, CvAttrList attributes=cvAttrList())
    
    :param fs: File storage 

    :param name: Name of the written structure. The structure can be accessed by this name when the storage is read. 

    :param struct_flags: A combination one of the following values: 
         
            * **CV_NODE_SEQ** the written structure is a sequence (see discussion of  :ocv:struct:`CvFileStorage` ), that is, its elements do not have a name. 
            
            * **CV_NODE_MAP** the written structure is a map (see discussion of  :ocv:struct:`CvFileStorage` ), that is, all its elements have names. 

         One and only one of the two above flags must be specified 

            * **CV_NODE_FLOW** the optional flag that makes sense only for YAML streams. It means that the structure is written as a flow (not as a block), which is more compact. It is recommended to use this flag for structures or arrays whose elements are all scalars. 

    :param typeName: Optional parameter - the object type name. In
        case of XML it is written as a  ``type_id``  attribute of the
        structure opening tag. In the case of YAML it is written after a colon
        following the structure name (see the example in  :ocv:struct:`CvFileStorage` 
        description). Mainly it is used with user objects. When the storage
        is read, the encoded type name is used to determine the object type
        (see  :ocv:struct:`CvTypeInfo`  and  :ocv:cfunc:`FindType` ). 

    :param attributes: This parameter is not used in the current implementation 

The function starts writing a compound structure (collection) that can be a sequence or a map. After all the structure fields, which can be scalars or structures, are written, :ocv:cfunc:`EndWriteStruct` should be called. The function can be used to group some objects or to implement the ``write`` function for a some user object (see :ocv:struct:`CvTypeInfo`).


TypeOf
------
Returns the type of an object.

.. ocv:cfunction:: CvTypeInfo* cvTypeOf( const void* structPtr )

    :param structPtr: The object pointer 

The function finds the type of a given object. It iterates through the list of registered types and calls the  ``is_instance`` function/method for every type info structure with that object until one of them returns non-zero or until the whole list has been traversed. In the latter case, the function returns NULL.


UnregisterType
--------------
Unregisters the type.

.. ocv:cfunction:: void cvUnregisterType( const char* typeName )
    
    :param typeName: Name of an unregistered type 

The function unregisters a type with a specified name. If the name is unknown, it is possible to locate the type info by an instance of the type using :ocv:cfunc:`TypeOf` or by iterating the type list, starting from  :ocv:cfunc:`FirstType`, and then calling ``cvUnregisterType(info->typeName)``.


Write
-----
Writes an object to file storage.

.. ocv:cfunction:: void  cvWrite( CvFileStorage* fs, const char* name, const void* ptr, CvAttrList attributes=cvAttrList() )
    
    :param fs: File storage 

    :param name: Name of the written object. Should be NULL if and only if the parent structure is a sequence. 

    :param ptr: Pointer to the object 

    :param attributes: The attributes of the object. They are specific for each particular type (see the discussion below).

The function writes an object to file storage. First, the appropriate type info is found using :ocv:cfunc:`TypeOf`. Then, the ``write`` method associated with the type info is called.

Attributes are used to customize the writing procedure. The standard types support the following attributes (all the ``dt`` attributes have the same format as in :ocv:cfunc:`WriteRawData`):

#.
    CvSeq

        * **header_dt** description of user fields of the sequence header that follow CvSeq, or CvChain (if the sequence is a Freeman chain) or CvContour (if the sequence is a contour or point sequence) 

        * **dt** description of the sequence elements. 

        * **recursive** if the attribute is present and is not equal to "0" or "false", the whole tree of sequences (contours) is stored.

#.
    CvGraph

        * **header_dt** description of user fields of the graph header that follows CvGraph; 

        * **vertex_dt** description of user fields of graph vertices 

        * **edge_dt** description of user fields of graph edges (note that the edge weight is always written, so there is no need to specify it explicitly)

Below is the code that creates the YAML file shown in the 
``CvFileStorage``
description:

::

    #include "cxcore.h"
    
    int main( int argc, char** argv )
    {
        CvMat* mat = cvCreateMat( 3, 3, CV_32F );
        CvFileStorage* fs = cvOpenFileStorage( "example.yml", 0, CV_STORAGE_WRITE );
    
        cvSetIdentity( mat );
        cvWrite( fs, "A", mat, cvAttrList(0,0) );
    
        cvReleaseFileStorage( &fs );
        cvReleaseMat( &mat );
        return 0;
    }

..


WriteComment
------------
Writes a comment.

.. ocv:cfunction:: void  cvWriteComment( CvFileStorage* fs, const char* comment, int eolComment)

    :param fs: File storage 

    :param comment: The written comment, single-line or multi-line 

    :param eolComment: If non-zero, the function tries to put the comment at the end of current line. If the flag is zero, if the comment is multi-line, or if it does not fit at the end of the current line, the comment starts  a new line. 

The function writes a comment into file storage. The comments are skipped when the storage is read.

WriteFileNode
-------------
Writes a file node to another file storage.

.. ocv:cfunction:: void cvWriteFileNode( CvFileStorage* fs, const char* new_node_name, const CvFileNode* node, int embed )
    
    :param fs: Destination file storage 

    :param new_node_name: New name of the file node in the destination file storage. To keep the existing name, use  :ocv:cfunc:`cvGetFileNodeName` 

    :param node: The written node 

    :param embed: If the written node is a collection and this parameter is not zero, no extra level of hierarchy is created. Instead, all the elements of  ``node``  are written into the currently written structure. Of course, map elements can only be embedded into another map, and sequence elements can only be embedded into another sequence. 

The function writes a copy of a file node to file storage. Possible applications of the function are merging several file storages into one and conversion between XML and YAML formats.

WriteInt
--------
Writes an integer value.

.. ocv:cfunction:: void  cvWriteInt( CvFileStorage* fs, const char* name, int value)

    :param fs: File storage 

    :param name: Name of the written value. Should be NULL if and only if the parent structure is a sequence. 

    :param value: The written value 

The function writes a single integer value (with or without a name) to the file storage.


WriteRawData
------------
Writes multiple numbers.

.. ocv:cfunction:: void  cvWriteRawData( CvFileStorage* fs, const void* src, int len, const char* dt )
    
    :param fs: File storage 

    :param src: Pointer to the written array 

    :param len: Number of the array elements to write 

    :param dt: Specification of each array element that has the following format  ``([count]{'u'|'c'|'w'|'s'|'i'|'f'|'d'})...`` 
        where the characters correspond to fundamental C types: 

            * **u** 8-bit unsigned number 

            * **c** 8-bit signed number 

            * **w** 16-bit unsigned number 

            * **s** 16-bit signed number 

            * **i** 32-bit signed number 

            * **f** single precision floating-point number 

            * **d** double precision floating-point number 

            * **r** pointer, 32 lower bits of which are written as a signed integer. The type can be used to store structures with links between the elements. ``count``  is the optional counter of values of a given type. For
                example,  ``2if``  means that each array element is a structure
                of 2 integers, followed by a single-precision floating-point number. The
                equivalent notations of the above specification are ' ``iif`` ',
                ' ``2i1f`` ' and so forth. Other examples:  ``u``  means that the
                array consists of bytes, and  ``2d``  means the array consists of pairs
                of doubles.

The function writes an array, whose elements consist
of single or multiple numbers. The function call can be replaced with
a loop containing a few 
:ocv:cfunc:`WriteInt`
and 
:ocv:cfunc:`WriteReal`
calls, but
a single call is more efficient. Note that because none of the elements
have a name, they should be written to a sequence rather than a map.


WriteReal
---------
Writes a floating-point value.

.. ocv:cfunction:: void  cvWriteReal( CvFileStorage* fs, const char* name, double value )
    
    :param fs: File storage 

    :param name: Name of the written value. Should be NULL if and only if the parent structure is a sequence. 

    :param value: The written value 

The function writes a single floating-point value (with or without a name) to file storage. Special values are encoded as follows: NaN (Not A Number) as .NaN, infinity as +.Inf or -.Inf.

The following example shows how to use the low-level writing functions to store custom structures, such as termination criteria, without registering a new type. ::

    void write_termcriteria( CvFileStorage* fs, const char* struct_name,
                             CvTermCriteria* termcrit )
    {
        cvStartWriteStruct( fs, struct_name, CV_NODE_MAP, NULL, cvAttrList(0,0));
        cvWriteComment( fs, "termination criteria", 1 ); // just a description
        if( termcrit->type & CV_TERMCRIT_ITER )
            cvWriteInteger( fs, "max_iterations", termcrit->max_iter );
        if( termcrit->type & CV_TERMCRIT_EPS )
            cvWriteReal( fs, "accuracy", termcrit->epsilon );
        cvEndWriteStruct( fs );
    }

..


WriteString
-----------
Writes a text string.

.. ocv:cfunction:: void  cvWriteString( CvFileStorage* fs, const char* name, const char* str, int quote=0 )

    :param fs: File storage 

    :param name: Name of the written string . Should be NULL if and only if the parent structure is a sequence. 

    :param str: The written text string 

    :param quote: If non-zero, the written string is put in quotes, regardless of whether they are required. Otherwise, if the flag is zero, quotes are used only when they are required (e.g. when the string starts with a digit or contains spaces). 

The function writes a text string to file storage.
