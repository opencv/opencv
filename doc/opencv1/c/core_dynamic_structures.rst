Dynamic Structures
==================

.. highlight:: c



.. index:: CvMemStorage

.. _CvMemStorage:

CvMemStorage
------------

`id=0.334804981773 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvMemStorage>`__

.. ctype:: CvMemStorage



Growing memory storage.




::


    
    typedef struct CvMemStorage
    {
        struct CvMemBlock* bottom;/* first allocated block */
        struct CvMemBlock* top; /* the current memory block - top of the stack */
        struct CvMemStorage* parent; /* borrows new blocks from */
        int block_size; /* block size */
        int free_space; /* free space in the ``top`` block (in bytes) */
    } CvMemStorage;
    

..

Memory storage is a low-level structure used to store dynamicly growing
data structures such as sequences, contours, graphs, subdivisions, etc. It
is organized as a list of memory blocks of equal size - 
``bottom``
field is the beginning of the list of blocks and 
``top``
is the
currently used block, but not necessarily the last block of the list. All
blocks between 
``bottom``
and 
``top``
, not including the
latter, are considered fully occupied; all blocks between 
``top``
and the last block, not including 
``top``
, are considered free
and 
``top``
itself is partly ocupied - 
``free_space``
contains the number of free bytes left in the end of 
``top``
.

A new memory buffer that may be allocated explicitly by
:ref:`MemStorageAlloc`
function or implicitly by higher-level functions,
such as 
:ref:`SeqPush`
, 
:ref:`GraphAddEdge`
, etc., 
``always``
starts in the end of the current block if it fits there. After allocation,
``free_space``
is decremented by the size of the allocated buffer
plus some padding to keep the proper alignment. When the allocated buffer
does not fit into the available portion of 
``top``
, the next storage
block from the list is taken as 
``top``
and 
``free_space``
is reset to the whole block size prior to the allocation.

If there are no more free blocks, a new block is allocated (or borrowed
from the parent, see 
:ref:`CreateChildMemStorage`
) and added to the end of
list. Thus, the storage behaves as a stack with 
``bottom``
indicating
bottom of the stack and the pair (
``top``
, 
``free_space``
)
indicating top of the stack. The stack top may be saved via
:ref:`SaveMemStoragePos`
, restored via 
:ref:`RestoreMemStoragePos`
,
or reset via 
:ref:`ClearStorage`
.

.. index:: CvMemBlock

.. _CvMemBlock:

CvMemBlock
----------

`id=0.108820280688 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvMemBlock>`__

.. ctype:: CvMemBlock



Memory storage block.




::


    
    typedef struct CvMemBlock
    {
        struct CvMemBlock* prev;
        struct CvMemBlock* next;
    } CvMemBlock;
    

..

The structure 
:ref:`CvMemBlock`
represents a single block of memory
storage. The actual data in the memory blocks follows the header, that is,
the 
:math:`i_{th}`
byte of the memory block can be retrieved with the expression
``((char*)(mem_block_ptr+1))[i]``
. However, there is normally no need
to access the storage structure fields directly.


.. index:: CvMemStoragePos

.. _CvMemStoragePos:

CvMemStoragePos
---------------

`id=0.832479670677 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvMemStoragePos>`__

.. ctype:: CvMemStoragePos



Memory storage position.




::


    
    typedef struct CvMemStoragePos
    {
        CvMemBlock* top;
        int free_space;
    } CvMemStoragePos;
    

..

The structure described above stores the position of the stack top that can be saved via 
:ref:`SaveMemStoragePos`
and restored via 
:ref:`RestoreMemStoragePos`
.


.. index:: CvSeq

.. _CvSeq:

CvSeq
-----

`id=0.387726368946 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvSeq>`__

.. ctype:: CvSeq



Growable sequence of elements.




::


    
    
    #define CV_SEQUENCE_FIELDS() \
        int flags; /* micsellaneous flags */ \
        int header_size; /* size of sequence header */ \
        struct CvSeq* h_prev; /* previous sequence */ \
        struct CvSeq* h_next; /* next sequence */ \
        struct CvSeq* v_prev; /* 2nd previous sequence */ \
        struct CvSeq* v_next; /* 2nd next sequence */ \
        int total; /* total number of elements */ \
        int elem_size;/* size of sequence element in bytes */ \
        char* block_max;/* maximal bound of the last block */ \
        char* ptr; /* current write pointer */ \
        int delta_elems; /* how many elements allocated when the sequence grows 
                            (sequence granularity) */ \
        CvMemStorage* storage; /* where the seq is stored */ \
        CvSeqBlock* free_blocks; /* free blocks list */ \
        CvSeqBlock* first; /* pointer to the first sequence block */
    
    typedef struct CvSeq
    {
        CV_SEQUENCE_FIELDS()
    } CvSeq;
    
    

..

The structure 
:ref:`CvSeq`
is a base for all of OpenCV dynamic data structures.

Such an unusual definition via a helper macro simplifies the extension
of the structure 
:ref:`CvSeq`
with additional parameters. To extend
:ref:`CvSeq`
the user may define a new structure and put user-defined
fields after all 
:ref:`CvSeq`
fields that are included via the macro
``CV_SEQUENCE_FIELDS()``
.

There are two types of sequences - dense and sparse. The base type for dense
sequences is 
:ref:`CvSeq`
and such sequences are used to represent
growable 1d arrays - vectors, stacks, queues, and deques. They have no gaps
in the middle - if an element is removed from the middle or inserted
into the middle of the sequence, the elements from the closer end are
shifted. Sparse sequences have 
:ref:`CvSet`
as a base class and they are
discussed later in more detail. They are sequences of nodes; each may be either occupied or free as indicated by the node flag. Such
sequences are used for unordered data structures such as sets of elements,
graphs, hash tables and so forth.

The field 
``header_size``
contains the actual size of the sequence
header and should be greater than or equal to 
``sizeof(CvSeq)``
.

The fields
``h_prev``
, 
``h_next``
, 
``v_prev``
, 
``v_next``
can be used to create hierarchical structures from separate sequences. The
fields 
``h_prev``
and 
``h_next``
point to the previous and
the next sequences on the same hierarchical level, while the fields
``v_prev``
and 
``v_next``
point to the previous and the
next sequences in the vertical direction, that is, the parent and its first
child. But these are just names and the pointers can be used in a
different way.

The field 
``first``
points to the first sequence block, whose structure is described below.

The field 
``total``
contains the actual number of dense sequence elements and number of allocated nodes in a sparse sequence.

The field 
``flags``
contains the particular dynamic type
signature (
``CV_SEQ_MAGIC_VAL``
for dense sequences and
``CV_SET_MAGIC_VAL``
for sparse sequences) in the highest 16
bits and miscellaneous information about the sequence. The lowest
``CV_SEQ_ELTYPE_BITS``
bits contain the ID of the element
type. Most of sequence processing functions do not use element type but rather
element size stored in 
``elem_size``
. If a sequence contains the
numeric data for one of the 
:ref:`CvMat`
type then the element type matches
to the corresponding 
:ref:`CvMat`
element type, e.g., 
``CV_32SC2``
may be
used for a sequence of 2D points, 
``CV_32FC1``
for sequences of floating-point
values, etc. A 
``CV_SEQ_ELTYPE(seq_header_ptr)``
macro retrieves the
type of sequence elements. Processing functions that work with numerical
sequences check that 
``elem_size``
is equal to that calculated from
the type element size. Besides 
:ref:`CvMat`
compatible types, there
are few extra element types defined in the 
``cvtypes.h``
header:

Standard Types of Sequence Elements




::


    
    
    #define CV_SEQ_ELTYPE_POINT          CV_32SC2  /* (x,y) */
    #define CV_SEQ_ELTYPE_CODE           CV_8UC1   /* freeman code: 0..7 */
    #define CV_SEQ_ELTYPE_GENERIC        0 /* unspecified type of 
                                            sequence elements */
    #define CV_SEQ_ELTYPE_PTR            CV_USRTYPE1 /* =6 */
    #define CV_SEQ_ELTYPE_PPOINT         CV_SEQ_ELTYPE_PTR  /* &elem: pointer to 
                                                    element of other sequence */
    #define CV_SEQ_ELTYPE_INDEX          CV_32SC1  /* #elem: index of element of 
                                                          some other sequence */
    #define CV_SEQ_ELTYPE_GRAPH_EDGE     CV_SEQ_ELTYPE_GENERIC  /* &next_o, 
                                                      &next_d, &vtx_o, &vtx_d */
    #define CV_SEQ_ELTYPE_GRAPH_VERTEX   CV_SEQ_ELTYPE_GENERIC  /* first_edge, 
                                                                       &(x,y) */
    #define CV_SEQ_ELTYPE_TRIAN_ATR      CV_SEQ_ELTYPE_GENERIC  /* vertex of the 
                                                                binary tree   */
    #define CV_SEQ_ELTYPE_CONNECTED_COMP CV_SEQ_ELTYPE_GENERIC  /* connected 
                                                                   component  */
    #define CV_SEQ_ELTYPE_POINT3D        CV_32FC3  /* (x,y,z)  */
    
    

..

The next 
``CV_SEQ_KIND_BITS``
bits specify the kind of sequence:

Standard Kinds of Sequences




::


    
    
    /* generic (unspecified) kind of sequence */
    #define CV_SEQ_KIND_GENERIC     (0 << CV_SEQ_ELTYPE_BITS)
    
    /* dense sequence suntypes */
    #define CV_SEQ_KIND_CURVE       (1 << CV_SEQ_ELTYPE_BITS)
    #define CV_SEQ_KIND_BIN_TREE    (2 << CV_SEQ_ELTYPE_BITS)
    
    /* sparse sequence (or set) subtypes */
    #define CV_SEQ_KIND_GRAPH       (3 << CV_SEQ_ELTYPE_BITS)
    #define CV_SEQ_KIND_SUBDIV2D    (4 << CV_SEQ_ELTYPE_BITS)
    
    

..

The remaining bits are used to identify different features specific
to certain sequence kinds and element types. For example, curves
made of points 
``(CV_SEQ_KIND_CURVE|CV_SEQ_ELTYPE_POINT)``
,
together with the flag 
``CV_SEQ_FLAG_CLOSED``
, belong to the
type 
``CV_SEQ_POLYGON``
or, if other flags are used, to its
subtype. Many contour processing functions check the type of the input
sequence and report an error if they do not support this type. The
file 
``cvtypes.h``
stores the complete list of all supported
predefined sequence types and helper macros designed to get the sequence
type of other properties. The definition of the building
blocks of sequences can be found below.


.. index:: CvSeqBlock

.. _CvSeqBlock:

CvSeqBlock
----------

`id=0.211082721332 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvSeqBlock>`__

.. ctype:: CvSeqBlock



Continuous sequence block.




::


    
    
    typedef struct CvSeqBlock
    {
        struct CvSeqBlock* prev; /* previous sequence block */
        struct CvSeqBlock* next; /* next sequence block */
        int start_index; /* index of the first element in the block +
        sequence->first->start_index */
        int count; /* number of elements in the block */
        char* data; /* pointer to the first element of the block */
    } CvSeqBlock;
    
    

..

Sequence blocks make up a circular double-linked list, so the pointers
``prev``
and 
``next``
are never 
``NULL``
and point to the
previous and the next sequence blocks within the sequence. It means that
``next``
of the last block is the first block and 
``prev``
of
the first block is the last block. The fields 
``startIndex``
and
``count``
help to track the block location within the sequence. For
example, if the sequence consists of 10 elements and splits into three
blocks of 3, 5, and 2 elements, and the first block has the parameter
``startIndex = 2``
, then pairs 
``(startIndex, count)``
for the sequence
blocks are
(2,3), (5, 5), and (10, 2)
correspondingly. The parameter
``startIndex``
of the first block is usually 
``0``
unless
some elements have been inserted at the beginning of the sequence.


.. index:: CvSlice

.. _CvSlice:

CvSlice
-------

`id=0.519045630752 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvSlice>`__

.. ctype:: CvSlice



A sequence slice.




::


    
    typedef struct CvSlice
    {
        int start_index;
        int end_index;
    } CvSlice;
    
    inline CvSlice cvSlice( int start, int end );
    #define CV_WHOLE_SEQ_END_INDEX 0x3fffffff
    #define CV_WHOLE_SEQ  cvSlice(0, CV_WHOLE_SEQ_END_INDEX)
    
    /* calculates the sequence slice length */
    int cvSliceLength( CvSlice slice, const CvSeq* seq );
    

..

Some of functions that operate on sequences take a 
``CvSlice slice``
parameter that is often set to the whole sequence (CV
_
WHOLE
_
SEQ) by
default. Either of the 
``startIndex``
and 
``endIndex``
may be negative or exceed the sequence length, 
``startIndex``
is
inclusive, and 
``endIndex``
is an exclusive boundary. If they are equal,
the slice is considered empty (i.e., contains no elements). Because
sequences are treated as circular structures, the slice may select a
few elements in the end of a sequence followed by a few elements at the
beginning of the sequence. For example, 
``cvSlice(-2, 3)``
in the case of
a 10-element sequence will select a 5-element slice, containing the pre-last
(8th), last (9th), the very first (0th), second (1th) and third (2nd)
elements. The functions normalize the slice argument in the following way:
first, 
:ref:`SliceLength`
is called to determine the length of the slice,
then, 
``startIndex``
of the slice is normalized similarly to the
argument of 
:ref:`GetSeqElem`
(i.e., negative indices are allowed). The
actual slice to process starts at the normalized 
``startIndex``
and lasts 
:ref:`SliceLength`
elements (again, assuming the sequence is
a circular structure).

If a function does not accept a slice argument, but you want to process
only a part of the sequence, the sub-sequence may be extracted
using the 
:ref:`SeqSlice`
function, or stored into a continuous
buffer with 
:ref:`CvtSeqToArray`
(optionally, followed by
:ref:`MakeSeqHeaderForArray`
).


.. index:: CvSet

.. _CvSet:

CvSet
-----

`id=0.825263988294 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvSet>`__

.. ctype:: CvSet



Collection of nodes.




::


    
    typedef struct CvSetElem
    {
        int flags; /* it is negative if the node is free and zero or positive otherwise */
        struct CvSetElem* next_free; /* if the node is free, the field is a
                                        pointer to next free node */
    }
    CvSetElem;
    
    #define CV_SET_FIELDS()    \
        CV_SEQUENCE_FIELDS()   /* inherits from [#CvSeq CvSeq] */ \
        struct CvSetElem* free_elems; /* list of free nodes */
    
    typedef struct CvSet
    {
        CV_SET_FIELDS()
    } CvSet;
    

..

The structure 
:ref:`CvSet`
is a base for OpenCV sparse data structures.

As follows from the above declaration, 
:ref:`CvSet`
inherits from
:ref:`CvSeq`
and it adds the 
``free_elems``
field, which
is a list of free nodes, to it. Every set node, whether free or not, is an
element of the underlying sequence. While there are no restrictions on
elements of dense sequences, the set (and derived structures) elements
must start with an integer field and be able to fit CvSetElem structure,
because these two fields (an integer followed by a pointer) are required
for the organization of a node set with the list of free nodes. If a node is
free, the 
``flags``
field is negative (the most-significant bit, or
MSB, of the field is set), and the 
``next_free``
points to the next
free node (the first free node is referenced by the 
``free_elems``
field of 
:ref:`CvSet`
). And if a node is occupied, the 
``flags``
field
is positive and contains the node index that may be retrieved using the
(
``set_elem->flags & CV_SET_ELEM_IDX_MASK``
) expressions, the rest of
the node content is determined by the user. In particular, the occupied
nodes are not linked as the free nodes are, so the second field can be
used for such a link as well as for some different purpose. The macro
``CV_IS_SET_ELEM(set_elem_ptr)``
can be used to determined whether
the specified node is occupied or not.

Initially the set and the list are empty. When a new node is requested
from the set, it is taken from the list of free nodes, which is then updated. If the list appears to be empty, a new sequence block is allocated
and all the nodes within the block are joined in the list of free
nodes. Thus, the 
``total``
field of the set is the total number of nodes
both occupied and free. When an occupied node is released, it is added
to the list of free nodes. The node released last will be occupied first.

In OpenCV 
:ref:`CvSet`
is used for representing graphs (
:ref:`CvGraph`
),
sparse multi-dimensional arrays (
:ref:`CvSparseMat`
), and planar subdivisions
:ref:`CvSubdiv2D`
.



.. index:: CvGraph

.. _CvGraph:

CvGraph
-------

`id=0.878989998624 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvGraph>`__

.. ctype:: CvGraph



Oriented or unoriented weighted graph.




::


    
    #define CV_GRAPH_VERTEX_FIELDS()    \
        int flags; /* vertex flags */   \
        struct CvGraphEdge* first; /* the first incident edge */
    
    typedef struct CvGraphVtx
    {
        CV_GRAPH_VERTEX_FIELDS()
    }
    CvGraphVtx;
    
    #define CV_GRAPH_EDGE_FIELDS()      \
        int flags; /* edge flags */     \
        float weight; /* edge weight */ \
        struct CvGraphEdge* next[2]; /* the next edges in the incidence lists for staring (0) */ \
                                      /* and ending (1) vertices */ \
        struct CvGraphVtx* vtx[2]; /* the starting (0) and ending (1) vertices */
    
    typedef struct CvGraphEdge
    {
        CV_GRAPH_EDGE_FIELDS()
    }
    CvGraphEdge;
    
    #define  CV_GRAPH_FIELDS()                  \
        CV_SET_FIELDS() /* set of vertices */   \
        CvSet* edges;   /* set of edges */
    
    typedef struct CvGraph
    {
        CV_GRAPH_FIELDS()
    }
    CvGraph;
    
    

..

The structure 
:ref:`CvGraph`
is a base for graphs used in OpenCV.

The graph structure inherits from 
:ref:`CvSet`
- which describes common graph properties and the graph vertices, and contains another set as a member - which describes the graph edges.

The vertex, edge, and the graph header structures are declared using the
same technique as other extendible OpenCV structures - via macros, which
simplify extension and customization of the structures. While the vertex
and edge structures do not inherit from 
:ref:`CvSetElem`
explicitly, they
satisfy both conditions of the set elements: having an integer field in
the beginning and fitting within the CvSetElem structure. The 
``flags``
fields are
used as for indicating occupied vertices and edges as well as for other
purposes, for example, for graph traversal (see 
:ref:`CreateGraphScanner`
et al.), so it is better not to use them directly.

The graph is represented as a set of edges each of which has a list of
incident edges. The incidence lists for different vertices are interleaved
to avoid information duplication as much as posssible.

The graph may be oriented or unoriented. In the latter case there is no
distiction between the edge connecting vertex 
:math:`A`
with vertex 
:math:`B`
and the edge
connecting vertex 
:math:`B`
with vertex 
:math:`A`
- only one of them can exist in the
graph at the same moment and it represents both 
:math:`A \rightarrow B`
and
:math:`B \rightarrow A`
edges.


.. index:: CvGraphScanner

.. _CvGraphScanner:

CvGraphScanner
--------------

`id=0.551304755988 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvGraphScanner>`__

.. ctype:: CvGraphScanner



Graph traversal state.




::


    
    typedef struct CvGraphScanner
    {
        CvGraphVtx* vtx;       /* current graph vertex (or current edge origin) */
        CvGraphVtx* dst;       /* current graph edge destination vertex */
        CvGraphEdge* edge;     /* current edge */
    
        CvGraph* graph;        /* the graph */
        CvSeq*   stack;        /* the graph vertex stack */
        int      index;        /* the lower bound of certainly visited vertices */
        int      mask;         /* event mask */
    }
    CvGraphScanner;
    
    

..

The structure 
:ref:`CvGraphScanner`
is used for depth-first graph traversal. See discussion of the functions below.

cvmacro
Helper macro for a tree node type declaration.

The macro 
``CV_TREE_NODE_FIELDS()``
is used to declare structures
that can be organized into hierarchical strucutures (trees), such as
:ref:`CvSeq`
- the basic type for all dynamic structures. The trees
created with nodes declared using this macro can be processed using the
functions described below in this section.


.. index:: CvTreeNodeIterator

.. _CvTreeNodeIterator:

CvTreeNodeIterator
------------------

`id=0.486956655882 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvTreeNodeIterator>`__

.. ctype:: CvTreeNodeIterator



Opens existing or creates new file storage.




::


    
    typedef struct CvTreeNodeIterator
    {
        const void* node;
        int level;
        int max_level;
    }
    CvTreeNodeIterator;
    

..




::


    
    #define CV_TREE_NODE_FIELDS(node_type)                          \
        int       flags;         /* micsellaneous flags */          \
        int       header_size;   /* size of sequence header */      \
        struct    node_type* h_prev; /* previous sequence */        \
        struct    node_type* h_next; /* next sequence */            \
        struct    node_type* v_prev; /* 2nd previous sequence */    \
        struct    node_type* v_next; /* 2nd next sequence */
    
    

..

The structure 
:ref:`CvTreeNodeIterator`
is used to traverse trees. Each tree node should start with the certain fields which are defined by 
``CV_TREE_NODE_FIELDS(...)``
macro. In C++ terms, each tree node should be a structure "derived" from




::


    
    struct _BaseTreeNode
    {
        CV_TREE_NODE_FIELDS(_BaseTreeNode);
    }
    

..

``CvSeq``
, 
``CvSet``
, 
``CvGraph``
and other dynamic structures derived from 
``CvSeq``
comply with the requirement.


.. index:: ClearGraph

.. _ClearGraph:

ClearGraph
----------

`id=0.332439919365 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ClearGraph>`__




.. cfunction:: void cvClearGraph( CvGraph* graph )

    Clears a graph.





    
    :param graph: Graph 
    
    
    
The function removes all vertices and edges from a graph. The function has O(1) time complexity.


.. index:: ClearMemStorage

.. _ClearMemStorage:

ClearMemStorage
---------------

`id=0.771544719824 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ClearMemStorage>`__




.. cfunction:: void cvClearMemStorage( CvMemStorage* storage )

    Clears memory storage.





    
    :param storage: Memory storage 
    
    
    
The function resets the top (free space
boundary) of the storage to the very beginning. This function does not
deallocate any memory. If the storage has a parent, the function returns
all blocks to the parent.


.. index:: ClearSeq

.. _ClearSeq:

ClearSeq
--------

`id=0.773624423506 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ClearSeq>`__




.. cfunction:: void cvClearSeq( CvSeq* seq )

    Clears a sequence.





    
    :param seq: Sequence 
    
    
    
The function removes all elements from a
sequence. The function does not return the memory to the storage block, but this
memory is reused later when new elements are added to the sequence. The function has
'O(1)' time complexity.



.. index:: ClearSet

.. _ClearSet:

ClearSet
--------

`id=0.561246622558 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ClearSet>`__




.. cfunction:: void cvClearSet( CvSet* setHeader )

    Clears a set.





    
    :param setHeader: Cleared set 
    
    
    
The function removes all elements from set. It has O(1) time complexity.



.. index:: CloneGraph

.. _CloneGraph:

CloneGraph
----------

`id=0.516560929963 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CloneGraph>`__




.. cfunction:: CvGraph* cvCloneGraph(  const CvGraph* graph, CvMemStorage* storage )

    Clones a graph.





    
    :param graph: The graph to copy 
    
    
    :param storage: Container for the copy 
    
    
    
The function creates a full copy of the specified graph. If the
graph vertices or edges have pointers to some external data, it can still be
shared between the copies. The vertex and edge indices in the new graph
may be different from the original because the function defragments
the vertex and edge sets.


.. index:: CloneSeq

.. _CloneSeq:

CloneSeq
--------

`id=0.219909371893 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CloneSeq>`__




.. cfunction:: CvSeq* cvCloneSeq(  const CvSeq* seq, CvMemStorage* storage=NULL )

    Creates a copy of a sequence.





    
    :param seq: Sequence 
    
    
    :param storage: The destination storage block to hold the new sequence header and the copied data, if any. If it is NULL, the function uses the storage block containing the input sequence. 
    
    
    
The function makes a complete copy of the input sequence and returns it.

The call



::


    
    cvCloneSeq( seq, storage )
    

..

is equivalent to




::


    
    cvSeqSlice( seq, CV_WHOLE_SEQ, storage, 1 )
    

..


.. index:: CreateChildMemStorage

.. _CreateChildMemStorage:

CreateChildMemStorage
---------------------

`id=0.901847234907 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateChildMemStorage>`__




.. cfunction:: CvMemStorage* cvCreateChildMemStorage(CvMemStorage* parent)

    Creates child memory storage.





    
    :param parent: Parent memory storage 
    
    
    
The function creates a child memory
storage that is similar to simple memory storage except for the
differences in the memory allocation/deallocation mechanism. When a
child storage needs a new block to add to the block list, it tries
to get this block from the parent. The first unoccupied parent block
available is taken and excluded from the parent block list. If no blocks
are available, the parent either allocates a block or borrows one from
its own parent, if any. In other words, the chain, or a more complex
structure, of memory storages where every storage is a child/parent of
another is possible. When a child storage is released or even cleared,
it returns all blocks to the parent. In other aspects, child storage
is the same as simple storage.

Child storage is useful in the following situation. Imagine
that the user needs to process dynamic data residing in a given storage area and
put the result back to that same storage area. With the simplest approach,
when temporary data is resided in the same storage area as the input and
output data, the storage area will look as follows after processing:

Dynamic data processing without using child storage



.. image:: ../pics/memstorage1.png



That is, garbage appears in the middle of the storage. However, if
one creates a child memory storage at the beginning of processing,
writes temporary data there, and releases the child storage at the end,
no garbage will appear in the source/destination storage:

Dynamic data processing using a child storage



.. image:: ../pics/memstorage2.png




.. index:: CreateGraph

.. _CreateGraph:

CreateGraph
-----------

`id=0.714927849129 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateGraph>`__




.. cfunction:: CvGraph* cvCreateGraph(  int graph_flags, int header_size, int vtx_size, int edge_size, CvMemStorage* storage )

    Creates an empty graph.





    
    :param graph_flags: Type of the created graph. Usually, it is either  ``CV_SEQ_KIND_GRAPH``  for generic unoriented graphs and ``CV_SEQ_KIND_GRAPH | CV_GRAPH_FLAG_ORIENTED``  for generic oriented graphs. 
    
    
    :param header_size: Graph header size; may not be less than  ``sizeof(CvGraph)`` 
    
    
    :param vtx_size: Graph vertex size; the custom vertex structure must start with  :ref:`CvGraphVtx`  (use  ``CV_GRAPH_VERTEX_FIELDS()`` ) 
    
    
    :param edge_size: Graph edge size; the custom edge structure must start with  :ref:`CvGraphEdge`  (use  ``CV_GRAPH_EDGE_FIELDS()`` ) 
    
    
    :param storage: The graph container 
    
    
    
The function creates an empty graph and returns a pointer to it.


.. index:: CreateGraphScanner

.. _CreateGraphScanner:

CreateGraphScanner
------------------

`id=0.761147235713 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateGraphScanner>`__




.. cfunction:: CvGraphScanner*  cvCreateGraphScanner(  CvGraph* graph, CvGraphVtx* vtx=NULL, int mask=CV_GRAPH_ALL_ITEMS )

    Creates structure for depth-first graph traversal.





    
    :param graph: Graph 
    
    
    :param vtx: Initial vertex to start from. If NULL, the traversal starts from the first vertex (a vertex with the minimal index in the sequence of vertices). 
    
    
    :param mask: Event mask indicating which events are of interest to the user (where  :ref:`NextGraphItem`  function returns control to the user) It can be  ``CV_GRAPH_ALL_ITEMS``  (all events are of interest) or a combination of the following flags:
        
         
            * **CV_GRAPH_VERTEX** stop at the graph vertices visited for the first time 
            
            * **CV_GRAPH_TREE_EDGE** stop at tree edges ( ``tree edge``  is the edge connecting the last visited vertex and the vertex to be visited next) 
            
            * **CV_GRAPH_BACK_EDGE** stop at back edges ( ``back edge``  is an edge connecting the last visited vertex with some of its ancestors in the search tree) 
            
            * **CV_GRAPH_FORWARD_EDGE** stop at forward edges ( ``forward edge``  is an edge conecting the last visited vertex with some of its descendants in the search tree. The forward edges are only possible during oriented graph traversal) 
            
            * **CV_GRAPH_CROSS_EDGE** stop at cross edges ( ``cross edge``  is an edge connecting different search trees or branches of the same tree. The  ``cross edges``  are only possible during oriented graph traversal) 
            
            * **CV_GRAPH_ANY_EDGE** stop at any edge ( ``tree, back, forward`` , and  ``cross edges`` ) 
            
            * **CV_GRAPH_NEW_TREE** stop in the beginning of every new search tree. When the traversal procedure visits all vertices and edges reachable from the initial vertex (the visited vertices together with tree edges make up a tree), it searches for some unvisited vertex in the graph and resumes the traversal process from that vertex. Before starting a new tree (including the very first tree when  ``cvNextGraphItem``  is called for the first time) it generates a  ``CV_GRAPH_NEW_TREE``  event. For unoriented graphs, each search tree corresponds to a connected component of the graph. 
            
            * **CV_GRAPH_BACKTRACKING** stop at every already visited vertex during backtracking - returning to already visited vertexes of the traversal tree. 
            
            
    
    
    
The function creates a structure for depth-first graph traversal/search. The initialized structure is used in the 
:ref:`NextGraphItem`
function - the incremental traversal procedure.


.. index:: CreateMemStorage

.. _CreateMemStorage:

CreateMemStorage
----------------

`id=0.484842854055 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateMemStorage>`__




.. cfunction:: CvMemStorage* cvCreateMemStorage( int blockSize=0 )

    Creates memory storage.





    
    :param blockSize: Size of the storage blocks in bytes. If it is 0, the block size is set to a default value - currently it is  about 64K. 
    
    
    
The function creates an empty memory storage. See 
:ref:`CvMemStorage`
description.


.. index:: CreateSeq

.. _CreateSeq:

CreateSeq
---------

`id=0.879299981261 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateSeq>`__




.. cfunction:: CvSeq* cvCreateSeq(  int seqFlags, int headerSize, int elemSize, CvMemStorage* storage)

    Creates a sequence.





    
    :param seqFlags: Flags of the created sequence. If the sequence is not passed to any function working with a specific type of sequences, the sequence value may be set to 0, otherwise the appropriate type must be selected from the list of predefined sequence types. 
    
    
    :param headerSize: Size of the sequence header; must be greater than or equal to  ``sizeof(CvSeq)`` . If a specific type or its extension is indicated, this type must fit the base type header. 
    
    
    :param elemSize: Size of the sequence elements in bytes. The size must be consistent with the sequence type. For example, for a sequence of points to be created, the element type    ``CV_SEQ_ELTYPE_POINT``  should be specified and the parameter  ``elemSize``  must be equal to  ``sizeof(CvPoint)`` . 
    
    
    :param storage: Sequence location 
    
    
    
The function creates a sequence and returns
the pointer to it. The function allocates the sequence header in
the storage block as one continuous chunk and sets the structure
fields 
``flags``
, 
``elemSize``
, 
``headerSize``
, and
``storage``
to passed values, sets 
``delta_elems``
to the
default value (that may be reassigned using the 
:ref:`SetSeqBlockSize`
function), and clears other header fields, including the space following
the first 
``sizeof(CvSeq)``
bytes.


.. index:: CreateSet

.. _CreateSet:

CreateSet
---------

`id=0.149633794529 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CreateSet>`__




.. cfunction:: CvSet* cvCreateSet(  int set_flags, int header_size, int elem_size, CvMemStorage* storage )

    Creates an empty set.





    
    :param set_flags: Type of the created set 
    
    
    :param header_size: Set header size; may not be less than  ``sizeof(CvSet)`` 
    
    
    :param elem_size: Set element size; may not be less than  :ref:`CvSetElem` 
    
    
    :param storage: Container for the set 
    
    
    
The function creates an empty set with a specified header size and element size, and returns the pointer to the set. This function is just a thin layer on top of 
:ref:`CreateSeq`
.


.. index:: CvtSeqToArray

.. _CvtSeqToArray:

CvtSeqToArray
-------------

`id=0.900164505728 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvtSeqToArray>`__




.. cfunction:: void* cvCvtSeqToArray(  const CvSeq* seq, void* elements, CvSlice slice=CV_WHOLE_SEQ )

    Copies a sequence to one continuous block of memory.





    
    :param seq: Sequence 
    
    
    :param elements: Pointer to the destination array that must be large enough. It should be a pointer to data, not a matrix header. 
    
    
    :param slice: The sequence portion to copy to the array 
    
    
    
The function copies the entire sequence or subsequence to the specified buffer and returns the pointer to the buffer.


.. index:: EndWriteSeq

.. _EndWriteSeq:

EndWriteSeq
-----------

`id=0.919895703214 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/EndWriteSeq>`__




.. cfunction:: CvSeq* cvEndWriteSeq( CvSeqWriter* writer )

    Finishes the process of writing a sequence.





    
    :param writer: Writer state 
    
    
    
The function finishes the writing process and
returns the pointer to the written sequence. The function also truncates
the last incomplete sequence block to return the remaining part of the
block to memory storage. After that, the sequence can be read and
modified safely. See 
:ref:`cvStartWriteSeq`
and 
:ref:`cvStartAppendToSeq`

.. index:: FindGraphEdge

.. _FindGraphEdge:

FindGraphEdge
-------------

`id=0.18087190834 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/FindGraphEdge>`__




.. cfunction:: CvGraphEdge* cvFindGraphEdge( const CvGraph* graph, int start_idx, int end_idx )

    Finds an edge in a graph.






::


    
    
    #define cvGraphFindEdge cvFindGraphEdge
    
    

..



    
    :param graph: Graph 
    
    
    :param start_idx: Index of the starting vertex of the edge 
    
    
    :param end_idx: Index of the ending vertex of the edge. For an unoriented graph, the order of the vertex parameters does not matter. 
    
    
    
The function finds the graph edge connecting two specified vertices and returns a pointer to it or NULL if the edge does not exist.


.. index:: FindGraphEdgeByPtr

.. _FindGraphEdgeByPtr:

FindGraphEdgeByPtr
------------------

`id=0.509139476588 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/FindGraphEdgeByPtr>`__




.. cfunction:: CvGraphEdge* cvFindGraphEdgeByPtr(  const CvGraph* graph, const CvGraphVtx* startVtx, const CvGraphVtx* endVtx )

    Finds an edge in a graph by using its pointer.






::


    
    #define cvGraphFindEdgeByPtr cvFindGraphEdgeByPtr
    

..



    
    :param graph: Graph 
    
    
    :param startVtx: Pointer to the starting vertex of the edge 
    
    
    :param endVtx: Pointer to the ending vertex of the edge. For an unoriented graph, the order of the vertex parameters does not matter. 
    
    
    
The function finds the graph edge connecting two specified vertices and returns pointer to it or NULL if the edge does not exists.


.. index:: FlushSeqWriter

.. _FlushSeqWriter:

FlushSeqWriter
--------------

`id=0.821406812895 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/FlushSeqWriter>`__




.. cfunction:: void cvFlushSeqWriter( CvSeqWriter* writer )

    Updates sequence headers from the writer.





    
    :param writer: Writer state 
    
    
    
The function is intended to enable the user to
read sequence elements, whenever required, during the writing process,
e.g., in order to check specific conditions. The function updates the
sequence headers to make reading from the sequence possible. The writer
is not closed, however, so that the writing process can be continued at
any time. If an algorithm requires frequent flushes, consider using
:ref:`SeqPush`
instead.


.. index:: GetGraphVtx

.. _GetGraphVtx:

GetGraphVtx
-----------

`id=0.802641800298 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetGraphVtx>`__




.. cfunction:: CvGraphVtx* cvGetGraphVtx(  CvGraph* graph, int vtx_idx )

    Finds a graph vertex by using its index.





    
    :param graph: Graph 
    
    
    :param vtx_idx: Index of the vertex 
    
    
    
The function finds the graph vertex by using its index and returns the pointer to it or NULL if the vertex does not belong to the graph.



.. index:: GetSeqElem

.. _GetSeqElem:

GetSeqElem
----------

`id=0.778073099468 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetSeqElem>`__




.. cfunction:: char* cvGetSeqElem( const CvSeq* seq, int index )

    Returns a pointer to a sequence element according to its index.






::


    
    #define CV_GET_SEQ_ELEM( TYPE, seq, index )  (TYPE*)cvGetSeqElem( (CvSeq*)(seq), (index) )
    

..



    
    :param seq: Sequence 
    
    
    :param index: Index of element 
    
    
    
The function finds the element with the given
index in the sequence and returns the pointer to it. If the element
is not found, the function returns 0. The function supports negative
indices, where -1 stands for the last sequence element, -2 stands for
the one before last, etc. If the sequence is most likely to consist of
a single sequence block or the desired element is likely to be located
in the first block, then the macro
``CV_GET_SEQ_ELEM( elemType, seq, index )``
should be used, where the parameter 
``elemType``
is the
type of sequence elements ( 
:ref:`CvPoint`
for example), the parameter
``seq``
is a sequence, and the parameter 
``index``
is the index
of the desired element. The macro checks first whether the desired element
belongs to the first block of the sequence and returns it if it does;
otherwise the macro calls the main function 
``GetSeqElem``
. Negative
indices always cause the 
:ref:`GetSeqElem`
call. The function has O(1)
time complexity assuming that the number of blocks is much smaller than the
number of elements.


.. index:: GetSeqReaderPos

.. _GetSeqReaderPos:

GetSeqReaderPos
---------------

`id=0.869101167847 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetSeqReaderPos>`__




.. cfunction:: int cvGetSeqReaderPos( CvSeqReader* reader )

    Returns the current reader position.





    
    :param reader: Reader state 
    
    
    
The function returns the current reader position (within 0 ... 
``reader->seq->total``
- 1).


.. index:: GetSetElem

.. _GetSetElem:

GetSetElem
----------

`id=0.506490712171 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GetSetElem>`__




.. cfunction:: CvSetElem* cvGetSetElem(  const CvSet* setHeader, int index )

    Finds a set element by its index.





    
    :param setHeader: Set 
    
    
    :param index: Index of the set element within a sequence 
    
    
    
The function finds a set element by its index. The function returns the pointer to it or 0 if the index is invalid or the corresponding node is free. The function supports negative indices as it uses 
:ref:`GetSeqElem`
to locate the node.


.. index:: GraphAddEdge

.. _GraphAddEdge:

GraphAddEdge
------------

`id=0.752253770377 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GraphAddEdge>`__




.. cfunction:: int cvGraphAddEdge(  CvGraph* graph, int start_idx, int end_idx, const CvGraphEdge* edge=NULL, CvGraphEdge** inserted_edge=NULL )

    Adds an edge to a graph.





    
    :param graph: Graph 
    
    
    :param start_idx: Index of the starting vertex of the edge 
    
    
    :param end_idx: Index of the ending vertex of the edge. For an unoriented graph, the order of the vertex parameters does not matter. 
    
    
    :param edge: Optional input parameter, initialization data for the edge 
    
    
    :param inserted_edge: Optional output parameter to contain the address of the inserted edge 
    
    
    
The function connects two specified vertices. The function returns 1 if the edge has been added successfully, 0 if the edge connecting the two vertices exists already and -1 if either of the vertices was not found, the starting and the ending vertex are the same, or there is some other critical situation. In the latter case (i.e., when the result is negative), the function also reports an error by default.


.. index:: GraphAddEdgeByPtr

.. _GraphAddEdgeByPtr:

GraphAddEdgeByPtr
-----------------

`id=0.313903446977 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GraphAddEdgeByPtr>`__




.. cfunction:: int cvGraphAddEdgeByPtr(  CvGraph* graph, CvGraphVtx* start_vtx, CvGraphVtx* end_vtx, const CvGraphEdge* edge=NULL, CvGraphEdge** inserted_edge=NULL )

    Adds an edge to a graph by using its pointer.





    
    :param graph: Graph 
    
    
    :param start_vtx: Pointer to the starting vertex of the edge 
    
    
    :param end_vtx: Pointer to the ending vertex of the edge. For an unoriented graph, the order of the vertex parameters does not matter. 
    
    
    :param edge: Optional input parameter, initialization data for the edge 
    
    
    :param inserted_edge: Optional output parameter to contain the address of the inserted edge within the edge set 
    
    
    
The function connects two specified vertices. The
function returns 1 if the edge has been added successfully, 0 if the
edge connecting the two vertices exists already, and -1 if either of the
vertices was not found, the starting and the ending vertex are the same
or there is some other critical situation. In the latter case (i.e., when
the result is negative), the function also reports an error by default.


.. index:: GraphAddVtx

.. _GraphAddVtx:

GraphAddVtx
-----------

`id=0.236553727886 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GraphAddVtx>`__




.. cfunction:: int cvGraphAddVtx(  CvGraph* graph, const CvGraphVtx* vtx=NULL, CvGraphVtx** inserted_vtx=NULL )

    Adds a vertex to a graph.





    
    :param graph: Graph 
    
    
    :param vtx: Optional input argument used to initialize the added vertex (only user-defined fields beyond  ``sizeof(CvGraphVtx)``  are copied) 
    
    
    :param inserted_vertex: Optional output argument. If not  ``NULL`` , the address of the new vertex is written here. 
    
    
    
The function adds a vertex to the graph and returns the vertex index.


.. index:: GraphEdgeIdx

.. _GraphEdgeIdx:

GraphEdgeIdx
------------

`id=0.571043881578 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GraphEdgeIdx>`__




.. cfunction:: int cvGraphEdgeIdx(  CvGraph* graph, CvGraphEdge* edge )

    Returns the index of a graph edge.





    
    :param graph: Graph 
    
    
    :param edge: Pointer to the graph edge 
    
    
    
The function returns the index of a graph edge.


.. index:: GraphRemoveEdge

.. _GraphRemoveEdge:

GraphRemoveEdge
---------------

`id=0.608632884153 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GraphRemoveEdge>`__




.. cfunction:: void cvGraphRemoveEdge(  CvGraph* graph, int start_idx, int end_idx )

    Removes an edge from a graph.





    
    :param graph: Graph 
    
    
    :param start_idx: Index of the starting vertex of the edge 
    
    
    :param end_idx: Index of the ending vertex of the edge. For an unoriented graph, the order of the vertex parameters does not matter. 
    
    
    
The function removes the edge connecting two specified vertices. If the vertices are not connected [in that order], the function does nothing.


.. index:: GraphRemoveEdgeByPtr

.. _GraphRemoveEdgeByPtr:

GraphRemoveEdgeByPtr
--------------------

`id=0.642579664169 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GraphRemoveEdgeByPtr>`__




.. cfunction:: void cvGraphRemoveEdgeByPtr(  CvGraph* graph, CvGraphVtx* start_vtx, CvGraphVtx* end_vtx )

    Removes an edge from a graph by using its pointer.





    
    :param graph: Graph 
    
    
    :param start_vtx: Pointer to the starting vertex of the edge 
    
    
    :param end_vtx: Pointer to the ending vertex of the edge. For an unoriented graph, the order of the vertex parameters does not matter. 
    
    
    
The function removes the edge connecting two specified vertices. If the vertices are not connected [in that order], the function does nothing.


.. index:: GraphRemoveVtx

.. _GraphRemoveVtx:

GraphRemoveVtx
--------------

`id=0.970005049786 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GraphRemoveVtx>`__




.. cfunction:: int cvGraphRemoveVtx(  CvGraph* graph, int index )

    Removes a vertex from a graph.





    
    :param graph: Graph 
    
    
    :param vtx_idx: Index of the removed vertex 
    
    
    
The function removes a vertex from a graph
together with all the edges incident to it. The function reports an error
if the input vertex does not belong to the graph. The return value is the
number of edges deleted, or -1 if the vertex does not belong to the graph.


.. index:: GraphRemoveVtxByPtr

.. _GraphRemoveVtxByPtr:

GraphRemoveVtxByPtr
-------------------

`id=0.605088135179 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GraphRemoveVtxByPtr>`__




.. cfunction:: int cvGraphRemoveVtxByPtr(  CvGraph* graph, CvGraphVtx* vtx )

    Removes a vertex from a graph by using its pointer.





    
    :param graph: Graph 
    
    
    :param vtx: Pointer to the removed vertex 
    
    
    
The function removes a vertex from the graph by using its pointer together with all the edges incident to it. The function reports an error if the vertex does not belong to the graph. The return value is the number of edges deleted, or -1 if the vertex does not belong to the graph.


.. index:: GraphVtxDegree

.. _GraphVtxDegree:

GraphVtxDegree
--------------

`id=0.257037043726 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GraphVtxDegree>`__




.. cfunction:: int cvGraphVtxDegree( const CvGraph* graph, int vtxIdx )

    Counts the number of edges indicent to the vertex.





    
    :param graph: Graph 
    
    
    :param vtxIdx: Index of the graph vertex 
    
    
    
The function returns the number of edges incident to the specified vertex, both incoming and outgoing. To count the edges, the following code is used:




::


    
    CvGraphEdge* edge = vertex->first; int count = 0;
    while( edge )
    {
        edge = CV_NEXT_GRAPH_EDGE( edge, vertex );
        count++;
    }
    

..

The macro 
``CV_NEXT_GRAPH_EDGE( edge, vertex )``
returns the edge incident to 
``vertex``
that follows after 
``edge``
.


.. index:: GraphVtxDegreeByPtr

.. _GraphVtxDegreeByPtr:

GraphVtxDegreeByPtr
-------------------

`id=0.739296929217 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GraphVtxDegreeByPtr>`__




.. cfunction:: int cvGraphVtxDegreeByPtr(  const CvGraph* graph, const CvGraphVtx* vtx )

    Finds an edge in a graph.





    
    :param graph: Graph 
    
    
    :param vtx: Pointer to the graph vertex 
    
    
    
The function returns the number of edges incident to the specified vertex, both incoming and outcoming.



.. index:: GraphVtxIdx

.. _GraphVtxIdx:

GraphVtxIdx
-----------

`id=0.717221417419 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/GraphVtxIdx>`__




.. cfunction:: int cvGraphVtxIdx(  CvGraph* graph, CvGraphVtx* vtx )

    Returns the index of a graph vertex.





    
    :param graph: Graph 
    
    
    :param vtx: Pointer to the graph vertex 
    
    
    
The function returns the index of a graph vertex.


.. index:: InitTreeNodeIterator

.. _InitTreeNodeIterator:

InitTreeNodeIterator
--------------------

`id=0.483111798793 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/InitTreeNodeIterator>`__




.. cfunction:: void cvInitTreeNodeIterator(  CvTreeNodeIterator* tree_iterator, const void* first, int max_level )

    Initializes the tree node iterator.





    
    :param tree_iterator: Tree iterator initialized by the function 
    
    
    :param first: The initial node to start traversing from 
    
    
    :param max_level: The maximal level of the tree ( ``first``  node assumed to be at the first level) to traverse up to. For example, 1 means that only nodes at the same level as  ``first``  should be visited, 2 means that the nodes on the same level as  ``first``  and their direct children should be visited, and so forth. 
    
    
    
The function initializes the tree iterator. The tree is traversed in depth-first order.


.. index:: InsertNodeIntoTree

.. _InsertNodeIntoTree:

InsertNodeIntoTree
------------------

`id=0.159347112834 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/InsertNodeIntoTree>`__




.. cfunction:: void cvInsertNodeIntoTree(  void* node, void* parent, void* frame )

    Adds a new node to a tree.





    
    :param node: The inserted node 
    
    
    :param parent: The parent node that is already in the tree 
    
    
    :param frame: The top level node. If  ``parent``  and  ``frame``  are the same, the  ``v_prev``  field of  ``node``  is set to NULL rather than  ``parent`` . 
    
    
    
The function adds another node into tree. The function does not allocate any memory, it can only modify links of the tree nodes.


.. index:: MakeSeqHeaderForArray

.. _MakeSeqHeaderForArray:

MakeSeqHeaderForArray
---------------------

`id=0.960790357917 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/MakeSeqHeaderForArray>`__




.. cfunction:: CvSeq* cvMakeSeqHeaderForArray(  int seq_type, int header_size, int elem_size, void* elements, int total, CvSeq* seq, CvSeqBlock* block )

    Constructs a sequence header for an array.





    
    :param seq_type: Type of the created sequence 
    
    
    :param header_size: Size of the header of the sequence. Parameter sequence must point to the structure of that size or greater 
    
    
    :param elem_size: Size of the sequence elements 
    
    
    :param elements: Elements that will form a sequence 
    
    
    :param total: Total number of elements in the sequence. The number of array elements must be equal to the value of this parameter. 
    
    
    :param seq: Pointer to the local variable that is used as the sequence header 
    
    
    :param block: Pointer to the local variable that is the header of the single sequence block 
    
    
    
The function initializes a sequence
header for an array. The sequence header as well as the sequence block are
allocated by the user (for example, on stack). No data is copied by the
function. The resultant sequence will consists of a single block and
have NULL storage pointer; thus, it is possible to read its elements,
but the attempts to add elements to the sequence will raise an error in
most cases.


.. index:: MemStorageAlloc

.. _MemStorageAlloc:

MemStorageAlloc
---------------

`id=0.301172131439 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/MemStorageAlloc>`__




.. cfunction:: void* cvMemStorageAlloc(  CvMemStorage* storage, size_t size )

    Allocates a memory buffer in a storage block.





    
    :param storage: Memory storage 
    
    
    :param size: Buffer size 
    
    
    
The function allocates a memory buffer in
a storage block. The buffer size must not exceed the storage block size,
otherwise a runtime error is raised. The buffer address is aligned by
``CV_STRUCT_ALIGN=sizeof(double)``
(for the moment) bytes.


.. index:: MemStorageAllocString

.. _MemStorageAllocString:

MemStorageAllocString
---------------------

`id=0.109838084699 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/MemStorageAllocString>`__




.. cfunction:: CvString cvMemStorageAllocString(CvMemStorage* storage, const char* ptr, int len=-1)

    Allocates a text string in a storage block.






::


    
    typedef struct CvString
    {
        int len;
        char* ptr;
    }
    CvString;
    

..



    
    :param storage: Memory storage 
    
    
    :param ptr: The string 
    
    
    :param len: Length of the string (not counting the ending  ``NUL`` ) . If the parameter is negative, the function computes the length. 
    
    
    
The function creates copy of the string
in memory storage. It returns the structure that contains user-passed
or computed length of the string and pointer to the copied string.


.. index:: NextGraphItem

.. _NextGraphItem:

NextGraphItem
-------------

`id=0.801658747963 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/NextGraphItem>`__




.. cfunction:: int cvNextGraphItem( CvGraphScanner* scanner )

    Executes one or more steps of the graph traversal procedure.





    
    :param scanner: Graph traversal state. It is updated by this function. 
    
    
    
The function traverses through the graph
until an event of interest to the user (that is, an event, specified
in the 
``mask``
in the 
:ref:`CreateGraphScanner`
call) is met or the
traversal is completed. In the first case, it returns one of the events
listed in the description of the 
``mask``
parameter above and with
the next call it resumes the traversal. In the latter case, it returns
``CV_GRAPH_OVER``
(-1). When the event is 
``CV_GRAPH_VERTEX``
,
``CV_GRAPH_BACKTRACKING``
, or 
``CV_GRAPH_NEW_TREE``
,
the currently observed vertex is stored in 
``scanner-:math:`>`vtx``
. And if the
event is edge-related, the edge itself is stored at 
``scanner-:math:`>`edge``
,
the previously visited vertex - at 
``scanner-:math:`>`vtx``
and the other ending
vertex of the edge - at 
``scanner-:math:`>`dst``
.


.. index:: NextTreeNode

.. _NextTreeNode:

NextTreeNode
------------

`id=0.892783495145 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/NextTreeNode>`__




.. cfunction:: void* cvNextTreeNode( CvTreeNodeIterator* tree_iterator )

    Returns the currently observed node and moves the iterator toward the next node.





    
    :param tree_iterator: Tree iterator initialized by the function 
    
    
    
The function returns the currently observed node and then updates the
iterator - moving it toward the next node. In other words, the function
behavior is similar to the 
``*p++``
expression on a typical C
pointer or C++ collection iterator. The function returns NULL if there
are no more nodes.



.. index:: PrevTreeNode

.. _PrevTreeNode:

PrevTreeNode
------------

`id=0.199395520003 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/PrevTreeNode>`__




.. cfunction:: void* cvPrevTreeNode( CvTreeNodeIterator* tree_iterator )

    Returns the currently observed node and moves the iterator toward the previous node.





    
    :param tree_iterator: Tree iterator initialized by the function 
    
    
    
The function returns the currently observed node and then updates
the iterator - moving it toward the previous node. In other words,
the function behavior is similar to the 
``*p--``
expression on a
typical C pointer or C++ collection iterator. The function returns NULL
if there are no more nodes.



.. index:: ReleaseGraphScanner

.. _ReleaseGraphScanner:

ReleaseGraphScanner
-------------------

`id=0.572499008135 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ReleaseGraphScanner>`__




.. cfunction:: void cvReleaseGraphScanner( CvGraphScanner** scanner )

    Completes the graph traversal procedure.





    
    :param scanner: Double pointer to graph traverser 
    
    
    
The function completes the graph traversal procedure and releases the traverser state.




.. index:: ReleaseMemStorage

.. _ReleaseMemStorage:

ReleaseMemStorage
-----------------

`id=0.449342726691 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/ReleaseMemStorage>`__




.. cfunction:: void cvReleaseMemStorage( CvMemStorage** storage )

    Releases memory storage.





    
    :param storage: Pointer to the released storage 
    
    
    
The function deallocates all storage memory
blocks or returns them to the parent, if any. Then it deallocates the
storage header and clears the pointer to the storage. All child storage 
associated with a given parent storage block must be released before the 
parent storage block is released.


.. index:: RestoreMemStoragePos

.. _RestoreMemStoragePos:

RestoreMemStoragePos
--------------------

`id=0.0596222862557 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/RestoreMemStoragePos>`__




.. cfunction:: void cvRestoreMemStoragePos( CvMemStorage* storage, CvMemStoragePos* pos)

    Restores memory storage position.





    
    :param storage: Memory storage 
    
    
    :param pos: New storage top position 
    
    
    
The function restores the position of the storage top from the parameter 
``pos``
. This function and the function 
``cvClearMemStorage``
are the only methods to release memory occupied in memory blocks. Note again that there is no way to free memory in the middle of an occupied portion of a storage block.



.. index:: SaveMemStoragePos

.. _SaveMemStoragePos:

SaveMemStoragePos
-----------------

`id=0.625300615076 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SaveMemStoragePos>`__




.. cfunction:: void cvSaveMemStoragePos( const CvMemStorage* storage, CvMemStoragePos* pos)

    Saves memory storage position.





    
    :param storage: Memory storage 
    
    
    :param pos: The output position of the storage top 
    
    
    
The function saves the current position
of the storage top to the parameter 
``pos``
. The function
``cvRestoreMemStoragePos``
can further retrieve this position.


.. index:: SeqElemIdx

.. _SeqElemIdx:

SeqElemIdx
----------

`id=0.724143019934 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqElemIdx>`__




.. cfunction:: int cvSeqElemIdx(  const CvSeq* seq, const void* element, CvSeqBlock** block=NULL )

    Returns the index of a specific sequence element.





    
    :param seq: Sequence 
    
    
    :param element: Pointer to the element within the sequence 
    
    
    :param block: Optional argument. If the pointer is not  ``NULL`` , the address of the sequence block that contains the element is stored in this location. 
    
    
    
The function returns the index of a sequence element or a negative number if the element is not found.


.. index:: SeqInsert

.. _SeqInsert:

SeqInsert
---------

`id=0.0992440051218 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqInsert>`__




.. cfunction:: char* cvSeqInsert(  CvSeq* seq, int beforeIndex, void* element=NULL )

    Inserts an element in the middle of a sequence.





    
    :param seq: Sequence 
    
    
    :param beforeIndex: Index before which the element is inserted. Inserting before 0 (the minimal allowed value of the parameter) is equal to  :ref:`SeqPushFront`  and inserting before  ``seq->total``  (the maximal allowed value of the parameter) is equal to  :ref:`SeqPush` . 
    
    
    :param element: Inserted element 
    
    
    
The function shifts the sequence elements from the inserted position to the nearest end of the sequence and copies the 
``element``
content there if the pointer is not NULL. The function returns a pointer to the inserted element.



.. index:: SeqInsertSlice

.. _SeqInsertSlice:

SeqInsertSlice
--------------

`id=0.819564817378 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqInsertSlice>`__




.. cfunction:: void cvSeqInsertSlice(  CvSeq* seq, int beforeIndex, const CvArr* fromArr )

    Inserts an array in the middle of a sequence.





    
    :param seq: Sequence 
    
    
    :param slice: The part of the sequence to remove 
    
    
    :param fromArr: The array to take elements from 
    
    
    
The function inserts all 
``fromArr``
array elements at the specified position of the sequence. The array
``fromArr``
can be a matrix or another sequence.


.. index:: SeqInvert

.. _SeqInvert:

SeqInvert
---------

`id=0.695189452157 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqInvert>`__




.. cfunction:: void cvSeqInvert( CvSeq* seq )

    Reverses the order of sequence elements.





    
    :param seq: Sequence 
    
    
    
The function reverses the sequence in-place - makes the first element go last, the last element go first and so forth.


.. index:: SeqPop

.. _SeqPop:

SeqPop
------

`id=0.891792572997 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqPop>`__




.. cfunction:: void cvSeqPop(  CvSeq* seq, void* element=NULL )

    Removes an element from the end of a sequence.





    
    :param seq: Sequence 
    
    
    :param element: Optional parameter . If the pointer is not zero, the function copies the removed element to this location. 
    
    
    
The function removes an element from a sequence. The function reports an error if the sequence is already empty. The function has O(1) complexity.


.. index:: SeqPopFront

.. _SeqPopFront:

SeqPopFront
-----------

`id=0.802844810483 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqPopFront>`__




.. cfunction:: void cvSeqPopFront(   CvSeq* seq, void* element=NULL )

    Removes an element from the beginning of a sequence.





    
    :param seq: Sequence 
    
    
    :param element: Optional parameter. If the pointer is not zero, the function copies the removed element to this location. 
    
    
    
The function removes an element from the beginning of a sequence. The function reports an error if the sequence is already empty. The function has O(1) complexity.


.. index:: SeqPopMulti

.. _SeqPopMulti:

SeqPopMulti
-----------

`id=0.260750127544 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqPopMulti>`__




.. cfunction:: void cvSeqPopMulti(  CvSeq* seq, void* elements, int count, int in_front=0 )

    Removes several elements from either end of a sequence.





    
    :param seq: Sequence 
    
    
    :param elements: Removed elements 
    
    
    :param count: Number of elements to pop 
    
    
    :param in_front: The flags specifying which end of the modified sequence. 
         
            * **CV_BACK** the elements are added to the end of the sequence 
            
            * **CV_FRONT** the elements are added to the beginning of the sequence 
            
            
    
    
    
The function removes several elements from either end of the sequence. If the number of the elements to be removed exceeds the total number of elements in the sequence, the function removes as many elements as possible.


.. index:: SeqPush

.. _SeqPush:

SeqPush
-------

`id=0.90060051534 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqPush>`__




.. cfunction:: char* cvSeqPush(  CvSeq* seq, void* element=NULL )

    Adds an element to the end of a sequence.





    
    :param seq: Sequence 
    
    
    :param element: Added element 
    
    
    
The function adds an element to the end of a sequence and returns a pointer to the allocated element. If the input 
``element``
is NULL, the function simply allocates a space for one more element.

The following code demonstrates how to create a new sequence using this function:




::


    
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* seq = cvCreateSeq( CV_32SC1, /* sequence of integer elements */
                              sizeof(CvSeq), /* header size - no extra fields */
                              sizeof(int), /* element size */
                              storage /* the container storage */ );
    int i;
    for( i = 0; i < 100; i++ )
    {
        int* added = (int*)cvSeqPush( seq, &i );
        printf( "
    }
    
    ...
    /* release memory storage in the end */
    cvReleaseMemStorage( &storage );
    

..

The function has O(1) complexity, but there is a faster method for writing large sequences (see 
:ref:`StartWriteSeq`
and related functions).



.. index:: SeqPushFront

.. _SeqPushFront:

SeqPushFront
------------

`id=0.862751238482 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqPushFront>`__




.. cfunction:: char* cvSeqPushFront( CvSeq* seq, void* element=NULL )

    Adds an element to the beginning of a sequence.





    
    :param seq: Sequence 
    
    
    :param element: Added element 
    
    
    
The function is similar to 
:ref:`SeqPush`
but it adds the new element to the beginning of the sequence. The function has O(1) complexity.


.. index:: SeqPushMulti

.. _SeqPushMulti:

SeqPushMulti
------------

`id=0.958302949543 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqPushMulti>`__




.. cfunction:: void cvSeqPushMulti(  CvSeq* seq, void* elements, int count, int in_front=0 )

    Pushes several elements to either end of a sequence.





    
    :param seq: Sequence 
    
    
    :param elements: Added elements 
    
    
    :param count: Number of elements to push 
    
    
    :param in_front: The flags specifying which end of the modified sequence. 
         
            * **CV_BACK** the elements are added to the end of the sequence 
            
            * **CV_FRONT** the elements are added to the beginning of the sequence 
            
            
    
    
    
The function adds several elements to either
end of a sequence. The elements are added to the sequence in the same
order as they are arranged in the input array but they can fall into
different sequence blocks.


.. index:: SeqRemove

.. _SeqRemove:

SeqRemove
---------

`id=0.432719803682 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqRemove>`__




.. cfunction:: void cvSeqRemove(  CvSeq* seq, int index )

    Removes an element from the middle of a sequence.





    
    :param seq: Sequence 
    
    
    :param index: Index of removed element 
    
    
    
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

`id=0.971861630547 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqRemoveSlice>`__




.. cfunction:: void cvSeqRemoveSlice( CvSeq* seq, CvSlice slice )

    Removes a sequence slice.





    
    :param seq: Sequence 
    
    
    :param slice: The part of the sequence to remove 
    
    
    
The function removes a slice from the sequence.


.. index:: SeqSearch

.. _SeqSearch:

SeqSearch
---------

`id=0.729745795436 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqSearch>`__




.. cfunction:: char* cvSeqSearch( CvSeq* seq, const void* elem, CvCmpFunc func,                    int is_sorted, int* elem_idx, void* userdata=NULL )

    Searches for an element in a sequence.





    
    :param seq: The sequence 
    
    
    :param elem: The element to look for 
    
    
    :param func: The comparison function that returns negative, zero or positive value depending on the relationships among the elements (see also  :ref:`SeqSort` ) 
    
    
    :param is_sorted: Whether the sequence is sorted or not 
    
    
    :param elem_idx: Output parameter; index of the found element 
    
    
    :param userdata: The user parameter passed to the compasion function; helps to avoid global variables in some cases 
    
    
    



::


    
    /* a < b ? -1 : a > b ? 1 : 0 */
    typedef int (CV_CDECL* CvCmpFunc)(const void* a, const void* b, void* userdata);
    

..

The function searches for the element in the sequence. If
the sequence is sorted, a binary O(log(N)) search is used; otherwise, a
simple linear search is used. If the element is not found, the function
returns a NULL pointer and the index is set to the number of sequence
elements if a linear search is used, or to the smallest index
``i, seq(i)>elem``
.


.. index:: SeqSlice

.. _SeqSlice:

SeqSlice
--------

`id=0.0557062585643 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqSlice>`__




.. cfunction:: CvSeq* cvSeqSlice(  const CvSeq* seq, CvSlice slice, CvMemStorage* storage=NULL, int copy_data=0 )

    Makes a separate header for a sequence slice.





    
    :param seq: Sequence 
    
    
    :param slice: The part of the sequence to be extracted 
    
    
    :param storage: The destination storage block to hold the new sequence header and the copied data, if any. If it is NULL, the function uses the storage block containing the input sequence. 
    
    
    :param copy_data: The flag that indicates whether to copy the elements of the extracted slice ( ``copy_data!=0`` ) or not ( ``copy_data=0`` ) 
    
    
    
The function creates a sequence that represents the specified slice of the input sequence. The new sequence either shares the elements with the original sequence or has its own copy of the elements. So if one needs to process a part of sequence but the processing function does not have a slice parameter, the required sub-sequence may be extracted using this function.


.. index:: SeqSort

.. _SeqSort:

SeqSort
-------

`id=0.290622936492 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SeqSort>`__




.. cfunction:: void cvSeqSort( CvSeq* seq, CvCmpFunc func, void* userdata=NULL )

    Sorts sequence element using the specified comparison function.






::


    
    /* a < b ? -1 : a > b ? 1 : 0 */
    typedef int (CV_CDECL* CvCmpFunc)(const void* a, const void* b, void* userdata);
    

..



    
    :param seq: The sequence to sort 
    
    
    :param func: The comparison function that returns a negative, zero, or positive value depending on the relationships among the elements (see the above declaration and the example below) - a similar function is used by  ``qsort``  from C runline except that in the latter,  ``userdata``  is not used 
    
    
    :param userdata: The user parameter passed to the compasion function; helps to avoid global variables in some cases 
    
    
    
The function sorts the sequence in-place using the specified criteria. Below is an example of using this function:




::


    
    /* Sort 2d points in top-to-bottom left-to-right order */
    static int cmp_func( const void* _a, const void* _b, void* userdata )
    {
        CvPoint* a = (CvPoint*)_a;
        CvPoint* b = (CvPoint*)_b;
        int y_diff = a->y - b->y;
        int x_diff = a->x - b->x;
        return y_diff ? y_diff : x_diff;
    }
    
    ...
    
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* seq = cvCreateSeq( CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage );
    int i;
    
    for( i = 0; i < 10; i++ )
    {
        CvPoint pt;
        pt.x = rand() 
        pt.y = rand() 
        cvSeqPush( seq, &pt );
    }
    
    cvSeqSort( seq, cmp_func, 0 /* userdata is not used here */ );
    
    /* print out the sorted sequence */
    for( i = 0; i < seq->total; i++ )
    {
        CvPoint* pt = (CvPoint*)cvSeqElem( seq, i );
        printf( "(
    }
    
    cvReleaseMemStorage( &storage );
    

..


.. index:: SetAdd

.. _SetAdd:

SetAdd
------

`id=0.151496822644 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetAdd>`__




.. cfunction:: int cvSetAdd(  CvSet* setHeader, CvSetElem* elem=NULL, CvSetElem** inserted_elem=NULL )

    Occupies a node in the set.





    
    :param setHeader: Set 
    
    
    :param elem: Optional input argument, an inserted element. If not NULL, the function copies the data to the allocated node (the MSB of the first integer field is cleared after copying). 
    
    
    :param inserted_elem: Optional output argument; the pointer to the allocated cell 
    
    
    
The function allocates a new node, optionally copies
input element data to it, and returns the pointer and the index to the
node. The index value is taken from the lower bits of the 
``flags``
field of the node. The function has O(1) complexity; however, there exists
a faster function for allocating set nodes (see 
:ref:`SetNew`
).


.. index:: SetNew

.. _SetNew:

SetNew
------

`id=0.448446991925 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetNew>`__




.. cfunction:: CvSetElem* cvSetNew( CvSet* setHeader )

    Adds an element to a set (fast variant).





    
    :param setHeader: Set 
    
    
    
The function is an inline lightweight variant of 
:ref:`SetAdd`
. It occupies a new node and returns a pointer to it rather than an index.



.. index:: SetRemove

.. _SetRemove:

SetRemove
---------

`id=0.513485030618 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetRemove>`__




.. cfunction:: void cvSetRemove(  CvSet* setHeader, int index )

    Removes an element from a set.





    
    :param setHeader: Set 
    
    
    :param index: Index of the removed element 
    
    
    
The function removes an element with a specified
index from the set. If the node at the specified location is not occupied,
the function does nothing. The function has O(1) complexity; however,
:ref:`SetRemoveByPtr`
provides a quicker way to remove a set element
if it is located already.


.. index:: SetRemoveByPtr

.. _SetRemoveByPtr:

SetRemoveByPtr
--------------

`id=0.511092796762 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetRemoveByPtr>`__




.. cfunction:: void cvSetRemoveByPtr(  CvSet* setHeader, void* elem )

    Removes a set element based on its pointer.





    
    :param setHeader: Set 
    
    
    :param elem: Removed element 
    
    
    
The function is an inline lightweight variant of 
:ref:`SetRemove`
that requires an element pointer. The function does not check whether the node is occupied or not - the user should take care of that.



.. index:: SetSeqBlockSize

.. _SetSeqBlockSize:

SetSeqBlockSize
---------------

`id=0.94569516135 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetSeqBlockSize>`__




.. cfunction:: void cvSetSeqBlockSize(  CvSeq* seq, int deltaElems )

    Sets up sequence block size.





    
    :param seq: Sequence 
    
    
    :param deltaElems: Desirable sequence block size for elements 
    
    
    
The function affects memory allocation
granularity. When the free space in the sequence buffers has run out,
the function allocates the space for 
``deltaElems``
sequence
elements. If this block immediately follows the one previously allocated,
the two blocks are concatenated; otherwise, a new sequence block is
created. Therefore, the bigger the parameter is, the lower the possible
sequence fragmentation, but the more space in the storage block is wasted. When
the sequence is created, the parameter 
``deltaElems``
is set to
the default value of about 1K. The function can be called any time after
the sequence is created and affects future allocations. The function
can modify the passed value of the parameter to meet memory storage
constraints.


.. index:: SetSeqReaderPos

.. _SetSeqReaderPos:

SetSeqReaderPos
---------------

`id=0.435675937023 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/SetSeqReaderPos>`__




.. cfunction:: void cvSetSeqReaderPos(  CvSeqReader* reader, int index, int is_relative=0 )

    Moves the reader to the specified position.





    
    :param reader: Reader state 
    
    
    :param index: The destination position. If the positioning mode is used (see the next parameter), the actual position will be  ``index``  mod  ``reader->seq->total`` . 
    
    
    :param is_relative: If it is not zero, then  ``index``  is a relative to the current position 
    
    
    
The function moves the read position to an absolute position or relative to the current position.



.. index:: StartAppendToSeq

.. _StartAppendToSeq:

StartAppendToSeq
----------------

`id=0.481797162299 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/StartAppendToSeq>`__




.. cfunction:: void cvStartAppendToSeq(  CvSeq* seq, CvSeqWriter* writer )

    Initializes the process of writing data to a sequence.





    
    :param seq: Pointer to the sequence 
    
    
    :param writer: Writer state; initialized by the function 
    
    
    
The function initializes the process of
writing data to a sequence. Written elements are added to the end of the
sequence by using the
``CV_WRITE_SEQ_ELEM( written_elem, writer )``
macro. Note
that during the writing process, other operations on the sequence may
yield an incorrect result or even corrupt the sequence (see description of
:ref:`FlushSeqWriter`
, which helps to avoid some of these problems).


.. index:: StartReadSeq

.. _StartReadSeq:

StartReadSeq
------------

`id=0.274476331583 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/StartReadSeq>`__




.. cfunction:: void cvStartReadSeq(  const CvSeq* seq, CvSeqReader* reader, int reverse=0 )

    Initializes the process of sequential reading from a sequence.





    
    :param seq: Sequence 
    
    
    :param reader: Reader state; initialized by the function 
    
    
    :param reverse: Determines the direction of the sequence traversal. If  ``reverse``  is 0, the reader is positioned at the first sequence element; otherwise it is positioned at the last element.  
    
    
    
The function initializes the reader state. After
that, all the sequence elements from the first one down to the last one
can be read by subsequent calls of the macro
``CV_READ_SEQ_ELEM( read_elem, reader )``
in the case of forward reading and by using
``CV_REV_READ_SEQ_ELEM( read_elem, reader )``
in the case of reverse
reading. Both macros put the sequence element to 
``read_elem``
and
move the reading pointer toward the next element. A circular structure
of sequence blocks is used for the reading process, that is, after the
last element has been read by the macro 
``CV_READ_SEQ_ELEM``
, the
first element is read when the macro is called again. The same applies to
``CV_REV_READ_SEQ_ELEM``
. There is no function to finish the reading
process, since it neither changes the sequence nor creates any temporary
buffers. The reader field 
``ptr``
points to the current element of
the sequence that is to be read next. The code below demonstrates how
to use the sequence writer and reader.




::


    
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* seq = cvCreateSeq( CV_32SC1, sizeof(CvSeq), sizeof(int), storage );
    CvSeqWriter writer;
    CvSeqReader reader;
    int i;
    
    cvStartAppendToSeq( seq, &writer );
    for( i = 0; i < 10; i++ )
    {
        int val = rand()
        CV_WRITE_SEQ_ELEM( val, writer );
        printf("
    }
    cvEndWriteSeq( &writer );
    
    cvStartReadSeq( seq, &reader, 0 );
    for( i = 0; i < seq->total; i++ )
    {
        int val;
    #if 1
        CV_READ_SEQ_ELEM( val, reader );
        printf("
    #else /* alternative way, that is prefferable if sequence elements are large,
             or their size/type is unknown at compile time */
        printf("
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    #endif
    }
    ...
    
    cvReleaseStorage( &storage );
    

..


.. index:: StartWriteSeq

.. _StartWriteSeq:

StartWriteSeq
-------------

`id=0.633886985438 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/StartWriteSeq>`__




.. cfunction:: void cvStartWriteSeq(  int seq_flags, int header_size, int elem_size, CvMemStorage* storage, CvSeqWriter* writer )

    Creates a new sequence and initializes a writer for it.





    
    :param seq_flags: Flags of the created sequence. If the sequence is not passed to any function working with a specific type of sequences, the sequence value may be equal to 0; otherwise the appropriate type must be selected from the list of predefined sequence types. 
    
    
    :param header_size: Size of the sequence header. The parameter value may not be less than  ``sizeof(CvSeq)`` . If a certain type or extension is specified, it must fit within the base type header. 
    
    
    :param elem_size: Size of the sequence elements in bytes; must be consistent with the sequence type. For example, if a sequence of points is created (element type  ``CV_SEQ_ELTYPE_POINT``  ), then the parameter  ``elem_size``  must be equal to  ``sizeof(CvPoint)`` . 
    
    
    :param storage: Sequence location 
    
    
    :param writer: Writer state; initialized by the function 
    
    
    
The function is a combination of
:ref:`CreateSeq`
and 
:ref:`StartAppendToSeq`
. The pointer to the
created sequence is stored at
``writer->seq``
and is also returned by the
:ref:`EndWriteSeq`
function that should be called at the end.


.. index:: TreeToNodeSeq

.. _TreeToNodeSeq:

TreeToNodeSeq
-------------

`id=0.995912413662 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/TreeToNodeSeq>`__




.. cfunction:: CvSeq* cvTreeToNodeSeq(  const void* first, int header_size, CvMemStorage* storage )

    Gathers all node pointers to a single sequence.





    
    :param first: The initial tree node 
    
    
    :param header_size: Header size of the created sequence (sizeof(CvSeq) is the most frequently used value) 
    
    
    :param storage: Container for the sequence 
    
    
    
The function puts pointers of all nodes reacheable from 
``first``
into a single sequence. The pointers are written sequentially in the depth-first order.

