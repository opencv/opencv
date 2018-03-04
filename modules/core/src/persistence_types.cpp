// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "persistence.hpp"



/****************************************************************************************\
*                          Reading/Writing etc. for standard types                       *
\****************************************************************************************/

/*#define CV_TYPE_NAME_MAT "opencv-matrix"
#define CV_TYPE_NAME_MATND "opencv-nd-matrix"
#define CV_TYPE_NAME_SPARSE_MAT "opencv-sparse-matrix"
#define CV_TYPE_NAME_IMAGE "opencv-image"
#define CV_TYPE_NAME_SEQ "opencv-sequence"
#define CV_TYPE_NAME_SEQ_TREE "opencv-sequence-tree"
#define CV_TYPE_NAME_GRAPH "opencv-graph"*/

/******************************* CvMat ******************************/

static int icvIsMat( const void* ptr )
{
    return CV_IS_MAT_HDR_Z(ptr);
}

static void icvWriteMat( CvFileStorage* fs, const char* name, const void* struct_ptr, CvAttrList /*attr*/ )
{
    const CvMat* mat = (const CvMat*)struct_ptr;
    char dt[16];
    CvSize size;
    int y;

    assert( CV_IS_MAT_HDR_Z(mat) );

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_MAT );
    cvWriteInt( fs, "rows", mat->rows );
    cvWriteInt( fs, "cols", mat->cols );
    cvWriteString( fs, "dt", icvEncodeFormat( CV_MAT_TYPE(mat->type), dt ), 0 );
    cvStartWriteStruct( fs, "data", CV_NODE_SEQ + CV_NODE_FLOW );

    size = cvGetSize(mat);
    if( size.height > 0 && size.width > 0 && mat->data.ptr )
    {
        if( CV_IS_MAT_CONT(mat->type) )
        {
            size.width *= size.height;
            size.height = 1;
        }

        for( y = 0; y < size.height; y++ )
            cvWriteRawData( fs, mat->data.ptr + (size_t)y*mat->step, size.width, dt );
    }
    cvEndWriteStruct( fs );
    cvEndWriteStruct( fs );
}


static int icvFileNodeSeqLen( CvFileNode* node )
{
    return CV_NODE_IS_COLLECTION(node->tag) ? node->data.seq->total :
        CV_NODE_TYPE(node->tag) != CV_NODE_NONE;
}


static void* icvReadMat( CvFileStorage* fs, CvFileNode* node )
{
    void* ptr = 0;
    CvMat* mat;
    const char* dt;
    CvFileNode* data;
    int rows, cols, elem_type;

    rows = cvReadIntByName( fs, node, "rows", -1 );
    cols = cvReadIntByName( fs, node, "cols", -1 );
    dt = cvReadStringByName( fs, node, "dt", 0 );

    if( rows < 0 || cols < 0 || !dt )
        CV_Error( CV_StsError, "Some of essential matrix attributes are absent" );

    elem_type = icvDecodeSimpleFormat( dt );

    data = cvGetFileNodeByName( fs, node, "data" );
    if( !data )
        CV_Error( CV_StsError, "The matrix data is not found in file storage" );

    int nelems = icvFileNodeSeqLen( data );
    if( nelems > 0 && nelems != rows*cols*CV_MAT_CN(elem_type) )
        CV_Error( CV_StsUnmatchedSizes,
                 "The matrix size does not match to the number of stored elements" );

    if( nelems > 0 )
    {
        mat = cvCreateMat( rows, cols, elem_type );
        cvReadRawData( fs, data, mat->data.ptr, dt );
    }
    else
        mat = cvCreateMatHeader( rows, cols, elem_type );

    ptr = mat;
    return ptr;
}


/******************************* CvMatND ******************************/

static int icvIsMatND( const void* ptr )
{
    return CV_IS_MATND_HDR(ptr);
}


static void icvWriteMatND( CvFileStorage* fs, const char* name, const void* struct_ptr, CvAttrList /*attr*/ )
{
    CvMatND* mat = (CvMatND*)struct_ptr;
    CvMatND stub;
    CvNArrayIterator iterator;
    int dims, sizes[CV_MAX_DIM];
    char dt[16];

    assert( CV_IS_MATND_HDR(mat) );

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_MATND );
    dims = cvGetDims( mat, sizes );
    cvStartWriteStruct( fs, "sizes", CV_NODE_SEQ + CV_NODE_FLOW );
    cvWriteRawData( fs, sizes, dims, "i" );
    cvEndWriteStruct( fs );
    cvWriteString( fs, "dt", icvEncodeFormat( cvGetElemType(mat), dt ), 0 );
    cvStartWriteStruct( fs, "data", CV_NODE_SEQ + CV_NODE_FLOW );

    if( mat->dim[0].size > 0 && mat->data.ptr )
    {
        cvInitNArrayIterator( 1, (CvArr**)&mat, 0, &stub, &iterator );

        do
            cvWriteRawData( fs, iterator.ptr[0], iterator.size.width, dt );
        while( cvNextNArraySlice( &iterator ));
    }
    cvEndWriteStruct( fs );
    cvEndWriteStruct( fs );
}


static void* icvReadMatND( CvFileStorage* fs, CvFileNode* node )
{
    void* ptr = 0;
    CvMatND* mat;
    const char* dt;
    CvFileNode* data;
    CvFileNode* sizes_node;
    int sizes[CV_MAX_DIM] = {0}, dims, elem_type;
    int i, total_size;

    sizes_node = cvGetFileNodeByName( fs, node, "sizes" );
    dt = cvReadStringByName( fs, node, "dt", 0 );

    if( !sizes_node || !dt )
        CV_Error( CV_StsError, "Some of essential matrix attributes are absent" );

    dims = CV_NODE_IS_SEQ(sizes_node->tag) ? sizes_node->data.seq->total :
           CV_NODE_IS_INT(sizes_node->tag) ? 1 : -1;

    if( dims <= 0 || dims > CV_MAX_DIM )
        CV_Error( CV_StsParseError, "Could not determine the matrix dimensionality" );

    cvReadRawData( fs, sizes_node, sizes, "i" );
    elem_type = icvDecodeSimpleFormat( dt );

    data = cvGetFileNodeByName( fs, node, "data" );
    if( !data )
        CV_Error( CV_StsError, "The matrix data is not found in file storage" );



    for( total_size = CV_MAT_CN(elem_type), i = 0; i < dims; i++ )
    {
        CV_Assert(sizes[i]);
        total_size *= sizes[i];
    }

    int nelems = icvFileNodeSeqLen( data );

    if( nelems > 0 && nelems != total_size )
        CV_Error( CV_StsUnmatchedSizes,
                 "The matrix size does not match to the number of stored elements" );

    if( nelems > 0 )
    {
        mat = cvCreateMatND( dims, sizes, elem_type );
        cvReadRawData( fs, data, mat->data.ptr, dt );
    }
    else
        mat = cvCreateMatNDHeader( dims, sizes, elem_type );

    ptr = mat;
    return ptr;
}


/******************************* CvSparseMat ******************************/

static int icvIsSparseMat( const void* ptr )
{
    return CV_IS_SPARSE_MAT(ptr);
}


static int icvSortIdxCmpFunc( const void* _a, const void* _b, void* userdata )
{
    int i, dims = *(int*)userdata;
    const int* a = *(const int**)_a;
    const int* b = *(const int**)_b;

    for( i = 0; i < dims; i++ )
    {
        int delta = a[i] - b[i];
        if( delta )
            return delta;
    }

    return 0;
}


static void icvWriteSparseMat( CvFileStorage* fs, const char* name, const void* struct_ptr, CvAttrList /*attr*/ )
{
    CvMemStorage* memstorage = 0;
    const CvSparseMat* mat = (const CvSparseMat*)struct_ptr;
    CvSparseMatIterator iterator;
    CvSparseNode* node;
    CvSeq* elements;
    CvSeqReader reader;
    int i, dims;
    int *prev_idx = 0;
    char dt[16];

    assert( CV_IS_SPARSE_MAT(mat) );

    memstorage = cvCreateMemStorage();

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_SPARSE_MAT );
    dims = cvGetDims( mat, 0 );

    cvStartWriteStruct( fs, "sizes", CV_NODE_SEQ + CV_NODE_FLOW );
    cvWriteRawData( fs, mat->size, dims, "i" );
    cvEndWriteStruct( fs );
    cvWriteString( fs, "dt", icvEncodeFormat( CV_MAT_TYPE(mat->type), dt ), 0 );
    cvStartWriteStruct( fs, "data", CV_NODE_SEQ + CV_NODE_FLOW );

    elements = cvCreateSeq( CV_SEQ_ELTYPE_PTR, sizeof(CvSeq), sizeof(int*), memstorage );

    node = cvInitSparseMatIterator( mat, &iterator );
    while( node )
    {
        int* idx = CV_NODE_IDX( mat, node );
        cvSeqPush( elements, &idx );
        node = cvGetNextSparseNode( &iterator );
    }

    cvSeqSort( elements, icvSortIdxCmpFunc, &dims );
    cvStartReadSeq( elements, &reader, 0 );

    for( i = 0; i < elements->total; i++ )
    {
        int* idx;
        void* val;
        int k = 0;

        CV_READ_SEQ_ELEM( idx, reader );
        if( i > 0 )
        {
            for( ; idx[k] == prev_idx[k]; k++ )
                assert( k < dims );
            if( k < dims - 1 )
                fs->write_int( fs, 0, k - dims + 1 );
        }
        for( ; k < dims; k++ )
            fs->write_int( fs, 0, idx[k] );
        prev_idx = idx;

        node = (CvSparseNode*)((uchar*)idx - mat->idxoffset );
        val = CV_NODE_VAL( mat, node );

        cvWriteRawData( fs, val, 1, dt );
    }

    cvEndWriteStruct( fs );
    cvEndWriteStruct( fs );
    cvReleaseMemStorage( &memstorage );
}


static void* icvReadSparseMat( CvFileStorage* fs, CvFileNode* node )
{
    void* ptr = 0;
    CvSparseMat* mat;
    const char* dt;
    CvFileNode* data;
    CvFileNode* sizes_node;
    CvSeqReader reader;
    CvSeq* elements;
    int sizes[CV_MAX_DIM_HEAP], dims, elem_type, cn;
    int i;

    sizes_node = cvGetFileNodeByName( fs, node, "sizes" );
    dt = cvReadStringByName( fs, node, "dt", 0 );

    if( !sizes_node || !dt )
        CV_Error( CV_StsError, "Some of essential matrix attributes are absent" );

    dims = CV_NODE_IS_SEQ(sizes_node->tag) ? sizes_node->data.seq->total :
           CV_NODE_IS_INT(sizes_node->tag) ? 1 : -1;

    if( dims <= 0 || dims > CV_MAX_DIM)
        CV_Error( CV_StsParseError, "Could not determine sparse matrix dimensionality" );

    cvReadRawData( fs, sizes_node, sizes, "i" );
    elem_type = icvDecodeSimpleFormat( dt );

    data = cvGetFileNodeByName( fs, node, "data" );
    if( !data || !CV_NODE_IS_SEQ(data->tag) )
        CV_Error( CV_StsError, "The matrix data is not found in file storage" );

    mat = cvCreateSparseMat( dims, sizes, elem_type );

    cn = CV_MAT_CN(elem_type);
    int idx[CV_MAX_DIM_HEAP];
    elements = data->data.seq;
    cvStartReadRawData( fs, data, &reader );

    for( i = 0; i < elements->total; )
    {
        CvFileNode* elem = (CvFileNode*)reader.ptr;
        uchar* val;
        int k;
        if( !CV_NODE_IS_INT(elem->tag ))
            CV_Error( CV_StsParseError, "Sparse matrix data is corrupted" );
        k = elem->data.i;
        if( i > 0 && k >= 0 )
            idx[dims-1] = k;
        else
        {
            if( i > 0 )
                k = dims + k - 1;
            else
                idx[0] = k, k = 1;
            for( ; k < dims; k++ )
            {
                CV_NEXT_SEQ_ELEM( elements->elem_size, reader );
                i++;
                elem = (CvFileNode*)reader.ptr;
                if( !CV_NODE_IS_INT(elem->tag ) || elem->data.i < 0 )
                    CV_Error( CV_StsParseError, "Sparse matrix data is corrupted" );
                idx[k] = elem->data.i;
            }
        }
        CV_NEXT_SEQ_ELEM( elements->elem_size, reader );
        i++;
        val = cvPtrND( mat, idx, 0, 1, 0 );
        cvReadRawDataSlice( fs, &reader, cn, val, dt );
        i += cn;
    }

    ptr = mat;
    return ptr;
}


/******************************* IplImage ******************************/

static int icvIsImage( const void* ptr )
{
    return CV_IS_IMAGE_HDR(ptr);
}

static void icvWriteImage( CvFileStorage* fs, const char* name, const void* struct_ptr, CvAttrList /*attr*/ )
{
    const IplImage* image = (const IplImage*)struct_ptr;
    char dt_buf[16], *dt;
    CvSize size;
    int y, depth;

    assert( CV_IS_IMAGE(image) );

    if( image->dataOrder == IPL_DATA_ORDER_PLANE )
        CV_Error( CV_StsUnsupportedFormat,
        "Images with planar data layout are not supported" );

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_IMAGE );
    cvWriteInt( fs, "width", image->width );
    cvWriteInt( fs, "height", image->height );
    cvWriteString( fs, "origin", image->origin == IPL_ORIGIN_TL
                   ? "top-left" : "bottom-left", 0 );
    cvWriteString( fs, "layout", image->dataOrder == IPL_DATA_ORDER_PLANE
                   ? "planar" : "interleaved", 0 );
    if( image->roi )
    {
        cvStartWriteStruct( fs, "roi", CV_NODE_MAP + CV_NODE_FLOW );
        cvWriteInt( fs, "x", image->roi->xOffset );
        cvWriteInt( fs, "y", image->roi->yOffset );
        cvWriteInt( fs, "width", image->roi->width );
        cvWriteInt( fs, "height", image->roi->height );
        cvWriteInt( fs, "coi", image->roi->coi );
        cvEndWriteStruct( fs );
    }

    depth = IPL2CV_DEPTH(image->depth);
    sprintf( dt_buf, "%d%c", image->nChannels, icvTypeSymbol(depth) );
    dt = dt_buf + (dt_buf[2] == '\0' && dt_buf[0] == '1');
    cvWriteString( fs, "dt", dt, 0 );

    size = cvSize(image->width, image->height);
    if( size.width*image->nChannels*CV_ELEM_SIZE(depth) == image->widthStep )
    {
        size.width *= size.height;
        size.height = 1;
    }

    cvStartWriteStruct( fs, "data", CV_NODE_SEQ + CV_NODE_FLOW );
    for( y = 0; y < size.height; y++ )
        cvWriteRawData( fs, image->imageData + y*image->widthStep, size.width, dt );
    cvEndWriteStruct( fs );
    cvEndWriteStruct( fs );
}


static void* icvReadImage( CvFileStorage* fs, CvFileNode* node )
{
    void* ptr = 0;
    IplImage* image;
    const char* dt;
    CvFileNode* data;
    CvFileNode* roi_node;
    CvSeqReader reader;
    CvRect roi;
    int y, width, height, elem_type, coi, depth;
    const char* origin, *data_order;

    width = cvReadIntByName( fs, node, "width", 0 );
    height = cvReadIntByName( fs, node, "height", 0 );
    dt = cvReadStringByName( fs, node, "dt", 0 );
    origin = cvReadStringByName( fs, node, "origin", 0 );

    if( width == 0 || height == 0 || dt == 0 || origin == 0 )
        CV_Error( CV_StsError, "Some of essential image attributes are absent" );

    elem_type = icvDecodeSimpleFormat( dt );
    data_order = cvReadStringByName( fs, node, "layout", "interleaved" );
    if( !data_order || strcmp( data_order, "interleaved" ) != 0 )
        CV_Error( CV_StsError, "Only interleaved images can be read" );

    data = cvGetFileNodeByName( fs, node, "data" );
    if( !data )
        CV_Error( CV_StsError, "The image data is not found in file storage" );

    if( icvFileNodeSeqLen( data ) != width*height*CV_MAT_CN(elem_type) )
        CV_Error( CV_StsUnmatchedSizes,
        "The matrix size does not match to the number of stored elements" );

    depth = cvIplDepth(elem_type);
    image = cvCreateImage( cvSize(width,height), depth, CV_MAT_CN(elem_type) );

    roi_node = cvGetFileNodeByName( fs, node, "roi" );
    if( roi_node )
    {
        roi.x = cvReadIntByName( fs, roi_node, "x", 0 );
        roi.y = cvReadIntByName( fs, roi_node, "y", 0 );
        roi.width = cvReadIntByName( fs, roi_node, "width", 0 );
        roi.height = cvReadIntByName( fs, roi_node, "height", 0 );
        coi = cvReadIntByName( fs, roi_node, "coi", 0 );

        cvSetImageROI( image, roi );
        cvSetImageCOI( image, coi );
    }

    if( width*CV_ELEM_SIZE(elem_type) == image->widthStep )
    {
        width *= height;
        height = 1;
    }

    width *= CV_MAT_CN(elem_type);
    cvStartReadRawData( fs, data, &reader );
    for( y = 0; y < height; y++ )
    {
        cvReadRawDataSlice( fs, &reader, width,
            image->imageData + y*image->widthStep, dt );
    }

    ptr = image;
    return ptr;
}


/******************************* CvSeq ******************************/

static int icvIsSeq( const void* ptr )
{
    return CV_IS_SEQ(ptr);
}


static void
icvReleaseSeq( void** ptr )
{
    if( !ptr )
        CV_Error( CV_StsNullPtr, "NULL double pointer" );
    *ptr = 0; // it's impossible now to release seq, so just clear the pointer
}


static void* icvCloneSeq( const void* ptr )
{
    return cvSeqSlice( (CvSeq*)ptr, CV_WHOLE_SEQ,
                       0 /* use the same storage as for the original sequence */, 1 );
}


static void icvWriteHeaderData( CvFileStorage* fs, const CvSeq* seq, CvAttrList* attr, int initial_header_size )
{
    char header_dt_buf[128];
    const char* header_dt = cvAttrValue( attr, "header_dt" );

    if( header_dt )
    {
        int dt_header_size;
        dt_header_size = icvCalcElemSize( header_dt, initial_header_size );
        if( dt_header_size > seq->header_size )
            CV_Error( CV_StsUnmatchedSizes,
            "The size of header calculated from \"header_dt\" is greater than header_size" );
    }
    else if( seq->header_size > initial_header_size )
    {
        if( CV_IS_SEQ(seq) && CV_IS_SEQ_POINT_SET(seq) &&
            seq->header_size == sizeof(CvPoint2DSeq) &&
            seq->elem_size == sizeof(int)*2 )
        {
            CvPoint2DSeq* point_seq = (CvPoint2DSeq*)seq;

            cvStartWriteStruct( fs, "rect", CV_NODE_MAP + CV_NODE_FLOW );
            cvWriteInt( fs, "x", point_seq->rect.x );
            cvWriteInt( fs, "y", point_seq->rect.y );
            cvWriteInt( fs, "width", point_seq->rect.width );
            cvWriteInt( fs, "height", point_seq->rect.height );
            cvEndWriteStruct( fs );
            cvWriteInt( fs, "color", point_seq->color );
        }
        else if( CV_IS_SEQ(seq) && CV_IS_SEQ_CHAIN(seq) &&
                 CV_MAT_TYPE(seq->flags) == CV_8UC1 )
        {
            CvChain* chain = (CvChain*)seq;

            cvStartWriteStruct( fs, "origin", CV_NODE_MAP + CV_NODE_FLOW );
            cvWriteInt( fs, "x", chain->origin.x );
            cvWriteInt( fs, "y", chain->origin.y );
            cvEndWriteStruct( fs );
        }
        else
        {
            unsigned extra_size = seq->header_size - initial_header_size;
            // a heuristic to provide nice defaults for sequences of int's & float's
            if( extra_size % sizeof(int) == 0 )
                sprintf( header_dt_buf, "%ui", (unsigned)(extra_size/sizeof(int)) );
            else
                sprintf( header_dt_buf, "%uu", extra_size );
            header_dt = header_dt_buf;
        }
    }

    if( header_dt )
    {
        cvWriteString( fs, "header_dt", header_dt, 0 );
        cvStartWriteStruct( fs, "header_user_data", CV_NODE_SEQ + CV_NODE_FLOW );
        cvWriteRawData( fs, (uchar*)seq + sizeof(CvSeq), 1, header_dt );
        cvEndWriteStruct( fs );
    }
}


static char* icvGetFormat( const CvSeq* seq, const char* dt_key, CvAttrList* attr, int initial_elem_size, char* dt_buf )
{
    char* dt = 0;
    dt = (char*)cvAttrValue( attr, dt_key );

    if( dt )
    {
        int dt_elem_size;
        dt_elem_size = icvCalcElemSize( dt, initial_elem_size );
        if( dt_elem_size != seq->elem_size )
            CV_Error( CV_StsUnmatchedSizes,
            "The size of element calculated from \"dt\" and "
            "the elem_size do not match" );
    }
    else if( CV_MAT_TYPE(seq->flags) != 0 || seq->elem_size == 1 )
    {
        if( CV_ELEM_SIZE(seq->flags) != seq->elem_size )
            CV_Error( CV_StsUnmatchedSizes,
            "Size of sequence element (elem_size) is inconsistent with seq->flags" );
        dt = icvEncodeFormat( CV_MAT_TYPE(seq->flags), dt_buf );
    }
    else if( seq->elem_size > initial_elem_size )
    {
        unsigned extra_elem_size = seq->elem_size - initial_elem_size;
        // a heuristic to provide nice defaults for sequences of int's & float's
        if( extra_elem_size % sizeof(int) == 0 )
            sprintf( dt_buf, "%ui", (unsigned)(extra_elem_size/sizeof(int)) );
        else
            sprintf( dt_buf, "%uu", extra_elem_size );
        dt = dt_buf;
    }

    return dt;
}


static void icvWriteSeq( CvFileStorage* fs, const char* name, const void* struct_ptr, CvAttrList attr, int level )
{
    const CvSeq* seq = (CvSeq*)struct_ptr;
    CvSeqBlock* block;
    char buf[128];
    char dt_buf[128], *dt;

    assert( CV_IS_SEQ( seq ));
    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_SEQ );

    if( level >= 0 )
        cvWriteInt( fs, "level", level );

    dt = icvGetFormat( seq, "dt", &attr, 0, dt_buf );

    strcpy(buf, "");
    if( CV_IS_SEQ_CLOSED(seq) )
        strcat(buf, " closed");
    if( CV_IS_SEQ_HOLE(seq) )
        strcat(buf, " hole");
    if( CV_IS_SEQ_CURVE(seq) )
        strcat(buf, " curve");
    if( CV_SEQ_ELTYPE(seq) == 0 && seq->elem_size != 1 )
        strcat(buf, " untyped");

    cvWriteString( fs, "flags", buf + (buf[0] ? 1 : 0), 1 );

    cvWriteInt( fs, "count", seq->total );

    cvWriteString( fs, "dt", dt, 0 );

    icvWriteHeaderData( fs, seq, &attr, sizeof(CvSeq) );
    cvStartWriteStruct( fs, "data", CV_NODE_SEQ + CV_NODE_FLOW );

    for( block = seq->first; block; block = block->next )
    {
        cvWriteRawData( fs, block->data, block->count, dt );
        if( block == seq->first->prev )
            break;
    }
    cvEndWriteStruct( fs );
    cvEndWriteStruct( fs );
}


static void icvWriteSeqTree( CvFileStorage* fs, const char* name, const void* struct_ptr, CvAttrList attr )
{
    const CvSeq* seq = (CvSeq*)struct_ptr;
    const char* recursive_value = cvAttrValue( &attr, "recursive" );
    int is_recursive = recursive_value &&
                       strcmp(recursive_value,"0") != 0 &&
                       strcmp(recursive_value,"false") != 0 &&
                       strcmp(recursive_value,"False") != 0 &&
                       strcmp(recursive_value,"FALSE") != 0;

    assert( CV_IS_SEQ( seq ));

    if( !is_recursive )
    {
        icvWriteSeq( fs, name, seq, attr, -1 );
    }
    else
    {
        CvTreeNodeIterator tree_iterator;

        cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_SEQ_TREE );
        cvStartWriteStruct( fs, "sequences", CV_NODE_SEQ );
        cvInitTreeNodeIterator( &tree_iterator, seq, INT_MAX );

        for(;;)
        {
            if( !tree_iterator.node )
                break;
            icvWriteSeq( fs, 0, tree_iterator.node, attr, tree_iterator.level );
            cvNextTreeNode( &tree_iterator );
        }

        cvEndWriteStruct( fs );
        cvEndWriteStruct( fs );
    }
}


static void* icvReadSeq( CvFileStorage* fs, CvFileNode* node )
{
    void* ptr = 0;
    CvSeq* seq;
    CvSeqBlock* block;
    CvFileNode *data, *header_node, *rect_node, *origin_node;
    CvSeqReader reader;
    int total, flags;
    int elem_size, header_size = sizeof(CvSeq);
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS], i, fmt_pair_count;
    int items_per_elem = 0;
    const char* flags_str;
    const char* header_dt;
    const char* dt;
    char* endptr = 0;

    flags_str = cvReadStringByName( fs, node, "flags", 0 );
    total = cvReadIntByName( fs, node, "count", -1 );
    dt = cvReadStringByName( fs, node, "dt", 0 );

    if( !flags_str || total == -1 || !dt )
        CV_Error( CV_StsError, "Some of essential sequence attributes are absent" );

    flags = CV_SEQ_MAGIC_VAL;

    if( cv_isdigit(flags_str[0]) )
    {
        const int OLD_SEQ_ELTYPE_BITS = 9;
        const int OLD_SEQ_ELTYPE_MASK = (1 << OLD_SEQ_ELTYPE_BITS) - 1;
        const int OLD_SEQ_KIND_BITS = 3;
        const int OLD_SEQ_KIND_MASK = ((1 << OLD_SEQ_KIND_BITS) - 1) << OLD_SEQ_ELTYPE_BITS;
        const int OLD_SEQ_KIND_CURVE = 1 << OLD_SEQ_ELTYPE_BITS;
        const int OLD_SEQ_FLAG_SHIFT = OLD_SEQ_KIND_BITS + OLD_SEQ_ELTYPE_BITS;
        const int OLD_SEQ_FLAG_CLOSED = 1 << OLD_SEQ_FLAG_SHIFT;
        const int OLD_SEQ_FLAG_HOLE = 8 << OLD_SEQ_FLAG_SHIFT;

        int flags0 = (int)strtol( flags_str, &endptr, 16 );
        if( endptr == flags_str || (flags0 & CV_MAGIC_MASK) != CV_SEQ_MAGIC_VAL )
            CV_Error( CV_StsError, "The sequence flags are invalid" );
        if( (flags0 & OLD_SEQ_KIND_MASK) == OLD_SEQ_KIND_CURVE )
            flags |= CV_SEQ_KIND_CURVE;
        if( flags0 & OLD_SEQ_FLAG_CLOSED )
            flags |= CV_SEQ_FLAG_CLOSED;
        if( flags0 & OLD_SEQ_FLAG_HOLE )
            flags |= CV_SEQ_FLAG_HOLE;
        flags |= flags0 & OLD_SEQ_ELTYPE_MASK;
    }
    else
    {
        if( strstr(flags_str, "curve") )
            flags |= CV_SEQ_KIND_CURVE;
        if( strstr(flags_str, "closed") )
            flags |= CV_SEQ_FLAG_CLOSED;
        if( strstr(flags_str, "hole") )
            flags |= CV_SEQ_FLAG_HOLE;
        if( !strstr(flags_str, "untyped") )
        {
            CV_TRY
            {
                flags |= icvDecodeSimpleFormat(dt);
            }
            CV_CATCH_ALL
            {
            }
        }
    }

    header_dt = cvReadStringByName( fs, node, "header_dt", 0 );
    header_node = cvGetFileNodeByName( fs, node, "header_user_data" );

    if( (header_dt != 0) ^ (header_node != 0) )
        CV_Error( CV_StsError,
        "One of \"header_dt\" and \"header_user_data\" is there, while the other is not" );

    rect_node = cvGetFileNodeByName( fs, node, "rect" );
    origin_node = cvGetFileNodeByName( fs, node, "origin" );

    if( (header_node != 0) + (rect_node != 0) + (origin_node != 0) > 1 )
        CV_Error( CV_StsError, "Only one of \"header_user_data\", \"rect\" and \"origin\" tags may occur" );

    if( header_dt )
    {
        header_size = icvCalcElemSize( header_dt, header_size );
    }
    else if( rect_node )
        header_size = sizeof(CvPoint2DSeq);
    else if( origin_node )
        header_size = sizeof(CvChain);

    elem_size = icvCalcElemSize( dt, 0 );
    seq = cvCreateSeq( flags, header_size, elem_size, fs->dststorage );

    if( header_node )
    {
        CV_Assert(header_dt);
        cvReadRawData( fs, header_node, (char*)seq + sizeof(CvSeq), header_dt );
    }
    else if( rect_node )
    {
        CvPoint2DSeq* point_seq = (CvPoint2DSeq*)seq;
        point_seq->rect.x = cvReadIntByName( fs, rect_node, "x", 0 );
        point_seq->rect.y = cvReadIntByName( fs, rect_node, "y", 0 );
        point_seq->rect.width = cvReadIntByName( fs, rect_node, "width", 0 );
        point_seq->rect.height = cvReadIntByName( fs, rect_node, "height", 0 );
        point_seq->color = cvReadIntByName( fs, node, "color", 0 );
    }
    else if( origin_node )
    {
        CvChain* chain = (CvChain*)seq;
        chain->origin.x = cvReadIntByName( fs, origin_node, "x", 0 );
        chain->origin.y = cvReadIntByName( fs, origin_node, "y", 0 );
    }

    cvSeqPushMulti( seq, 0, total, 0 );
    fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
    fmt_pair_count *= 2;
    for( i = 0; i < fmt_pair_count; i += 2 )
        items_per_elem += fmt_pairs[i];

    data = cvGetFileNodeByName( fs, node, "data" );
    if( !data )
        CV_Error( CV_StsError, "The image data is not found in file storage" );

    if( icvFileNodeSeqLen( data ) != total*items_per_elem )
        CV_Error( CV_StsError, "The number of stored elements does not match to \"count\"" );

    cvStartReadRawData( fs, data, &reader );
    for( block = seq->first; block; block = block->next )
    {
        int delta = block->count*items_per_elem;
        cvReadRawDataSlice( fs, &reader, delta, block->data, dt );
        if( block == seq->first->prev )
            break;
    }

    ptr = seq;
    return ptr;
}


static void* icvReadSeqTree( CvFileStorage* fs, CvFileNode* node )
{
    void* ptr = 0;
    CvFileNode *sequences_node = cvGetFileNodeByName( fs, node, "sequences" );
    CvSeq* sequences;
    CvSeq* root = 0;
    CvSeq* parent = 0;
    CvSeq* prev_seq = 0;
    CvSeqReader reader;
    int i, total;
    int prev_level = 0;

    if( !sequences_node || !CV_NODE_IS_SEQ(sequences_node->tag) )
        CV_Error( CV_StsParseError,
        "opencv-sequence-tree instance should contain a field \"sequences\" that should be a sequence" );

    sequences = sequences_node->data.seq;
    total = sequences->total;

    cvStartReadSeq( sequences, &reader, 0 );
    for( i = 0; i < total; i++ )
    {
        CvFileNode* elem = (CvFileNode*)reader.ptr;
        CvSeq* seq;
        int level;
        seq = (CvSeq*)cvRead( fs, elem );
        CV_Assert(seq);
        level = cvReadIntByName( fs, elem, "level", -1 );
        if( level < 0 )
            CV_Error( CV_StsParseError, "All the sequence tree nodes should contain \"level\" field" );
        if( !root )
            root = seq;
        if( level > prev_level )
        {
            assert( level == prev_level + 1 );
            parent = prev_seq;
            prev_seq = 0;
            if( parent )
                parent->v_next = seq;
        }
        else if( level < prev_level )
        {
            for( ; prev_level > level; prev_level-- )
                prev_seq = prev_seq->v_prev;
            parent = prev_seq->v_prev;
        }
        seq->h_prev = prev_seq;
        if( prev_seq )
            prev_seq->h_next = seq;
        seq->v_prev = parent;
        prev_seq = seq;
        prev_level = level;
        CV_NEXT_SEQ_ELEM( sequences->elem_size, reader );
    }

    ptr = root;
    return ptr;
}

/******************************* CvGraph ******************************/

static int icvIsGraph( const void* ptr )
{
    return CV_IS_GRAPH(ptr);
}


static void icvReleaseGraph( void** ptr )
{
    if( !ptr )
        CV_Error( CV_StsNullPtr, "NULL double pointer" );

    *ptr = 0; // it's impossible now to release graph, so just clear the pointer
}


static void* icvCloneGraph( const void* ptr )
{
    return cvCloneGraph( (const CvGraph*)ptr, 0 );
}


static void icvWriteGraph( CvFileStorage* fs, const char* name, const void* struct_ptr, CvAttrList attr )
{
    int* flag_buf = 0;
    char* write_buf = 0;
    const CvGraph* graph = (const CvGraph*)struct_ptr;
    CvSeqReader reader;
    char buf[128];
    int i, k, vtx_count, edge_count;
    char vtx_dt_buf[128], *vtx_dt;
    char edge_dt_buf[128], *edge_dt;
    int write_buf_size;

    assert( CV_IS_GRAPH(graph) );
    vtx_count = cvGraphGetVtxCount( graph );
    edge_count = cvGraphGetEdgeCount( graph );
    flag_buf = (int*)cvAlloc( vtx_count*sizeof(flag_buf[0]));

    // count vertices
    cvStartReadSeq( (CvSeq*)graph, &reader );
    for( i = 0, k = 0; i < graph->total; i++ )
    {
        if( CV_IS_SET_ELEM( reader.ptr ))
        {
            CvGraphVtx* vtx = (CvGraphVtx*)reader.ptr;
            flag_buf[k] = vtx->flags;
            vtx->flags = k++;
        }
        CV_NEXT_SEQ_ELEM( graph->elem_size, reader );
    }

    // write header
    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_GRAPH );

    cvWriteString(fs, "flags", CV_IS_GRAPH_ORIENTED(graph) ? "oriented" : "", 1);

    cvWriteInt( fs, "vertex_count", vtx_count );
    vtx_dt = icvGetFormat( (CvSeq*)graph, "vertex_dt",
                    &attr, sizeof(CvGraphVtx), vtx_dt_buf );
    if( vtx_dt )
        cvWriteString( fs, "vertex_dt", vtx_dt, 0 );

    cvWriteInt( fs, "edge_count", edge_count );
    edge_dt = icvGetFormat( (CvSeq*)graph->edges, "edge_dt",
                                &attr, sizeof(CvGraphEdge), buf );
    sprintf( edge_dt_buf, "2if%s", edge_dt ? edge_dt : "" );
    edge_dt = edge_dt_buf;
    cvWriteString( fs, "edge_dt", edge_dt, 0 );

    icvWriteHeaderData( fs, (CvSeq*)graph, &attr, sizeof(CvGraph) );

    write_buf_size = MAX( 3*graph->elem_size, 1 << 16 );
    write_buf_size = MAX( 3*graph->edges->elem_size, write_buf_size );
    write_buf = (char*)cvAlloc( write_buf_size );

    // as vertices and edges are written in similar way,
    // do it as a parametrized 2-iteration loop
    for( k = 0; k < 2; k++ )
    {
        const char* dt = k == 0 ? vtx_dt : edge_dt;
        if( dt )
        {
            CvSet* data = k == 0 ? (CvSet*)graph : graph->edges;
            int elem_size = data->elem_size;
            int write_elem_size = icvCalcElemSize( dt, 0 );
            char* src_ptr = write_buf;
            int write_max = write_buf_size / write_elem_size, write_count = 0;

            // alignment of user part of the edge data following 2if
            int edge_user_align = sizeof(float);

            if( k == 1 )
            {
                int fmt_pairs[CV_FS_MAX_FMT_PAIRS], fmt_pair_count;
                fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
                if( fmt_pair_count > 2 && CV_ELEM_SIZE(fmt_pairs[2*2+1]) >= (int)sizeof(double))
                    edge_user_align = sizeof(double);
            }

            cvStartWriteStruct( fs, k == 0 ? "vertices" : "edges",
                                CV_NODE_SEQ + CV_NODE_FLOW );
            cvStartReadSeq( (CvSeq*)data, &reader );
            for( i = 0; i < data->total; i++ )
            {
                if( CV_IS_SET_ELEM( reader.ptr ))
                {
                    if( k == 0 ) // vertices
                        memcpy( src_ptr, reader.ptr + sizeof(CvGraphVtx), write_elem_size );
                    else
                    {
                        CvGraphEdge* edge = (CvGraphEdge*)reader.ptr;
                        src_ptr = (char*)cvAlignPtr( src_ptr, sizeof(int) );
                        ((int*)src_ptr)[0] = edge->vtx[0]->flags;
                        ((int*)src_ptr)[1] = edge->vtx[1]->flags;
                        *(float*)(src_ptr + sizeof(int)*2) = edge->weight;
                        if( elem_size > (int)sizeof(CvGraphEdge) )
                        {
                            char* src_ptr2 = (char*)cvAlignPtr( src_ptr + 2*sizeof(int)
                                                + sizeof(float), edge_user_align );
                            memcpy( src_ptr2, edge + 1, elem_size - sizeof(CvGraphEdge) );
                        }
                    }
                    src_ptr += write_elem_size;
                    if( ++write_count >= write_max )
                    {
                        cvWriteRawData( fs, write_buf, write_count, dt );
                        write_count = 0;
                        src_ptr = write_buf;
                    }
                }
                CV_NEXT_SEQ_ELEM( data->elem_size, reader );
            }

            if( write_count > 0 )
                cvWriteRawData( fs, write_buf, write_count, dt );
            cvEndWriteStruct( fs );
        }
    }

    cvEndWriteStruct( fs );

    // final stage. restore the graph flags
    cvStartReadSeq( (CvSeq*)graph, &reader );
    vtx_count = 0;
    for( i = 0; i < graph->total; i++ )
    {
        if( CV_IS_SET_ELEM( reader.ptr ))
            ((CvGraphVtx*)reader.ptr)->flags = flag_buf[vtx_count++];
        CV_NEXT_SEQ_ELEM( graph->elem_size, reader );
    }

    cvFree( &write_buf );
    cvFree( &flag_buf );
}


static void* icvReadGraph( CvFileStorage* fs, CvFileNode* node )
{
    void* ptr = 0;
    char* read_buf = 0;
    CvGraphVtx** vtx_buf = 0;
    CvGraph* graph;
    CvFileNode *header_node, *vtx_node, *edge_node;
    int flags, vtx_count, edge_count;
    int vtx_size = sizeof(CvGraphVtx), edge_size, header_size = sizeof(CvGraph);
    int src_vtx_size = 0, src_edge_size;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS], fmt_pair_count;
    int vtx_items_per_elem = 0, edge_items_per_elem = 0;
    int edge_user_align = sizeof(float);
    int read_buf_size;
    int i, k;
    const char* flags_str;
    const char* header_dt;
    const char* vtx_dt;
    const char* edge_dt;
    char* endptr = 0;

    flags_str = cvReadStringByName( fs, node, "flags", 0 );
    vtx_dt = cvReadStringByName( fs, node, "vertex_dt", 0 );
    edge_dt = cvReadStringByName( fs, node, "edge_dt", 0 );
    vtx_count = cvReadIntByName( fs, node, "vertex_count", -1 );
    edge_count = cvReadIntByName( fs, node, "edge_count", -1 );

    if( !flags_str || vtx_count == -1 || edge_count == -1 || !edge_dt )
        CV_Error( CV_StsError, "Some of essential graph attributes are absent" );

    flags = CV_SET_MAGIC_VAL + CV_GRAPH;

    if( isxdigit(flags_str[0]) )
    {
        const int OLD_SEQ_ELTYPE_BITS = 9;
        const int OLD_SEQ_KIND_BITS = 3;
        const int OLD_SEQ_FLAG_SHIFT = OLD_SEQ_KIND_BITS + OLD_SEQ_ELTYPE_BITS;
        const int OLD_GRAPH_FLAG_ORIENTED = 1 << OLD_SEQ_FLAG_SHIFT;

        int flags0 = (int)strtol( flags_str, &endptr, 16 );
        if( endptr == flags_str || (flags0 & CV_MAGIC_MASK) != CV_SET_MAGIC_VAL )
            CV_Error( CV_StsError, "The sequence flags are invalid" );
        if( flags0 & OLD_GRAPH_FLAG_ORIENTED )
            flags |= CV_GRAPH_FLAG_ORIENTED;
    }
    else
    {
        if( strstr(flags_str, "oriented") )
            flags |= CV_GRAPH_FLAG_ORIENTED;
    }

    header_dt = cvReadStringByName( fs, node, "header_dt", 0 );
    header_node = cvGetFileNodeByName( fs, node, "header_user_data" );

    if( (header_dt != 0) ^ (header_node != 0) )
        CV_Error( CV_StsError,
        "One of \"header_dt\" and \"header_user_data\" is there, while the other is not" );

    if( header_dt )
        header_size = icvCalcElemSize( header_dt, header_size );

    if( vtx_dt )
    {
        src_vtx_size = icvCalcElemSize( vtx_dt, 0 );
        vtx_size = icvCalcElemSize( vtx_dt, vtx_size );
        fmt_pair_count = icvDecodeFormat( edge_dt,
                            fmt_pairs, CV_FS_MAX_FMT_PAIRS );
        fmt_pair_count *= 2;
        for( i = 0; i < fmt_pair_count; i += 2 )
            vtx_items_per_elem += fmt_pairs[i];
    }

    {
        char dst_edge_dt_buf[128];
        const char* dst_edge_dt = 0;

        fmt_pair_count = icvDecodeFormat( edge_dt,
                            fmt_pairs, CV_FS_MAX_FMT_PAIRS );
        if( fmt_pair_count < 2 ||
            fmt_pairs[0] != 2 || fmt_pairs[1] != CV_32S ||
            fmt_pairs[2] < 1 || fmt_pairs[3] != CV_32F )
            CV_Error( CV_StsBadArg,
            "Graph edges should start with 2 integers and a float" );

        // alignment of user part of the edge data following 2if
        if( fmt_pair_count > 2 && CV_ELEM_SIZE(fmt_pairs[5]) >= (int)sizeof(double))
            edge_user_align = sizeof(double);

        fmt_pair_count *= 2;
        for( i = 0; i < fmt_pair_count; i += 2 )
            edge_items_per_elem += fmt_pairs[i];

        if( edge_dt[2] == 'f' || (edge_dt[2] == '1' && edge_dt[3] == 'f') )
            dst_edge_dt = edge_dt + 3 + cv_isdigit(edge_dt[2]);
        else
        {
            int val = (int)strtol( edge_dt + 2, &endptr, 10 );
            sprintf( dst_edge_dt_buf, "%df%s", val-1, endptr );
            dst_edge_dt = dst_edge_dt_buf;
        }

        edge_size = icvCalcElemSize( dst_edge_dt, sizeof(CvGraphEdge) );
        src_edge_size = icvCalcElemSize( edge_dt, 0 );
    }

    graph = cvCreateGraph( flags, header_size, vtx_size, edge_size, fs->dststorage );

    if( header_node )
    {
        CV_Assert(header_dt);
        cvReadRawData( fs, header_node, (char*)graph + sizeof(CvGraph), header_dt );
    }

    read_buf_size = MAX( src_vtx_size*3, 1 << 16 );
    read_buf_size = MAX( src_edge_size*3, read_buf_size );
    read_buf = (char*)cvAlloc( read_buf_size );
    vtx_buf = (CvGraphVtx**)cvAlloc( vtx_count * sizeof(vtx_buf[0]) );

    vtx_node = cvGetFileNodeByName( fs, node, "vertices" );
    edge_node = cvGetFileNodeByName( fs, node, "edges" );
    if( !edge_node )
        CV_Error( CV_StsBadArg, "No edges data" );
    if( vtx_dt && !vtx_node )
        CV_Error( CV_StsBadArg, "No vertices data" );

    // as vertices and edges are read in similar way,
    // do it as a parametrized 2-iteration loop
    for( k = 0; k < 2; k++ )
    {
        const char* dt = k == 0 ? vtx_dt : edge_dt;
        int elem_size = k == 0 ? vtx_size : edge_size;
        int src_elem_size = k == 0 ? src_vtx_size : src_edge_size;
        int items_per_elem = k == 0 ? vtx_items_per_elem : edge_items_per_elem;
        int elem_count = k == 0 ? vtx_count : edge_count;
        char* dst_ptr = read_buf;
        int read_max = read_buf_size /MAX(src_elem_size, 1), read_count = 0;
        CvSeqReader reader;
        if(dt)
            cvStartReadRawData( fs, k == 0 ? vtx_node : edge_node, &reader );

        for( i = 0; i < elem_count; i++ )
        {
            if( read_count == 0 && dt )
            {
                int count = MIN( elem_count - i, read_max )*items_per_elem;
                cvReadRawDataSlice( fs, &reader, count, read_buf, dt );
                read_count = count;
                dst_ptr = read_buf;
            }

            if( k == 0 )
            {
                CvGraphVtx* vtx;
                cvGraphAddVtx( graph, 0, &vtx );
                vtx_buf[i] = vtx;
                if( dt )
                    memcpy( vtx + 1, dst_ptr, src_elem_size );
            }
            else
            {
                CvGraphEdge* edge = 0;
                int vtx1 = ((int*)dst_ptr)[0];
                int vtx2 = ((int*)dst_ptr)[1];
                int result;

                if( (unsigned)vtx1 >= (unsigned)vtx_count ||
                    (unsigned)vtx2 >= (unsigned)vtx_count )
                    CV_Error( CV_StsOutOfRange,
                    "Some of stored vertex indices are out of range" );

                result = cvGraphAddEdgeByPtr( graph,
                    vtx_buf[vtx1], vtx_buf[vtx2], 0, &edge );

                if( result == 0 )
                    CV_Error( CV_StsBadArg, "Duplicated edge has occurred" );

                edge->weight = *(float*)(dst_ptr + sizeof(int)*2);
                if( elem_size > (int)sizeof(CvGraphEdge) )
                {
                    char* dst_ptr2 = (char*)cvAlignPtr( dst_ptr + sizeof(int)*2 +
                                                sizeof(float), edge_user_align );
                    memcpy( edge + 1, dst_ptr2, elem_size - sizeof(CvGraphEdge) );
                }
            }

            dst_ptr += src_elem_size;
            read_count--;
        }
    }

    ptr = graph;
    cvFree( &read_buf );
    cvFree( &vtx_buf );

    return ptr;
}

/****************************************************************************************\
*                                    RTTI Functions                                      *
\****************************************************************************************/

CvTypeInfo *CvType::first = 0, *CvType::last = 0;

CvType::CvType( const char* type_name,
                CvIsInstanceFunc is_instance, CvReleaseFunc release,
                CvReadFunc read, CvWriteFunc write, CvCloneFunc clone )
{
    CvTypeInfo _info;
    _info.flags = 0;
    _info.header_size = sizeof(_info);
    _info.type_name = type_name;
    _info.prev = _info.next = 0;
    _info.is_instance = is_instance;
    _info.release = release;
    _info.clone = clone;
    _info.read = read;
    _info.write = write;

    cvRegisterType( &_info );
    info = first;
}


CvType::~CvType()
{
    cvUnregisterType( info->type_name );
}


CvType seq_type( CV_TYPE_NAME_SEQ, icvIsSeq, icvReleaseSeq, icvReadSeq,
                 icvWriteSeqTree /* this is the entry point for
                 writing a single sequence too */, icvCloneSeq );

CvType seq_tree_type( CV_TYPE_NAME_SEQ_TREE, icvIsSeq, icvReleaseSeq,
                      icvReadSeqTree, icvWriteSeqTree, icvCloneSeq );

CvType seq_graph_type( CV_TYPE_NAME_GRAPH, icvIsGraph, icvReleaseGraph,
                       icvReadGraph, icvWriteGraph, icvCloneGraph );

CvType sparse_mat_type( CV_TYPE_NAME_SPARSE_MAT, icvIsSparseMat,
                        (CvReleaseFunc)cvReleaseSparseMat, icvReadSparseMat,
                        icvWriteSparseMat, (CvCloneFunc)cvCloneSparseMat );

CvType image_type( CV_TYPE_NAME_IMAGE, icvIsImage, (CvReleaseFunc)cvReleaseImage,
                   icvReadImage, icvWriteImage, (CvCloneFunc)cvCloneImage );

CvType mat_type( CV_TYPE_NAME_MAT, icvIsMat, (CvReleaseFunc)cvReleaseMat,
                 icvReadMat, icvWriteMat, (CvCloneFunc)cvCloneMat );

CvType matnd_type( CV_TYPE_NAME_MATND, icvIsMatND, (CvReleaseFunc)cvReleaseMatND,
                   icvReadMatND, icvWriteMatND, (CvCloneFunc)cvCloneMatND );
