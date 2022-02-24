// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef  struct  CvTsSimpleSeq
{
    schar* array;
    int   count;
    int   max_count;
    int   elem_size;
} CvTsSimpleSeq;


static CvTsSimpleSeq*  cvTsCreateSimpleSeq( int max_count, int elem_size )
{
    CvTsSimpleSeq* seq = (CvTsSimpleSeq*)cvAlloc( sizeof(*seq) + max_count * elem_size );
    seq->elem_size = elem_size;
    seq->max_count = max_count;
    seq->count = 0;
    seq->array = (schar*)(seq + 1);
    return seq;
}


static void cvTsReleaseSimpleSeq( CvTsSimpleSeq** seq )
{
    cvFree( seq );
}


static schar*  cvTsSimpleSeqElem( CvTsSimpleSeq* seq, int index )
{
    CV_Assert( 0 <= index && index < seq->count );
    return seq->array + index * seq->elem_size;
}


static void  cvTsClearSimpleSeq( CvTsSimpleSeq* seq )
{
    seq->count = 0;
}


static void cvTsSimpleSeqShiftAndCopy( CvTsSimpleSeq* seq, int from_idx, int to_idx, void* elem=0 )
{
    int elem_size = seq->elem_size;

    if( from_idx == to_idx )
        return;

    if (elem)
        CV_Assert(from_idx < to_idx);
    else
        CV_Assert(from_idx > to_idx);

    if( from_idx < seq->count )
    {
        memmove( seq->array + to_idx*elem_size, seq->array + from_idx*elem_size,
                (seq->count - from_idx)*elem_size );
    }
    seq->count += to_idx - from_idx;
    if( elem )
        memcpy( seq->array + from_idx*elem_size, elem, (to_idx - from_idx)*elem_size );
}

static void cvTsSimpleSeqInvert( CvTsSimpleSeq* seq )
{
    int i, k, len = seq->count, elem_size = seq->elem_size;
    schar *data = seq->array, t;

    for( i = 0; i < len/2; i++ )
    {
        schar* a = data + i*elem_size;
        schar* b = data + (len - i - 1)*elem_size;
        for( k = 0; k < elem_size; k++ )
            CV_SWAP( a[k], b[k], t );
    }
}

/****************************************************************************************\
 *                                simple cvset implementation                               *
 \****************************************************************************************/

typedef  struct  CvTsSimpleSet
{
    schar* array;
    int   count, max_count;
    int   elem_size;
    int*  free_stack;
    int   free_count;
} CvTsSimpleSet;


static void  cvTsClearSimpleSet( CvTsSimpleSet* set_header )
{
    int i;
    int elem_size = set_header->elem_size;

    for( i = 0; i < set_header->max_count; i++ )
    {
        set_header->array[i*elem_size] = 0;
        set_header->free_stack[i] = set_header->max_count - i - 1;
    }
    set_header->free_count = set_header->max_count;
    set_header->count = 0;
}


static CvTsSimpleSet*  cvTsCreateSimpleSet( int max_count, int elem_size )
{
    CvTsSimpleSet* set_header = (CvTsSimpleSet*)cvAlloc( sizeof(*set_header) + max_count *
                                                        (elem_size + 1 + sizeof(int)));
    set_header->elem_size = elem_size + 1;
    set_header->max_count = max_count;
    set_header->free_stack = (int*)(set_header + 1);
    set_header->array = (schar*)(set_header->free_stack + max_count);

    cvTsClearSimpleSet( set_header );
    return set_header;
}


static void cvTsReleaseSimpleSet( CvTsSimpleSet** set_header )
{
    cvFree( set_header );
}


static schar*  cvTsSimpleSetFind( CvTsSimpleSet* set_header, int index )
{
    int idx = index * set_header->elem_size;
    CV_Assert( 0 <= index && index < set_header->max_count );
    return set_header->array[idx] ? set_header->array + idx + 1 : 0;
}


static int  cvTsSimpleSetAdd( CvTsSimpleSet* set_header, void* elem )
{
    int idx, idx2;
    CV_Assert( set_header->free_count > 0 );

    idx = set_header->free_stack[--set_header->free_count];
    idx2 = idx * set_header->elem_size;
    CV_Assert( set_header->array[idx2] == 0 );
    set_header->array[idx2] = 1;
    if( set_header->elem_size > 1 )
        memcpy( set_header->array + idx2 + 1, elem, set_header->elem_size - 1 );
    set_header->count = MAX( set_header->count, idx + 1 );

    return idx;
}


static void  cvTsSimpleSetRemove( CvTsSimpleSet* set_header, int index )
{
    CV_Assert( set_header->free_count < set_header->max_count &&
               0 <= index && index < set_header->max_count );
    CV_Assert( set_header->array[index * set_header->elem_size] == 1 );

    set_header->free_stack[set_header->free_count++] = index;
    set_header->array[index * set_header->elem_size] = 0;
}


/****************************************************************************************\
 *                              simple graph implementation                               *
 \****************************************************************************************/

typedef  struct  CvTsSimpleGraph
{
    char* matrix;
    int   edge_size;
    int   oriented;
    CvTsSimpleSet* vtx;
} CvTsSimpleGraph;


static void  cvTsClearSimpleGraph( CvTsSimpleGraph* graph )
{
    int max_vtx_count = graph->vtx->max_count;
    cvTsClearSimpleSet( graph->vtx );
    memset( graph->matrix, 0, max_vtx_count * max_vtx_count * graph->edge_size );
}


static CvTsSimpleGraph*  cvTsCreateSimpleGraph( int max_vtx_count, int vtx_size,
                                               int edge_size, int oriented )
{
    CvTsSimpleGraph* graph;

    CV_Assert( max_vtx_count > 1 && vtx_size >= 0 && edge_size >= 0 );
    graph = (CvTsSimpleGraph*)cvAlloc( sizeof(*graph) +
                                      max_vtx_count * max_vtx_count * (edge_size + 1));
    graph->vtx = cvTsCreateSimpleSet( max_vtx_count, vtx_size );
    graph->edge_size = edge_size + 1;
    graph->matrix = (char*)(graph + 1);
    graph->oriented = oriented;

    cvTsClearSimpleGraph( graph );
    return graph;
}


static void cvTsReleaseSimpleGraph( CvTsSimpleGraph** graph )
{
    if( *graph )
    {
        cvTsReleaseSimpleSet( &(graph[0]->vtx) );
        cvFree( graph );
    }
}


static int  cvTsSimpleGraphAddVertex( CvTsSimpleGraph* graph, void* vertex )
{
    return cvTsSimpleSetAdd( graph->vtx, vertex );
}


static void  cvTsSimpleGraphRemoveVertex( CvTsSimpleGraph* graph, int index )
{
    int i, max_vtx_count = graph->vtx->max_count;
    int edge_size = graph->edge_size;
    cvTsSimpleSetRemove( graph->vtx, index );

    /* remove all the corresponding edges */
    for( i = 0; i < max_vtx_count; i++ )
    {
        graph->matrix[(i*max_vtx_count + index)*edge_size] =
        graph->matrix[(index*max_vtx_count + i)*edge_size] = 0;
    }
}


static void cvTsSimpleGraphAddEdge( CvTsSimpleGraph* graph, int idx1, int idx2, void* edge )
{
    int i, t, n = graph->oriented ? 1 : 2;

    CV_Assert( cvTsSimpleSetFind( graph->vtx, idx1 ) &&
               cvTsSimpleSetFind( graph->vtx, idx2 ));

    for( i = 0; i < n; i++ )
    {
        int ofs = (idx1*graph->vtx->max_count + idx2)*graph->edge_size;
        CV_Assert( graph->matrix[ofs] == 0 );
        graph->matrix[ofs] = 1;
        if( graph->edge_size > 1 )
            memcpy( graph->matrix + ofs + 1, edge, graph->edge_size - 1 );

        CV_SWAP( idx1, idx2, t );
    }
}


static void  cvTsSimpleGraphRemoveEdge( CvTsSimpleGraph* graph, int idx1, int idx2 )
{
    int i, t, n = graph->oriented ? 1 : 2;

    CV_Assert( cvTsSimpleSetFind( graph->vtx, idx1 ) &&
           cvTsSimpleSetFind( graph->vtx, idx2 ));

    for( i = 0; i < n; i++ )
    {
        int ofs = (idx1*graph->vtx->max_count + idx2)*graph->edge_size;
        CV_Assert( graph->matrix[ofs] == 1 );
        graph->matrix[ofs] = 0;
        CV_SWAP( idx1, idx2, t );
    }
}


static schar*  cvTsSimpleGraphFindVertex( CvTsSimpleGraph* graph, int index )
{
    return cvTsSimpleSetFind( graph->vtx, index );
}


static char*  cvTsSimpleGraphFindEdge( CvTsSimpleGraph* graph, int idx1, int idx2 )
{
    if( cvTsSimpleGraphFindVertex( graph, idx1 ) &&
       cvTsSimpleGraphFindVertex( graph, idx2 ))
    {
        char* edge = graph->matrix + (idx1 * graph->vtx->max_count + idx2)*graph->edge_size;
        if( edge[0] ) return edge + 1;
    }
    return 0;
}


static int  cvTsSimpleGraphVertexDegree( CvTsSimpleGraph* graph, int index )
{
    int i, count = 0;
    int edge_size = graph->edge_size;
    int max_vtx_count = graph->vtx->max_count;
    CV_Assert( cvTsSimpleGraphFindVertex( graph, index ) != 0 );

    for( i = 0; i < max_vtx_count; i++ )
    {
        count += graph->matrix[(i*max_vtx_count + index)*edge_size] +
        graph->matrix[(index*max_vtx_count + i)*edge_size];
    }

    if( !graph->oriented )
    {
        CV_Assert( count % 2 == 0 );
        count /= 2;
    }
    return count;
}


///////////////////////////////////// the tests //////////////////////////////////

#define CV_TS_SEQ_CHECK_CONDITION( expr, err_msg )          \
if( !(expr) )                                               \
{                                                           \
set_error_context( #expr, err_msg, __FILE__, __LINE__ );    \
ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );\
throw -1;                                                   \
}

class Core_DynStructBaseTest : public cvtest::BaseTest
{
public:
    Core_DynStructBaseTest();
    virtual ~Core_DynStructBaseTest();
    bool can_do_fast_forward();
    void clear();

protected:
    int read_params( const cv::FileStorage& fs );
    void run_func(void);
    void set_error_context( const char* condition,
                           const char* err_msg,
                           const char* file, int line );
    int test_seq_block_consistence( int _struct_idx, CvSeq* seq, int total );
    void update_progressbar();

    int struct_count, max_struct_size, iterations, generations;
    int min_log_storage_block_size, max_log_storage_block_size;
    int min_log_elem_size, max_log_elem_size;
    int gen, struct_idx, iter;
    int test_progress;
    int64 start_time;
    double cpu_freq;
    vector<void*> cxcore_struct;
    vector<void*> simple_struct;
    Ptr<CvMemStorage> storage;
};


Core_DynStructBaseTest::Core_DynStructBaseTest()
{
    struct_count = 2;
    max_struct_size = 2000;
    min_log_storage_block_size = 7;
    max_log_storage_block_size = 12;
    min_log_elem_size = 0;
    max_log_elem_size = 8;
    generations = 10;
    iterations = max_struct_size*2;
    gen = struct_idx = iter = -1;
    test_progress = -1;
}


Core_DynStructBaseTest::~Core_DynStructBaseTest()
{
    clear();
}


void Core_DynStructBaseTest::run_func()
{
}

bool Core_DynStructBaseTest::can_do_fast_forward()
{
    return false;
}


void Core_DynStructBaseTest::clear()
{
    cvtest::BaseTest::clear();
}


int Core_DynStructBaseTest::read_params( const cv::FileStorage& fs )
{
    int code = cvtest::BaseTest::read_params( fs );
    double sqrt_scale = sqrt(ts->get_test_case_count_scale());
    if( code < 0 )
        return code;

    read( find_param( fs, "struct_count" ), struct_count, struct_count );
    read( find_param( fs, "max_struct_size" ), max_struct_size, max_struct_size );
    read( find_param( fs, "generations" ), generations, generations );
    read( find_param( fs, "iterations" ), iterations, iterations );
    generations = cvRound(generations*sqrt_scale);
    iterations = cvRound(iterations*sqrt_scale);

    read( find_param( fs, "min_log_storage_block_size" ),
              min_log_storage_block_size, min_log_storage_block_size );
    read( find_param( fs, "max_log_storage_block_size" ),
              max_log_storage_block_size, max_log_storage_block_size );
    read( find_param( fs, "min_log_elem_size" ), min_log_elem_size, min_log_elem_size );
    read( find_param( fs, "max_log_elem_size" ), max_log_elem_size, max_log_elem_size );

    struct_count = cvtest::clipInt( struct_count, 1, 100 );
    max_struct_size = cvtest::clipInt( max_struct_size, 1, 1<<20 );
    generations = cvtest::clipInt( generations, 1, 100 );
    iterations = cvtest::clipInt( iterations, 100, 1<<20 );

    min_log_storage_block_size = cvtest::clipInt( min_log_storage_block_size, 7, 20 );
    max_log_storage_block_size = cvtest::clipInt( max_log_storage_block_size,
                                             min_log_storage_block_size, 20 );

    min_log_elem_size = cvtest::clipInt( min_log_elem_size, 0, 8 );
    max_log_elem_size = cvtest::clipInt( max_log_elem_size, min_log_elem_size, 10 );

    return 0;
}


void Core_DynStructBaseTest::update_progressbar()
{
    int64 t;

    if( test_progress < 0 )
    {
        test_progress = 0;
        cpu_freq = cv::getTickFrequency();
        start_time = cv::getTickCount();
    }

    t = cv::getTickCount();
    test_progress = update_progress( test_progress, 0, 0, (double)(t - start_time)/cpu_freq );
}


void Core_DynStructBaseTest::set_error_context( const char* condition,
                                                const char* err_msg,
                                                const char* filename, int lineno )
{
    ts->printf( cvtest::TS::LOG, "file %s, line %d: %s\n(\"%s\" failed).\n"
               "generation = %d, struct_idx = %d, iter = %d\n",
               filename, lineno, err_msg, condition, gen, struct_idx, iter );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
}


int Core_DynStructBaseTest::test_seq_block_consistence( int _struct_idx, CvSeq* seq, int total )
{
    int sum = 0;
    struct_idx = _struct_idx;

    CV_TS_SEQ_CHECK_CONDITION( seq != 0, "Null sequence pointer" );

    if( seq->first )
    {
        CvSeqBlock* block = seq->first;
        CvSeqBlock* prev_block = block->prev;

        int delta_idx = seq->first->start_index;

        for( ;; )
        {
            CV_TS_SEQ_CHECK_CONDITION( sum == block->start_index - delta_idx &&
                                      block->count > 0 && block->prev == prev_block &&
                                      prev_block->next == block,
                                      "sequence blocks are inconsistent" );
            sum += block->count;
            prev_block = block;
            block = block->next;
            if( block == seq->first ) break;
        }

        CV_TS_SEQ_CHECK_CONDITION( block->prev->count * seq->elem_size +
                                  block->prev->data <= seq->block_max,
                                  "block->data or block_max pointer are incorrect" );
    }

    CV_TS_SEQ_CHECK_CONDITION( seq->total == sum && sum == total,
                              "total number of elements is incorrect" );

    return 0;
}


/////////////////////////////////// sequence tests ////////////////////////////////////

class Core_SeqBaseTest : public Core_DynStructBaseTest
{
public:
    Core_SeqBaseTest();
    virtual ~Core_SeqBaseTest();
    void clear();
    void run( int );

protected:
    int test_multi_create();
    int test_get_seq_elem( int _struct_idx, int iters );
    int test_get_seq_reading( int _struct_idx, int iters );
    int test_seq_ops( int iters );
};

Core_SeqBaseTest::Core_SeqBaseTest()
{
}

Core_SeqBaseTest::~Core_SeqBaseTest()
{
    clear();
}

void Core_SeqBaseTest::clear()
{
    for( size_t i = 0; i < simple_struct.size(); i++ )
        cvTsReleaseSimpleSeq( (CvTsSimpleSeq**)&simple_struct[i] );
    Core_DynStructBaseTest::clear();
}


int Core_SeqBaseTest::test_multi_create()
{
    vector<CvSeqWriter> writer(struct_count);
    vector<int> pos(struct_count);
    vector<int> index(struct_count);
    int  cur_count, elem_size;
    RNG& rng = ts->get_rng();

    for( int i = 0; i < struct_count; i++ )
    {
        double t;
        CvTsSimpleSeq* sseq;

        pos[i] = -1;
        index[i] = i;

        t = cvtest::randReal(rng)*(max_log_elem_size - min_log_elem_size) + min_log_elem_size;
        elem_size = cvRound( exp(t * CV_LOG2) );
        elem_size = MIN( elem_size, (int)(storage->block_size - sizeof(void*) -
                                          sizeof(CvSeqBlock) - sizeof(CvMemBlock)) );

        cvTsReleaseSimpleSeq( (CvTsSimpleSeq**)&simple_struct[i] );
        simple_struct[i] = sseq = cvTsCreateSimpleSeq( max_struct_size, elem_size );
        cxcore_struct[i] = 0;
        sseq->count = cvtest::randInt( rng ) % max_struct_size;
        Mat m( 1, MAX(sseq->count,1)*elem_size, CV_8UC1, sseq->array );
        cvtest::randUni( rng, m, Scalar::all(0), Scalar::all(256) );
    }

    for( cur_count = struct_count; cur_count > 0; cur_count-- )
    {
        for(;;)
        {
            int k = cvtest::randInt( rng ) % cur_count;
            struct_idx = index[k];
            CvTsSimpleSeq* sseq = (CvTsSimpleSeq*)simple_struct[struct_idx];

            if( pos[struct_idx] < 0 )
            {
                int hdr_size = (cvtest::randInt(rng) % 10)*4 + sizeof(CvSeq);
                hdr_size = MIN( hdr_size, (int)(storage->block_size - sizeof(CvMemBlock)) );
                elem_size = sseq->elem_size;

                if( cvtest::randInt(rng) % 2 )
                {
                    cvStartWriteSeq( 0, hdr_size, elem_size, storage, &writer[struct_idx] );
                }
                else
                {
                    CvSeq* s;
                    s = cvCreateSeq( 0, hdr_size, elem_size, storage );
                    cvStartAppendToSeq( s, &writer[struct_idx] );
                }

                cvSetSeqBlockSize( writer[struct_idx].seq, cvtest::randInt( rng ) % 10000 );
                pos[struct_idx] = 0;
            }

            update_progressbar();
            if( pos[struct_idx] == sseq->count )
            {
                cxcore_struct[struct_idx] = cvEndWriteSeq( &writer[struct_idx] );
                /* del index */
                for( ; k < cur_count-1; k++ )
                    index[k] = index[k+1];
                break;
            }

            {
                schar* el = cvTsSimpleSeqElem( sseq, pos[struct_idx] );
                CV_WRITE_SEQ_ELEM_VAR( el, writer[struct_idx] );
            }
            pos[struct_idx]++;
        }
    }

    return 0;
}


int  Core_SeqBaseTest::test_get_seq_elem( int _struct_idx, int iters )
{
    RNG& rng = ts->get_rng();

    CvSeq* seq = (CvSeq*)cxcore_struct[_struct_idx];
    CvTsSimpleSeq* sseq = (CvTsSimpleSeq*)simple_struct[_struct_idx];
    struct_idx = _struct_idx;

    CV_Assert( seq->total == sseq->count );

    if( sseq->count == 0 )
        return 0;

    for( int i = 0; i < iters; i++ )
    {
        int idx = cvtest::randInt(rng) % (sseq->count*3) - sseq->count*3/2;
        int idx0 = (unsigned)idx < (unsigned)(sseq->count) ? idx : idx < 0 ?
        idx + sseq->count : idx - sseq->count;
        int bad_range = (unsigned)idx0 >= (unsigned)(sseq->count);
        schar* elem;
         elem = cvGetSeqElem( seq, idx );

        if( bad_range )
        {
            CV_TS_SEQ_CHECK_CONDITION( elem == 0,
                                      "cvGetSeqElem doesn't "
                                      "handle \"out of range\" properly" );
        }
        else
        {
            CV_TS_SEQ_CHECK_CONDITION( elem != 0 &&
                                      !memcmp( elem, cvTsSimpleSeqElem(sseq, idx0), sseq->elem_size ),
                                      "cvGetSeqElem returns wrong element" );

             idx = cvSeqElemIdx(seq, elem );
            CV_TS_SEQ_CHECK_CONDITION( idx >= 0 && idx == idx0,
                                      "cvSeqElemIdx is incorrect" );
        }
    }

    return 0;
}


int  Core_SeqBaseTest::test_get_seq_reading( int _struct_idx, int iters )
{
    const int max_val = 3*5 + 2;
    CvSeq* seq = (CvSeq*)cxcore_struct[_struct_idx];
    CvTsSimpleSeq* sseq = (CvTsSimpleSeq*)simple_struct[_struct_idx];
    int total = seq->total;
    RNG& rng = ts->get_rng();
    CvSeqReader reader;
    vector<schar> _elem(sseq->elem_size);
    schar* elem = &_elem[0];

    CV_Assert( total == sseq->count );
    this->struct_idx = _struct_idx;

    int pos = cvtest::randInt(rng) % 2;
    cvStartReadSeq( seq, &reader, pos );

    if( total == 0 )
    {
        CV_TS_SEQ_CHECK_CONDITION( reader.ptr == 0, "Empty sequence reader pointer is not NULL" );
        return 0;
    }

    pos = pos ? seq->total - 1 : 0;

    CV_TS_SEQ_CHECK_CONDITION( pos == cvGetSeqReaderPos(&reader),
                              "initial reader position is wrong" );

    for( iter = 0; iter < iters; iter++ )
    {
        int op = cvtest::randInt(rng) % max_val;

        if( op >= max_val - 2 )
        {
            int new_pos, new_pos0;
            int bad_range;
            int is_relative = op == max_val - 1;

            new_pos = cvtest::randInt(rng) % (total*2) - total;
            new_pos0 = new_pos + (is_relative ? pos : 0 );

            if( new_pos0 < 0 ) new_pos0 += total;
            if( new_pos0 >= total ) new_pos0 -= total;

            bad_range = (unsigned)new_pos0 >= (unsigned)total;
             cvSetSeqReaderPos( &reader, new_pos, is_relative );

            if( !bad_range )
            {
                CV_TS_SEQ_CHECK_CONDITION( new_pos0 == cvGetSeqReaderPos( &reader ),
                                          "cvset reader position doesn't work" );
                pos = new_pos0;
            }
            else
            {
                CV_TS_SEQ_CHECK_CONDITION( pos == cvGetSeqReaderPos( &reader ),
                                          "reader doesn't stay at the current position after wrong positioning" );
            }
        }
        else
        {
            int direction = (op % 3) - 1;
            memcpy( elem, reader.ptr, sseq->elem_size );

            if( direction > 0 )
            {
                CV_NEXT_SEQ_ELEM( sseq->elem_size, reader );
            }
            else if( direction < 0 )
            {
                CV_PREV_SEQ_ELEM( sseq->elem_size, reader );
            }

            CV_TS_SEQ_CHECK_CONDITION( memcmp(elem, cvTsSimpleSeqElem(sseq, pos),
                                              sseq->elem_size) == 0, "reading is incorrect" );
            pos += direction;
            if( -pos > 0 ) pos += total;
            if( pos >= total ) pos -= total;

            CV_TS_SEQ_CHECK_CONDITION( pos == cvGetSeqReaderPos( &reader ),
                                      "reader doesn't move correctly after reading" );
        }
    }

    return 0;
}


int  Core_SeqBaseTest::test_seq_ops( int iters )
{
    const int max_op = 14;
    int max_elem_size = 0;
    schar* elem2 = 0;
    RNG& rng = ts->get_rng();

    for( int i = 0; i < struct_count; i++ )
        max_elem_size = MAX( max_elem_size, ((CvSeq*)cxcore_struct[i])->elem_size );

    vector<schar> elem_buf(max_struct_size*max_elem_size);
    schar* elem = (schar*)&elem_buf[0];
    Mat elem_mat;

    for( iter = 0; iter < iters; iter++ )
    {
        struct_idx = cvtest::randInt(rng) % struct_count;
        int op = cvtest::randInt(rng) % max_op;
        CvSeq* seq = (CvSeq*)cxcore_struct[struct_idx];
        CvTsSimpleSeq* sseq = (CvTsSimpleSeq*)simple_struct[struct_idx];
        int elem_size = sseq->elem_size;
        int whence = 0, pos = 0, count = 0;

        switch( op )
        {
            case 0:
            case 1:
            case 2:  // push/pushfront/insert
                if( sseq->count == sseq->max_count )
                    break;

                elem_mat = Mat(1, elem_size, CV_8U, elem);
                cvtest::randUni( rng, elem_mat, cvScalarAll(0), cvScalarAll(255) );

                whence = op - 1;
                if( whence < 0 )
                {
                    pos = 0;
                    cvSeqPushFront( seq, elem );
                }
                else if( whence > 0 )
                {
                    pos = sseq->count;
                    cvSeqPush( seq, elem );
                }
                else
                {
                    pos = cvtest::randInt(rng) % (sseq->count + 1);
                    cvSeqInsert( seq, pos, elem );
                }

                cvTsSimpleSeqShiftAndCopy( sseq, pos, pos + 1, elem );
                elem2 = cvGetSeqElem( seq, pos );
                CV_TS_SEQ_CHECK_CONDITION( elem2 != 0, "The inserted element could not be retrieved" );
                CV_TS_SEQ_CHECK_CONDITION( seq->total == sseq->count &&
                                          memcmp(elem2, cvTsSimpleSeqElem(sseq,pos), elem_size) == 0,
                                          "The inserted sequence element is wrong" );
                break;

            case 3:
            case 4:
            case 5: // pop/popfront/remove
                if( sseq->count == 0 )
                    break;

                whence = op - 4;
                if( whence < 0 )
                {
                    pos = 0;
                     cvSeqPopFront( seq, elem );
                }
                else if( whence > 0 )
                {
                    pos = sseq->count-1;
                     cvSeqPop( seq, elem );
                }
                else
                {
                    pos = cvtest::randInt(rng) % sseq->count;
                     cvSeqRemove( seq, pos );
                }

                if( whence != 0 )
                    CV_TS_SEQ_CHECK_CONDITION( seq->total == sseq->count - 1 &&
                                              memcmp( elem, cvTsSimpleSeqElem(sseq,pos), elem_size) == 0,
                                              "The popped sequence element isn't correct" );

                cvTsSimpleSeqShiftAndCopy( sseq, pos + 1, pos );

                if( sseq->count > 0 )
                {
                     elem2 = cvGetSeqElem( seq, pos < sseq->count ? pos : -1 );
                    CV_TS_SEQ_CHECK_CONDITION( elem2 != 0, "GetSeqElem fails after removing the element" );

                    CV_TS_SEQ_CHECK_CONDITION( memcmp( elem2,
                                                      cvTsSimpleSeqElem(sseq, pos - (pos == sseq->count)), elem_size) == 0,
                                              "The first shifted element is not correct after removing another element" );
                }
                else
                {
                    CV_TS_SEQ_CHECK_CONDITION( seq->first == 0,
                                              "The sequence doesn't become empty after the final remove" );
                }
                break;

            case 6:
            case 7:
            case 8: // push [front] multi/insert slice
                if( sseq->count == sseq->max_count )
                    break;

                count = cvtest::randInt( rng ) % (sseq->max_count - sseq->count + 1);
                elem_mat = Mat(1, MAX(count,1) * elem_size, CV_8U, elem);
                cvtest::randUni( rng, elem_mat, cvScalarAll(0), cvScalarAll(255) );

                whence = op - 7;
                pos = whence < 0 ? 0 : whence > 0 ? sseq->count : (int)(cvtest::randInt(rng) % (sseq->count+1));
                if( whence != 0 )
                {
                     cvSeqPushMulti( seq, elem, count, whence < 0 );
                }
                else
                {
                    CvSeq header;
                    CvSeqBlock block;
                    cvMakeSeqHeaderForArray( CV_SEQ_KIND_GENERIC, sizeof(CvSeq),
                                                     sseq->elem_size,
                                                     elem, count,
                                                     &header, &block );

                    cvSeqInsertSlice( seq, pos, &header );
                }
                cvTsSimpleSeqShiftAndCopy( sseq, pos, pos + count, elem );

                if( sseq->count > 0 )
                {
                    // choose the random element among the added
                    pos = count > 0 ? (int)(cvtest::randInt(rng) % count + pos) : MAX(pos-1,0);
                    elem2 = cvGetSeqElem( seq, pos );
                    CV_TS_SEQ_CHECK_CONDITION( elem2 != 0, "multi push operation doesn't add elements" );
                    CV_TS_SEQ_CHECK_CONDITION( seq->total == sseq->count &&
                                              memcmp( elem2, cvTsSimpleSeqElem(sseq,pos), elem_size) == 0,
                                              "One of the added elements is wrong" );
                }
                else
                {
                    CV_TS_SEQ_CHECK_CONDITION( seq->total == 0 && seq->first == 0,
                                              "Adding no elements to empty sequence fails" );
                }
                break;

            case 9:
            case 10:
            case 11: // pop [front] multi
                if( sseq->count == 0 )
                    break;

                count = cvtest::randInt(rng) % (sseq->count+1);
                whence = op - 10;
                pos = whence < 0 ? 0 : whence > 0 ? sseq->count - count :
                    (int)(cvtest::randInt(rng) % (sseq->count - count + 1));

                if( whence != 0 )
                {
                     cvSeqPopMulti( seq, elem, count, whence < 0 );

                    if( count > 0 )
                    {
                        CV_TS_SEQ_CHECK_CONDITION( memcmp(elem,
                                                          cvTsSimpleSeqElem(sseq,pos), elem_size) == 0,
                                                  "The first (in the sequence order) removed element is wrong after popmulti" );
                    }
                }
                else
                {
                     cvSeqRemoveSlice( seq, cvSlice(pos, pos + count) );
                }

                CV_TS_SEQ_CHECK_CONDITION( seq->total == sseq->count - count,
                                          "The popmulti left a wrong number of elements in the sequence" );

                cvTsSimpleSeqShiftAndCopy( sseq, pos + count, pos, 0 );
                if( sseq->count > 0 )
                {
                    pos = whence < 0 ? 0 : MIN( pos, sseq->count - 1 );
                    elem2 = cvGetSeqElem( seq, pos );
                    CV_TS_SEQ_CHECK_CONDITION( elem2 &&
                                              memcmp( elem2, cvTsSimpleSeqElem(sseq,pos), elem_size) == 0,
                                              "The last sequence element is wrong after POP" );
                }
                else
                {
                    CV_TS_SEQ_CHECK_CONDITION( seq->total == 0 && seq->first == 0,
                                              "The sequence doesn't become empty after final POP" );
                }
                break;
            case 12: // seqslice
            {
                CvMemStoragePos storage_pos;
                cvSaveMemStoragePos( storage, &storage_pos );

                int copy_data = cvtest::randInt(rng) % 2;
                count = cvtest::randInt(rng) % (seq->total + 1);
                pos = cvtest::randInt(rng) % (seq->total - count + 1);
                CvSeq* seq_slice = cvSeqSlice( seq, cvSlice(pos, pos + count), storage, copy_data );

                CV_TS_SEQ_CHECK_CONDITION( seq_slice && seq_slice->total == count,
                                          "cvSeqSlice returned incorrect slice" );

                if( count > 0 )
                {
                    int test_idx = cvtest::randInt(rng) % count;
                    elem2 = cvGetSeqElem( seq_slice, test_idx );
                    schar* elem3 = cvGetSeqElem( seq, pos + test_idx );
                    CV_TS_SEQ_CHECK_CONDITION( elem2 &&
                                              memcmp( elem2, cvTsSimpleSeqElem(sseq,pos + test_idx), elem_size) == 0,
                                              "The extracted slice elements are not correct" );
                    CV_TS_SEQ_CHECK_CONDITION( (elem2 == elem3) ^ copy_data,
                                              "copy_data flag is handled incorrectly" );
                }

                cvRestoreMemStoragePos( storage, &storage_pos );
            }
                break;
            case 13: // clear
                cvTsClearSimpleSeq( sseq );
                cvClearSeq( seq );
                CV_TS_SEQ_CHECK_CONDITION( seq->total == 0 && seq->first == 0,
                                          "The sequence doesn't become empty after clear" );
                break;
            default:
                CV_Assert(0);
                return -1;
        }

        if( test_seq_block_consistence(struct_idx, seq, sseq->count) < 0 )
            return -1;

        if( test_get_seq_elem(struct_idx, 7) < 0 )
            return -1;

        update_progressbar();
    }

    return 0;
}


void Core_SeqBaseTest::run( int )
{
    try
    {
        RNG& rng = ts->get_rng();
        int i;
        double t;

        clear();
        test_progress = -1;

        simple_struct.resize(struct_count, 0);
        cxcore_struct.resize(struct_count, 0);

        for( gen = 0; gen < generations; gen++ )
        {
            struct_idx = iter = -1;

            if( !storage )
            {
                t = cvtest::randReal(rng)*(max_log_storage_block_size - min_log_storage_block_size)
                + min_log_storage_block_size;
                storage.reset(cvCreateMemStorage( cvRound( exp(t * CV_LOG2) ) ));
            }

            iter = struct_idx = -1;
            test_multi_create();

            for( i = 0; i < struct_count; i++ )
            {
                if( test_seq_block_consistence(i, (CvSeq*)cxcore_struct[i],
                                               ((CvTsSimpleSeq*)simple_struct[i])->count) < 0 )
                    return;

                if( test_get_seq_elem( i, MAX(iterations/3,7) ) < 0 )
                    return;

                if( test_get_seq_reading( i, MAX(iterations/3,7) ) < 0 )
                    return;
                update_progressbar();
            }

            if( test_seq_ops( iterations ) < 0 )
                return;

            if( cvtest::randInt(rng) % 2 )
                storage.release();
            else
                cvClearMemStorage( storage );
        }
    }
    catch(const int &)
    {
    }
}


////////////////////////////// more sequence tests //////////////////////////////////////

class Core_SeqSortInvTest : public Core_SeqBaseTest
{
public:
    Core_SeqSortInvTest();
    void run( int );

protected:
};


Core_SeqSortInvTest::Core_SeqSortInvTest()
{
}


static int icvCmpSeqElems( const void* a, const void* b, void* userdata )
{
    return memcmp( a, b, ((CvSeq*)userdata)->elem_size );
}

static int icvCmpSeqElems2_elem_size = 0;
static int icvCmpSeqElems2( const void* a, const void* b )
{
    return memcmp( a, b, icvCmpSeqElems2_elem_size );
}


void Core_SeqSortInvTest::run( int )
{
    try
    {
        RNG& rng = ts->get_rng();
        int i, k;
        double t;
        schar *elem0, *elem, *elem2;
        vector<uchar> buffer;

        clear();
        test_progress = -1;

        simple_struct.resize(struct_count, 0);
        cxcore_struct.resize(struct_count, 0);

        for( gen = 0; gen < generations; gen++ )
        {
            struct_idx = iter = -1;

            if( !storage )
            {
                t = cvtest::randReal(rng)*(max_log_storage_block_size - min_log_storage_block_size)
                + min_log_storage_block_size;
                storage.reset(cvCreateMemStorage( cvRound( exp(t * CV_LOG2) ) ));
            }

            for( iter = 0; iter < iterations/10; iter++ )
            {
                int max_size = 0;
                test_multi_create();

                for( i = 0; i < struct_count; i++ )
                {
                    CvTsSimpleSeq* sseq = (CvTsSimpleSeq*)simple_struct[i];
                    max_size = MAX( max_size, sseq->count*sseq->elem_size );
                }

                buffer.resize(max_size);

                for( i = 0; i < struct_count; i++ )
                {
                    CvSeq* seq = (CvSeq*)cxcore_struct[i];
                    CvTsSimpleSeq* sseq = (CvTsSimpleSeq*)simple_struct[i];
                    CvSlice slice = CV_WHOLE_SEQ;

                    //printf("%d. %d. %d-th size = %d\n", gen, iter, i, sseq->count );

                    cvSeqInvert( seq );
                    cvTsSimpleSeqInvert( sseq );

                    if( test_seq_block_consistence( i, seq, sseq->count ) < 0 )
                        return;

                    if( sseq->count > 0 && cvtest::randInt(rng) % 2 == 0 )
                    {
                        slice.end_index = cvtest::randInt(rng) % sseq->count + 1;
                        slice.start_index = cvtest::randInt(rng) % (sseq->count - slice.end_index + 1);
                        slice.end_index += slice.start_index;
                    }

                    cvCvtSeqToArray( seq, &buffer[0], slice );

                    slice.end_index = MIN( slice.end_index, sseq->count );
                    CV_TS_SEQ_CHECK_CONDITION( sseq->count == 0 || memcmp( &buffer[0],
                                                                          sseq->array + slice.start_index*sseq->elem_size,
                                                                          (slice.end_index - slice.start_index)*sseq->elem_size ) == 0,
                                              "cvSeqInvert returned wrong result" );

                    for( k = 0; k < (sseq->count > 0 ? 10 : 0); k++ )
                    {
                        int idx0 = cvtest::randInt(rng) % sseq->count, idx = 0;
                        elem0 = cvTsSimpleSeqElem( sseq, idx0 );
                        elem = cvGetSeqElem( seq, idx0 );
                        elem2 = cvSeqSearch( seq, elem0, k % 2 ? icvCmpSeqElems : 0, 0, &idx, seq );

                        CV_TS_SEQ_CHECK_CONDITION( elem != 0 &&
                                                  memcmp( elem0, elem, seq->elem_size ) == 0,
                                                  "cvSeqInvert gives incorrect result" );
                        CV_TS_SEQ_CHECK_CONDITION( elem2 != 0 &&
                                                  memcmp( elem0, elem2, seq->elem_size ) == 0 &&
                                                  elem2 == cvGetSeqElem( seq, idx ),
                                                  "cvSeqSearch failed (linear search)" );
                    }

                    cvSeqSort( seq, icvCmpSeqElems, seq );

                    if( test_seq_block_consistence( i, seq, sseq->count ) < 0 )
                        return;

                    if( sseq->count > 0 )
                    {
                        // !!! This is not thread-safe !!!
                        icvCmpSeqElems2_elem_size = sseq->elem_size;
                        qsort( sseq->array, sseq->count, sseq->elem_size, icvCmpSeqElems2 );

                        if( cvtest::randInt(rng) % 2 == 0 )
                        {
                            slice.end_index = cvtest::randInt(rng) % sseq->count + 1;
                            slice.start_index = cvtest::randInt(rng) % (sseq->count - slice.end_index + 1);
                            slice.end_index += slice.start_index;
                        }
                    }

                    cvCvtSeqToArray( seq, &buffer[0], slice );
                    CV_TS_SEQ_CHECK_CONDITION( sseq->count == 0 || memcmp( &buffer[0],
                                                                          sseq->array + slice.start_index*sseq->elem_size,
                                                                          (slice.end_index - slice.start_index)*sseq->elem_size ) == 0,
                                              "cvSeqSort returned wrong result" );

                    for( k = 0; k < (sseq->count > 0 ? 10 : 0); k++ )
                    {
                        int idx0 = cvtest::randInt(rng) % sseq->count, idx = 0;
                        elem0 = cvTsSimpleSeqElem( sseq, idx0 );
                        elem = cvGetSeqElem( seq, idx0 );
                        elem2 = cvSeqSearch( seq, elem0, icvCmpSeqElems, 1, &idx, seq );

                        CV_TS_SEQ_CHECK_CONDITION( elem != 0 &&
                                                  memcmp( elem0, elem, seq->elem_size ) == 0,
                                                  "cvSeqSort gives incorrect result" );
                        CV_TS_SEQ_CHECK_CONDITION( elem2 != 0 &&
                                                  memcmp( elem0, elem2, seq->elem_size ) == 0 &&
                                                  elem2 == cvGetSeqElem( seq, idx ),
                                                  "cvSeqSearch failed (binary search)" );
                    }
                }

                cvClearMemStorage( storage );
            }

            storage.release();
        }
    }
    catch (const int &)
    {
    }
}


/////////////////////////////////////// set tests ///////////////////////////////////////

class Core_SetTest : public Core_DynStructBaseTest
{
public:
    Core_SetTest();
    virtual ~Core_SetTest();
    void clear();
    void run( int );

protected:
    //int test_seq_block_consistence( int struct_idx );
    int test_set_ops( int iters );
};


Core_SetTest::Core_SetTest()
{
}

Core_SetTest::~Core_SetTest()
{
    clear();
}

void Core_SetTest::clear()
{
    for( size_t i = 0; i < simple_struct.size(); i++ )
        cvTsReleaseSimpleSet( (CvTsSimpleSet**)&simple_struct[i] );
    Core_DynStructBaseTest::clear();
}


int  Core_SetTest::test_set_ops( int iters )
{
    const int max_op = 4;
    int max_elem_size = 0;
    int idx, idx0;
    CvSetElem *elem = 0, *elem2 = 0, *elem3 = 0;
    schar* elem_data = 0;
    RNG& rng = ts->get_rng();
    //int max_active_count = 0, mean_active_count = 0;

    for( int i = 0; i < struct_count; i++ )
        max_elem_size = MAX( max_elem_size, ((CvSeq*)cxcore_struct[i])->elem_size );

    vector<schar> elem_buf(max_elem_size);
    Mat elem_mat;

    for( iter = 0; iter < iters; iter++ )
    {
        struct_idx = cvtest::randInt(rng) % struct_count;

        CvSet* cvset = (CvSet*)cxcore_struct[struct_idx];
        CvTsSimpleSet* sset = (CvTsSimpleSet*)simple_struct[struct_idx];
        int pure_elem_size = sset->elem_size - 1;
        int prev_total = cvset->total, prev_count = cvset->active_count;
        int op = cvtest::randInt(rng) % (iter <= iters/10 ? 2 : max_op);
        int by_ptr = op % 2 == 0;
        CvSetElem* first_free = cvset->free_elems;
        CvSetElem* next_free = first_free ? first_free->next_free : 0;
        int pass_data = 0;

        if( iter > iters/10 && cvtest::randInt(rng)%200 == 0 ) // clear set
        {
            prev_count = cvset->total;
            cvClearSet( cvset );
            cvTsClearSimpleSet( sset );

            CV_TS_SEQ_CHECK_CONDITION( cvset->active_count == 0 && cvset->total == 0 &&
                                      cvset->first == 0 && cvset->free_elems == 0 &&
                                      (cvset->free_blocks != 0 || prev_count == 0),
                                      "cvClearSet doesn't remove all the elements" );
            continue;
        }
        else if( op == 0 || op == 1 ) // add element
        {
            if( sset->free_count == 0 )
                continue;

            elem_mat = Mat(1, cvset->elem_size, CV_8U, &elem_buf[0]);
            cvtest::randUni( rng, elem_mat, cvScalarAll(0), cvScalarAll(255) );
            elem = (CvSetElem*)&elem_buf[0];

            if( by_ptr )
            {
                elem2 = cvSetNew( cvset );
                CV_TS_SEQ_CHECK_CONDITION( elem2 != 0, "cvSetNew returned NULL pointer" );
            }
            else
            {
                pass_data = cvtest::randInt(rng) % 2;
                idx = cvSetAdd( cvset, pass_data ? elem : 0, &elem2 );
                CV_TS_SEQ_CHECK_CONDITION( elem2 != 0 && elem2->flags == idx,
                                          "cvSetAdd returned NULL pointer or a wrong index" );
            }

            elem_data = (schar*)elem + sizeof(int);

            if( !pass_data )
                memcpy( (schar*)elem2 + sizeof(int), elem_data, pure_elem_size );

            idx = elem2->flags;
            idx0 = cvTsSimpleSetAdd( sset, elem_data );
            elem3 = cvGetSetElem( cvset, idx );

            CV_TS_SEQ_CHECK_CONDITION( CV_IS_SET_ELEM(elem3) &&
                                      idx == idx0 && elem3 == elem2 && (!pass_data ||
                                                                        memcmp( (char*)elem3 + sizeof(int), elem_data, pure_elem_size) == 0),
                                      "The added element is not correct" );

            CV_TS_SEQ_CHECK_CONDITION( (!first_free || elem3 == first_free) &&
                                      (!next_free || cvset->free_elems == next_free) &&
                                      cvset->active_count == prev_count + 1,
                                      "The free node list is modified incorrectly" );
        }
        else if( op == 2 || op == 3 ) // remove element
        {
            idx = cvtest::randInt(rng) % sset->max_count;

            if( sset->free_count == sset->max_count || idx >= sset->count )
                continue;

            elem_data = cvTsSimpleSetFind(sset, idx);
            if( elem_data == 0 )
                continue;

            elem = cvGetSetElem( cvset, idx );
            CV_TS_SEQ_CHECK_CONDITION( CV_IS_SET_ELEM(elem) && elem->flags == idx &&
                                      memcmp((char*)elem + sizeof(int), elem_data, pure_elem_size) == 0,
                                      "cvGetSetElem returned wrong element" );

            if( by_ptr )
            {
                 cvSetRemoveByPtr( cvset, elem );
            }
            else
            {
                 cvSetRemove( cvset, idx );
            }

            cvTsSimpleSetRemove( sset, idx );

            CV_TS_SEQ_CHECK_CONDITION( !CV_IS_SET_ELEM(elem) && !cvGetSetElem(cvset, idx) &&
                                      (elem->flags & CV_SET_ELEM_IDX_MASK) == idx,
                                      "cvSetRemove[ByPtr] didn't release the element properly" );

            CV_TS_SEQ_CHECK_CONDITION( elem->next_free == first_free &&
                                      cvset->free_elems == elem &&
                                      cvset->active_count == prev_count - 1,
                                      "The free node list has not been updated properly" );
        }

        //max_active_count = MAX( max_active_count, cvset->active_count );
        //mean_active_count += cvset->active_count;
        CV_TS_SEQ_CHECK_CONDITION( cvset->active_count == sset->max_count - sset->free_count &&
                                  cvset->total >= cvset->active_count &&
                                  (cvset->total == 0 || cvset->total >= prev_total),
                                  "The total number of cvset elements is not correct" );

        // CvSet and simple set do not necessary have the same "total" (active & free) number,
        // so pass "set->total" to skip that check
        test_seq_block_consistence( struct_idx, (CvSeq*)cvset, cvset->total );
        update_progressbar();
    }

    return 0;
}


void Core_SetTest::run( int )
{
    try
    {
        RNG& rng = ts->get_rng();
        double t;

        clear();
        test_progress = -1;

        simple_struct.resize(struct_count, 0);
        cxcore_struct.resize(struct_count, 0);

        for( gen = 0; gen < generations; gen++ )
        {
            struct_idx = iter = -1;
            t = cvtest::randReal(rng)*(max_log_storage_block_size - min_log_storage_block_size) + min_log_storage_block_size;
            storage.reset(cvCreateMemStorage( cvRound( exp(t * CV_LOG2) ) ));

            for( int i = 0; i < struct_count; i++ )
            {
                t = cvtest::randReal(rng)*(max_log_elem_size - min_log_elem_size) + min_log_elem_size;
                int pure_elem_size = cvRound( exp(t * CV_LOG2) );
                int elem_size = pure_elem_size + sizeof(int);
                elem_size = (elem_size + sizeof(size_t) - 1) & ~(sizeof(size_t)-1);
                elem_size = MAX( elem_size, (int)sizeof(CvSetElem) );
                elem_size = MIN( elem_size, (int)(storage->block_size - sizeof(void*) - sizeof(CvMemBlock) - sizeof(CvSeqBlock)) );
                pure_elem_size = MIN( pure_elem_size, elem_size-(int)sizeof(CvSetElem) );

                cvTsReleaseSimpleSet( (CvTsSimpleSet**)&simple_struct[i] );
                simple_struct[i] = cvTsCreateSimpleSet( max_struct_size, pure_elem_size );
                cxcore_struct[i] = cvCreateSet( 0, sizeof(CvSet), elem_size, storage );
            }

            if( test_set_ops( iterations*100 ) < 0 )
                return;

            storage.release();
        }
    }
    catch(const int &)
    {
    }
}


/////////////////////////////////////// graph tests //////////////////////////////////

class Core_GraphTest : public Core_DynStructBaseTest
{
public:
    Core_GraphTest();
    virtual ~Core_GraphTest();
    void clear();
    void run( int );

protected:
    //int test_seq_block_consistence( int struct_idx );
    int test_graph_ops( int iters );
};


Core_GraphTest::Core_GraphTest()
{
}

Core_GraphTest::~Core_GraphTest()
{
    clear();
}

void Core_GraphTest::clear()
{
    for( size_t i = 0; i < simple_struct.size(); i++ )
        cvTsReleaseSimpleGraph( (CvTsSimpleGraph**)&simple_struct[i] );
    Core_DynStructBaseTest::clear();
}


int  Core_GraphTest::test_graph_ops( int iters )
{
    const int max_op = 4;
    int i, k;
    int max_elem_size = 0;
    int idx, idx0;
    CvGraphVtx *vtx = 0, *vtx2 = 0, *vtx3 = 0;
    CvGraphEdge* edge = 0, *edge2 = 0;
    RNG& rng = ts->get_rng();
    //int max_active_count = 0, mean_active_count = 0;

    for( i = 0; i < struct_count; i++ )
    {
        CvGraph* graph = (CvGraph*)cxcore_struct[i];
        max_elem_size = MAX( max_elem_size, graph->elem_size );
        max_elem_size = MAX( max_elem_size, graph->edges->elem_size );
    }

    vector<schar> elem_buf(max_elem_size);
    Mat elem_mat;

    for( iter = 0; iter < iters; iter++ )
    {
        struct_idx = cvtest::randInt(rng) % struct_count;
        CvGraph* graph = (CvGraph*)cxcore_struct[struct_idx];
        CvTsSimpleGraph* sgraph = (CvTsSimpleGraph*)simple_struct[struct_idx];
        CvSet* edges = graph->edges;
        schar *vtx_data;
        char *edge_data;
        int pure_vtx_size = sgraph->vtx->elem_size - 1,
        pure_edge_size = sgraph->edge_size - 1;
        int prev_vtx_total = graph->total,
        prev_edge_total = graph->edges->total,
        prev_vtx_count = graph->active_count,
        prev_edge_count = graph->edges->active_count;
        int op = cvtest::randInt(rng) % max_op;
        int pass_data = 0, vtx_degree0 = 0, vtx_degree = 0;
        CvSetElem *first_free, *next_free;

        if( cvtest::randInt(rng) % 200 == 0 ) // clear graph
        {
            int prev_vtx_count2 = graph->total, prev_edge_count2 = graph->edges->total;

            cvClearGraph( graph );
            cvTsClearSimpleGraph( sgraph );

            CV_TS_SEQ_CHECK_CONDITION( graph->active_count == 0 && graph->total == 0 &&
                                      graph->first == 0 && graph->free_elems == 0 &&
                                      (graph->free_blocks != 0 || prev_vtx_count2 == 0),
                                      "The graph is not empty after clearing" );

            CV_TS_SEQ_CHECK_CONDITION( edges->active_count == 0 && edges->total == 0 &&
                                      edges->first == 0 && edges->free_elems == 0 &&
                                      (edges->free_blocks != 0 || prev_edge_count2 == 0),
                                      "The graph is not empty after clearing" );
        }
        else if( op == 0 ) // add vertex
        {
            if( sgraph->vtx->free_count == 0 )
                continue;

            first_free = graph->free_elems;
            next_free = first_free ? first_free->next_free : 0;

            if( pure_vtx_size )
            {
                elem_mat = Mat(1, graph->elem_size, CV_8U, &elem_buf[0]);
                cvtest::randUni( rng, elem_mat, cvScalarAll(0), cvScalarAll(255) );
            }

            vtx = (CvGraphVtx*)&elem_buf[0];
            idx0 = cvTsSimpleGraphAddVertex( sgraph, vtx + 1 );

            pass_data = cvtest::randInt(rng) % 2;
            idx = cvGraphAddVtx( graph, pass_data ? vtx : 0, &vtx2 );

            if( !pass_data && pure_vtx_size > 0 )
                memcpy( vtx2 + 1, vtx + 1, pure_vtx_size );

            vtx3 = cvGetGraphVtx( graph, idx );

            CV_TS_SEQ_CHECK_CONDITION( (CV_IS_SET_ELEM(vtx3) && vtx3->flags == idx &&
                                        vtx3->first == 0) || (idx == idx0 && vtx3 == vtx2 &&
                                                              (!pass_data || pure_vtx_size == 0 ||
                                                               memcmp(vtx3 + 1, vtx + 1, pure_vtx_size) == 0)),
                                      "The added element is not correct" );

            CV_TS_SEQ_CHECK_CONDITION( (!first_free || first_free == (CvSetElem*)vtx3) &&
                                      (!next_free || graph->free_elems == next_free) &&
                                      graph->active_count == prev_vtx_count + 1,
                                      "The free node list is modified incorrectly" );
        }
        else if( op == 1 ) // remove vertex
        {
            idx = cvtest::randInt(rng) % sgraph->vtx->max_count;
            if( sgraph->vtx->free_count == sgraph->vtx->max_count || idx >= sgraph->vtx->count )
                continue;

            vtx_data = cvTsSimpleGraphFindVertex(sgraph, idx);
            if( vtx_data == 0 )
                continue;

            vtx_degree0 = cvTsSimpleGraphVertexDegree( sgraph, idx );
            first_free = graph->free_elems;

            vtx = cvGetGraphVtx( graph, idx );
            CV_TS_SEQ_CHECK_CONDITION( CV_IS_SET_ELEM(vtx) && vtx->flags == idx &&
                                      (pure_vtx_size == 0 || memcmp( vtx + 1, vtx_data, pure_vtx_size) == 0),
                                      "cvGetGraphVtx returned wrong element" );

            if( cvtest::randInt(rng) % 2 )
            {
                 vtx_degree = cvGraphVtxDegreeByPtr( graph, vtx );
                 cvGraphRemoveVtxByPtr( graph, vtx );
            }
            else
            {
                 vtx_degree = cvGraphVtxDegree( graph, idx );
                 cvGraphRemoveVtx( graph, idx );
            }

            cvTsSimpleGraphRemoveVertex( sgraph, idx );

            CV_TS_SEQ_CHECK_CONDITION( vtx_degree == vtx_degree0,
                                      "Number of incident edges is different in two graph representations" );

            CV_TS_SEQ_CHECK_CONDITION( !CV_IS_SET_ELEM(vtx) && !cvGetGraphVtx(graph, idx) &&
                                      (vtx->flags & CV_SET_ELEM_IDX_MASK) == idx,
                                      "cvGraphRemoveVtx[ByPtr] didn't release the vertex properly" );

            CV_TS_SEQ_CHECK_CONDITION( graph->edges->active_count == prev_edge_count - vtx_degree,
                                      "cvGraphRemoveVtx[ByPtr] didn't remove all the incident edges "
                                      "(or removed some extra)" );

            CV_TS_SEQ_CHECK_CONDITION( ((CvSetElem*)vtx)->next_free == first_free &&
                                      graph->free_elems == (CvSetElem*)vtx &&
                                      graph->active_count == prev_vtx_count - 1,
                                      "The free node list has not been updated properly" );
        }
        else if( op == 2 ) // add edge
        {
            int v_idx[2] = {0,0}, res = 0;
            int v_prev_degree[2] = {0,0}, v_degree[2] = {0,0};

            if( sgraph->vtx->free_count >= sgraph->vtx->max_count-1 )
                continue;

            for( i = 0, k = 0; i < 10; i++ )
            {
                int j = cvtest::randInt(rng) % sgraph->vtx->count;
                vtx_data = cvTsSimpleGraphFindVertex( sgraph, j );
                if( vtx_data )
                {
                    v_idx[k] = j;
                    if( k == 0 )
                        k++;
                    else if( v_idx[0] != v_idx[1] &&
                            cvTsSimpleGraphFindEdge( sgraph, v_idx[0], v_idx[1] ) == 0 )
                    {
                        k++;
                        break;
                    }
                }
            }

            if( k < 2 )
                continue;

            first_free = graph->edges->free_elems;
            next_free = first_free ? first_free->next_free : 0;

            edge = cvFindGraphEdge( graph, v_idx[0], v_idx[1] );
            CV_TS_SEQ_CHECK_CONDITION( edge == 0, "Extra edge appeared in the graph" );

            if( pure_edge_size > 0 )
            {
                elem_mat = Mat(1, graph->edges->elem_size, CV_8U, &elem_buf[0]);
                cvtest::randUni( rng, elem_mat, cvScalarAll(0), cvScalarAll(255) );
            }
            edge = (CvGraphEdge*)&elem_buf[0];

            // assign some default weight that is easy to check for
            // consistensy, 'cause an edge weight is not stored
            // in the simple graph
            edge->weight = (float)(v_idx[0] + v_idx[1]);
            pass_data = cvtest::randInt(rng) % 2;

            vtx = cvGetGraphVtx( graph, v_idx[0] );
            vtx2 = cvGetGraphVtx( graph, v_idx[1] );
            CV_TS_SEQ_CHECK_CONDITION( vtx != 0 && vtx2 != 0 && vtx->flags == v_idx[0] &&
                                      vtx2->flags == v_idx[1], "Some of the vertices are missing" );

            if( cvtest::randInt(rng) % 2 )
            {
                 v_prev_degree[0] = cvGraphVtxDegreeByPtr( graph, vtx );
                 v_prev_degree[1] = cvGraphVtxDegreeByPtr( graph, vtx2 );
                 res = cvGraphAddEdgeByPtr(graph, vtx, vtx2, pass_data ? edge : 0, &edge2);
                 v_degree[0] = cvGraphVtxDegreeByPtr( graph, vtx );
                 v_degree[1] = cvGraphVtxDegreeByPtr( graph, vtx2 );
            }
            else
            {
                 v_prev_degree[0] = cvGraphVtxDegree( graph, v_idx[0] );
                 v_prev_degree[1] = cvGraphVtxDegree( graph, v_idx[1] );
                 res = cvGraphAddEdge(graph, v_idx[0], v_idx[1], pass_data ? edge : 0, &edge2);
                 v_degree[0] = cvGraphVtxDegree( graph, v_idx[0] );
                 v_degree[1] = cvGraphVtxDegree( graph, v_idx[1] );
            }

            //edge3 = (CvGraphEdge*)cvGetSetElem( graph->edges, idx );
            CV_TS_SEQ_CHECK_CONDITION( res == 1 && edge2 != 0 && CV_IS_SET_ELEM(edge2) &&
                                      ((edge2->vtx[0] == vtx && edge2->vtx[1] == vtx2) ||
                                       (!CV_IS_GRAPH_ORIENTED(graph) && edge2->vtx[0] == vtx2 && edge2->vtx[1] == vtx)) &&
                                      (!pass_data || pure_edge_size == 0 || memcmp( edge2 + 1, edge + 1, pure_edge_size ) == 0),
                                      "The edge has been added incorrectly" );

            if( !pass_data )
            {
                if( pure_edge_size > 0 )
                    memcpy( edge2 + 1, edge + 1, pure_edge_size );
                edge2->weight = edge->weight;
            }

            CV_TS_SEQ_CHECK_CONDITION( v_degree[0] == v_prev_degree[0] + 1 &&
                                      v_degree[1] == v_prev_degree[1] + 1,
                                      "The vertices lists have not been updated properly" );

            cvTsSimpleGraphAddEdge( sgraph, v_idx[0], v_idx[1], edge + 1 );

            CV_TS_SEQ_CHECK_CONDITION( (!first_free || first_free == (CvSetElem*)edge2) &&
                                      (!next_free || graph->edges->free_elems == next_free) &&
                                      graph->edges->active_count == prev_edge_count + 1,
                                      "The free node list is modified incorrectly" );
        }
        else if( op == 3 ) // find & remove edge
        {
            int v_idx[2] = {0,0}, by_ptr;
            int v_prev_degree[2] = {0,0}, v_degree[2] = {0,0};

            if( sgraph->vtx->free_count >= sgraph->vtx->max_count-1 )
                continue;

            edge_data = 0;
            for( i = 0, k = 0; i < 10; i++ )
            {
                int j = cvtest::randInt(rng) % sgraph->vtx->count;
                vtx_data = cvTsSimpleGraphFindVertex( sgraph, j );
                if( vtx_data )
                {
                    v_idx[k] = j;
                    if( k == 0 )
                        k++;
                    else
                    {
                        edge_data = cvTsSimpleGraphFindEdge( sgraph, v_idx[0], v_idx[1] );
                        if( edge_data )
                        {
                            k++;
                            break;
                        }
                    }
                }
            }

            if( k < 2 )
                continue;

            by_ptr = cvtest::randInt(rng) % 2;
            first_free = graph->edges->free_elems;

            vtx = cvGetGraphVtx( graph, v_idx[0] );
            vtx2 = cvGetGraphVtx( graph, v_idx[1] );
            CV_TS_SEQ_CHECK_CONDITION( vtx != 0 && vtx2 != 0 && vtx->flags == v_idx[0] &&
                                      vtx2->flags == v_idx[1], "Some of the vertices are missing" );

            if( by_ptr )
            {
                 edge = cvFindGraphEdgeByPtr( graph, vtx, vtx2 );
                 v_prev_degree[0] = cvGraphVtxDegreeByPtr( graph, vtx );
                 v_prev_degree[1] = cvGraphVtxDegreeByPtr( graph, vtx2 );
            }
            else
            {
                 edge = cvFindGraphEdge( graph, v_idx[0], v_idx[1] );
                 v_prev_degree[0] = cvGraphVtxDegree( graph, v_idx[0] );
                 v_prev_degree[1] = cvGraphVtxDegree( graph, v_idx[1] );
            }

            idx = edge->flags;

            CV_TS_SEQ_CHECK_CONDITION( edge != 0 && edge->weight == v_idx[0] + v_idx[1] &&
                                      ((edge->vtx[0] == vtx && edge->vtx[1] == vtx2) ||
                                       (!CV_IS_GRAPH_ORIENTED(graph) && edge->vtx[1] == vtx && edge->vtx[0] == vtx2)) &&
                                      (pure_edge_size == 0 || memcmp(edge + 1, edge_data, pure_edge_size) == 0),
                                      "An edge is missing or incorrect" );

            if( by_ptr )
            {
                 cvGraphRemoveEdgeByPtr( graph, vtx, vtx2 );
                 edge2 = cvFindGraphEdgeByPtr( graph, vtx, vtx2 );
                 v_degree[0] = cvGraphVtxDegreeByPtr( graph, vtx );
                 v_degree[1] = cvGraphVtxDegreeByPtr( graph, vtx2 );
            }
            else
            {
                 cvGraphRemoveEdge(graph, v_idx[0], v_idx[1] );
                 edge2 = cvFindGraphEdge( graph, v_idx[0], v_idx[1] );
                 v_degree[0] = cvGraphVtxDegree( graph, v_idx[0] );
                 v_degree[1] = cvGraphVtxDegree( graph, v_idx[1] );
            }

            CV_TS_SEQ_CHECK_CONDITION( !edge2 && !CV_IS_SET_ELEM(edge),
                                      "The edge has not been removed from the edge set" );

            CV_TS_SEQ_CHECK_CONDITION( v_degree[0] == v_prev_degree[0] - 1 &&
                                      v_degree[1] == v_prev_degree[1] - 1,
                                      "The vertices lists have not been updated properly" );

            cvTsSimpleGraphRemoveEdge( sgraph, v_idx[0], v_idx[1] );

            CV_TS_SEQ_CHECK_CONDITION( graph->edges->free_elems == (CvSetElem*)edge &&
                                      graph->edges->free_elems->next_free == first_free &&
                                      graph->edges->active_count == prev_edge_count - 1,
                                      "The free edge list has not been modified properly" );
        }

        //max_active_count = MAX( max_active_count, graph->active_count );
        //mean_active_count += graph->active_count;

        CV_TS_SEQ_CHECK_CONDITION( graph->active_count == sgraph->vtx->max_count - sgraph->vtx->free_count &&
                                  graph->total >= graph->active_count &&
                                  (graph->total == 0 || graph->total >= prev_vtx_total),
                                  "The total number of graph vertices is not correct" );

        CV_TS_SEQ_CHECK_CONDITION( graph->edges->total >= graph->edges->active_count &&
                                  (graph->edges->total == 0 || graph->edges->total >= prev_edge_total),
                                  "The total number of graph vertices is not correct" );

        // CvGraph and simple graph do not necessary have the same "total" (active & free) number,
        // so pass "graph->total" (or "graph->edges->total") to skip that check
        test_seq_block_consistence( struct_idx, (CvSeq*)graph, graph->total );
        test_seq_block_consistence( struct_idx, (CvSeq*)graph->edges, graph->edges->total );
        update_progressbar();
    }

    return 0;
}


void Core_GraphTest::run( int )
{
    try
    {
        RNG& rng = ts->get_rng();
        int i, k;
        double t;

        clear();
        test_progress = -1;

        simple_struct.resize(struct_count, 0);
        cxcore_struct.resize(struct_count, 0);

        for( gen = 0; gen < generations; gen++ )
        {
            struct_idx = iter = -1;
            t = cvtest::randReal(rng)*(max_log_storage_block_size - min_log_storage_block_size) + min_log_storage_block_size;
            int block_size = cvRound( exp(t * CV_LOG2) );
            block_size = MAX(block_size, (int)(sizeof(CvGraph) + sizeof(CvMemBlock) + sizeof(CvSeqBlock)));

            storage.reset(cvCreateMemStorage(block_size));

            for( i = 0; i < struct_count; i++ )
            {
                int pure_elem_size[2], elem_size[2];
                int is_oriented = (gen + i) % 2;
                for( k = 0; k < 2; k++ )
                {
                    t = cvtest::randReal(rng)*(max_log_elem_size - min_log_elem_size) + min_log_elem_size;
                    int pe = cvRound( exp(t * CV_LOG2) ) - 1; // pure_elem_size==0 does also make sense
                    int delta = k == 0 ? sizeof(CvGraphVtx) : sizeof(CvGraphEdge);
                    int e = pe + delta;
                    e = (e + sizeof(size_t) - 1) & ~(sizeof(size_t)-1);
                    e = MIN( e, (int)(storage->block_size - sizeof(CvMemBlock) -
                                      sizeof(CvSeqBlock) - sizeof(void*)) );
                    pe = MIN(pe, e - delta);
                    pure_elem_size[k] = pe;
                    elem_size[k] = e;
                }

                cvTsReleaseSimpleGraph( (CvTsSimpleGraph**)&simple_struct[i] );
                simple_struct[i] = cvTsCreateSimpleGraph( max_struct_size/4, pure_elem_size[0],
                                                         pure_elem_size[1], is_oriented );
                cxcore_struct[i] = cvCreateGraph( is_oriented ? CV_ORIENTED_GRAPH : CV_GRAPH,
                                                          sizeof(CvGraph), elem_size[0], elem_size[1],
                                                          storage );
            }

            if( test_graph_ops( iterations*10 ) < 0 )
                return;

            storage.release();
        }
    }
    catch(const int &)
    {
    }
}


//////////// graph scan test //////////////

class Core_GraphScanTest : public Core_DynStructBaseTest
{
public:
    Core_GraphScanTest();
    void run( int );

protected:
    //int test_seq_block_consistence( int struct_idx );
    int create_random_graph( int );
};


Core_GraphScanTest::Core_GraphScanTest()
{
    iterations = 100;
    struct_count = 1;
}


int Core_GraphScanTest::create_random_graph( int _struct_idx )
{
    RNG& rng = ts->get_rng();
    int is_oriented = cvtest::randInt(rng) % 2;
    int i, vtx_count = cvtest::randInt(rng) % max_struct_size;
    int edge_count = cvtest::randInt(rng) % MAX(vtx_count*20, 1);
    CvGraph* graph;

    struct_idx = _struct_idx;
    cxcore_struct[_struct_idx] = graph =
        cvCreateGraph(is_oriented ? CV_ORIENTED_GRAPH : CV_GRAPH,
                      sizeof(CvGraph), sizeof(CvGraphVtx),
                      sizeof(CvGraphEdge), storage );

    for( i = 0; i < vtx_count; i++ )
         cvGraphAddVtx( graph );

    CV_Assert( graph->active_count == vtx_count );

    for( i = 0; i < edge_count; i++ )
    {
        int j = cvtest::randInt(rng) % vtx_count;
        int k = cvtest::randInt(rng) % vtx_count;

        if( j != k )
             cvGraphAddEdge( graph, j, k );
    }

    CV_Assert( graph->active_count == vtx_count && graph->edges->active_count <= edge_count );

    return 0;
}


void Core_GraphScanTest::run( int )
{
    CvGraphScanner* scanner = 0;
    try
    {
        RNG& rng = ts->get_rng();
        vector<uchar> vtx_mask, edge_mask;
        double t;
        int i;

        clear();
        test_progress = -1;

        cxcore_struct.resize(struct_count, 0);

        for( gen = 0; gen < generations; gen++ )
        {
            struct_idx = iter = -1;
            t = cvtest::randReal(rng)*(max_log_storage_block_size - min_log_storage_block_size) + min_log_storage_block_size;
            int storage_blocksize = cvRound( exp(t * CV_LOG2) );
            storage_blocksize = MAX(storage_blocksize, (int)(sizeof(CvGraph) + sizeof(CvMemBlock) + sizeof(CvSeqBlock)));
            storage_blocksize = MAX(storage_blocksize, (int)(sizeof(CvGraphEdge) + sizeof(CvMemBlock) + sizeof(CvSeqBlock)));
            storage_blocksize = MAX(storage_blocksize, (int)(sizeof(CvGraphVtx) + sizeof(CvMemBlock) + sizeof(CvSeqBlock)));
            storage.reset(cvCreateMemStorage(storage_blocksize));

            if( gen == 0 )
            {
                // special regression test for one sample graph.
                // !!! ATTENTION !!! The test relies on the particular order of the inserted edges
                // (LIFO: the edge inserted last goes first in the list of incident edges).
                // if it is changed, the test will have to be modified.

                int vtx_count = -1, edge_count = 0, edges[][3] =
                {
                    {0,4,'f'}, {0,1,'t'}, {1,4,'t'}, {1,2,'t'}, {2,3,'t'}, {4,3,'c'}, {3,1,'b'},
                    {5,7,'t'}, {7,5,'b'}, {5,6,'t'}, {6,0,'c'}, {7,6,'c'}, {6,4,'c'}, {-1,-1,0}
                };

                CvGraph* graph = cvCreateGraph( CV_ORIENTED_GRAPH, sizeof(CvGraph),
                                               sizeof(CvGraphVtx), sizeof(CvGraphEdge), storage );

                for( i = 0; edges[i][0] >= 0; i++ )
                {
                    vtx_count = MAX( vtx_count, edges[i][0] );
                    vtx_count = MAX( vtx_count, edges[i][1] );
                }
                vtx_count++;

                for( i = 0; i < vtx_count; i++ )
                     cvGraphAddVtx( graph );

                for( i = 0; edges[i][0] >= 0; i++ )
                {
                    CvGraphEdge* edge;
                     cvGraphAddEdge( graph, edges[i][0], edges[i][1], 0, &edge );
                    edge->weight = (float)edges[i][2];
                }

                edge_count = i;
                scanner = cvCreateGraphScanner( graph, 0, CV_GRAPH_ALL_ITEMS );

                for(;;)
                {
                    int code, a = -1, b = -1;
                    const char* event = "";
                     code = cvNextGraphItem( scanner );

                    switch( code )
                    {
                        case CV_GRAPH_VERTEX:
                            event = "Vertex";
                            vtx_count--;
                            a = cvGraphVtxIdx( graph, scanner->vtx );
                            break;
                        case CV_GRAPH_TREE_EDGE:
                            event = "Tree Edge";
                            edge_count--;
                            CV_TS_SEQ_CHECK_CONDITION( scanner->edge->weight == (float)'t',
                                                      "Invalid edge type" );
                            a = cvGraphVtxIdx( graph, scanner->vtx );
                            b = cvGraphVtxIdx( graph, scanner->dst );
                            break;
                        case CV_GRAPH_BACK_EDGE:
                            event = "Back Edge";
                            edge_count--;
                            CV_TS_SEQ_CHECK_CONDITION( scanner->edge->weight == (float)'b',
                                                      "Invalid edge type" );
                            a = cvGraphVtxIdx( graph, scanner->vtx );
                            b = cvGraphVtxIdx( graph, scanner->dst );
                            break;
                        case CV_GRAPH_CROSS_EDGE:
                            event = "Cross Edge";
                            edge_count--;
                            CV_TS_SEQ_CHECK_CONDITION( scanner->edge->weight == (float)'c',
                                                      "Invalid edge type" );
                            a = cvGraphVtxIdx( graph, scanner->vtx );
                            b = cvGraphVtxIdx( graph, scanner->dst );
                            break;
                        case CV_GRAPH_FORWARD_EDGE:
                            event = "Forward Edge";
                            edge_count--;
                            CV_TS_SEQ_CHECK_CONDITION( scanner->edge->weight == (float)'f',
                                                      "Invalid edge type" );
                            a = cvGraphVtxIdx( graph, scanner->vtx );
                            b = cvGraphVtxIdx( graph, scanner->dst );
                            break;
                        case CV_GRAPH_BACKTRACKING:
                            event = "Backtracking";
                            a = cvGraphVtxIdx( graph, scanner->vtx );
                            break;
                        case CV_GRAPH_NEW_TREE:
                            event = "New search tree";
                            break;
                        case CV_GRAPH_OVER:
                            event = "End of procedure";
                            break;
                        default:
                            CV_TS_SEQ_CHECK_CONDITION( 0, "Invalid code appeared during graph scan" );
                    }

                    ts->printf( cvtest::TS::LOG, "%s", event );
                    if( a >= 0 )
                    {
                        if( b >= 0 )
                            ts->printf( cvtest::TS::LOG, ": (%d,%d)", a, b );
                        else
                            ts->printf( cvtest::TS::LOG, ": %d", a );
                    }

                    ts->printf( cvtest::TS::LOG, "\n" );

                    if( code < 0 )
                        break;
                }

                CV_TS_SEQ_CHECK_CONDITION( vtx_count == 0 && edge_count == 0,
                                          "Not every vertex/edge has been visited" );
                update_progressbar();

                cvReleaseGraphScanner( &scanner );
            }

            // for a random graph the test just checks that every graph vertex and
            // every edge is vitisted during the scan
            for( iter = 0; iter < iterations; iter++ )
            {
                create_random_graph(0);
                CvGraph* graph = (CvGraph*)cxcore_struct[0];

                // iterate twice to check that scanner doesn't damage the graph
                for( i = 0; i < 2; i++ )
                {
                    CvGraphVtx* start_vtx = cvtest::randInt(rng) % 2 || graph->active_count == 0 ? 0 :
                    cvGetGraphVtx( graph, cvtest::randInt(rng) % graph->active_count );

                    scanner = cvCreateGraphScanner( graph, start_vtx, CV_GRAPH_ALL_ITEMS );

                    vtx_mask.resize(0);
                    vtx_mask.resize(graph->active_count, 0);
                    edge_mask.resize(0);
                    edge_mask.resize(graph->edges->active_count, 0);

                    for(;;)
                    {
                        int code = cvNextGraphItem( scanner );

                        if( code == CV_GRAPH_OVER )
                            break;
                        else if( code & CV_GRAPH_ANY_EDGE )
                        {
                            int edge_idx = scanner->edge->flags & CV_SET_ELEM_IDX_MASK;

                            CV_TS_SEQ_CHECK_CONDITION( edge_idx < graph->edges->active_count &&
                                                      edge_mask[edge_idx] == 0,
                                                      "The edge is not found or visited for the second time" );
                            edge_mask[edge_idx] = 1;
                        }
                        else if( code & CV_GRAPH_VERTEX )
                        {
                            int vtx_idx = scanner->vtx->flags & CV_SET_ELEM_IDX_MASK;

                            CV_TS_SEQ_CHECK_CONDITION( vtx_idx < graph->active_count &&
                                                      vtx_mask[vtx_idx] == 0,
                                                      "The vtx is not found or visited for the second time" );
                            vtx_mask[vtx_idx] = 1;
                        }
                    }

                    cvReleaseGraphScanner( &scanner );

                    CV_TS_SEQ_CHECK_CONDITION( cvtest::norm(Mat(vtx_mask),CV_L1) == graph->active_count &&
                                              cvtest::norm(Mat(edge_mask),CV_L1) == graph->edges->active_count,
                                              "Some vertices or edges have not been visited" );
                    update_progressbar();
                }
                cvClearMemStorage( storage );
            }

            storage.release();
        }
    }
    catch(const int &)
    {
    }
}


TEST(Core_DS_Seq, basic_operations) { Core_SeqBaseTest test; test.safe_run(); }
TEST(Core_DS_Seq, sort_invert) { Core_SeqSortInvTest test; test.safe_run(); }
TEST(Core_DS_Set, basic_operations) { Core_SetTest test; test.safe_run(); }
TEST(Core_DS_Graph, basic_operations) { Core_GraphTest test; test.safe_run(); }
TEST(Core_DS_Graph, scan) { Core_GraphScanTest test; test.safe_run(); }

}} // namespace
