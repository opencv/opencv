/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "_cxts.h"
#include <ctype.h>
#include <stdarg.h>
#include <fcntl.h>
#include <time.h>
#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
#include <io.h>
#else
#include <unistd.h>
#endif

CvTest* CvTest::first = 0;
CvTest* CvTest::last = 0;
int CvTest::test_count = 0;

/*****************************************************************************************\
*                                Exception and memory handlers                            *
\*****************************************************************************************/

// a few platform-dependent declarations

#define CV_TS_NORMAL 0
#define CV_TS_BLUE   1
#define CV_TS_GREEN  2
#define CV_TS_RED    4

#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
#include <windows.h>

#ifdef _MSC_VER
#include <eh.h>
#endif

#ifdef _MSC_VER
static void cv_seh_translator( unsigned int /*u*/, EXCEPTION_POINTERS* pExp )
{
    int code = CvTS::FAIL_EXCEPTION;
    switch( pExp->ExceptionRecord->ExceptionCode )
    {
    case EXCEPTION_ACCESS_VIOLATION:
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
    case EXCEPTION_DATATYPE_MISALIGNMENT:
    case EXCEPTION_FLT_STACK_CHECK:
    case EXCEPTION_STACK_OVERFLOW:
    case EXCEPTION_IN_PAGE_ERROR:
        code = CvTS::FAIL_MEMORY_EXCEPTION;
        break;
    case EXCEPTION_FLT_DENORMAL_OPERAND:
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
    case EXCEPTION_FLT_INEXACT_RESULT:
    case EXCEPTION_FLT_INVALID_OPERATION:
    case EXCEPTION_FLT_OVERFLOW:
    case EXCEPTION_FLT_UNDERFLOW:
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
    case EXCEPTION_INT_OVERFLOW:
        code = CvTS::FAIL_ARITHM_EXCEPTION;
        break;
    case EXCEPTION_BREAKPOINT:
    case EXCEPTION_ILLEGAL_INSTRUCTION:
    case EXCEPTION_INVALID_DISPOSITION:
    case EXCEPTION_NONCONTINUABLE_EXCEPTION:
    case EXCEPTION_PRIV_INSTRUCTION:
    case EXCEPTION_SINGLE_STEP:
        code = CvTS::FAIL_EXCEPTION;
    }
    throw code;
}
#endif

static void change_color( int color )
{
    static int normal_attributes = -1;
    HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
    fflush(stdout);

    if( normal_attributes < 0 )
    {
        CONSOLE_SCREEN_BUFFER_INFO info;
        GetConsoleScreenBufferInfo( hstdout, &info );
        normal_attributes = info.wAttributes;
    }

    SetConsoleTextAttribute( hstdout,
        (WORD)(color == CV_TS_NORMAL ? normal_attributes :
        ((color & CV_TS_BLUE ? FOREGROUND_BLUE : 0)|
        (color & CV_TS_GREEN ? FOREGROUND_GREEN : 0)|
        (color & CV_TS_RED ? FOREGROUND_RED : 0)|FOREGROUND_INTENSITY)) );
}

#else

#include <signal.h>

static const int cv_ts_sig_id[] = { SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, -1 };

static jmp_buf cv_ts_jmp_mark;

void cv_signal_handler( int sig_code )
{
    int code = CvTS::FAIL_EXCEPTION;
    switch( sig_code )
    {
    case SIGFPE:
        code = CvTS::FAIL_ARITHM_EXCEPTION;
        break;
    case SIGSEGV:
    case SIGBUS:
        code = CvTS::FAIL_ARITHM_EXCEPTION;
        break;
    case SIGILL:
        code = CvTS::FAIL_EXCEPTION;
    }

    longjmp( cv_ts_jmp_mark, code );
}

static void change_color( int color )
{
    static const uchar ansi_tab[] = { 30, 34, 32, 36, 31, 35, 33, 37 };
    char buf[16];
    int code = 0;
    fflush( stdout );
    if( color != CV_TS_NORMAL )
        code = ansi_tab[color & (CV_TS_BLUE|CV_TS_GREEN|CV_TS_RED)];
    sprintf( buf, "\x1b[%dm", code );
    fputs( buf, stdout );
}

#endif


// reads 16-digit hexadecimal number (i.e. 64-bit integer)
static int64 read_seed( const char* str )
{
    int64 val = 0;
    if( str && strlen(str) == 16 )
    {
        for( int i = 0; str[i]; i++ )
        {
            int c = tolower(str[i]);
            if( !isxdigit(c) )
                return 0;
            val = val * 16 +
            (str[i] < 'a' ? str[i] - '0' : str[i] - 'a' + 10);
        }
    }
    return val;
}


/***************************** memory manager *****************************/

typedef struct CvTestAllocBlock
{
    struct CvTestAllocBlock* prev;
    struct CvTestAllocBlock* next;
    char* origin;
    char* data;
    size_t size;
    int index;
}
CvTestAllocBlock;


class CvTestMemoryManager
{
public:
    CvTestMemoryManager( CvTS* ts );
    virtual ~CvTestMemoryManager();

    virtual void clear_and_check( int min_index = -1 );
    virtual void start_tracking( int index_to_stop_at=-1 );
    virtual void stop_tracking_and_check();
    int get_alloc_index() { return index; }

    static void* alloc_proxy( size_t size, void* userdata );
    static int free_proxy( void* ptr, void* userdata );

protected:
    virtual void* alloc( size_t size );
    virtual int free( void* ptr );
    virtual int free_block( CvTestAllocBlock* block );

    int index;
    int track_blocks;
    int show_msg_box;
    int index_to_stop_at;
    const char* guard_pattern;
    int guard_size;
    int block_align;
    enum { MAX_MARKS = 1024 };
    int marks[MAX_MARKS];
    int marks_top;
    CvTS* ts;
    CvTestAllocBlock* first;
    CvTestAllocBlock* last;
};


void* CvTestMemoryManager::alloc_proxy( size_t size, void* userdata )
{
    return ((CvTestMemoryManager*)userdata)->alloc( size );
}


int CvTestMemoryManager::free_proxy( void* ptr, void* userdata )
{
    return ((CvTestMemoryManager*)userdata)->free( ptr );
}


CvTestMemoryManager::CvTestMemoryManager( CvTS* _test_system )
{
    ts = _test_system;
    guard_pattern = "THIS IS A GUARD PATTERN!";
    guard_size = (int)strlen(guard_pattern);
    block_align = CV_MALLOC_ALIGN;
    track_blocks = 0;
    marks_top = 0;
    first = last = 0;
    index = 0;
    index_to_stop_at = -1;
    show_msg_box = 1;
}


CvTestMemoryManager::~CvTestMemoryManager()
{
    clear_and_check();
}


void CvTestMemoryManager::clear_and_check( int min_index )
{
    int alloc_index = -1;
    CvTestAllocBlock* block;
    int leak_size = 0, leak_block_count = 0, mem_size = 0;
    void* mem_addr = 0;

    while( marks_top > 0 && marks[marks_top - 1] >= min_index )
        marks_top--;

    for( block = last; block != 0; )
    {
        CvTestAllocBlock* prev = block->prev;
        if( block->index < min_index )
            break;
        leak_size += (int)block->size;
        leak_block_count++;
        alloc_index = block->index;
        mem_addr = block->data;
        mem_size = (int)block->size;
        free_block( block );
        block = prev;
    }
    track_blocks--;
    if( leak_block_count > 0 )
    {
        ts->set_failed_test_info( CvTS::FAIL_MEMORY_LEAK, alloc_index );
        ts->printf( CvTS::LOG, "Memory leaks: %u blocks, %u bytes total\n"
                    "%s leaked block: %p, %u bytes\n",
                    leak_block_count, leak_size, leak_block_count > 1 ? "The first" : "The",
                    mem_addr, mem_size );
    }

    index = block ? block->index + 1 : 0;
}


void CvTestMemoryManager::start_tracking( int _index_to_stop_at )
{
    track_blocks--;
    marks[marks_top++] = index;
    assert( marks_top <= MAX_MARKS );
    track_blocks+=2;
    index_to_stop_at = _index_to_stop_at >= index ? _index_to_stop_at : -1;
}


void CvTestMemoryManager::stop_tracking_and_check()
{
    if( marks_top > 0 )
    {
        int min_index = marks[--marks_top];
        clear_and_check( min_index );
    }
}


int CvTestMemoryManager::free_block( CvTestAllocBlock* block )
{
    int code = 0;
    char* data = block->data;

    if( block->origin == 0 || ((size_t)block->origin & (sizeof(double)-1)) != 0 )
        code = CvTS::FAIL_MEMORY_CORRUPTION_BEGIN;

    if( memcmp( data - guard_size, guard_pattern, guard_size ) != 0 )
        code = CvTS::FAIL_MEMORY_CORRUPTION_BEGIN;
    else if( memcmp( data + block->size, guard_pattern, guard_size ) != 0 )
        code = CvTS::FAIL_MEMORY_CORRUPTION_END;

    if( code >= 0 )
    {
        if( block->prev )
            block->prev->next = block->next;
        else if( first == block )
            first = block->next;

        if( block->next )
            block->next->prev = block->prev;
        else if( last == block )
            last = block->prev;

        free( block->origin );
    }
    else
    {
        ts->set_failed_test_info( code, block->index );
        ts->printf( CvTS::LOG, "Corrupted block (%s): %p, %u bytes\n",
                    code == CvTS::FAIL_MEMORY_CORRUPTION_BEGIN ? "beginning" : "end",
                    block->data, block->size );
    }

    return code;
}


void* CvTestMemoryManager::alloc( size_t size )
{
    char* data;
    CvTestAllocBlock* block;
    size_t new_size = sizeof(*block) + size + guard_size*2 + block_align + sizeof(size_t)*2;
    char* ptr = (char*)malloc( new_size );

    if( !ptr )
        return 0;

    data = (char*)cvAlignPtr( ptr + sizeof(size_t) + sizeof(*block) + guard_size, block_align );
    block = (CvTestAllocBlock*)cvAlignPtr( data - guard_size -
            sizeof(size_t) - sizeof(*block), sizeof(size_t) );
    block->origin = ptr;
    block->data = data;
    block->size = 0;
    block->index = -1;
    block->next = block->prev = 0;
    memcpy( data - guard_size, guard_pattern, guard_size );
    memcpy( data + size, guard_pattern, guard_size );

    if( track_blocks > 0 )
    {
        track_blocks--;
        block->size = size;

        if( index == index_to_stop_at )
        {
            if( show_msg_box )
            {
        #if defined WIN32 || defined _WIN32
                MessageBox( NULL, "The block that is corrupted and/or not deallocated has been just allocated\n"
                            "Press Ok to start debugging", "Memory Manager", MB_ICONERROR|MB_OK|MB_SYSTEMMODAL );
        #endif
            }
            CV_DBG_BREAK();
        }

        block->index = index++;

        block->prev = last;
        block->next = 0;
        if( last )
            last = last->next = block;
        else
            first = last = block;

        track_blocks++;
    }

    return data;
}


int CvTestMemoryManager::free( void* ptr )
{
    char* data = (char*)ptr;
    CvTestAllocBlock* block = (CvTestAllocBlock*)
        cvAlignPtr( data - guard_size - sizeof(size_t) - sizeof(*block), sizeof(size_t) );

    int code = free_block( block );
    if( code < 0 && ts->is_debug_mode() )
        CV_DBG_BREAK();
    return 0;
}


/***************************** error handler *****************************/

#if 0
static int cvTestErrorCallback( int status, const char* func_name, const char* err_msg,
                         const char* file_name, int line, void* userdata )
{
    if( status < 0 && status != CV_StsBackTrace && status != CV_StsAutoTrace )
        ((CvTS*)userdata)->set_failed_test_info( CvTS::FAIL_ERROR_IN_CALLED_FUNC );

    // print error message
    return cvStdErrReport( status, func_name, err_msg, file_name, line, 0 );
}
#endif

/*****************************************************************************************\
*                                    Base Class for Tests                                 *
\*****************************************************************************************/

CvTest::CvTest( const char* _test_name, const char* _test_funcs, const char* _test_descr ) :
    name(_test_name ? _test_name : ""), tested_functions(_test_funcs ? _test_funcs : ""),
    description(_test_descr ? _test_descr : ""), ts(0)
{
    if( last )
        last->next = this;
    else
        first = this;
    last = this;
    test_count++;
    ts = 0;
    hdr_state = 0;

    timing_param_names = 0;
    timing_param_current = 0;
    timing_param_seqs = 0;
    timing_param_idxs = 0;
    timing_param_count = -1;

    test_case_count = -1;
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}

CvTest::~CvTest()
{
    clear();
}

CvTest* CvTest::get_first_test()
{
    return first;
}

void CvTest::clear()
{
    if( timing_param_current )
        free( timing_param_current );
    if( timing_param_seqs )
        free( timing_param_seqs );
    if( timing_param_idxs )
        free( timing_param_idxs );

    timing_param_current = 0;
    timing_param_seqs = 0;
    timing_param_idxs = 0;
    timing_param_count = -1;
}


int CvTest::init( CvTS* _test_system )
{
    clear();
    ts = _test_system;
    return read_params( ts->get_file_storage() );
}


const char* CvTest::get_parent_name( const char* name, char* buffer )
{
    const char* dash_pos = strrchr( name ? name : "", '-' );
    if( !dash_pos )
        return 0;

    if( name != (const char*)buffer )
        strncpy( buffer, name, dash_pos - name );
    buffer[dash_pos - name] = '\0';
    return buffer;
}


const CvFileNode* CvTest::find_param( CvFileStorage* fs, const char* param_name )
{
    char buffer[256];
    const char* name = get_name();
    CvFileNode* node = 0;

    for(;;)
    {
        if( !name )
            break;
        node = cvGetFileNodeByName( fs, 0, name );
        if( node )
        {
            node = cvGetFileNodeByName( fs, node, param_name );
            if( node )
                break;
        }
        name = get_parent_name( name, buffer );
    }

    return node;
}


void CvTest::start_write_param( CvFileStorage* fs )
{
    if( hdr_state == 0 )
    {
        cvStartWriteStruct( fs, get_name(), CV_NODE_MAP );
        hdr_state = 1;
    }
}


void CvTest::write_param( CvFileStorage* fs, const char* paramname, int val )
{
    if( !ts->find_written_param( this, paramname, CV_NODE_INT, &val) )
    {
        start_write_param( fs );
        cvWriteInt( fs, paramname, val );
    }
}


void CvTest::write_param( CvFileStorage* fs, const char* paramname, double val )
{
    if( !ts->find_written_param( this, paramname, CV_NODE_REAL, &val) )
    {
        start_write_param( fs );
        cvWriteReal( fs, paramname, val );
    }
}


void CvTest::write_param( CvFileStorage* fs, const char* paramname, const char* val )
{
    if( !ts->find_written_param( this, paramname, CV_NODE_STRING, &val) )
    {
        start_write_param( fs );
        cvWriteString( fs, paramname, val );
    }
}


void CvTest::write_string_list( CvFileStorage* fs, const char* paramname, const char** val, int count )
{
    if( val )
    {
        start_write_param( fs );
        int i;
        if( count < 0 )
            count = INT_MAX;

        cvStartWriteStruct( fs, paramname, CV_NODE_SEQ + CV_NODE_FLOW );
        for( i = 0; i < count && val[i] != 0; i++ )
            cvWriteString( fs, 0, val[i] );
        cvEndWriteStruct( fs );
    }
}


void CvTest::write_int_list( CvFileStorage* fs, const char* paramname,
                             const int* val, int count, int stop_value )
{
    if( val )
    {
        start_write_param( fs );
        int i;
        if( count < 0 )
            count = INT_MAX;

        cvStartWriteStruct( fs, paramname, CV_NODE_SEQ + CV_NODE_FLOW );
        for( i = 0; i < count && val[i] != stop_value; i++ )
            cvWriteInt( fs, 0, val[i] );
        cvEndWriteStruct( fs );
    }
}


void CvTest::write_real_list( CvFileStorage* fs, const char* paramname,
                              const double* val, int count, double stop_value )
{
    if( val )
    {
        start_write_param( fs );
        int i;
        if( count < 0 )
            count = INT_MAX;

        cvStartWriteStruct( fs, paramname, CV_NODE_SEQ + CV_NODE_FLOW );
        for( i = 0; i < count && val[i] != stop_value; i++ )
            cvWriteReal( fs, 0, val[i] );
        cvEndWriteStruct( fs );
    }
}


int CvTest::read_params( CvFileStorage* fs )
{
    int code = 0;

    if(fs == NULL)
      return code;

    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        timing_param_names = find_param( fs, "timing_params" );
        if(!timing_param_names)
          return code;

        if( CV_NODE_IS_SEQ(timing_param_names->tag) )
        {
            CvSeq* seq = timing_param_names->data.seq;
            CvSeqReader reader;
            cvStartReadSeq( seq, &reader );
            int i;

            timing_param_count = seq->total;
            timing_param_seqs = (const CvFileNode**)malloc( timing_param_count*sizeof(timing_param_seqs[0]));
            timing_param_idxs = (int*)malloc( timing_param_count*sizeof(timing_param_idxs[0]));
            timing_param_current = (const CvFileNode**)malloc( timing_param_count*sizeof(timing_param_current[0]));
            test_case_count = 1;

            for( i = 0; i < timing_param_count; i++ )
            {
                CvFileNode* param_name = (CvFileNode*)(reader.ptr);

                if( !CV_NODE_IS_STRING(param_name->tag) )
                {
                    ts->printf( CvTS::LOG, "ERROR: name of timing parameter #%d is not a string\n", i );
                    code = -1;
                    break;
                }

                timing_param_idxs[i] = 0;
                timing_param_current[i] = 0;
                timing_param_seqs[i] = find_param( fs, param_name->data.str.ptr );
                if( !timing_param_seqs[i] )
                {
                    ts->printf( CvTS::LOG, "ERROR: timing parameter %s is not found\n", param_name->data.str.ptr );
                    code = -1;
                    break;
                }

                if( CV_NODE_IS_SEQ(timing_param_seqs[i]->tag) )
                    test_case_count *= timing_param_seqs[i]->data.seq->total;

                CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
            }

            if( i < timing_param_count )
                timing_param_count = 0;
        }
        else
        {
            ts->printf( CvTS::LOG, "ERROR: \"timing_params\" is not found" );
            code = -1;
        }
    }

    return code;
}


int CvTest::get_next_timing_param_tuple(void)
{
    bool increment;
    int i;

    if( timing_param_count <= 0 || !timing_param_names || !timing_param_seqs )
        return -1;

    increment = timing_param_current[0] != 0; // if already have some valid test tuple, move to the next
    for( i = 0; i < timing_param_count; i++ )
    {
        const CvFileNode* node = timing_param_seqs[i];
        int total = CV_NODE_IS_SEQ(node->tag) ? node->data.seq->total : 1;
        int new_idx = timing_param_idxs[i];

        if( !timing_param_current[i] )
            timing_param_idxs[i] = new_idx = 0;
        else if( increment )
        {
            new_idx++;
            if( new_idx >= total )
                new_idx = 0;
            else if( total > 1 )
                increment = false;
        }

        if( !timing_param_current[i] || new_idx != timing_param_idxs[i] )
        {
            if( CV_NODE_IS_SEQ(node->tag) )
                timing_param_current[i] = (CvFileNode*)cvGetSeqElem( node->data.seq, new_idx );
            else
                timing_param_current[i] = node;
            timing_param_idxs[i] = new_idx;
        }
    }

    return !increment; // return 0 in case of overflow (i.e. if there is no more test cases)
}


const CvFileNode* CvTest::find_timing_param( const char* paramname )
{
    if( timing_param_names )
    {
        int i;
        CvSeqReader reader;
        cvStartReadSeq( timing_param_names->data.seq, &reader, 0 );

        for( i = 0; i < timing_param_count; i++ )
        {
            const char* ptr = ((const CvFileNode*)(reader.ptr))->data.str.ptr;
            if( ptr[0] == paramname[0] && strcmp(ptr, paramname) == 0 )
                return timing_param_current[i];
            CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        }
    }
    return 0;
}


int CvTest::write_defaults(CvTS* _ts)
{
    ts = _ts;
    hdr_state = 0;
    write_default_params( ts->get_file_storage() );
    if( hdr_state )
        cvEndWriteStruct( ts->get_file_storage() );
    return 0;
}


int CvTest::write_default_params( CvFileStorage* fs )
{
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
        write_string_list( fs, "timing_params", default_timing_param_names, timing_param_count );
    return 0;
}


bool CvTest::can_do_fast_forward()
{
    return true;
}


int CvTest::get_support_testing_modes()
{
    return support_testing_modes;
}

void CvTest::safe_run( int start_from )
{
    if(ts->is_debug_mode())
        run( start_from );
    else
    {
        try
        {
        #if !defined WIN32 && !defined _WIN32
        int _code = setjmp( cv_ts_jmp_mark );
        if( !_code )
            run( start_from );
        else
            throw _code;
        #else
            run( start_from );
        #endif
        }
        catch (const cv::Exception& exc)
        {
            const char* errorStr = cvErrorStr(exc.code);
            char buf[1 << 16];
            
            sprintf( buf, "OpenCV Error: %s (%s) in %s, file %s, line %d",
                    errorStr, exc.err.c_str(), exc.func.size() > 0 ?
                    exc.func.c_str() : "unknown function", exc.file.c_str(), exc.line );
            ts->printf(CvTS::LOG, "%s\n", buf);
            ts->set_failed_test_info( CvTS::FAIL_ERROR_IN_CALLED_FUNC );
        }
        catch (...)
        {
            ts->set_failed_test_info( CvTS::FAIL_EXCEPTION );
        }
    }
}


void CvTest::run( int start_from )
{
    int i, test_case_idx, count = get_test_case_count();
    int64 t_start = cvGetTickCount();
    double freq = cv::getTickFrequency();
    bool ff = can_do_fast_forward();
    int progress = 0, code;
    std::vector<double> v_cpe, v_time;
    int64 t1 = t_start;

    for( test_case_idx = ff && start_from >= 0 ? start_from : 0;
         count < 0 || test_case_idx < count; test_case_idx++ )
    {
        ts->update_context( this, test_case_idx, ff );
        progress = update_progress( progress, test_case_idx, count, (double)(t1 - t_start)/(freq*1000) );

        int64 t00 = 0, t0 = 0, t2 = 0, t3 = 0;
        double t_acc = 0, t_cpu_acc = 0;

        if( ts->get_testing_mode() == CvTS::TIMING_MODE )
        {
            const int iterations = 20;
            code = prepare_test_case( test_case_idx );

            if( code < 0 || ts->get_err_code() < 0 )
                return;

            if( code == 0 )
                continue;

            v_cpe.resize(0);
            v_time.resize(0);

            for( i = 0; i < iterations; i++ )
            {
                for(;;)
                {
                    t0 = cv::getTickCount();
                    t2 = cv::getCPUTickCount();

                    run_func();

                    t3 = cv::getCPUTickCount();
                    t1 = cv::getTickCount();

                    if( ts->get_err_code() < 0 )
                        return;

                    if( t3 - t2 > 0 && t1 - t0 > 1 )
                        break;
                }

                if( i == 0 )
                    t00 = t0;

                v_cpe.push_back((double)(t3 - t2));
                v_time.push_back((double)(t1 - t0));

                if( i >= 5 && t1 - t00 > freq*5 )
                    break;
            }

            std::sort(v_cpe.begin(), v_cpe.end());
            std::sort(v_time.begin(), v_time.end());

            t_cpu_acc = v_cpe[i/2];
            t_acc     = v_time[i/2];

            print_time( test_case_idx, t_acc, t_cpu_acc );
        }
        else
        {
            code = prepare_test_case( test_case_idx );
            if( code < 0 || ts->get_err_code() < 0 )
                return;

            if( code == 0 )
                continue;

            run_func();

            if( ts->get_err_code() < 0 )
                return;

            if( validate_test_results( test_case_idx ) < 0 || ts->get_err_code() < 0 )
                return;
        }
    }
}


void CvTest::run_func(void)
{
    assert(0);
}


int CvTest::get_test_case_count(void)
{
    return test_case_count;
}


int CvTest::prepare_test_case( int )
{
    return 0;
}


int CvTest::validate_test_results( int )
{
    return 0;
}


void CvTest::print_time( int /*test_case_idx*/, double /*time_usecs*/, double /*time_cpu_clocks*/ )
{
}


int CvTest::update_progress( int progress, int test_case_idx, int count, double dt )
{
    int width = 60 - (int)strlen(get_name());
    if( count > 0 )
    {
        int t = cvRound( ((double)test_case_idx * width)/count );
        if( t > progress )
        {
            ts->printf( CvTS::CONSOLE, "." );
            progress = t;
        }
    }
    else if( cvRound(dt) > progress )
    {
        ts->printf( CvTS::CONSOLE, "." );
        progress = cvRound(dt);
    }

    return progress;
}


CvBadArgTest::CvBadArgTest( const char* _test_name, const char* _test_funcs, const char* _test_descr )
  : CvTest( _test_name, _test_funcs, _test_descr )
{
    progress      = -1;
    test_case_idx = -1;
    freq          = cv::getTickFrequency();
}

CvBadArgTest::~CvBadArgTest(void)
{
}

int CvBadArgTest::run_test_case( int expected_code, const char* descr )
{
    double new_t = (double)cv::getTickCount(), dt;
    if( test_case_idx < 0 )
    {
        test_case_idx = 0;
        progress      = 0;
        dt            = 0;
    }
    else
    {
        dt = (new_t - t)/(freq*1000);
        t  = new_t;
    }
    progress = update_progress(progress, test_case_idx, 0, dt);

    int errcount = 0;
    bool thrown = false;

    if(!descr)
        descr = "";

    try
    {
        run_func();
    }

    catch(const cv::Exception& e)
    {
        thrown = true;
        if( e.code != expected_code )
        {
            ts->printf(CvTS::LOG, "%s (test case #%d): the error code %d is different from the expected %d\n",
                descr, test_case_idx, e.code, expected_code);
            errcount = 1;
        }
    }

    catch(...)
    {
        thrown = true;
        ts->printf(CvTS::LOG, "%s  (test case #%d): unknown exception was thrown (the function has likely crashed)\n",
                   descr, test_case_idx);
        errcount = 1;
    }

    if(!thrown)
    {
        ts->printf(CvTS::LOG, "%s  (test case #%d): no expected exception was thrown\n",
                   descr, test_case_idx);
        errcount = 1;
    }
    test_case_idx++;

    return errcount;
}


/*****************************************************************************************\
*                                 Base Class for Test System                              *
\*****************************************************************************************/

/******************************** Constructors/Destructors ******************************/

CvTS::CvTS(const char* _module_name)
{
    module_name    = _module_name;
    start_time     = 0;
    version        = CV_TS_VERSION;
    memory_manager = 0;

    /*
    memory_manager = new CvTestMemoryManager(this);
    cvSetMemoryManager( CvTestMemoryManager::alloc_proxy,
                        CvTestMemoryManager::free_proxy,
                        memory_manager );*/

    ostrm_suffixes[SUMMARY_IDX] = ".sum";
    ostrm_suffixes[LOG_IDX]     = ".log";
    ostrm_suffixes[CSV_IDX]     = ".csv";
    ostrm_suffixes[CONSOLE_IDX] = 0;

    ostrm_base_name = 0;

    memset( output_streams, 0, sizeof(output_streams) );
    memset( &params, 0, sizeof(params) );

    selected_tests = new CvTestPtrVec();
    failed_tests   = new CvTestInfoVec();
    written_params = new CvTestPtrVec();

    clear();

    return;
} // ctor


void CvTS::clear(void)
{
    int i;
    CvTest* test;

    for( test = get_first_test(); test != 0; test = test->get_next() )
        test->clear();

    for( i = 0; i <= CONSOLE_IDX; i++ )
    {
        if( i == LOG_IDX )
            fflush( stderr );
        else if( i == CONSOLE_IDX )
            fflush( stdout );

        if( i < CONSOLE_IDX && output_streams[i].f )
        {
            fclose( output_streams[i].f );
            output_streams[i].f = 0;
        }

        if( i == LOG_IDX && output_streams[i].default_handle > 0 )
        {
            dup2( output_streams[i].default_handle, 2 );
            output_streams[i].default_handle = 0;
        }
        output_streams[i].enable = 1;
    }

    cvReleaseFileStorage( &fs );

    selected_tests->clear();
    failed_tests->clear();

    if( ostrm_base_name )
    {
        free( ostrm_base_name );
        ostrm_base_name = 0;
    }

    params.rng_seed          = 0;
    params.debug_mode        = -1;
    params.print_only_failed = 0;
    params.skip_header = -1;
    params.ignore_blacklist = -1;
    params.test_mode = CORRECTNESS_CHECK_MODE;
    params.timing_mode = MIN_TIME;
    params.use_optimized = -1;
    params.color_terminal = 1;

    if( memory_manager )
        memory_manager->clear_and_check();
} // clear()


CvTS::~CvTS(void)
{
    clear();
    set_data_path(0);

    if( written_params )
    {
        for( int i = 0; i < written_params->size(); i++ )
            free( written_params->at(i) );
        delete written_params;
    }

    delete selected_tests;
    delete failed_tests;

    return;
} // dtor


const char* CvTS::str_from_code( int code )
{
    switch( code )
    {
    case OK:                           return "Ok";
    case FAIL_GENERIC:                 return "Generic/Unknown";
    case FAIL_MISSING_TEST_DATA:       return "No test data";
    case FAIL_INVALID_TEST_DATA:       return "Invalid test data";
    case FAIL_ERROR_IN_CALLED_FUNC:    return "cvError invoked";
    case FAIL_EXCEPTION:               return "Hardware/OS exception";
    case FAIL_MEMORY_EXCEPTION:        return "Invalid memory access";
    case FAIL_ARITHM_EXCEPTION:        return "Arithmetic exception";
    case FAIL_MEMORY_CORRUPTION_BEGIN: return "Corrupted memblock (beginning)";
    case FAIL_MEMORY_CORRUPTION_END:   return "Corrupted memblock (end)";
    case FAIL_MEMORY_LEAK:             return "Memory leak";
    case FAIL_INVALID_OUTPUT:          return "Invalid function output";
    case FAIL_MISMATCH:                return "Unexpected output";
    case FAIL_BAD_ACCURACY:            return "Bad accuracy";
    case FAIL_HANG:                    return "Infinite loop(?)";
    case FAIL_BAD_ARG_CHECK:           return "Incorrect handling of bad arguments";
    default:                           return "Generic/Unknown";
    }
}

/************************************** Running tests **********************************/

void CvTS::make_output_stream_base_name( const char* config_name )
{
    int k, len = (int)strlen( config_name );

    if( ostrm_base_name )
        free( ostrm_base_name );

    for( k = len-1; k >= 0; k-- )
    {
        char c = config_name[k];
        if( c == '.' || c == '/' || c == '\\' || c == ':' )
            break;
    }

    if( k > 0 && config_name[k] == '.' )
        len = k;

    ostrm_base_name = (char*)malloc( len + 1 );
    memcpy( ostrm_base_name, config_name, len );
    ostrm_base_name[len] = '\0';
}


void CvTS::set_handlers( bool on )
{
    if( on )
    {
        cvSetErrMode( CV_ErrModeParent );
        cvRedirectError( cvStdErrReport );
    #if defined WIN32 || defined _WIN32
        #ifdef _MSC_VER
        _set_se_translator( cv_seh_translator );
        #endif
    #else
        for( int i = 0; cv_ts_sig_id[i] >= 0; i++ )
            signal( cv_ts_sig_id[i], cv_signal_handler );
    #endif
    }
    else
    {
        cvSetErrMode( CV_ErrModeLeaf );
        cvRedirectError( cvGuiBoxReport );
    #if defined WIN32 || defined _WIN32
        #ifdef _MSC_VER
        _set_se_translator( 0 );
        #endif
    #else
        for( int i = 0; cv_ts_sig_id[i] >= 0; i++ )
            signal( cv_ts_sig_id[i], SIG_DFL );
    #endif
    }
}


void CvTS::set_data_path( const char* data_path )
{
    if( data_path == params.data_path )
        return;

    if( params.data_path )
        delete[] params.data_path;
    if( data_path )
    {
        int size = (int)strlen(data_path)+1;
        bool append_slash = data_path[size-1] != '/' && data_path[size-1] != '\\';
        params.data_path = new char[size+1];
        memcpy( params.data_path, data_path, size );
        if( append_slash )
            strcat( params.data_path, "/" );
    }
}


typedef struct CvTsParamVal
{
    const char* fullname;
    const void* val;

} CvTsParamVal;


int CvTS::find_written_param( CvTest* test, const char* paramname, int valtype, const void* val )
{
    const char* testname = test->get_name();
    bool add_to_list = test->get_func_list()[0] == '\0';
    char buffer[256];
    int paramname_len = (int)strlen(paramname);
    int paramval_len = valtype == CV_NODE_INT ? (int)sizeof(int) :
        valtype == CV_NODE_REAL ? (int)sizeof(double) : -1;
    const char* name = CvTest::get_parent_name( testname, buffer );

    if( !fs )
        return -1;

    if( paramval_len < 0 )
    {
        assert(0); // unsupported parameter type
        return -1;
    }

    while( name )
    {
        int i, len = (int)strlen(buffer);
        buffer[len] = '.';
        memcpy( buffer + len + 1, paramname, paramname_len + 1 );
        for( i = 0; i < written_params->size(); i++ )
        {
            CvTsParamVal* param = (CvTsParamVal*)written_params->at(i);
            if( strcmp( param->fullname, buffer ) == 0 )
            {
                if( (paramval_len > 0 && memcmp( param->val, val, paramval_len ) == 0) ||
                    (paramval_len < 0 && strcmp( (const char*)param->val, (const char*)val ) == 0) )
                    return 1;
                break;
            }
        }
        if( i < written_params->size() )
            break;
        buffer[len] = '\0';
        name = CvTest::get_parent_name( buffer, buffer );
    }

    if( add_to_list )
    {
        int bufsize, fullname_len = (int)strlen(testname) + paramname_len + 2;
        CvTsParamVal* param;
        if( paramval_len < 0 )
            paramval_len = (int)strlen((const char*)val) + 1;
        bufsize = sizeof(*param) + fullname_len + paramval_len;
        param = (CvTsParamVal*)malloc(bufsize);
        param->fullname = (const char*)(param + 1);
        param->val = param->fullname + fullname_len;
        sprintf( (char*)param->fullname, "%s.%s", testname, paramname );
        memcpy( (void*)param->val, val, paramval_len );
        written_params->push( param );
    }

    return 0;
}


#ifndef MAX_PATH
#define MAX_PATH 1024
#endif

static int CV_CDECL cmp_test_names( const void* a, const void* b )
{
    return strcmp( (*(const CvTest**)a)->get_name(), (*(const CvTest**)b)->get_name() );
}

int CvTS::run( int argc, char** argv, const char** blacklist )
{
    time( &start_time );

    int i, write_params = 0;
    int list_tests = 0;
    CvTestPtrVec all_tests;
    CvTest* test;

    // 0. reset all the parameters, reorder tests
    clear();

/*#if defined WIN32 || defined _WIN32
	cv::setBreakOnError(true);
#endif*/

    for( test = get_first_test(), i = 0; test != 0; test = test->get_next(), i++ )
        all_tests.push(test);

    if( all_tests.size() > 0 && all_tests.data() )
        qsort( all_tests.data(), all_tests.size(), sizeof(CvTest*), cmp_test_names );

    // 1. parse command line options
    for( i = 1; i < argc; i++ )
    {
        if( strcmp( argv[i], "-h" ) == 0 || strcmp( argv[i], "--help" ) == 0 )
        {
            print_help();
            return 0;
        }
        else if( strcmp( argv[i], "-f" ) == 0 )
            config_name = argv[++i];
        else if( strcmp( argv[i], "-w" ) == 0 )
            write_params = 1;
        else if( strcmp( argv[i], "-t" ) == 0 )
            params.test_mode = TIMING_MODE;
        else if( strcmp( argv[i], "-O0" ) == 0 || strcmp( argv[i], "-O1" ) == 0 )
            params.use_optimized = argv[i][2] - '0';
        else if( strcmp( argv[i], "-l" ) == 0 )
            list_tests = 1;
        else if( strcmp( argv[i], "-d" ) == 0 )
            set_data_path(argv[++i]);
        else if( strcmp( argv[i], "-nc" ) == 0 )
            params.color_terminal = 0;
        else if( strcmp( argv[i], "-nh" ) == 0 )
            params.skip_header = 1;
        else if( strcmp( argv[i], "-nb" ) == 0 )
            params.ignore_blacklist = 1;
        else if( strcmp( argv[i], "-r" ) == 0 )
            params.debug_mode = 0;
        else if( strcmp( argv[i], "-tn" ) == 0 )
        {
            params.test_filter_pattern = argv[++i];
            params.test_filter_mode = CHOOSE_TESTS;
        }
        else if( strcmp( argv[i], "-seed" ) == 0 )
        {
            params.rng_seed = read_seed(argv[++i]);
            if( params.rng_seed == 0 )
                fprintf(stderr, "Invalid or zero RNG seed. Will use the seed from the config file or default one\n");
        }
    }

    // this is the fallback for the current OpenCV autotools setup
    if( !params.data_path || !params.data_path[0] )
    {
        char* datapath_dir = getenv("OPENCV_TEST_DATA_PATH");
        char buf[1024];
        if( datapath_dir )
        {
            sprintf( buf, "%s/%s", datapath_dir, module_name ? module_name : "" );
            //printf( LOG + SUMMARY, "Data Path = %s\n", buf);
            set_data_path(buf);
        }
    }

    if( write_params )
    {
        if( !config_name )
        {
            printf( LOG, "ERROR: output config name is not specified\n" );
            return -1;
        }

        fs = cvOpenFileStorage( config_name, 0, CV_STORAGE_WRITE );
        if( !fs )
        {
            printf( LOG, "ERROR: could not open config file %s\n", config_name );
            return -1;
        }

        cvWriteComment( fs, CV_TS_VERSION " config file", 0 );
        cvStartWriteStruct( fs, "common", CV_NODE_MAP );
        write_default_params( fs );
        cvEndWriteStruct( fs );

        for( i = 0; i < all_tests.size(); i++ )
        {
            test = (CvTest*)all_tests[i];
            if( !(test->get_support_testing_modes() & get_testing_mode()) )
                continue;

            test->write_defaults( this );
            test->clear();
        }

        cvReleaseFileStorage( &fs );

        return 0;
    }

    if( !config_name )
        ;
        //printf( LOG, "WARNING: config name is not specified, using default parameters\n" );
    else
    {
        // 2. read common parameters of test system
        fs = cvOpenFileStorage( config_name, 0, CV_STORAGE_READ );
        if( !fs )
        {
            printf( LOG, "ERROR: could not open config file %s", config_name );
            return -1;
        }
    }

    if( params.test_mode == CORRECTNESS_CHECK_MODE || fs )
    {
        // in the case of algorithmic tests we always run read_params,
        // even if there is no config file
        if( read_params(fs) < 0 )
            return -1;
    }

    if( !ostrm_base_name )
        make_output_stream_base_name( config_name ? config_name : argv[0] );

    ostream_testname_mask = -1; // disable printing test names at initial stage

    // 3. open file streams
    for( i = 0; i < CONSOLE_IDX; i++ )
    {
        char filename[MAX_PATH];
        sprintf( filename, "%s%s", ostrm_base_name, ostrm_suffixes[i] );
        output_streams[i].f = fopen( filename, "wt" );
        if( !output_streams[i].f )
        {
            printf( LOG, "ERROR: could not open %s\n", filename );
            return -1;
        }

        if( i == LOG_IDX )
        {
            // redirect stderr to log file
            fflush( stderr );
            output_streams[i].default_handle = dup(2);
            dup2( fileno(output_streams[i].f), 2 );
        }
    }

    int filter_state = 0;

    // 4. traverse through the list of all registered tests.
    // Initialize the selected tests and put them into the separate sequence
    for( i = 0; i < all_tests.size(); i++ )
    {
        test = (CvTest*)all_tests[i];
        if( !(test->get_support_testing_modes() & get_testing_mode()) )
            continue;

        if( strcmp( test->get_func_list(), "" ) != 0 && filter(test, filter_state, blacklist) )
        {
            if( test->init(this) >= 0 )
            {
                selected_tests->push( test );
                if( list_tests )
                    ::printf( "%s\n", test->get_name() );
            }
            else
                printf( LOG, "WARNING: an error occured during test %s initialization\n", test->get_name() );
        }
    }

    if( list_tests )
    {
        clear();
        return 0;
    }

    // 5. setup all the neccessary handlers and print header
    set_handlers( !params.debug_mode );

    if( params.use_optimized == 0 )
        cvUseOptimized(0);

    if( !params.skip_header )
        print_summary_header( SUMMARY + LOG + CONSOLE + CSV );
    rng = params.rng_seed;
    update_context( 0, -1, true );

    // 6. run all the tests
    for( i = 0; i < selected_tests->size(); i++ )
    {
        CvTest* test = (CvTest*)selected_tests->at(i);
        int code;
        CvTestInfo temp;

        if( memory_manager )
            memory_manager->start_tracking();

        update_context( test, -1, true );
        current_test_info.rng_seed0 = current_test_info.rng_seed;

        ostream_testname_mask = 0; // reset "test name was printed" flags
        logbuf = std::string();
        if( output_streams[LOG_IDX].f )
            fflush( output_streams[LOG_IDX].f );

        temp = current_test_info;
        test->safe_run(0);
        if( get_err_code() >= 0 )
        {
            update_context( test, -1, false );
            current_test_info.rng_seed = temp.rng_seed;
            current_test_info.base_alloc_index = temp.base_alloc_index;
        }

        test->clear();

        if( memory_manager )
            memory_manager->stop_tracking_and_check();

        code = get_err_code();
        if( code >= 0 )
        {
            if( !params.print_only_failed )
            {
                printf( SUMMARY + CONSOLE, "\t" );
                set_color( CV_TS_GREEN );
                printf( SUMMARY + CONSOLE, "Ok\n" );
                set_color( CV_TS_NORMAL );
            }
        }
        else
        {
            printf( SUMMARY + CONSOLE, "\t" );
            set_color( CV_TS_RED );
            printf( SUMMARY + CONSOLE, "FAIL(%s)\n", str_from_code(code) );
            set_color( CV_TS_NORMAL );
            printf( LOG, "context: test case = %d, seed = %08x%08x\n",
                    current_test_info.test_case_idx,
                    (unsigned)(current_test_info.rng_seed>>32),
                    (unsigned)(current_test_info.rng_seed));
            if(logbuf.size() > 0)
            {
                printf( SUMMARY + CONSOLE, ">>>\n%s\n", logbuf.c_str());
            }
            failed_tests->push(current_test_info);
            if( params.rerun_immediately )
                break;
        }
    }

    ostream_testname_mask = -1;
    print_summary_tailer( SUMMARY + CONSOLE + LOG );

    if( !params.debug_mode && (params.rerun_failed || params.rerun_immediately) )
    {
        set_handlers(0);
        update_context( 0, -1, true );
        for( i = 0; i < failed_tests->size(); i++ )
        {
            CvTestInfo info = failed_tests->at(i);
            if( (info.code == FAIL_MEMORY_CORRUPTION_BEGIN ||
                 info.code == FAIL_MEMORY_CORRUPTION_END ||
                 info.code == FAIL_MEMORY_LEAK) && memory_manager )
                memory_manager->start_tracking( info.alloc_index - info.base_alloc_index
                                                + memory_manager->get_alloc_index() );
            rng = info.rng_seed;
            test->safe_run( info.test_case_idx );
        }
    }

    int nfailed = failed_tests ? (int)failed_tests->size() : 0;

    clear();

    return nfailed;
}


void CvTS::print_help(void)
{
    ::printf(
        "Usage: <test_executable> [{-h|--help}][-l] [-r] [-w] [-t] [-f <config_name>] [-d <data_path>] [-O{0|1}] [-tn <test_name>]\n\n"
        "-d - specify the test data path\n"
        "-f - use parameters from the provided XML/YAML config file\n"
        "     instead of the default parameters\n"
        "-h or --help - print this help information\n"
        "-l - list all the registered tests or subset of the tests,\n"
        "     selected in the config file, and exit\n"
        "-tn - only run a specific test\n"
        "-nc - do not use colors in the console output\n"
        "-nh - do not print the header\n"
        "-O{0|1} - disable/enable on-fly detection of IPP and other\n"
        "          supported optimized libs. It's enabled by default\n"
        "-r - continue running tests after OS/Hardware exception occured\n"
        "-t - switch to the performance testing mode instead of\n"
        "     the default algorithmic/correctness testing mode\n"
        "-w - write default parameters of the algorithmic or\n"
        "     performance (when -t is passed) tests to the specifed\n"
        "     config file (see -f) and exit\n\n"
        //"Test data path and config file can also be specified by the environment variables 'config' and 'datapath'.\n\n"
        );
}


#if defined WIN32 || defined _WIN32
const char* default_data_path = "../tests/cv/testdata/";
#else
const char* default_data_path = "../../../../tests/cv/testdata/";
#endif


int CvTS::read_params( CvFileStorage* fs )
{
    CvFileNode* node = fs ? cvGetFileNodeByName( fs, 0, "common" ) : 0;

    if(params.debug_mode < 0)
        params.debug_mode = cvReadIntByName( fs, node, "debug_mode", 1 ) != 0;

    if( params.skip_header < 0 )
        params.skip_header = cvReadIntByName( fs, node, "skip_header", 0 ) > 0;

    if( params.ignore_blacklist < 0 )
        params.ignore_blacklist = cvReadIntByName( fs, node, "ignore_blacklist", 0 ) > 0;

    params.print_only_failed  = cvReadIntByName( fs, node, "print_only_failed", 0 ) != 0;
    params.rerun_failed       = cvReadIntByName( fs, node, "rerun_failed", 0 ) != 0;
    params.rerun_immediately  = cvReadIntByName( fs, node, "rerun_immediately", 0 ) != 0;
    const char* str           = cvReadStringByName( fs, node, "filter_mode", "tests" );
    params.test_filter_mode   = strcmp( str, "functions" ) == 0 ? CHOOSE_FUNCTIONS : CHOOSE_TESTS;

    str = cvReadStringByName( fs, node, "test_mode", params.test_mode == TIMING_MODE ? "timing" : "correctness" );

    params.test_mode          = strcmp( str, "timing" ) == 0 || strcmp( str, "performance" ) == 0 ?
                                  TIMING_MODE : CORRECTNESS_CHECK_MODE;

    str = cvReadStringByName( fs, node, "timing_mode", params.timing_mode == AVG_TIME ? "avg" : "min" );

    params.timing_mode         = strcmp( str, "average" ) == 0 || strcmp( str, "avg" ) == 0 ? AVG_TIME : MIN_TIME;
    params.test_filter_pattern = params.test_filter_pattern != 0 &&
      strlen(params.test_filter_pattern) > 0 ? params.test_filter_pattern :
        cvReadStringByName( fs, node, params.test_filter_mode == CHOOSE_FUNCTIONS ?
        "functions" : "tests", "" );
    params.resource_path = cvReadStringByName( fs, node, "." );

    if( params.use_optimized < 0 )
        params.use_optimized = cvReadIntByName( fs, node, "use_optimized", -1 );

    if( !params.data_path || !params.data_path[0] )
    {
        const char* data_path =
            cvReadStringByName( fs, node, "data_path", default_data_path );
        set_data_path(data_path);
    }

    params.test_case_count_scale = cvReadRealByName( fs, node, "test_case_count_scale", 1. );

    if( params.test_case_count_scale <= 0 )
        params.test_case_count_scale = 1.;

    str = cvReadStringByName( fs, node, "seed", 0 );

    if( str && params.rng_seed == 0 )
        params.rng_seed = read_seed(str);

    if( params.rng_seed == 0 )
        params.rng_seed = cvGetTickCount();

    str = cvReadStringByName( fs, node, "output_file_base_name", 0 );
    if( str )
        make_output_stream_base_name( str );

    return 0;
}


void CvTS::write_default_params( CvFileStorage* fs )
{
    read_params(0); // fill parameters with default values

    cvWriteInt( fs, "debug_mode", params.debug_mode );
    cvWriteInt( fs, "skip_header", params.skip_header );
    cvWriteInt( fs, "print_only_failed", params.print_only_failed );
    cvWriteInt( fs, "rerun_failed", params.rerun_failed );
    cvWriteInt( fs, "rerun_immediately", params.rerun_immediately );
    cvWriteString( fs, "filter_mode", params.test_filter_mode == CHOOSE_FUNCTIONS ? "functions" : "tests" );
    cvWriteString( fs, "test_mode", params.test_mode == TIMING_MODE ? "timing" : "correctness" );
    cvWriteString( fs, "data_path", params.data_path ? params.data_path : default_data_path, 1 );
    if( params.test_mode == TIMING_MODE )
        cvWriteString( fs, "timing_mode", params.timing_mode == AVG_TIME ? "avg" : "min" );
    // test_filter, seed & output_file_base_name are not written
}


void CvTS::enable_output_streams( int stream_mask, int value )
{
    for( int i = 0; i < MAX_IDX; i++ )
        if( stream_mask & (1 << i) )
            output_streams[i].enable = value != 0;
}


void CvTS::update_context( CvTest* test, int test_case_idx, bool update_ts_context )
{
    current_test_info.test = test;
    current_test_info.test_case_idx = test_case_idx;
    current_test_info.alloc_index = 0;
    current_test_info.code = 0;
    cvSetErrStatus( CV_StsOk );
    if( update_ts_context )
    {
        current_test_info.rng_seed = rng;
        current_test_info.base_alloc_index = memory_manager ?
            memory_manager->get_alloc_index() : 0;
    }
}


void CvTS::set_failed_test_info( int fail_code, int alloc_index )
{
    if( fail_code == FAIL_MEMORY_CORRUPTION_BEGIN ||
        fail_code == FAIL_MEMORY_CORRUPTION_END ||
        current_test_info.code >= 0 )
    {
        current_test_info.code = fail_code;
        current_test_info.alloc_index = alloc_index;
    }
}


const char* CvTS::get_libs_info( const char** addon_modules )
{
    const char* all_info = 0;
    cvGetModuleInfo( 0, &all_info, addon_modules );
    return all_info;
}


void CvTS::print_summary_header( int streams )
{
    char csv_header[256], *ptr = csv_header;
    int i;

    printf( streams, "Engine: %s\n", version );
    time_t t1;
    time( &t1 );
    struct tm *t2 = localtime( &t1 );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );
    printf( streams, "Execution Date & Time: %s\n", buf );
    printf( streams, "Config File: %s\n", config_name );
    const char* plugins = 0;
    const char* lib_verinfo = get_libs_info( &plugins );
    printf( streams, "Tested Libraries: %s\n", lib_verinfo );
    printf( streams, "Optimized Low-level Plugin\'s: %s\n", plugins );
    printf( streams, "=================================================\n");

    sprintf( ptr, "funcName,dataType,channels,size," );
    ptr += strlen(ptr);

    for( i = 0; i < CvTest::TIMING_EXTRA_PARAMS; i++ )
    {
        sprintf( ptr, "param%d,", i );
        ptr += strlen(ptr);
    }

    sprintf( ptr, "CPE,Time(uSecs)" );
    printf( CSV, "%s\n", csv_header );
}


void CvTS::print_summary_tailer( int streams )
{
    printf( streams, "=================================================\n");
    if( selected_tests && failed_tests )
    {
        time_t end_time;
        time( &end_time );
        double total_time = difftime( end_time, start_time );
        printf( streams, "Summary: %d out of %d tests failed\n",
            failed_tests->size(), selected_tests->size() );
        int minutes = cvFloor(total_time/60.);
        int seconds = cvRound(total_time - minutes*60);
        int hours = minutes / 60;
        minutes %= 60;
        printf( streams, "Running time: %02d:%02d:%02d\n", hours, minutes, seconds );
    }
}

#if defined _MSC_VER && _MSC_VER < 1400
#undef vsnprintf
#define vsnprintf _vsnprintf
#endif

void CvTS::vprintf( int streams, const char* fmt, va_list l )
{
    if( streams )
    {
        char str[1 << 14];
        vsnprintf( str, sizeof(str)-1, fmt, l );

        for( int i = 0; i < MAX_IDX; i++ )
        {
            if( (streams & (1 << i)) && output_streams[i].enable )
            {
                FILE* f = i == CONSOLE_IDX ? stdout :
                          i == LOG_IDX ? stderr : output_streams[i].f;
                if( f )
                {
                    if( i != CSV_IDX && !(ostream_testname_mask & (1 << i)) && current_test_info.test )
                    {
                        fprintf( f, "-------------------------------------------------\n" );
                        if( i == CONSOLE_IDX || i == SUMMARY_IDX )
                          fprintf( f, "[%08x%08x]\n", (int)(current_test_info.rng_seed0 >> 32),
                            (int)(current_test_info.rng_seed0));
                        fprintf( f, "%s: ", current_test_info.test->get_name() );
                        fflush( f );
                        ostream_testname_mask |= 1 << i;
                        if( i == LOG_IDX )
                            logbuf = std::string();
                    }
                    fputs( str, f );
                    if( i == LOG_IDX )
                        logbuf += std::string(str);
                    if( i == CONSOLE_IDX )
                        fflush(f);
                }
            }
        }
    }
}


void CvTS::printf( int streams, const char* fmt, ... )
{
    if( streams )
    {
        va_list l;
        va_start( l, fmt );
        vprintf( streams, fmt, l );
        va_end( l );
    }
}


void CvTS::set_color(int color)
{
    if( params.color_terminal )
        change_color(color);
}


static char* cv_strnstr( const char* str, int len,
                         const char* pattern,
                         int pattern_len = -1,
                         int whole_word = 1 )
{
    int i;

    if( len < 0 && pattern_len < 0 )
        return (char*)strstr( str, pattern );

    if( len < 0 )
        len = (int)strlen( str );

    if( pattern_len < 0 )
        pattern_len = (int)strlen( pattern );

    for( i = 0; i < len - pattern_len + 1; i++ )
    {
        int j = i + pattern_len;
        if( str[i] == pattern[0] &&
            memcmp( str + i, pattern, pattern_len ) == 0 &&
            (!whole_word ||
            ((i == 0 || (!isalnum(str[i-1]) && str[i-1] != '_')) &&
             (j == len || (!isalnum(str[j]) && str[j] != '_')))))
            return (char*)(str + i);
    }

    return 0;
}


int CvTS::filter( CvTest* test, int& filter_state, const char** blacklist )
{
    const char* pattern = params.test_filter_pattern;
    const char* test_name = test->get_name();
    int inverse = 0;
    int greater_or_equal = 0;

    if( blacklist && !params.ignore_blacklist )
    {
        for( ; *blacklist != 0; blacklist++ )
        {
            if( strcmp( *blacklist, test_name ) == 0 )
                return 0;
        }
    }
    
    if( pattern && pattern[0] == '!' )
    {
        inverse = 1;
        pattern++;
    }
    
    if( pattern && pattern[0] == '>' )
    {
        greater_or_equal = 1;
        pattern++;
        if( pattern[0] == '=' )
        {
            greater_or_equal = 2;
            pattern++;
        }
    }
    
    if( !pattern || strcmp( pattern, "" ) == 0 || strcmp( pattern, "*" ) == 0 )
        return 1 ^ inverse;
    
    if( params.test_filter_mode == CHOOSE_TESTS )
    {
        int found = 0;

        while( pattern && *pattern )
        {
            char *ptr, *endptr = (char*)strchr( pattern, ',' );
            int len, have_wildcard;
            int t_name_len;

            if( endptr )
                *endptr = '\0';

            ptr = (char*)strchr( pattern, '*' );
            if( ptr )
            {
                len = (int)(ptr - pattern);
                have_wildcard = 1;
            }
            else
            {
                len = (int)strlen( pattern );
                have_wildcard = 0;
            }

            t_name_len = (int)strlen( test_name );
            found = (t_name_len == len || (have_wildcard && t_name_len > len)) &&
                    (len == 0 || memcmp( test_name, pattern, len ) == 0);
            if( endptr )
            {
                *endptr = ',';
                pattern = endptr + 1;
                while( isspace(*pattern) )
                    pattern++;
            }

            if( found || !endptr )
                break;
        }

        if( greater_or_equal == 0 )
            return found ^ inverse;
        if( filter_state )
            return inverse^1;
        if( !found )
            return inverse;
        if( greater_or_equal == 1 )
        {
            filter_state = 1;
            return inverse;
        }
        else
        {
            assert(filter_state == 2);
            filter_state = 1;
            return inverse ^ 1;
        }
    }
    else
    {
        assert( params.test_filter_mode == CHOOSE_FUNCTIONS );
        int glob_len = (int)strlen( pattern );
        const char* ptr = test->get_func_list();
        const char *tmp_ptr;

        while( ptr && *ptr )
        {
            const char* endptr = ptr - 1;
            const char* name_ptr;
            const char* name_first_match;
            int name_len;
            char c;

            do c = *++endptr;
            while( isspace(c) );

            if( !c )
                break;

            assert( isalpha(c) );
            name_ptr = endptr;

            do c = *++endptr;
            while( isalnum(c) || c == '_' );

            if( c == ':' ) // class
            {
                assert( endptr[1] == ':' );
                endptr = endptr + 2;
                name_len = (int)(endptr - name_ptr);

                // find the first occurence of the class name
                // in pattern
                name_first_match = cv_strnstr( pattern,
                                      glob_len, name_ptr, name_len, 1 );

                if( *endptr == '*' )
                {
                    if( name_first_match )
                        return 1 ^ inverse;
                }
                else
                {
                    assert( *endptr == '{' ); // a list of methods

                    if( !name_first_match )
                    {
                        // skip all the methods, if there is no such a class name
                        // in pattern
                        endptr = strchr( endptr, '}' );
                        assert( endptr != 0 );
                        endptr--;
                    }

                    for( ;; )
                    {
                        const char* method_name_ptr;
                        int method_name_len;

                        do c = *++endptr;
                        while( isspace(c) );

                        if( c == '}' )
                            break;
                        assert( isalpha(c) );

                        method_name_ptr = endptr;

                        do c = *++endptr;
                        while( isalnum(c) || c == '_' );

                        method_name_len = (int)(endptr - method_name_ptr);

                        // search for class_name::* or
                        // class_name::{...method_name...}
                        tmp_ptr = name_first_match;
                        do
                        {
                            const char* tmp_ptr2;
                            tmp_ptr += name_len;
                            if( *tmp_ptr == '*' )
                                return 1;
                            assert( *tmp_ptr == '{' );
                            tmp_ptr2 = strchr( tmp_ptr, '}' );
                            assert( tmp_ptr2 );

                            if( cv_strnstr( tmp_ptr, (int)(tmp_ptr2 - tmp_ptr) + 1,
                                             method_name_ptr, method_name_len, 1 ))
                                return 1 ^ inverse;

                            tmp_ptr = cv_strnstr( tmp_ptr2, glob_len -
                                                   (int)(tmp_ptr2 - pattern),
                                                   name_ptr, name_len, 1 );
                        }
                        while( tmp_ptr );

                        endptr--;
                        do c = *++endptr;
                        while( isspace(c) );

                        if( c != ',' )
                            endptr--;
                    }
                }
            }
            else
            {
                assert( !c || isspace(c) || c == ',' );
                name_len = (int)(endptr - name_ptr);
                tmp_ptr = pattern;

                for(;;)
                {
                    const char *tmp_ptr2, *tmp_ptr3;

                    tmp_ptr = cv_strnstr( tmp_ptr, glob_len -
                        (int)(tmp_ptr - pattern), name_ptr, name_len, 1 );

                    if( !tmp_ptr )
                        break;

                    // make sure it is not a method
                    tmp_ptr2 = strchr( tmp_ptr, '}' );
                    if( !tmp_ptr2 )
                        return 1 ^ inverse;

                    tmp_ptr3 = strchr( tmp_ptr, '{' );
                    if( tmp_ptr3 < tmp_ptr2 )
                        return 1 ^ inverse;

                    tmp_ptr = tmp_ptr2 + 1;
                }

                endptr--;
            }

            do c = *++endptr;
            while( isspace(c) );

            if( c == ',' )
                endptr++;
            ptr = endptr;
        }

        return 0 ^ inverse;
    }
}

/* End of file. */
