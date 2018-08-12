// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#ifndef SRC_PERSISTENCE_HPP
#define SRC_PERSISTENCE_HPP

#include "opencv2/core/types_c.h"
#include <deque>
#include <sstream>
#include <string>
#include <iterator>

#define USE_ZLIB 1
#if USE_ZLIB
#  ifndef _LFS64_LARGEFILE
#    define _LFS64_LARGEFILE 0
#  endif
#  ifndef _FILE_OFFSET_BITS
#    define _FILE_OFFSET_BITS 0
#  endif
#  include <zlib.h>
#else
typedef void* gzFile;
#endif

//=====================================================================================

static const size_t PARSER_BASE64_BUFFER_SIZE = 1024U * 1024U / 8U;

namespace base64 {

namespace fs {
enum State
{
    Uncertain,
    NotUse,
    InUse,
};
} // fs::

static const size_t HEADER_SIZE         = 24U;
static const size_t ENCODED_HEADER_SIZE = 32U;

size_t base64_encode(uint8_t const * src, uint8_t * dst, size_t off,      size_t cnt);
size_t base64_encode(   char const * src,    char * dst, size_t off = 0U, size_t cnt = 0U);
size_t base64_decode(uint8_t const * src, uint8_t * dst, size_t off,      size_t cnt);
size_t base64_decode(   char const * src,    char * dst, size_t off = 0U, size_t cnt = 0U);
bool   base64_valid (uint8_t const * src, size_t off,      size_t cnt);
bool   base64_valid (   char const * src, size_t off = 0U, size_t cnt = 0U);
size_t base64_encode_buffer_size(size_t cnt, bool is_end_with_zero = true);
size_t base64_decode_buffer_size(size_t cnt, bool is_end_with_zero = true);
size_t base64_decode_buffer_size(size_t cnt, char  const * src, bool is_end_with_zero = true);
size_t base64_decode_buffer_size(size_t cnt, uchar const * src, bool is_end_with_zero = true);
std::string make_base64_header(const char * dt);
bool read_base64_header(std::vector<char> const & header, std::string & dt);
void make_seq(void * binary_data, int elem_cnt, const char * dt, CvSeq & seq);
void cvWriteRawDataBase64(::CvFileStorage* fs, const void* _data, int len, const char* dt);

class Base64ContextEmitter;

class Base64Writer
{
public:
    Base64Writer(::CvFileStorage * fs);
    ~Base64Writer();
    void write(const void* _data, size_t len, const char* dt);
    template<typename _to_binary_convertor_t> void write(_to_binary_convertor_t & convertor, const char* dt);

private:
    void check_dt(const char* dt);

private:
    // disable copy and assignment
    Base64Writer(const Base64Writer &);
    Base64Writer & operator=(const Base64Writer &);

private:

    Base64ContextEmitter * emitter;
    std::string data_type_string;
};

class Base64ContextParser
{
public:
    explicit Base64ContextParser(uchar * buffer, size_t size);
    ~Base64ContextParser();
    Base64ContextParser & read(const uchar * beg, const uchar * end);
    bool flush();
private:
    static const size_t BUFFER_LEN = 120U;
    uchar * dst_cur;
    uchar * dst_end;
    std::vector<uchar> base64_buffer;
    uchar * src_beg;
    uchar * src_cur;
    uchar * src_end;
    std::vector<uchar> binary_buffer;
};

} // base64::

//=====================================================================================

#define CV_FS_MAX_LEN 4096
#define CV_FS_MAX_FMT_PAIRS  128

#define CV_FILE_STORAGE ('Y' + ('A' << 8) + ('M' << 16) + ('L' << 24))

#define CV_IS_FILE_STORAGE(fs) ((fs) != 0 && (fs)->flags == CV_FILE_STORAGE)

#define CV_CHECK_FILE_STORAGE(fs)                       \
{                                                       \
    if( !CV_IS_FILE_STORAGE(fs) )                       \
        CV_Error( (fs) ? CV_StsBadArg : CV_StsNullPtr,  \
                  "Invalid pointer to file storage" );  \
}

#define CV_CHECK_OUTPUT_FILE_STORAGE(fs)                \
{                                                       \
    CV_CHECK_FILE_STORAGE(fs);                          \
    if( !fs->write_mode )                               \
        CV_Error( CV_StsError, "The file storage is opened for reading" ); \
}

#define CV_PARSE_ERROR( errmsg )                                    \
    icvParseError( fs, CV_Func, (errmsg), __FILE__, __LINE__ )

typedef struct CvGenericHash
{
    CV_SET_FIELDS()
    int tab_size;
    void** table;
}
CvGenericHash;
typedef CvGenericHash CvStringHash;

//typedef void (*CvParse)( struct CvFileStorage* fs );
typedef void (*CvStartWriteStruct)( struct CvFileStorage* fs, const char* key,
                                    int struct_flags, const char* type_name );
typedef void (*CvEndWriteStruct)( struct CvFileStorage* fs );
typedef void (*CvWriteInt)( struct CvFileStorage* fs, const char* key, int value );
typedef void (*CvWriteReal)( struct CvFileStorage* fs, const char* key, double value );
typedef void (*CvWriteString)( struct CvFileStorage* fs, const char* key,
                               const char* value, int quote );
typedef void (*CvWriteComment)( struct CvFileStorage* fs, const char* comment, int eol_comment );
typedef void (*CvStartNextStream)( struct CvFileStorage* fs );

typedef struct CvFileStorage
{
    int flags;
    int fmt;
    int write_mode;
    int is_first;
    CvMemStorage* memstorage;
    CvMemStorage* dststorage;
    CvMemStorage* strstorage;
    CvStringHash* str_hash;
    CvSeq* roots;
    CvSeq* write_stack;
    int struct_indent;
    int struct_flags;
    CvString struct_tag;
    int space;
    char* filename;
    FILE* file;
    gzFile gzfile;
    char* buffer;
    char* buffer_start;
    char* buffer_end;
    int wrap_margin;
    int lineno;
    int dummy_eof;
    const char* errmsg;
    char errmsgbuf[128];

    CvStartWriteStruct start_write_struct;
    CvEndWriteStruct end_write_struct;
    CvWriteInt write_int;
    CvWriteReal write_real;
    CvWriteString write_string;
    CvWriteComment write_comment;
    CvStartNextStream start_next_stream;

    const char* strbuf;
    size_t strbufsize, strbufpos;
    std::deque<char>* outbuf;

    base64::Base64Writer * base64_writer;
    bool is_default_using_base64;
    base64::fs::State state_of_writing_base64;  /**< used in WriteRawData only */

    bool is_write_struct_delayed;
    char* delayed_struct_key;
    int   delayed_struct_flags;
    char* delayed_type_name;

    bool is_opened;
}
CvFileStorage;

typedef struct CvFileMapNode
{
    CvFileNode value;
    const CvStringHashNode* key;
    struct CvFileMapNode* next;
}
CvFileMapNode;

/****************************************************************************************\
*                            Common macros and type definitions                          *
\****************************************************************************************/

#define cv_isprint(c)     ((uchar)(c) >= (uchar)' ')
#define cv_isprint_or_tab(c)  ((uchar)(c) >= (uchar)' ' || (c) == '\t')

inline bool cv_isalnum(char c)
{
    return ('0' <= c && c <= '9') || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
}

inline bool cv_isalpha(char c)
{
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
}

inline bool cv_isdigit(char c)
{
    return '0' <= c && c <= '9';
}

inline bool cv_isspace(char c)
{
    return (9 <= c && c <= 13) || c == ' ';
}

inline char* cv_skip_BOM(char* ptr)
{
    if((uchar)ptr[0] == 0xef && (uchar)ptr[1] == 0xbb && (uchar)ptr[2] == 0xbf) //UTF-8 BOM
    {
      return ptr + 3;
    }
    return ptr;
}

char* icv_itoa( int _val, char* buffer, int /*radix*/ );
double icv_strtod( CvFileStorage* fs, char* ptr, char** endptr );
char* icvFloatToString( char* buf, float value );
char* icvDoubleToString( char* buf, double value );

char icvTypeSymbol(int depth);
void icvClose( CvFileStorage* fs, cv::String* out );
void icvCloseFile( CvFileStorage* fs );
void icvPuts( CvFileStorage* fs, const char* str );
char* icvGets( CvFileStorage* fs, char* str, int maxCount );
int icvEof( CvFileStorage* fs );
void icvRewind( CvFileStorage* fs );
char* icvFSFlush( CvFileStorage* fs );
void icvFSCreateCollection( CvFileStorage* fs, int tag, CvFileNode* collection );
char* icvFSResizeWriteBuffer( CvFileStorage* fs, char* ptr, int len );
int icvCalcStructSize( const char* dt, int initial_size );
int icvCalcElemSize( const char* dt, int initial_size );
void CV_NORETURN icvParseError( CvFileStorage* fs, const char* func_name, const char* err_msg, const char* source_file, int source_line );
char* icvEncodeFormat( int elem_type, char* dt );
int icvDecodeFormat( const char* dt, int* fmt_pairs, int max_len );
int icvDecodeSimpleFormat( const char* dt );
void icvWriteFileNode( CvFileStorage* fs, const char* name, const CvFileNode* node );
void icvWriteCollection( CvFileStorage* fs, const CvFileNode* node );
void switch_to_Base64_state( CvFileStorage* fs, base64::fs::State state );
void make_write_struct_delayed( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name );
void check_if_write_struct_is_delayed( CvFileStorage* fs, bool change_type_to_base64 = false );
CvGenericHash* cvCreateMap( int flags, int header_size, int elem_size, CvMemStorage* storage, int start_tab_size );

//
// XML
//
void icvXMLParse( CvFileStorage* fs );
void icvXMLStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name CV_DEFAULT(0));
void icvXMLEndWriteStruct( CvFileStorage* fs );
void icvXMLStartNextStream( CvFileStorage* fs );
void icvXMLWriteScalar( CvFileStorage* fs, const char* key, const char* data, int len );
void icvXMLWriteInt( CvFileStorage* fs, const char* key, int value );
void icvXMLWriteReal( CvFileStorage* fs, const char* key, double value );
void icvXMLWriteString( CvFileStorage* fs, const char* key, const char* str, int quote );
void icvXMLWriteComment( CvFileStorage* fs, const char* comment, int eol_comment );

typedef struct CvXMLStackRecord
{
    CvMemStoragePos pos;
    CvString struct_tag;
    int struct_indent;
    int struct_flags;
}
CvXMLStackRecord;

//
// YML
//
void icvYMLParse( CvFileStorage* fs );
void icvYMLWrite( CvFileStorage* fs, const char* key, const char* data );
void icvYMLStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name CV_DEFAULT(0));
void icvYMLEndWriteStruct( CvFileStorage* fs );
void icvYMLStartNextStream( CvFileStorage* fs );
void icvYMLWriteInt( CvFileStorage* fs, const char* key, int value );
void icvYMLWriteReal( CvFileStorage* fs, const char* key, double value );
void icvYMLWriteString( CvFileStorage* fs, const char* key, const char* str, int quote CV_DEFAULT(0));
void icvYMLWriteComment( CvFileStorage* fs, const char* comment, int eol_comment );

//
// JSON
//
void icvJSONParse( CvFileStorage* fs );
void icvJSONWrite( CvFileStorage* fs, const char* key, const char* data );
void icvJSONStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name CV_DEFAULT(0));
void icvJSONEndWriteStruct( CvFileStorage* fs );
void icvJSONStartNextStream( CvFileStorage* fs );
void icvJSONWriteInt( CvFileStorage* fs, const char* key, int value );
void icvJSONWriteReal( CvFileStorage* fs, const char* key, double value );
void icvJSONWriteString( CvFileStorage* fs, const char* key, const char* str, int quote CV_DEFAULT(0));
void icvJSONWriteComment( CvFileStorage* fs, const char* comment, int eol_comment );

// Adding icvGets is not enough - we need to merge buffer contents (see #11061)
#define CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG() \
    CV_Assert((ptr[0] != 0 || ptr != fs->buffer_end - 1) && "OpenCV persistence doesn't support very long lines")

#endif // SRC_PERSISTENCE_HPP
