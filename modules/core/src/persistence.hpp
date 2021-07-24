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

enum
{
    HEADER_SIZE         = 24,
    ENCODED_HEADER_SIZE = 32
};

} // base64::

//=====================================================================================

#define CV_FS_MAX_LEN 4096
#define CV_FS_MAX_FMT_PAIRS  128

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

namespace cv
{
namespace fs
{
int strcasecmp(const char* str1, const char* str2);
char* itoa( int _val, char* buffer, int /*radix*/ );
char* floatToString( char* buf, float value, bool halfprecision, bool explicitZero );
char* doubleToString( char* buf, double value, bool explicitZero );

int calcStructSize( const char* dt, int initial_size );
int calcElemSize( const char* dt, int initial_size );
char* encodeFormat( int elem_type, char* dt );
int decodeFormat( const char* dt, int* fmt_pairs, int max_len );
int decodeSimpleFormat( const char* dt );
}


#ifdef CV_STATIC_ANALYSIS
#define CV_PARSE_ERROR_CPP(errmsg) do { (void)fs; abort(); } while (0)
#else
#define CV_PARSE_ERROR_CPP( errmsg ) \
    fs->parseError( CV_Func, (errmsg), __FILE__, __LINE__ )
#endif


#define CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP() do { \
    CV_DbgAssert(ptr); \
    if((ptr)[0] == 0 && (ptr) == fs->bufferEnd() - 1) CV_PARSE_ERROR_CPP("OpenCV persistence doesn't support very long lines"); \
} while (0)


class FileStorageParser;
class FileStorageEmitter;

struct FStructData
{
    FStructData() { indent = flags = 0; }
    FStructData( const std::string& _struct_tag,
                 int _struct_flags, int _struct_indent )
    {
        tag = _struct_tag;
        flags = _struct_flags;
        indent = _struct_indent;
    }

    std::string tag;
    int flags;
    int indent;
};

class FileStorage_API
{
public:
    virtual ~FileStorage_API();
    virtual FileStorage* getFS() = 0;

    virtual void puts( const char* str ) = 0;
    virtual char* gets() = 0;
    virtual bool eof() = 0;
    virtual void setEof() = 0;
    virtual void closeFile() = 0;
    virtual void rewind() = 0;
    virtual char* resizeWriteBuffer( char* ptr, int len ) = 0;
    virtual char* bufferPtr() const = 0;
    virtual char* bufferStart() const = 0;
    virtual char* bufferEnd() const = 0;
    virtual void setBufferPtr(char* ptr) = 0;
    virtual char* flush() = 0;
    virtual void setNonEmpty() = 0;
    virtual int wrapMargin() const = 0;

    virtual FStructData& getCurrentStruct() = 0;

    virtual void convertToCollection( int type, FileNode& node ) = 0;
    virtual FileNode addNode( FileNode& collection, const std::string& key,
                               int type, const void* value=0, int len=-1 ) = 0;
    virtual void finalizeCollection( FileNode& collection ) = 0;
    virtual double strtod(char* ptr, char** endptr) = 0;

    virtual char* parseBase64(char* ptr, int indent, FileNode& collection) = 0;
    CV_NORETURN
    virtual void parseError(const char* funcname, const std::string& msg,
                            const char* filename, int lineno) = 0;

private:
    enum Base64State{
        Uncertain,
        NotUse,
        InUse,
    };

    friend class cv::FileStorage::Impl;
    friend class cv::FileStorage;
    friend class JSONEmitter;
    friend class XMLEmitter;
    friend class YAMLEmitter;

    virtual void check_if_write_struct_is_delayed(bool change_type_to_base64 = false) = 0;
    virtual void switch_to_Base64_state(Base64State state) = 0;
    virtual Base64State get_state_of_writing_base64() = 0;
    virtual int get_space() = 0;
};

class FileStorageEmitter
{
public:
    virtual ~FileStorageEmitter() {}

    virtual FStructData startWriteStruct( const FStructData& parent, const char* key,
                                          int struct_flags, const char* type_name=0 ) = 0;
    virtual void endWriteStruct(const FStructData& current_struct) = 0;
    virtual void write(const char* key, int value) = 0;
    virtual void write(const char* key, double value) = 0;
    virtual void write(const char* key, const char* value, bool quote) = 0;
    virtual void writeScalar(const char* key, const char* value) = 0;
    virtual void writeComment(const char* comment, bool eol_comment) = 0;
    virtual void startNextStream() = 0;
};

class FileStorageParser
{
public:
    virtual ~FileStorageParser() {}
    virtual bool parse(char* ptr) = 0;
    virtual bool getBase64Row(char* ptr, int indent, char* &beg, char* &end) = 0;
};

Ptr<FileStorageEmitter> createXMLEmitter(FileStorage_API* fs);
Ptr<FileStorageEmitter> createYAMLEmitter(FileStorage_API* fs);
Ptr<FileStorageEmitter> createJSONEmitter(FileStorage_API* fs);

Ptr<FileStorageParser> createXMLParser(FileStorage_API* fs);
Ptr<FileStorageParser> createYAMLParser(FileStorage_API* fs);
Ptr<FileStorageParser> createJSONParser(FileStorage_API* fs);

}

#endif // SRC_PERSISTENCE_HPP
