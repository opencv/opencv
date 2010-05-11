/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include "precomp.hpp"
#include <ctype.h>
#include <wchar.h>
#include <zlib.h>

/****************************************************************************************\
*                            Common macros and type definitions                          *
\****************************************************************************************/

#define cv_isprint(c)     ((signed char)(c) >= (signed char)' ')
#define cv_isprint_or_tab(c)  ((signed char)(c) >= (signed char)' ' || (c) == '\t')

static char* icv_itoa( int _val, char* buffer, int /*radix*/ )
{
    const int radix = 10;
    char* ptr=buffer + 23 /* enough even for 64-bit integers */;
    unsigned val = abs(_val);

    *ptr = '\0';
    do
    {
        unsigned r = val / radix;
        *--ptr = (char)(val - (r*radix) + '0');
        val = r;
    }
    while( val != 0 );

    if( _val < 0 )
        *--ptr = '-';

    return ptr;
}

cv::string cv::FileStorage::getDefaultObjectName(const string& _filename)
{
    static const char* stubname = "unnamed";
    const char* filename = _filename.c_str();
    const char* ptr2 = filename + _filename.size();
    const char* ptr = ptr2 - 1;
    cv::AutoBuffer<char> name_buf(_filename.size()+1);

    while( ptr >= filename && *ptr != '\\' && *ptr != '/' && *ptr != ':' )
    {
        if( *ptr == '.' && (!*ptr2 || strncmp(ptr2, ".gz", 3) == 0) )
            ptr2 = ptr;
        ptr--;
    }
    ptr++;
    if( ptr == ptr2 )
        CV_Error( CV_StsBadArg, "Invalid filename" );

    char* name = name_buf;

    // name must start with letter or '_'
    if( !isalpha(*ptr) && *ptr!= '_' ){
        *name++ = '_';
    }

    while( ptr < ptr2 )
    {
        char c = *ptr++;
        if( !isalnum(c) && c != '-' && c != '_' )
            c = '_';
        *name++ = c;
    }
    *name = '\0';
    name = name_buf;
    if( strcmp( name, "_" ) == 0 )
        strcpy( name, stubname );
    return cv::string(name);
}

namespace cv
{

string fromUtf16(const WString& str)
{
    cv::AutoBuffer<char> _buf(str.size()*4 + 1);
    char* buf = _buf;
        
    size_t sz = wcstombs(buf, str.c_str(), str.size());
    if( sz == (size_t)-1 )
        return string();
    buf[sz] = '\0';
    return string(buf);
}

WString toUtf16(const string& str)
{
    cv::AutoBuffer<wchar_t> _buf(str.size() + 1);
    wchar_t* buf = _buf;
        
    size_t sz = mbstowcs(buf, str.c_str(), str.size());
    if( sz == (size_t)-1 )
        return WString();
    buf[sz] = '\0';
    return WString(buf);
}

}

typedef struct CvGenericHash
{
    CV_SET_FIELDS()
    int tab_size;
    void** table;
}
CvGenericHash;

typedef CvGenericHash CvStringHash;

typedef struct CvFileMapNode
{
    CvFileNode value;
    const CvStringHashNode* key;
    struct CvFileMapNode* next;
}
CvFileMapNode;

typedef struct CvXMLStackRecord
{
    CvMemStoragePos pos;
    CvString struct_tag;
    int struct_indent;
    int struct_flags;
}
CvXMLStackRecord;

#define CV_XML_OPENING_TAG 1
#define CV_XML_CLOSING_TAG 2
#define CV_XML_EMPTY_TAG 3
#define CV_XML_HEADER_TAG 4
#define CV_XML_DIRECTIVE_TAG 5

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
    int is_xml;
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
    //CvParse parse;
}
CvFileStorage;

static void icvPuts( CvFileStorage* fs, const char* str )
{
    CV_Assert( fs->file || fs->gzfile );
    if( fs->file )
        fputs( str, fs->file );
    else
        gzputs( fs->gzfile, str );
}

static char* icvGets( CvFileStorage* fs, char* str, int maxCount )
{
    CV_Assert( fs->file || fs->gzfile );
    if( fs->file )
        return fgets( str, maxCount, fs->file );
    return gzgets( fs->gzfile, str, maxCount );
}

static int icvEof( CvFileStorage* fs )
{
    CV_Assert( fs->file || fs->gzfile );
    if( fs->file )
        return feof(fs->file);
    return gzeof(fs->gzfile);
}

static void icvClose( CvFileStorage* fs )
{
    if( fs->file )
        fclose( fs->file );
    if( fs->gzfile )
        gzclose( fs->gzfile );
    fs->file = 0;
    fs->gzfile = 0;
}

static void icvRewind( CvFileStorage* fs )
{
    CV_Assert( fs->file || fs->gzfile );
    if( fs->file )
        rewind(fs->file);
    else
        gzrewind(fs->gzfile);
}

#define CV_YML_INDENT  3
#define CV_XML_INDENT  2
#define CV_YML_INDENT_FLOW  1
#define CV_FS_MAX_LEN 4096

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

CV_IMPL const char*
cvAttrValue( const CvAttrList* attr, const char* attr_name )
{
    while( attr && attr->attr )
    {
        int i;
        for( i = 0; attr->attr[i*2] != 0; i++ )
        {
            if( strcmp( attr_name, attr->attr[i*2] ) == 0 )
                return attr->attr[i*2+1];
        }
        attr = attr->next;
    }

    return 0;
}


static CvGenericHash*
cvCreateMap( int flags, int header_size, int elem_size,
             CvMemStorage* storage, int start_tab_size )
{
    if( header_size < (int)sizeof(CvGenericHash) )
        CV_Error( CV_StsBadSize, "Too small map header_size" );

    if( start_tab_size <= 0 )
        start_tab_size = 16;

    CvGenericHash* map = (CvGenericHash*)cvCreateSet( flags, header_size, elem_size, storage );

    map->tab_size = start_tab_size;
    start_tab_size *= sizeof(map->table[0]);
    map->table = (void**)cvMemStorageAlloc( storage, start_tab_size );
    memset( map->table, 0, start_tab_size );

    return map;
}

#ifdef __GNUC__
#define CV_PARSE_ERROR( errmsg )                                    \
    icvParseError( fs, __func__, (errmsg), __FILE__, __LINE__ )
#else
#define CV_PARSE_ERROR( errmsg )                                    \
    icvParseError( fs, "", (errmsg), __FILE__, __LINE__ )
#endif

static void
icvParseError( CvFileStorage* fs, const char* func_name,
               const char* err_msg, const char* source_file, int source_line )
{
    char buf[1<<10];
    sprintf( buf, "%s(%d): %s", fs->filename, fs->lineno, err_msg );
    cvError( CV_StsParseError, func_name, buf, source_file, source_line );
}


static void
icvFSCreateCollection( CvFileStorage* fs, int tag, CvFileNode* collection )
{
    if( CV_NODE_IS_MAP(tag) )
    {
        if( collection->tag != CV_NODE_NONE )
        {
            assert( fs->is_xml != 0 );
            CV_PARSE_ERROR( "Sequence element should not have name (use <_></_>)" );
        }

        collection->data.map = cvCreateMap( 0, sizeof(CvFileNodeHash),
                            sizeof(CvFileMapNode), fs->memstorage, 16 );
    }
    else
    {
        CvSeq* seq;
        seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvFileNode), fs->memstorage );

        // if <collection> contains some scalar element, add it to the newly created collection
        if( CV_NODE_TYPE(collection->tag) != CV_NODE_NONE )
            cvSeqPush( seq, collection );

        collection->data.seq = seq;
    }

    collection->tag = tag;
    cvSetSeqBlockSize( collection->data.seq, 8 );
}


/*static void
icvFSReleaseCollection( CvSeq* seq )
{
    if( seq )
    {
        int is_map = CV_IS_SET(seq);
        CvSeqReader reader;
        int i, total = seq->total;
        cvStartReadSeq( seq, &reader, 0 );

        for( i = 0; i < total; i++ )
        {
            CvFileNode* node = (CvFileNode*)reader.ptr;

            if( (!is_map || CV_IS_SET_ELEM( node )) && CV_NODE_IS_COLLECTION(node->tag) )
            {
                if( CV_NODE_IS_USER(node->tag) && node->info && node->data.obj.decoded )
                    cvRelease( (void**)&node->data.obj.decoded );
                if( !CV_NODE_SEQ_IS_SIMPLE( node->data.seq ))
                    icvFSReleaseCollection( node->data.seq );
            }
            CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
        }
    }
}*/


static char*
icvFSDoResize( CvFileStorage* fs, char* ptr, int len )
{
    char* new_ptr = 0;
    int written_len = (int)(ptr - fs->buffer_start);
    int new_size = (int)((fs->buffer_end - fs->buffer_start)*3/2);
    new_size = MAX( written_len + len, new_size );
    new_ptr = (char*)cvAlloc( new_size + 256 );
    fs->buffer = new_ptr + (fs->buffer - fs->buffer_start);
    if( written_len > 0 )
        memcpy( new_ptr, fs->buffer_start, written_len );
    fs->buffer_start = new_ptr;
    fs->buffer_end = fs->buffer_start + new_size;
    new_ptr += written_len;
    return new_ptr;
}


inline char* icvFSResizeWriteBuffer( CvFileStorage* fs, char* ptr, int len )
{
    return ptr + len < fs->buffer_end ? ptr : icvFSDoResize( fs, ptr, len );
}


static char*
icvFSFlush( CvFileStorage* fs )
{
    char* ptr = fs->buffer;
    int indent;

    if( ptr > fs->buffer_start + fs->space )
    {
        ptr[0] = '\n';
        ptr[1] = '\0';
        icvPuts( fs, fs->buffer_start );
        fs->buffer = fs->buffer_start;
    }

    indent = fs->struct_indent;

    if( fs->space != indent )
    {
        if( fs->space < indent )
            memset( fs->buffer_start + fs->space, ' ', indent - fs->space );
        fs->space = indent;
    }

    ptr = fs->buffer = fs->buffer_start + fs->space;

    return ptr;
}


/* closes file storage and deallocates buffers */
CV_IMPL  void
cvReleaseFileStorage( CvFileStorage** p_fs )
{
    if( !p_fs )
        CV_Error( CV_StsNullPtr, "NULL double pointer to file storage" );

    if( *p_fs )
    {
        CvFileStorage* fs = *p_fs;
        *p_fs = 0;

        if( fs->write_mode && (fs->file || fs->gzfile) )
        {
            if( fs->write_stack )
            {
                while( fs->write_stack->total > 0 )
                    cvEndWriteStruct(fs);
            }
            icvFSFlush(fs);
            if( fs->is_xml )
                icvPuts( fs, "</opencv_storage>\n" );
        }

        //icvFSReleaseCollection( fs->roots ); // delete all the user types recursively

        icvClose(fs);

        cvReleaseMemStorage( &fs->strstorage );

        cvFree( &fs->buffer_start );
        cvReleaseMemStorage( &fs->memstorage );

        memset( fs, 0, sizeof(*fs) );
        cvFree( &fs );
    }
}


#define CV_HASHVAL_SCALE 33

CV_IMPL CvStringHashNode*
cvGetHashedKey( CvFileStorage* fs, const char* str, int len, int create_missing )
{
    CvStringHashNode* node = 0;
    unsigned hashval = 0;
    int i, tab_size;
    CvStringHash* map = fs->str_hash;

    if( !fs )
        return 0;

    if( len < 0 )
    {
        for( i = 0; str[i] != '\0'; i++ )
            hashval = hashval*CV_HASHVAL_SCALE + (unsigned char)str[i];
        len = i;
    }
    else for( i = 0; i < len; i++ )
        hashval = hashval*CV_HASHVAL_SCALE + (unsigned char)str[i];

    hashval &= INT_MAX;
    tab_size = map->tab_size;
    if( (tab_size & (tab_size - 1)) == 0 )
        i = (int)(hashval & (tab_size - 1));
    else
        i = (int)(hashval % tab_size);

    for( node = (CvStringHashNode*)(map->table[i]); node != 0; node = node->next )
    {
        if( node->hashval == hashval &&
            node->str.len == len &&
            memcmp( node->str.ptr, str, len ) == 0 )
            break;
    }

    if( !node && create_missing )
    {
        node = (CvStringHashNode*)cvSetNew( (CvSet*)map );
        node->hashval = hashval;
        node->str = cvMemStorageAllocString( map->storage, str, len );
        node->next = (CvStringHashNode*)(map->table[i]);
        map->table[i] = node;
    }

    return node;
}


CV_IMPL CvFileNode*
cvGetFileNode( CvFileStorage* fs, CvFileNode* _map_node,
               const CvStringHashNode* key,
               int create_missing )
{
    CvFileNode* value = 0;
    int k = 0, attempts = 1;

    if( !fs )
        return 0;

    CV_CHECK_FILE_STORAGE(fs);

    if( !key )
        CV_Error( CV_StsNullPtr, "Null key element" );

    if( _map_node )
    {
        if( !fs->roots )
            return 0;
        attempts = fs->roots->total;
    }

    for( k = 0; k < attempts; k++ )
    {
        int i, tab_size;
        CvFileNode* map_node = _map_node;
        CvFileMapNode* another;
        CvFileNodeHash* map;

        if( !map_node )
            map_node = (CvFileNode*)cvGetSeqElem( fs->roots, k );

        if( !CV_NODE_IS_MAP(map_node->tag) )
        {
            if( (!CV_NODE_IS_SEQ(map_node->tag) || map_node->data.seq->total != 0) &&
                CV_NODE_TYPE(map_node->tag) != CV_NODE_NONE )
                CV_Error( CV_StsError, "The node is neither a map nor an empty collection" );
            return 0;
        }

        map = map_node->data.map;
        tab_size = map->tab_size;

        if( (tab_size & (tab_size - 1)) == 0 )
            i = (int)(key->hashval & (tab_size - 1));
        else
            i = (int)(key->hashval % tab_size);

        for( another = (CvFileMapNode*)(map->table[i]); another != 0; another = another->next )
            if( another->key == key )
            {
                if( !create_missing )
                {
                    value = &another->value;
                    return value;
                }
                CV_PARSE_ERROR( "Duplicated key" );
            }

        if( k == attempts - 1 && create_missing )
        {
            CvFileMapNode* node = (CvFileMapNode*)cvSetNew( (CvSet*)map );
            node->key = key;

            node->next = (CvFileMapNode*)(map->table[i]);
            map->table[i] = node;
            value = (CvFileNode*)node;
        }
    }

    return value;
}


CV_IMPL CvFileNode*
cvGetFileNodeByName( const CvFileStorage* fs, const CvFileNode* _map_node, const char* str )
{
    CvFileNode* value = 0;
    int i, len, tab_size;
    unsigned hashval = 0;
    int k = 0, attempts = 1;

    if( !fs )
        return 0;

    CV_CHECK_FILE_STORAGE(fs);

    if( !str )
        CV_Error( CV_StsNullPtr, "Null element name" );

    for( i = 0; str[i] != '\0'; i++ )
        hashval = hashval*CV_HASHVAL_SCALE + (unsigned char)str[i];
    hashval &= INT_MAX;
    len = i;

    if( !_map_node )
    {
        if( !fs->roots )
            return 0;
        attempts = fs->roots->total;
    }

    for( k = 0; k < attempts; k++ )
    {
        CvFileNodeHash* map;
        const CvFileNode* map_node = _map_node;
        CvFileMapNode* another;

        if( !map_node )
            map_node = (CvFileNode*)cvGetSeqElem( fs->roots, k );

        if( !CV_NODE_IS_MAP(map_node->tag) )
        {
            if( (!CV_NODE_IS_SEQ(map_node->tag) || map_node->data.seq->total != 0) &&
                CV_NODE_TYPE(map_node->tag) != CV_NODE_NONE )
                CV_Error( CV_StsError, "The node is neither a map nor an empty collection" );
            return 0;
        }

        map = map_node->data.map;
        tab_size = map->tab_size;

        if( (tab_size & (tab_size - 1)) == 0 )
            i = (int)(hashval & (tab_size - 1));
        else
            i = (int)(hashval % tab_size);

        for( another = (CvFileMapNode*)(map->table[i]); another != 0; another = another->next )
        {
            const CvStringHashNode* key = another->key;

            if( key->hashval == hashval &&
                key->str.len == len &&
                memcmp( key->str.ptr, str, len ) == 0 )
            {
                value = &another->value;
                return value;
            }
        }
    }

    return value;
}


CV_IMPL CvFileNode*
cvGetRootFileNode( const CvFileStorage* fs, int stream_index )
{
    CV_CHECK_FILE_STORAGE(fs);

    if( !fs->roots || (unsigned)stream_index >= (unsigned)fs->roots->total )
        return 0;

    return (CvFileNode*)cvGetSeqElem( fs->roots, stream_index );
}


/* returns the sequence element by its index */
/*CV_IMPL CvFileNode*
cvGetFileNodeFromSeq( CvFileStorage* fs,
                      CvFileNode* seq_node, int index )
{
    CvFileNode* value = 0;
    CvSeq* seq;

    if( !seq_node )
        seq = fs->roots;
    else if( !CV_NODE_IS_SEQ(seq_node->tag) )
    {
        if( CV_NODE_IS_MAP(seq_node->tag) )
            CV_Error( CV_StsError, "The node is map. Use cvGetFileNodeFromMap()." );
        if( CV_NODE_TYPE(seq_node->tag) == CV_NODE_NONE )
            CV_Error( CV_StsError, "The node is an empty object (None)." );
        if( index != 0 && index != -1 )
            CV_Error( CV_StsOutOfRange, "" );
        value = seq_node;
        EXIT;
    }
    else
        seq = seq_node->data.seq;

    if( !seq )
        CV_Error( CV_StsNullPtr, "The file storage is empty" );

    value = (CvFileNode*)cvGetSeqElem( seq, index, 0 );

    

    return value;
}*/


static char*
icvDoubleToString( char* buf, double value )
{
    Cv64suf val;
    unsigned ieee754_hi;

    val.f = value;
    ieee754_hi = (unsigned)(val.u >> 32);

    if( (ieee754_hi & 0x7ff00000) != 0x7ff00000 )
    {
        int ivalue = cvRound(value);
        if( ivalue == value )
            sprintf( buf, "%d.", ivalue );
        else
        {
            static const char* fmt = "%.16e";
            char* ptr = buf;
            sprintf( buf, fmt, value );
            if( *ptr == '+' || *ptr == '-' )
                ptr++;
            for( ; isdigit(*ptr); ptr++ )
                ;
            if( *ptr == ',' )
                *ptr = '.';
        }
    }
    else
    {
        unsigned ieee754_lo = (unsigned)val.u;
        if( (ieee754_hi & 0x7fffffff) + (ieee754_lo != 0) > 0x7ff00000 )
            strcpy( buf, ".Nan" );
        else
            strcpy( buf, (int)ieee754_hi < 0 ? "-.Inf" : ".Inf" );
    }

    return buf;
}


static char*
icvFloatToString( char* buf, float value )
{
    Cv32suf val;
    unsigned ieee754;
    val.f = value;
    ieee754 = val.u;

    if( (ieee754 & 0x7f800000) != 0x7f800000 )
    {
        int ivalue = cvRound(value);
        if( ivalue == value )
            sprintf( buf, "%d.", ivalue );
        else
        {
            static const char* fmt = "%.8e";
            char* ptr = buf;
            sprintf( buf, fmt, value );
            if( *ptr == '+' || *ptr == '-' )
                ptr++;
            for( ; isdigit(*ptr); ptr++ )
                ;
            if( *ptr == ',' )
                *ptr = '.';
        }
    }
    else
    {
        if( (ieee754 & 0x7fffffff) != 0x7f800000 )
            strcpy( buf, ".Nan" );
        else
            strcpy( buf, (int)ieee754 < 0 ? "-.Inf" : ".Inf" );
    }

    return buf;
}


static void
icvProcessSpecialDouble( CvFileStorage* fs, char* buf, double* value, char** endptr )
{
    char c = buf[0];
    int inf_hi = 0x7ff00000;

    if( c == '-' || c == '+' )
    {
        inf_hi = c == '-' ? 0xfff00000 : 0x7ff00000;
        c = *++buf;
    }

    if( c != '.' )
        CV_PARSE_ERROR( "Bad format of floating-point constant" );

    if( toupper(buf[1]) == 'I' && toupper(buf[2]) == 'N' && toupper(buf[3]) == 'F' )
        *(uint64*)value = ((uint64)inf_hi << 32);
    else if( toupper(buf[1]) == 'N' && toupper(buf[2]) == 'A' && toupper(buf[3]) == 'N' )
        *(uint64*)value = (uint64)-1;
    else
        CV_PARSE_ERROR( "Bad format of floating-point constant" );

    *endptr = buf + 4;
}


static double icv_strtod( CvFileStorage* fs, char* ptr, char** endptr )
{
    double fval = strtod( ptr, endptr );
    if( **endptr == '.' )
    {
        char* dot_pos = *endptr;
        *dot_pos = ',';
        double fval2 = strtod( ptr, endptr );
        *dot_pos = '.';
        if( *endptr > dot_pos )
            fval = fval2;
        else
            *endptr = dot_pos;
    }

    if( *endptr == ptr || isalpha(**endptr) )
        icvProcessSpecialDouble( fs, ptr, &fval, endptr );

    return fval;
}


/****************************************************************************************\
*                                       YAML Parser                                      *
\****************************************************************************************/

static char*
icvYMLSkipSpaces( CvFileStorage* fs, char* ptr, int min_indent, int max_comment_indent )
{
    for(;;)
    {
        while( *ptr == ' ' )
            ptr++;
        if( *ptr == '#' )
        {
            if( ptr - fs->buffer_start > max_comment_indent )
                return ptr;
            *ptr = '\0';
        }
        else if( cv_isprint(*ptr) )
        {
            if( ptr - fs->buffer_start < min_indent )
                CV_PARSE_ERROR( "Incorrect indentation" );
            break;
        }
        else if( *ptr == '\0' || *ptr == '\n' || *ptr == '\r' )
        {
            int max_size = (int)(fs->buffer_end - fs->buffer_start);
            ptr = icvGets( fs, fs->buffer_start, max_size );
            if( !ptr )
            {
                // emulate end of stream
                ptr = fs->buffer_start;
                ptr[0] = ptr[1] = ptr[2] = '.';
                ptr[3] = '\0';
                fs->dummy_eof = 1;
                break;
            }
            else
            {
                int l = (int)strlen(ptr);
                if( ptr[l-1] != '\n' && ptr[l-1] != '\r' && !icvEof(fs) )
                    CV_PARSE_ERROR( "Too long string or a last string w/o newline" );
            }

            fs->lineno++;
        }
        else
            CV_PARSE_ERROR( *ptr == '\t' ? "Tabs are prohibited in YAML!" : "Invalid character" );
    }

    return ptr;
}


static char*
icvYMLParseKey( CvFileStorage* fs, char* ptr,
                CvFileNode* map_node, CvFileNode** value_placeholder )
{
    char c;
    char *endptr = ptr - 1, *saveptr;
    CvStringHashNode* str_hash_node;

    if( *ptr == '-' )
        CV_PARSE_ERROR( "Key may not start with \'-\'" );

    do c = *++endptr;
    while( cv_isprint(c) && c != ':' );

    if( c != ':' )
        CV_PARSE_ERROR( "Missing \':\'" );

    saveptr = endptr + 1;
    do c = *--endptr;
    while( c == ' ' );

    ++endptr;
    if( endptr == ptr )
        CV_PARSE_ERROR( "An empty key" );

    str_hash_node = cvGetHashedKey( fs, ptr, (int)(endptr - ptr), 1 );
    *value_placeholder = cvGetFileNode( fs, map_node, str_hash_node, 1 );
    ptr = saveptr;

    return ptr;
}


static char*
icvYMLParseValue( CvFileStorage* fs, char* ptr, CvFileNode* node,
                  int parent_flags, int min_indent )
{
    char buf[CV_FS_MAX_LEN + 1024];
    char* endptr = 0;
    char c = ptr[0], d = ptr[1];
    int is_parent_flow = CV_NODE_IS_FLOW(parent_flags);
    int value_type = CV_NODE_NONE;
    int len;

    memset( node, 0, sizeof(*node) );

    if( c == '!' ) // handle explicit type specification
    {
        if( d == '!' || d == '^' )
        {
            ptr++;
            value_type |= CV_NODE_USER;
        }

        endptr = ptr++;
        do d = *++endptr;
        while( cv_isprint(d) && d != ' ' );
        len = (int)(endptr - ptr);
        if( len == 0 )
            CV_PARSE_ERROR( "Empty type name" );
        d = *endptr;
        *endptr = '\0';

        if( len == 3 && !CV_NODE_IS_USER(value_type) )
        {
            if( memcmp( ptr, "str", 3 ) == 0 )
                value_type = CV_NODE_STRING;
            else if( memcmp( ptr, "int", 3 ) == 0 )
                value_type = CV_NODE_INT;
            else if( memcmp( ptr, "seq", 3 ) == 0 )
                value_type = CV_NODE_SEQ;
            else if( memcmp( ptr, "map", 3 ) == 0 )
                value_type = CV_NODE_MAP;
        }
        else if( len == 5 && !CV_NODE_IS_USER(value_type) )
        {
            if( memcmp( ptr, "float", 5 ) == 0 )
                value_type = CV_NODE_REAL;
        }
        else if( CV_NODE_IS_USER(value_type) )
        {
            node->info = cvFindType( ptr );
            if( !node->info )
                node->tag &= ~CV_NODE_USER;
        }

        *endptr = d;
        ptr = icvYMLSkipSpaces( fs, endptr, min_indent, INT_MAX );

        c = *ptr;

        if( !CV_NODE_IS_USER(value_type) )
        {
            if( value_type == CV_NODE_STRING && c != '\'' && c != '\"' )
                goto force_string;
            if( value_type == CV_NODE_INT )
                goto force_int;
            if( value_type == CV_NODE_REAL )
                goto force_real;
        }
    }

    if( isdigit(c) ||
        ((c == '-' || c == '+') && (isdigit(d) || d == '.')) ||
        (c == '.' && isalnum(d))) // a number
    {
        double fval;
        int ival;
        endptr = ptr + (c == '-' || c == '+');
        while( isdigit(*endptr) )
            endptr++;
        if( *endptr == '.' || *endptr == 'e' )
        {
force_real:
            fval = icv_strtod( fs, ptr, &endptr );
            /*if( endptr == ptr || isalpha(*endptr) )
                icvProcessSpecialDouble( fs, endptr, &fval, &endptr ));*/

            node->tag = CV_NODE_REAL;
            node->data.f = fval;
        }
        else
        {
force_int:
            ival = (int)strtol( ptr, &endptr, 0 );
            node->tag = CV_NODE_INT;
            node->data.i = ival;
        }

        if( !endptr || endptr == ptr )
            CV_PARSE_ERROR( "Invalid numeric value (inconsistent explicit type specification?)" );

        ptr = endptr;
    }
    else if( c == '\'' || c == '\"' ) // an explicit string
    {
        node->tag = CV_NODE_STRING;
        if( c == '\'' )
            for( len = 0; len < CV_FS_MAX_LEN; )
            {
                c = *++ptr;
                if( isalnum(c) || (c != '\'' && cv_isprint(c)))
                    buf[len++] = c;
                else if( c == '\'' )
                {
                    c = *++ptr;
                    if( c != '\'' )
                        break;
                    buf[len++] = c;
                }
                else
                    CV_PARSE_ERROR( "Invalid character" );
            }
        else
            for( len = 0; len < CV_FS_MAX_LEN; )
            {
                c = *++ptr;
                if( isalnum(c) || (c != '\\' && c != '\"' && cv_isprint(c)))
                    buf[len++] = c;
                else if( c == '\"' )
                {
                    ++ptr;
                    break;
                }
                else if( c == '\\' )
                {
                    d = *++ptr;
                    if( d == '\'' )
                        buf[len++] = d;
                    else if( d == '\"' || d == '\\' || d == '\'' )
                        buf[len++] = d;
                    else if( d == 'n' )
                        buf[len++] = '\n';
                    else if( d == 'r' )
                        buf[len++] = '\r';
                    else if( d == 't' )
                        buf[len++] = '\t';
                    else if( d == 'x' || (isdigit(d) && d < '8') )
                    {
                        int val, is_hex = d == 'x';
                        c = ptr[3];
                        ptr[3] = '\0';
                        val = strtol( ptr + is_hex, &endptr, is_hex ? 8 : 16 );
                        ptr[3] = c;
                        if( endptr == ptr + is_hex )
                            buf[len++] = 'x';
                        else
                        {
                            buf[len++] = (char)val;
                            ptr = endptr;
                        }
                    }
                }
                else
                    CV_PARSE_ERROR( "Invalid character" );
            }

        if( len >= CV_FS_MAX_LEN )
            CV_PARSE_ERROR( "Too long string literal" );

        node->data.str = cvMemStorageAllocString( fs->memstorage, buf, len );
    }
    else if( c == '[' || c == '{' ) // collection as a flow
    {
        int new_min_indent = min_indent + !is_parent_flow;
        int struct_flags = CV_NODE_FLOW + (c == '{' ? CV_NODE_MAP : CV_NODE_SEQ);
        int is_simple = 1;

        icvFSCreateCollection( fs, CV_NODE_TYPE(struct_flags) +
                                        (node->info ? CV_NODE_USER : 0), node );

        d = c == '[' ? ']' : '}';

        for( ++ptr ;;)
        {
            CvFileNode* elem = 0;

            ptr = icvYMLSkipSpaces( fs, ptr, new_min_indent, INT_MAX );
            if( *ptr == '}' || *ptr == ']' )
            {
                if( *ptr != d )
                    CV_PARSE_ERROR( "The wrong closing bracket" );
                ptr++;
                break;
            }

            if( node->data.seq->total != 0 )
            {
                if( *ptr != ',' )
                    CV_PARSE_ERROR( "Missing , between the elements" );
                ptr = icvYMLSkipSpaces( fs, ptr + 1, new_min_indent, INT_MAX );
            }

            if( CV_NODE_IS_MAP(struct_flags) )
            {
                ptr = icvYMLParseKey( fs, ptr, node, &elem );
                ptr = icvYMLSkipSpaces( fs, ptr, new_min_indent, INT_MAX );
            }
            else
            {
                if( *ptr == ']' )
                    break;
                elem = (CvFileNode*)cvSeqPush( node->data.seq, 0 );
            }
            ptr = icvYMLParseValue( fs, ptr, elem, struct_flags, new_min_indent );
            if( CV_NODE_IS_MAP(struct_flags) )
                elem->tag |= CV_NODE_NAMED;
            is_simple &= !CV_NODE_IS_COLLECTION(elem->tag);
        }
        node->data.seq->flags |= is_simple ? CV_NODE_SEQ_SIMPLE : 0;
    }
    else
    {
        int indent, struct_flags, is_simple;

        if( is_parent_flow || c != '-' )
        {
            // implicit (one-line) string or nested block-style collection
            if( !is_parent_flow )
            {
                if( c == '?' )
                    CV_PARSE_ERROR( "Complex keys are not supported" );
                if( c == '|' || c == '>' )
                    CV_PARSE_ERROR( "Multi-line text literals are not supported" );
            }

force_string:
            endptr = ptr - 1;

            do c = *++endptr;
            while( cv_isprint(c) &&
                   (!is_parent_flow || (c != ',' && c != '}' && c != ']')) &&
                   (is_parent_flow || c != ':' || value_type == CV_NODE_STRING));

            if( endptr == ptr )
                CV_PARSE_ERROR( "Invalid character" );

            if( is_parent_flow || c != ':' )
            {
                char* str_end = endptr;
                node->tag = CV_NODE_STRING;
                // strip spaces in the end of string
                do c = *--str_end;
                while( str_end > ptr && c == ' ' );
                str_end++;
                node->data.str = cvMemStorageAllocString( fs->memstorage, ptr, (int)(str_end - ptr) );
                ptr = endptr;
                return ptr;
            }
            struct_flags = CV_NODE_MAP;
        }
        else
            struct_flags = CV_NODE_SEQ;

        icvFSCreateCollection( fs, struct_flags +
                    (node->info ? CV_NODE_USER : 0), node );

        indent = (int)(ptr - fs->buffer_start);
        is_simple = 1;

        for(;;)
        {
            CvFileNode* elem = 0;

            if( CV_NODE_IS_MAP(struct_flags) )
            {
                ptr = icvYMLParseKey( fs, ptr, node, &elem );
            }
            else
            {
                c = *ptr++;
                if( c != '-' )
                    CV_PARSE_ERROR( "Block sequence elements must be preceded with \'-\'" );

                elem = (CvFileNode*)cvSeqPush( node->data.seq, 0 );
            }

            ptr = icvYMLSkipSpaces( fs, ptr, indent + 1, INT_MAX );
            ptr = icvYMLParseValue( fs, ptr, elem, struct_flags, indent + 1 );
            if( CV_NODE_IS_MAP(struct_flags) )
                elem->tag |= CV_NODE_NAMED;
            is_simple &= !CV_NODE_IS_COLLECTION(elem->tag);

            ptr = icvYMLSkipSpaces( fs, ptr, 0, INT_MAX );
            if( ptr - fs->buffer_start != indent )
            {
                if( ptr - fs->buffer_start < indent )
                    break;
                else
                    CV_PARSE_ERROR( "Incorrect indentation" );
            }
            if( memcmp( ptr, "...", 3 ) == 0 )
                break;
        }

        node->data.seq->flags |= is_simple ? CV_NODE_SEQ_SIMPLE : 0;
    }

    return ptr;
}


static void
icvYMLParse( CvFileStorage* fs )
{
    char* ptr = fs->buffer_start;
    int is_first = 1;

    for(;;)
    {
        // 0. skip leading comments and directives  and ...
        // 1. reach the first item
        for(;;)
        {
            ptr = icvYMLSkipSpaces( fs, ptr, 0, INT_MAX );
            if( !ptr )
                return;

            if( *ptr == '%' )
            {
                if( memcmp( ptr, "%YAML:", 6 ) == 0 &&
                    memcmp( ptr, "%YAML:1.", 8 ) != 0 )
                    CV_PARSE_ERROR( "Unsupported YAML version (it must be 1.x)" );
                *ptr = '\0';
            }
            else if( *ptr == '-' )
            {
                if( memcmp(ptr, "---", 3) == 0 )
                {
                    ptr += 3;
                    break;
                }
                else if( is_first )
                    break;
            }
            else if( isalnum(*ptr) || *ptr=='_')
            {
                if( !is_first )
                    CV_PARSE_ERROR( "The YAML streams must start with '---', except the first one" );
                break;
            }
            else
                CV_PARSE_ERROR( "Invalid or unsupported syntax" );
        }

        ptr = icvYMLSkipSpaces( fs, ptr, 0, INT_MAX );
        if( memcmp( ptr, "...", 3 ) != 0 )
        {
            // 2. parse the collection
            CvFileNode* root_node = (CvFileNode*)cvSeqPush( fs->roots, 0 );

            ptr = icvYMLParseValue( fs, ptr, root_node, CV_NODE_NONE, 0 );
            if( !CV_NODE_IS_COLLECTION(root_node->tag) )
                CV_PARSE_ERROR( "Only collections as YAML streams are supported by this parser" );

            // 3. parse until the end of file or next collection
            ptr = icvYMLSkipSpaces( fs, ptr, 0, INT_MAX );
            if( !ptr )
                return;
        }

        if( fs->dummy_eof )
            break;
        ptr += 3;
        is_first = 0;
    }
}


/****************************************************************************************\
*                                       YAML Emitter                                     *
\****************************************************************************************/

static void
icvYMLWrite( CvFileStorage* fs, const char* key, const char* data )
{
    int i, keylen = 0;
    int datalen = 0;
    int struct_flags;
    char* ptr;

    struct_flags = fs->struct_flags;

    if( key && key[0] == '\0' )
        key = 0;

    if( CV_NODE_IS_COLLECTION(struct_flags) )
    {
        if( (CV_NODE_IS_MAP(struct_flags) ^ (key != 0)) )
            CV_Error( CV_StsBadArg, "An attempt to add element without a key to a map, "
                                    "or add element with key to sequence" );
    }
    else
    {
        fs->is_first = 0;
        struct_flags = CV_NODE_EMPTY | (key ? CV_NODE_MAP : CV_NODE_SEQ);
    }

    if( key )
    {
        keylen = (int)strlen(key);
        if( keylen == 0 )
            CV_Error( CV_StsBadArg, "The key is an empty" );

        if( keylen > CV_FS_MAX_LEN )
            CV_Error( CV_StsBadArg, "The key is too long" );
    }

    if( data )
        datalen = (int)strlen(data);

    if( CV_NODE_IS_FLOW(struct_flags) )
    {
        int new_offset;
        ptr = fs->buffer;
        if( !CV_NODE_IS_EMPTY(struct_flags) )
            *ptr++ = ',';
        new_offset = (int)(ptr - fs->buffer_start) + keylen + datalen;
        if( new_offset > fs->wrap_margin && new_offset - fs->struct_indent > 10 )
        {
            fs->buffer = ptr;
            ptr = icvFSFlush(fs);
        }
        else
            *ptr++ = ' ';
    }
    else
    {
        ptr = icvFSFlush(fs);
        if( !CV_NODE_IS_MAP(struct_flags) )
        {
            *ptr++ = '-';
            if( data )
                *ptr++ = ' ';
        }
    }

    if( key )
    {
        if( !isalpha(key[0]) && key[0] != '_' )
            CV_Error( CV_StsBadArg, "Key must start with a letter or _" );

        ptr = icvFSResizeWriteBuffer( fs, ptr, keylen );

        for( i = 0; i < keylen; i++ )
        {
            int c = key[i];

            ptr[i] = (char)c;
            if( !isalnum(c) && c != '-' && c != '_' && c != ' ' )
                CV_Error( CV_StsBadArg, "Key names may only contain alphanumeric characters [a-zA-Z0-9], '-', '_' and ' '" );
        }

        ptr += keylen;
        *ptr++ = ':';
        if( !CV_NODE_IS_FLOW(struct_flags) && data )
            *ptr++ = ' ';
    }

    if( data )
    {
        ptr = icvFSResizeWriteBuffer( fs, ptr, datalen );
        memcpy( ptr, data, datalen );
        ptr += datalen;
    }

    fs->buffer = ptr;
    fs->struct_flags = struct_flags & ~CV_NODE_EMPTY;
}


static void
icvYMLStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags,
                        const char* type_name CV_DEFAULT(0))
{
    int parent_flags;
    char buf[CV_FS_MAX_LEN + 1024];
    const char* data = 0;

    struct_flags = (struct_flags & (CV_NODE_TYPE_MASK|CV_NODE_FLOW)) | CV_NODE_EMPTY;
    if( !CV_NODE_IS_COLLECTION(struct_flags))
        CV_Error( CV_StsBadArg,
        "Some collection type - CV_NODE_SEQ or CV_NODE_MAP, must be specified" );

    if( CV_NODE_IS_FLOW(struct_flags) )
    {
        char c = CV_NODE_IS_MAP(struct_flags) ? '{' : '[';
        struct_flags |= CV_NODE_FLOW;

        if( type_name )
            sprintf( buf, "!!%s %c", type_name, c );
        else
        {
            buf[0] = c;
            buf[1] = '\0';
        }
        data = buf;
    }
    else if( type_name )
    {
        sprintf( buf, "!!%s", type_name );
        data = buf;
    }

    icvYMLWrite( fs, key, data );

    parent_flags = fs->struct_flags;
    cvSeqPush( fs->write_stack, &parent_flags );
    fs->struct_flags = struct_flags;

    if( !CV_NODE_IS_FLOW(parent_flags) )
        fs->struct_indent += CV_YML_INDENT + CV_NODE_IS_FLOW(struct_flags);
}


static void
icvYMLEndWriteStruct( CvFileStorage* fs )
{
    int parent_flags = 0, struct_flags;
    char* ptr;

    struct_flags = fs->struct_flags;
    if( fs->write_stack->total == 0 )
        CV_Error( CV_StsError, "EndWriteStruct w/o matching StartWriteStruct" );

    cvSeqPop( fs->write_stack, &parent_flags );

    if( CV_NODE_IS_FLOW(struct_flags) )
    {
        ptr = fs->buffer;
        if( ptr > fs->buffer_start + fs->struct_indent && !CV_NODE_IS_EMPTY(struct_flags) )
            *ptr++ = ' ';
        *ptr++ = CV_NODE_IS_MAP(struct_flags) ? '}' : ']';
        fs->buffer = ptr;
    }
    else if( CV_NODE_IS_EMPTY(struct_flags) )
    {
        ptr = icvFSFlush(fs);
        memcpy( ptr, CV_NODE_IS_MAP(struct_flags) ? "{}" : "[]", 2 );
        fs->buffer = ptr + 2;
    }

    if( !CV_NODE_IS_FLOW(parent_flags) )
        fs->struct_indent -= CV_YML_INDENT + CV_NODE_IS_FLOW(struct_flags);
    assert( fs->struct_indent >= 0 );

    fs->struct_flags = parent_flags;
}


static void
icvYMLStartNextStream( CvFileStorage* fs )
{
    if( !fs->is_first )
    {
        while( fs->write_stack->total > 0 )
            icvYMLEndWriteStruct(fs);

        fs->struct_indent = 0;
        icvFSFlush(fs);
        icvPuts( fs, "...\n" );
        icvPuts( fs, "---\n" );
        fs->buffer = fs->buffer_start;
    }
}


static void
icvYMLWriteInt( CvFileStorage* fs, const char* key, int value )
{
    char buf[128];
    icvYMLWrite( fs, key, icv_itoa( value, buf, 10 ));
}


static void
icvYMLWriteReal( CvFileStorage* fs, const char* key, double value )
{
    char buf[128];
    icvYMLWrite( fs, key, icvDoubleToString( buf, value ));
}


static void
icvYMLWriteString( CvFileStorage* fs, const char* key,
                   const char* str, int quote CV_DEFAULT(0))
{
    char buf[CV_FS_MAX_LEN*4+16];
    char* data = (char*)str;
    int i, len;

    if( !str )
        CV_Error( CV_StsNullPtr, "Null string pointer" );

    len = (int)strlen(str);
    if( len > CV_FS_MAX_LEN )
        CV_Error( CV_StsBadArg, "The written string is too long" );

    if( quote || len == 0 || str[0] != str[len-1] || (str[0] != '\"' && str[0] != '\'') )
    {
        int need_quote = quote || len == 0;
        data = buf;
        *data++ = '\"';
        for( i = 0; i < len; i++ )
        {
            char c = str[i];

            if( !need_quote && !isalnum(c) && c != '_' && c != ' ' && c != '-' &&
                c != '(' && c != ')' && c != '/' && c != '+' && c != ';' )
                need_quote = 1;

            if( !isalnum(c) && (!cv_isprint(c) || c == '\\' || c == '\'' || c == '\"') )
            {
                *data++ = '\\';
                if( cv_isprint(c) )
                    *data++ = c;
                else if( c == '\n' )
                    *data++ = 'n';
                else if( c == '\r' )
                    *data++ = 'r';
                else if( c == '\t' )
                    *data++ = 't';
                else
                {
                    sprintf( data, "x%02x", c );
                    data += 3;
                }
            }
            else
                *data++ = c;
        }
        if( !need_quote && (isdigit(str[0]) ||
            str[0] == '+' || str[0] == '-' || str[0] == '.' ))
            need_quote = 1;

        if( need_quote )
            *data++ = '\"';
        *data++ = '\0';
        data = buf + !need_quote;
    }

    icvYMLWrite( fs, key, data );
}


static void
icvYMLWriteComment( CvFileStorage* fs, const char* comment, int eol_comment )
{
    int len; //, indent;
    int multiline;
    const char* eol;
    char* ptr;

    if( !comment )
        CV_Error( CV_StsNullPtr, "Null comment" );

    len = (int)strlen(comment);
    eol = strchr(comment, '\n');
    multiline = eol != 0;
    ptr = fs->buffer;

    if( !eol_comment || multiline ||
        fs->buffer_end - ptr < len || ptr == fs->buffer_start )
        ptr = icvFSFlush( fs );
    else
        *ptr++ = ' ';

    while( comment )
    {
        *ptr++ = '#';
        *ptr++ = ' ';
        if( eol )
        {
            ptr = icvFSResizeWriteBuffer( fs, ptr, (int)(eol - comment) + 1 );
            memcpy( ptr, comment, eol - comment + 1 );
            fs->buffer = ptr + (eol - comment);
            comment = eol + 1;
            eol = strchr( comment, '\n' );
        }
        else
        {
            len = (int)strlen(comment);
            ptr = icvFSResizeWriteBuffer( fs, ptr, len );
            memcpy( ptr, comment, len );
            fs->buffer = ptr + len;
            comment = 0;
        }
        ptr = icvFSFlush( fs );
    }
}


/****************************************************************************************\
*                                       XML Parser                                       *
\****************************************************************************************/

#define CV_XML_INSIDE_COMMENT 1
#define CV_XML_INSIDE_TAG 2
#define CV_XML_INSIDE_DIRECTIVE 3

static char*
icvXMLSkipSpaces( CvFileStorage* fs, char* ptr, int mode )
{
    int level = 0;

    for(;;)
    {
        char c;
        ptr--;

        if( mode == CV_XML_INSIDE_COMMENT )
        {
            do c = *++ptr;
            while( cv_isprint_or_tab(c) && (c != '-' || ptr[1] != '-' || ptr[2] != '>') );

            if( c == '-' )
            {
                assert( ptr[1] == '-' && ptr[2] == '>' );
                mode = 0;
                ptr += 3;
            }
        }
        else if( mode == CV_XML_INSIDE_DIRECTIVE )
        {
            // !!!NOTE!!! This is not quite correct, but should work in most cases
            do
            {
                c = *++ptr;
                level += c == '<';
                level -= c == '>';
                if( level < 0 )
                    return ptr;
            } while( cv_isprint_or_tab(c) );
        }
        else
        {
            do c = *++ptr;
            while( c == ' ' || c == '\t' );

            if( c == '<' && ptr[1] == '!' && ptr[2] == '-' && ptr[3] == '-' )
            {
                if( mode != 0 )
                    CV_PARSE_ERROR( "Comments are not allowed here" );
                mode = CV_XML_INSIDE_COMMENT;
                ptr += 4;
            }
            else if( cv_isprint(c) )
                break;
        }

        if( !cv_isprint(*ptr) )
        {
            int max_size = (int)(fs->buffer_end - fs->buffer_start);
            if( *ptr != '\0' && *ptr != '\n' && *ptr != '\r' )
                CV_PARSE_ERROR( "Invalid character in the stream" );
            ptr = icvGets( fs, fs->buffer_start, max_size );
            if( !ptr )
            {
                ptr = fs->buffer_start;
                *ptr = '\0';
                fs->dummy_eof = 1;
                break;
            }
            else
            {
                int l = (int)strlen(ptr);
                if( ptr[l-1] != '\n' && ptr[l-1] != '\r' && !icvEof(fs) )
                    CV_PARSE_ERROR( "Too long string or a last string w/o newline" );
            }
            fs->lineno++;
        }
    }
    return ptr;
}


static char*
icvXMLParseTag( CvFileStorage* fs, char* ptr, CvStringHashNode** _tag,
                CvAttrList** _list, int* _tag_type );

static char*
icvXMLParseValue( CvFileStorage* fs, char* ptr, CvFileNode* node,
                  int value_type CV_DEFAULT(CV_NODE_NONE))
{
    CvFileNode *elem = node;
    int have_space = 1, is_simple = 1;
    int is_user_type = CV_NODE_IS_USER(value_type);
    memset( node, 0, sizeof(*node) );

    value_type = CV_NODE_TYPE(value_type);

    for(;;)
    {
        char c = *ptr, d;
        char* endptr;

        if( isspace(c) || c == '\0' || (c == '<' && ptr[1] == '!' && ptr[2] == '-') )
        {
            ptr = icvXMLSkipSpaces( fs, ptr, 0 );
            have_space = 1;
            c = *ptr;
        }

        d = ptr[1];

        if( c =='<' )
        {
            CvStringHashNode *key = 0, *key2 = 0;
            CvAttrList* list = 0;
            CvTypeInfo* info = 0;
            int tag_type = 0;
            int is_noname = 0;
            const char* type_name = 0;
            int elem_type = CV_NODE_NONE;

            if( d == '/' )
                break;

            ptr = icvXMLParseTag( fs, ptr, &key, &list, &tag_type );

            if( tag_type == CV_XML_DIRECTIVE_TAG )
                CV_PARSE_ERROR( "Directive tags are not allowed here" );
            if( tag_type == CV_XML_EMPTY_TAG )
                CV_PARSE_ERROR( "Empty tags are not supported" );

            assert( tag_type == CV_XML_OPENING_TAG );

            type_name = list ? cvAttrValue( list, "type_id" ) : 0;
            if( type_name )
            {
                if( strcmp( type_name, "str" ) == 0 )
                    elem_type = CV_NODE_STRING;
                else if( strcmp( type_name, "map" ) == 0 )
                    elem_type = CV_NODE_MAP;
                else if( strcmp( type_name, "seq" ) == 0 )
                    elem_type = CV_NODE_SEQ;
                else
                {
                    info = cvFindType( type_name );
                    if( info )
                        elem_type = CV_NODE_USER;
                }
            }

            is_noname = key->str.len == 1 && key->str.ptr[0] == '_';
            if( !CV_NODE_IS_COLLECTION(node->tag) )
            {
                icvFSCreateCollection( fs, is_noname ? CV_NODE_SEQ : CV_NODE_MAP, node );
            }
            else if( is_noname ^ CV_NODE_IS_SEQ(node->tag) )
                CV_PARSE_ERROR( is_noname ? "Map element should have a name" :
                              "Sequence element should not have name (use <_></_>)" );

            if( is_noname )
                elem = (CvFileNode*)cvSeqPush( node->data.seq, 0 );
            else
                elem = cvGetFileNode( fs, node, key, 1 );

            ptr = icvXMLParseValue( fs, ptr, elem, elem_type);
            if( !is_noname )
                elem->tag |= CV_NODE_NAMED;
            is_simple &= !CV_NODE_IS_COLLECTION(elem->tag);
            elem->info = info;
            ptr = icvXMLParseTag( fs, ptr, &key2, &list, &tag_type );
            if( tag_type != CV_XML_CLOSING_TAG || key2 != key )
                CV_PARSE_ERROR( "Mismatched closing tag" );
            have_space = 1;
        }
        else
        {
            if( !have_space )
                CV_PARSE_ERROR( "There should be space between literals" );

            elem = node;
            if( node->tag != CV_NODE_NONE )
            {
                if( !CV_NODE_IS_COLLECTION(node->tag) )
                    icvFSCreateCollection( fs, CV_NODE_SEQ, node );

                elem = (CvFileNode*)cvSeqPush( node->data.seq, 0 );
                elem->info = 0;
            }

            if( value_type != CV_NODE_STRING &&
                (isdigit(c) || ((c == '-' || c == '+') &&
                (isdigit(d) || d == '.')) || (c == '.' && isalnum(d))) ) // a number
            {
                double fval;
                int ival;
                endptr = ptr + (c == '-' || c == '+');
                while( isdigit(*endptr) )
                    endptr++;
                if( *endptr == '.' || *endptr == 'e' )
                {
                    fval = icv_strtod( fs, ptr, &endptr );
                    /*if( endptr == ptr || isalpha(*endptr) )
                        icvProcessSpecialDouble( fs, ptr, &fval, &endptr ));*/
                    elem->tag = CV_NODE_REAL;
                    elem->data.f = fval;
                }
                else
                {
                    ival = (int)strtol( ptr, &endptr, 0 );
                    elem->tag = CV_NODE_INT;
                    elem->data.i = ival;
                }

                if( endptr == ptr )
                    CV_PARSE_ERROR( "Invalid numeric value (inconsistent explicit type specification?)" );

                ptr = endptr;
            }
            else
            {
                // string
                char buf[CV_FS_MAX_LEN+16];
                int i = 0, len, is_quoted = 0;
                elem->tag = CV_NODE_STRING;
                if( c == '\"' )
                    is_quoted = 1;
                else
                    --ptr;

                for( ;; )
                {
                    c = *++ptr;
                    if( !isalnum(c) )
                    {
                        if( c == '\"' )
                        {
                            if( !is_quoted )
                                CV_PARSE_ERROR( "Literal \" is not allowed within a string. Use &quot;" );
                            ++ptr;
                            break;
                        }
                        else if( !cv_isprint(c) || c == '<' || (!is_quoted && isspace(c)))
                        {
                            if( is_quoted )
                                CV_PARSE_ERROR( "Closing \" is expected" );
                            break;
                        }
                        else if( c == '\'' || c == '>' )
                        {
                            CV_PARSE_ERROR( "Literal \' or > are not allowed. Use &apos; or &gt;" );
                        }
                        else if( c == '&' )
                        {
                            if( *ptr == '#' )
                            {
                                int val;
                                ptr++;
                                val = (int)strtol( ptr, &endptr, 0 );
                                if( (unsigned)val > (unsigned)255 ||
                                    !endptr || *endptr != ';' )
                                    CV_PARSE_ERROR( "Invalid numeric value in the string" );
                                c = (char)val;
                            }
                            else
                            {
                                endptr = ptr++;
                                do c = *++endptr;
                                while( isalnum(c) );
                                if( c != ';' )
                                    CV_PARSE_ERROR( "Invalid character in the symbol entity name" );
                                len = (int)(endptr - ptr);
                                if( len == 2 && memcmp( ptr, "lt", len ) == 0 )
                                    c = '<';
                                else if( len == 2 && memcmp( ptr, "gt", len ) == 0 )
                                    c = '>';
                                else if( len == 3 && memcmp( ptr, "amp", len ) == 0 )
                                    c = '&';
                                else if( len == 4 && memcmp( ptr, "apos", len ) == 0 )
                                    c = '\'';
                                else if( len == 4 && memcmp( ptr, "quot", len ) == 0 )
                                    c = '\"';
                                else
                                {
                                    memcpy( buf + i, ptr-1, len + 2 );
                                    i += len + 2;
                                }
                            }
                            ptr = endptr;
                        }
                    }
                    buf[i++] = c;
                    if( i >= CV_FS_MAX_LEN )
                        CV_PARSE_ERROR( "Too long string literal" );
                }
                elem->data.str = cvMemStorageAllocString( fs->memstorage, buf, i );
            }

            if( !CV_NODE_IS_COLLECTION(value_type) && value_type != CV_NODE_NONE )
                break;
            have_space = 0;
        }
    }

    if( (CV_NODE_TYPE(node->tag) == CV_NODE_NONE ||
        (CV_NODE_TYPE(node->tag) != value_type &&
        !CV_NODE_IS_COLLECTION(node->tag))) &&
        CV_NODE_IS_COLLECTION(value_type) )
    {
        icvFSCreateCollection( fs, CV_NODE_IS_MAP(value_type) ?
                                        CV_NODE_MAP : CV_NODE_SEQ, node );
    }

    if( value_type != CV_NODE_NONE &&
        value_type != CV_NODE_TYPE(node->tag) )
        CV_PARSE_ERROR( "The actual type is different from the specified type" );

    if( CV_NODE_IS_COLLECTION(node->tag) && is_simple )
            node->data.seq->flags |= CV_NODE_SEQ_SIMPLE;

    node->tag |= is_user_type ? CV_NODE_USER : 0;
    return ptr;
}


static char*
icvXMLParseTag( CvFileStorage* fs, char* ptr, CvStringHashNode** _tag,
                CvAttrList** _list, int* _tag_type )
{
    int tag_type = 0;
    CvStringHashNode* tagname = 0;
    CvAttrList *first = 0, *last = 0;
    int count = 0, max_count = 4;
    int attr_buf_size = (max_count*2 + 1)*sizeof(char*) + sizeof(CvAttrList);
    char* endptr;
    char c;
    int have_space;

    if( *ptr != '<' )
        CV_PARSE_ERROR( "Tag should start with \'<\'" );

    ptr++;
    if( isalnum(*ptr) || *ptr == '_' )
        tag_type = CV_XML_OPENING_TAG;
    else if( *ptr == '/' )
    {
        tag_type = CV_XML_CLOSING_TAG;
        ptr++;
    }
    else if( *ptr == '?' )
    {
        tag_type = CV_XML_HEADER_TAG;
        ptr++;
    }
    else if( *ptr == '!' )
    {
        tag_type = CV_XML_DIRECTIVE_TAG;
        assert( ptr[1] != '-' || ptr[2] != '-' );
        ptr++;
    }
    else
        CV_PARSE_ERROR( "Unknown tag type" );

    for(;;)
    {
        CvStringHashNode* attrname;

        if( !isalpha(*ptr) && *ptr != '_' )
            CV_PARSE_ERROR( "Name should start with a letter or underscore" );

        endptr = ptr - 1;
        do c = *++endptr;
        while( isalnum(c) || c == '_' || c == '-' );

        attrname = cvGetHashedKey( fs, ptr, (int)(endptr - ptr), 1 );
        ptr = endptr;

        if( !tagname )
            tagname = attrname;
        else
        {
            if( tag_type == CV_XML_CLOSING_TAG )
                CV_PARSE_ERROR( "Closing tag should not contain any attributes" );

            if( !last || count >= max_count )
            {
                CvAttrList* chunk;

                chunk = (CvAttrList*)cvMemStorageAlloc( fs->memstorage, attr_buf_size );
                memset( chunk, 0, attr_buf_size );
                chunk->attr = (const char**)(chunk + 1);
                count = 0;
                if( !last )
                    first = last = chunk;
                else
                    last = last->next = chunk;
            }
            last->attr[count*2] = attrname->str.ptr;
        }

        if( last )
        {
            CvFileNode stub;

            if( *ptr != '=' )
            {
                ptr = icvXMLSkipSpaces( fs, ptr, CV_XML_INSIDE_TAG );
                if( *ptr != '=' )
                    CV_PARSE_ERROR( "Attribute name should be followed by \'=\'" );
            }

            c = *++ptr;
            if( c != '\"' && c != '\'' )
            {
                ptr = icvXMLSkipSpaces( fs, ptr, CV_XML_INSIDE_TAG );
                if( *ptr != '\"' && *ptr != '\'' )
                    CV_PARSE_ERROR( "Attribute value should be put into single or double quotes" );
            }

            ptr = icvXMLParseValue( fs, ptr, &stub, CV_NODE_STRING );
            assert( stub.tag == CV_NODE_STRING );
            last->attr[count*2+1] = stub.data.str.ptr;
            count++;
        }

        c = *ptr;
        have_space = isspace(c) || c == '\0';

        if( c != '>' )
        {
            ptr = icvXMLSkipSpaces( fs, ptr, CV_XML_INSIDE_TAG );
            c = *ptr;
        }

        if( c == '>' )
        {
            if( tag_type == CV_XML_HEADER_TAG )
                CV_PARSE_ERROR( "Invalid closing tag for <?xml ..." );
            ptr++;
            break;
        }
        else if( c == '?' && tag_type == CV_XML_HEADER_TAG )
        {
            if( ptr[1] != '>'  )
                CV_PARSE_ERROR( "Invalid closing tag for <?xml ..." );
            ptr += 2;
            break;
        }
        else if( c == '/' && ptr[1] == '>' && tag_type == CV_XML_OPENING_TAG )
        {
            tag_type = CV_XML_EMPTY_TAG;
            ptr += 2;
            break;
        }

        if( !have_space )
            CV_PARSE_ERROR( "There should be space between attributes" );
    }

    *_tag = tagname;
    *_tag_type = tag_type;
    *_list = first;

    return ptr;
}


static void
icvXMLParse( CvFileStorage* fs )
{
    char* ptr = fs->buffer_start;
    CvStringHashNode *key = 0, *key2 = 0;
    CvAttrList* list = 0;
    int tag_type = 0;

    // CV_XML_INSIDE_TAG is used to prohibit leading comments
    ptr = icvXMLSkipSpaces( fs, ptr, CV_XML_INSIDE_TAG );

    if( memcmp( ptr, "<?xml", 5 ) != 0 )
        CV_PARSE_ERROR( "Valid XML should start with \'<?xml ...?>\'" );

    ptr = icvXMLParseTag( fs, ptr, &key, &list, &tag_type );

    /*{
        const char* version = cvAttrValue( list, "version" );
        if( version && strncmp( version, "1.", 2 ) != 0 )
            CV_Error( CV_StsParseError, "Unsupported version of XML" );
    }*/
    {
        const char* encoding = cvAttrValue( list, "encoding" );
        if( encoding && strcmp( encoding, "ASCII" ) != 0 )
            CV_PARSE_ERROR( "Unsupported encoding" );
    }

    while( *ptr != '\0' )
    {
        ptr = icvXMLSkipSpaces( fs, ptr, 0 );

        if( *ptr != '\0' )
        {
            CvFileNode* root_node;
            ptr = icvXMLParseTag( fs, ptr, &key, &list, &tag_type );
            if( tag_type != CV_XML_OPENING_TAG ||
                strcmp(key->str.ptr,"opencv_storage") != 0 )
                CV_PARSE_ERROR( "<opencv_storage> tag is missing" );

            root_node = (CvFileNode*)cvSeqPush( fs->roots, 0 );
            ptr = icvXMLParseValue( fs, ptr, root_node, CV_NODE_NONE );
            ptr = icvXMLParseTag( fs, ptr, &key2, &list, &tag_type );
            if( tag_type != CV_XML_CLOSING_TAG || key != key2 )
                CV_PARSE_ERROR( "</opencv_storage> tag is missing" );
            ptr = icvXMLSkipSpaces( fs, ptr, 0 );
        }
    }

    assert( fs->dummy_eof != 0 );
}


/****************************************************************************************\
*                                       XML Emitter                                      *
\****************************************************************************************/

#define icvXMLFlush icvFSFlush

static void
icvXMLWriteTag( CvFileStorage* fs, const char* key, int tag_type, CvAttrList list )
{
    char* ptr = fs->buffer;
    int i, len = 0;
    int struct_flags = fs->struct_flags;

    if( key && key[0] == '\0' )
        key = 0;

    if( tag_type == CV_XML_OPENING_TAG || tag_type == CV_XML_EMPTY_TAG )
    {
        if( CV_NODE_IS_COLLECTION(struct_flags) )
        {
            if( CV_NODE_IS_MAP(struct_flags) ^ (key != 0) )
                CV_Error( CV_StsBadArg, "An attempt to add element without a key to a map, "
                                        "or add element with key to sequence" );
        }
        else
        {
            struct_flags = CV_NODE_EMPTY + (key ? CV_NODE_MAP : CV_NODE_SEQ);
            fs->is_first = 0;
        }

        if( !CV_NODE_IS_EMPTY(struct_flags) )
            ptr = icvXMLFlush(fs);
    }

    if( !key )
        key = "_";
    else if( key[0] == '_' && key[1] == '\0' )
        CV_Error( CV_StsBadArg, "A single _ is a reserved tag name" );

    len = (int)strlen( key );
    *ptr++ = '<';
    if( tag_type == CV_XML_CLOSING_TAG )
    {
        if( list.attr )
            CV_Error( CV_StsBadArg, "Closing tag should not include any attributes" );
        *ptr++ = '/';
    }

    if( !isalpha(key[0]) && key[0] != '_' )
        CV_Error( CV_StsBadArg, "Key should start with a letter or _" );

    ptr = icvFSResizeWriteBuffer( fs, ptr, len );
    for( i = 0; i < len; i++ )
    {
        char c = key[i];
        if( !isalnum(c) && c != '_' && c != '-' )
            CV_Error( CV_StsBadArg, "Key name may only contain alphanumeric characters [a-zA-Z0-9], '-' and '_'" );
        ptr[i] = c;
    }
    ptr += len;

    for(;;)
    {
        const char** attr = list.attr;

        for( ; attr && attr[0] != 0; attr += 2 )
        {
            int len0 = (int)strlen(attr[0]);
            int len1 = (int)strlen(attr[1]);

            ptr = icvFSResizeWriteBuffer( fs, ptr, len0 + len1 + 4 );
            *ptr++ = ' ';
            memcpy( ptr, attr[0], len0 );
            ptr += len0;
            *ptr++ = '=';
            *ptr++ = '\"';
            memcpy( ptr, attr[1], len1 );
            ptr += len1;
            *ptr++ = '\"';
        }
        if( !list.next )
            break;
        list = *list.next;
    }

    if( tag_type == CV_XML_EMPTY_TAG )
        *ptr++ = '/';
    *ptr++ = '>';
    fs->buffer = ptr;
    fs->struct_flags = struct_flags & ~CV_NODE_EMPTY;
}


static void
icvXMLStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags,
                        const char* type_name CV_DEFAULT(0))
{
    CvXMLStackRecord parent;
    const char* attr[10];
    int idx = 0;

    struct_flags = (struct_flags & (CV_NODE_TYPE_MASK|CV_NODE_FLOW)) | CV_NODE_EMPTY;
    if( !CV_NODE_IS_COLLECTION(struct_flags))
        CV_Error( CV_StsBadArg,
        "Some collection type: CV_NODE_SEQ or CV_NODE_MAP must be specified" );

    if( type_name )
    {
        attr[idx++] = "type_id";
        attr[idx++] = type_name;
    }
    attr[idx++] = 0;

    icvXMLWriteTag( fs, key, CV_XML_OPENING_TAG, cvAttrList(attr,0) );

    parent.struct_flags = fs->struct_flags & ~CV_NODE_EMPTY;
    parent.struct_indent = fs->struct_indent;
    parent.struct_tag = fs->struct_tag;
    cvSaveMemStoragePos( fs->strstorage, &parent.pos );
    cvSeqPush( fs->write_stack, &parent );

    fs->struct_indent += CV_XML_INDENT;
    if( !CV_NODE_IS_FLOW(struct_flags) )
        icvXMLFlush( fs );

    fs->struct_flags = struct_flags;
    if( key )
    {
        fs->struct_tag = cvMemStorageAllocString( fs->strstorage, (char*)key, -1 );
    }
    else
    {
        fs->struct_tag.ptr = 0;
        fs->struct_tag.len = 0;
    }
}


static void
icvXMLEndWriteStruct( CvFileStorage* fs )
{
    CvXMLStackRecord parent;

    if( fs->write_stack->total == 0 )
        CV_Error( CV_StsError, "An extra closing tag" );

    icvXMLWriteTag( fs, fs->struct_tag.ptr, CV_XML_CLOSING_TAG, cvAttrList(0,0) );
    cvSeqPop( fs->write_stack, &parent );

    fs->struct_indent = parent.struct_indent;
    fs->struct_flags = parent.struct_flags;
    fs->struct_tag = parent.struct_tag;
    cvRestoreMemStoragePos( fs->strstorage, &parent.pos );
}


static void
icvXMLStartNextStream( CvFileStorage* fs )
{
    if( !fs->is_first )
    {
        while( fs->write_stack->total > 0 )
            icvXMLEndWriteStruct(fs);

        fs->struct_indent = 0;
        icvXMLFlush(fs);
        /* XML does not allow multiple top-level elements,
           so we just put a comment and continue
           the current (and the only) "stream" */
        icvPuts( fs, "\n<!-- next stream -->\n" );
        /*fputs( "</opencv_storage>\n", fs->file );
        fputs( "<opencv_storage>\n", fs->file );*/
        fs->buffer = fs->buffer_start;
    }
}


static void
icvXMLWriteScalar( CvFileStorage* fs, const char* key, const char* data, int len )
{
    if( CV_NODE_IS_MAP(fs->struct_flags) ||
        (!CV_NODE_IS_COLLECTION(fs->struct_flags) && key) )
    {
        icvXMLWriteTag( fs, key, CV_XML_OPENING_TAG, cvAttrList(0,0) );
        char* ptr = icvFSResizeWriteBuffer( fs, fs->buffer, len );
        memcpy( ptr, data, len );
        fs->buffer = ptr + len;
        icvXMLWriteTag( fs, key, CV_XML_CLOSING_TAG, cvAttrList(0,0) );
    }
    else
    {
        char* ptr = fs->buffer;
        int new_offset = (int)(ptr - fs->buffer_start) + len;

        if( key )
            CV_Error( CV_StsBadArg, "elements with keys can not be written to sequence" );

        fs->struct_flags = CV_NODE_SEQ;

        if( (new_offset > fs->wrap_margin && new_offset - fs->struct_indent > 10) ||
            (ptr > fs->buffer_start && ptr[-1] == '>' && !CV_NODE_IS_EMPTY(fs->struct_flags)) )
        {
            ptr = icvXMLFlush(fs);
        }
        else if( ptr > fs->buffer_start + fs->struct_indent && ptr[-1] != '>' )
            *ptr++ = ' ';

        memcpy( ptr, data, len );
        fs->buffer = ptr + len;
    }
}


static void
icvXMLWriteInt( CvFileStorage* fs, const char* key, int value )
{
    char buf[128], *ptr = icv_itoa( value, buf, 10 );
    int len = (int)strlen(ptr);
    icvXMLWriteScalar( fs, key, ptr, len );
}


static void
icvXMLWriteReal( CvFileStorage* fs, const char* key, double value )
{
    char buf[128];
    int len = (int)strlen( icvDoubleToString( buf, value ));
    icvXMLWriteScalar( fs, key, buf, len );
}


static void
icvXMLWriteString( CvFileStorage* fs, const char* key, const char* str, int quote )
{
    char buf[CV_FS_MAX_LEN*6+16];
    char* data = (char*)str;
    int i, len;

    if( !str )
        CV_Error( CV_StsNullPtr, "Null string pointer" );

    len = (int)strlen(str);
    if( len > CV_FS_MAX_LEN )
        CV_Error( CV_StsBadArg, "The written string is too long" );

    if( quote || len == 0 || str[0] != '\"' || str[0] != str[len-1] )
    {
        int need_quote = quote || len == 0;
        data = buf;
        *data++ = '\"';
        for( i = 0; i < len; i++ )
        {
            char c = str[i];

            if( !isalnum(c) && (!cv_isprint(c) || c == '<' || c == '>' ||
                c == '&' || c == '\'' || c == '\"') )
            {
                *data++ = '&';
                if( c == '<' )
                {
                    memcpy(data, "lt", 2);
                    data += 2;
                }
                else if( c == '>' )
                {
                    memcpy(data, "gt", 2);
                    data += 2;
                }
                else if( c == '&' )
                {
                    memcpy(data, "amp", 3);
                    data += 3;
                }
                else if( c == '\'' )
                {
                    memcpy(data, "apos", 4);
                    data += 4;
                }
                else if( c == '\"' )
                {
                    memcpy( data, "quot", 4);
                    data += 4;
                }
                else
                {
                    sprintf( data, "#x%02x", c );
                    data += 4;
                }
                *data++ = ';';
            }
            else
            {
                if( c == ' ' )
                    need_quote = 1;
                *data++ = c;
            }
        }
        if( !need_quote && (isdigit(str[0]) ||
            str[0] == '+' || str[0] == '-' || str[0] == '.' ))
            need_quote = 1;

        if( need_quote )
            *data++ = '\"';
        len = (int)(data - buf) - !need_quote;
        *data++ = '\0';
        data = buf + !need_quote;
    }

    icvXMLWriteScalar( fs, key, data, len );
}


static void
icvXMLWriteComment( CvFileStorage* fs, const char* comment, int eol_comment )
{
    int len;
    int multiline;
    const char* eol;
    char* ptr;

    if( !comment )
        CV_Error( CV_StsNullPtr, "Null comment" );

    if( strstr(comment, "--") != 0 )
        CV_Error( CV_StsBadArg, "Double hyphen \'--\' is not allowed in the comments" );

    len = (int)strlen(comment);
    eol = strchr(comment, '\n');
    multiline = eol != 0;
    ptr = fs->buffer;

    if( multiline || !eol_comment || fs->buffer_end - ptr < len + 5 )
        ptr = icvXMLFlush( fs );
    else if( ptr > fs->buffer_start + fs->struct_indent )
        *ptr++ = ' ';

    if( !multiline )
    {
        ptr = icvFSResizeWriteBuffer( fs, ptr, len + 9 );
        sprintf( ptr, "<!-- %s -->", comment );
        len = (int)strlen(ptr);
    }
    else
    {
        strcpy( ptr, "<!--" );
        len = 4;
    }

    fs->buffer = ptr + len;
    ptr = icvXMLFlush(fs);

    if( multiline )
    {
        while( comment )
        {
            if( eol )
            {
                ptr = icvFSResizeWriteBuffer( fs, ptr, (int)(eol - comment) + 1 );
                memcpy( ptr, comment, eol - comment + 1 );
                ptr += eol - comment;
                comment = eol + 1;
                eol = strchr( comment, '\n' );
            }
            else
            {
                len = (int)strlen(comment);
                ptr = icvFSResizeWriteBuffer( fs, ptr, len );
                memcpy( ptr, comment, len );
                ptr += len;
                comment = 0;
            }
            fs->buffer = ptr;
            ptr = icvXMLFlush( fs );
        }
        sprintf( ptr, "-->" );
        fs->buffer = ptr + 3;
        icvXMLFlush( fs );
    }
}


/****************************************************************************************\
*                              Common High-Level Functions                               *
\****************************************************************************************/

CV_IMPL CvFileStorage*
cvOpenFileStorage( const char* filename, CvMemStorage* dststorage, int flags )
{
    CvFileStorage* fs = 0;
    char* xml_buf = 0;
    int default_block_size = 1 << 18;
    bool append = (flags & 3) == CV_STORAGE_APPEND;
    bool isGZ = false;

    if( !filename )
        CV_Error( CV_StsNullPtr, "NULL filename" );

    fs = (CvFileStorage*)cvAlloc( sizeof(*fs) );
    memset( fs, 0, sizeof(*fs));

    fs->memstorage = cvCreateMemStorage( default_block_size );
    fs->dststorage = dststorage ? dststorage : fs->memstorage;

    int fnamelen = (int)strlen(filename);
    if( !fnamelen )
        CV_Error( CV_StsError, "Empty filename" );

    fs->filename = (char*)cvMemStorageAlloc( fs->memstorage, fnamelen+1 );
    strcpy( fs->filename, filename );
    
    char* dot_pos = strrchr(fs->filename, '.');
    char compression = '\0';

    if( dot_pos && dot_pos[1] == 'g' && dot_pos[2] == 'z' &&
        (dot_pos[3] == '\0' || (isdigit(dot_pos[3]) && dot_pos[4] == '\0')) )
    {
        if( append )
            CV_Error(CV_StsNotImplemented, "Appending data to compressed file is not implemented" );
        isGZ = true;
        compression = dot_pos[3];
        if( compression )
            dot_pos[3] = '\0', fnamelen--;
    }

    fs->flags = CV_FILE_STORAGE;
    fs->write_mode = (flags & 3) != 0;

    if( !isGZ )
    {
        fs->file = fopen(fs->filename, !fs->write_mode ? "rt" : !append ? "wt" : "a+t" );
        if( !fs->file )
            goto _exit_;
    }
    else
    {
        char mode[] = { fs->write_mode ? 'w' : 'r', 'b', compression ? compression : '3', '\0' };
        fs->gzfile = gzopen(fs->filename, mode);
        if( !fs->gzfile )
            goto _exit_;
    }

    fs->roots = 0;
    fs->struct_indent = 0;
    fs->struct_flags = 0;
    fs->wrap_margin = 71;

    if( fs->write_mode )
    {
        // we use factor=6 for XML (the longest characters (' and ") are encoded with 6 bytes (&apos; and &quot;)
        // and factor=4 for YAML ( as we use 4 bytes for non ASCII characters (e.g. \xAB))
        int buf_size = CV_FS_MAX_LEN*(fs->is_xml ? 6 : 4) + 1024;

        dot_pos = fs->filename + fnamelen - (isGZ ? 7 : 4);
        fs->is_xml = dot_pos > fs->filename && (memcmp( dot_pos, ".xml", 4) == 0 ||
                      memcmp(dot_pos, ".XML", 4) == 0 || memcmp(dot_pos, ".Xml", 4) == 0);

        if( append )
            fseek( fs->file, 0, SEEK_END );

        fs->write_stack = cvCreateSeq( 0, sizeof(CvSeq), fs->is_xml ?
                sizeof(CvXMLStackRecord) : sizeof(int), fs->memstorage );
        fs->is_first = 1;
        fs->struct_indent = 0;
        fs->struct_flags = CV_NODE_EMPTY;
        fs->buffer_start = fs->buffer = (char*)cvAlloc( buf_size + 1024 );
        fs->buffer_end = fs->buffer_start + buf_size;
        if( fs->is_xml )
        {
            int file_size = fs->file ? (int)ftell( fs->file ) : 0;
            fs->strstorage = cvCreateChildMemStorage( fs->memstorage );
            if( !append || file_size == 0 )
            {
                icvPuts( fs, "<?xml version=\"1.0\"?>\n" );
                icvPuts( fs, "<opencv_storage>\n" );
            }
            else
            {
                int xml_buf_size = 1 << 10;
                char substr[] = "</opencv_storage>";
                int last_occurence = -1;
                xml_buf_size = MIN(xml_buf_size, file_size);
                fseek( fs->file, -xml_buf_size, SEEK_END );
                xml_buf = (char*)cvAlloc( xml_buf_size+2 );
                // find the last occurence of </opencv_storage>
                for(;;)
                {
                    int line_offset = ftell( fs->file );
                    char* ptr0 = icvGets( fs, xml_buf, xml_buf_size ), *ptr;
                    if( !ptr0 )
                        break;
                    ptr = ptr0;
                    for(;;)
                    {
                        ptr = strstr( ptr, substr );
                        if( !ptr )
                            break;
                        last_occurence = line_offset + (int)(ptr - ptr0);
                        ptr += strlen(substr);
                    }
                }
                if( last_occurence < 0 )
                    CV_Error( CV_StsError, "Could not find </opencv_storage> in the end of file.\n" );
                icvClose( fs );
                fs->file = fopen( fs->filename, "r+t" );
                fseek( fs->file, last_occurence, SEEK_SET );
                // replace the last "</opencv_storage>" with " <!-- resumed -->", which has the same length
                icvPuts( fs, " <!-- resumed -->" );
                fseek( fs->file, 0, SEEK_END );
                icvPuts( fs, "\n" );
            }
            fs->start_write_struct = icvXMLStartWriteStruct;
            fs->end_write_struct = icvXMLEndWriteStruct;
            fs->write_int = icvXMLWriteInt;
            fs->write_real = icvXMLWriteReal;
            fs->write_string = icvXMLWriteString;
            fs->write_comment = icvXMLWriteComment;
            fs->start_next_stream = icvXMLStartNextStream;
        }
        else
        {
            if( !append )
                icvPuts( fs, "%YAML:1.0\n" );
            else
                icvPuts( fs, "...\n---\n" );
            fs->start_write_struct = icvYMLStartWriteStruct;
            fs->end_write_struct = icvYMLEndWriteStruct;
            fs->write_int = icvYMLWriteInt;
            fs->write_real = icvYMLWriteReal;
            fs->write_string = icvYMLWriteString;
            fs->write_comment = icvYMLWriteComment;
            fs->start_next_stream = icvYMLStartNextStream;
        }
    }
    else
    {
        int buf_size = 1 << 20;
        const char* yaml_signature = "%YAML:";
        char buf[16];
        icvGets( fs, buf, sizeof(buf)-2 );
        fs->is_xml = strncmp( buf, yaml_signature, strlen(yaml_signature) ) != 0;

        if( !isGZ )
        {
            fseek( fs->file, 0, SEEK_END );
            buf_size = ftell( fs->file );
            buf_size = MIN( buf_size, (1 << 20) );
            buf_size = MAX( buf_size, CV_FS_MAX_LEN*2 + 1024 );
        }
        icvRewind(fs);

        fs->str_hash = cvCreateMap( 0, sizeof(CvStringHash),
                        sizeof(CvStringHashNode), fs->memstorage, 256 );

        fs->roots = cvCreateSeq( 0, sizeof(CvSeq),
                        sizeof(CvFileNode), fs->memstorage );

        fs->buffer = fs->buffer_start = (char*)cvAlloc( buf_size + 256 );
        fs->buffer_end = fs->buffer_start + buf_size;
        fs->buffer[0] = '\n';
        fs->buffer[1] = '\0';

        //mode = cvGetErrMode();
        //cvSetErrMode( CV_ErrModeSilent );
        if( fs->is_xml )
            icvXMLParse( fs );
        else
            icvYMLParse( fs );
        //cvSetErrMode( mode );

        // release resources that we do not need anymore
        cvFree( &fs->buffer_start );
        fs->buffer = fs->buffer_end = 0;
    }
_exit_:
    if( fs )
    {
        if( cvGetErrStatus() < 0 || (!fs->file && !fs->gzfile) )
        {
            cvReleaseFileStorage( &fs );
        }
        else if( !fs->write_mode )
        {
            icvClose(fs);
        }
    }

    cvFree( &xml_buf );
    return  fs;
}


CV_IMPL void
cvStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags,
                    const char* type_name, CvAttrList /*attributes*/ )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->start_write_struct( fs, key, struct_flags, type_name );
}


CV_IMPL void
cvEndWriteStruct( CvFileStorage* fs )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->end_write_struct( fs );
}


CV_IMPL void
cvWriteInt( CvFileStorage* fs, const char* key, int value )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->write_int( fs, key, value );
}


CV_IMPL void
cvWriteReal( CvFileStorage* fs, const char* key, double value )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->write_real( fs, key, value );
}


CV_IMPL void
cvWriteString( CvFileStorage* fs, const char* key, const char* value, int quote )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->write_string( fs, key, value, quote );
}


CV_IMPL void
cvWriteComment( CvFileStorage* fs, const char* comment, int eol_comment )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->write_comment( fs, comment, eol_comment );
}


CV_IMPL void
cvStartNextStream( CvFileStorage* fs )
{
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);
    fs->start_next_stream( fs );
}


static const char icvTypeSymbol[] = "ucwsifdr";
#define CV_FS_MAX_FMT_PAIRS  128

static char*
icvEncodeFormat( int elem_type, char* dt )
{
    sprintf( dt, "%d%c", CV_MAT_CN(elem_type), icvTypeSymbol[CV_MAT_DEPTH(elem_type)] );
    return dt + ( dt[2] == '\0' && dt[0] == '1' );
}

static int
icvDecodeFormat( const char* dt, int* fmt_pairs, int max_len )
{
    int fmt_pair_count = 0;
    int i = 0, k = 0, len = dt ? (int)strlen(dt) : 0;

    if( !dt || !len )
        return 0;

    assert( fmt_pairs != 0 && max_len > 0 );
    fmt_pairs[0] = 0;
    max_len *= 2;

    for( ; k < len; k++ )
    {
        char c = dt[k];

        if( isdigit(c) )
        {
            int count = c - '0';
            if( isdigit(dt[k+1]) )
            {
                char* endptr = 0;
                count = (int)strtol( dt+k, &endptr, 10 );
                k = (int)(endptr - dt) - 1;
            }

            if( count <= 0 )
                CV_Error( CV_StsBadArg, "Invalid data type specification" );

            fmt_pairs[i] = count;
        }
        else
        {
            const char* pos = strchr( icvTypeSymbol, c );
            if( !pos )
                CV_Error( CV_StsBadArg, "Invalid data type specification" );
            if( fmt_pairs[i] == 0 )
                fmt_pairs[i] = 1;
            fmt_pairs[i+1] = (int)(pos - icvTypeSymbol);
            if( i > 0 && fmt_pairs[i+1] == fmt_pairs[i-1] )
                fmt_pairs[i-2] += fmt_pairs[i];
            else
            {
                i += 2;
                if( i >= max_len )
                    CV_Error( CV_StsBadArg, "Too long data type specification" );
            }
            fmt_pairs[i] = 0;
        }
    }

    fmt_pair_count = i/2;
    return fmt_pair_count;
}


static int
icvCalcElemSize( const char* dt, int initial_size )
{
    int size = 0;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS], i, fmt_pair_count;
    int comp_size;

    fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
    fmt_pair_count *= 2;
    for( i = 0, size = initial_size; i < fmt_pair_count; i += 2 )
    {
        comp_size = CV_ELEM_SIZE(fmt_pairs[i+1]);
        size = cvAlign( size, comp_size );
        size += comp_size * fmt_pairs[i];
    }
    if( initial_size == 0 )
    {
        comp_size = CV_ELEM_SIZE(fmt_pairs[1]);
        size = cvAlign( size, comp_size );
    }
    return size;
}


static int
icvDecodeSimpleFormat( const char* dt )
{
    int elem_type = -1;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS], fmt_pair_count;

    fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
    if( fmt_pair_count != 1 || fmt_pairs[0] > 4 )
        CV_Error( CV_StsError, "Too complex format for the matrix" );

    elem_type = CV_MAKETYPE( fmt_pairs[1], fmt_pairs[0] );

    return elem_type;
}


CV_IMPL void
cvWriteRawData( CvFileStorage* fs, const void* _data, int len, const char* dt )
{
    const char* data0 = (const char*)_data;
    int offset = 0;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS*2], k, fmt_pair_count;
    char buf[256] = "";

    CV_CHECK_OUTPUT_FILE_STORAGE( fs );

    if( !data0 )
        CV_Error( CV_StsNullPtr, "Null data pointer" );

    if( len < 0 )
        CV_Error( CV_StsOutOfRange, "Negative number of elements" );

    fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );

    if( !len )
        return;

    if( fmt_pair_count == 1 )
    {
        fmt_pairs[0] *= len;
        len = 1;
    }

    for(;len--;)
    {
        for( k = 0; k < fmt_pair_count; k++ )
        {
            int i, count = fmt_pairs[k*2];
            int elem_type = fmt_pairs[k*2+1];
            int elem_size = CV_ELEM_SIZE(elem_type);
            const char* data, *ptr;

            offset = cvAlign( offset, elem_size );
            data = data0 + offset;

            for( i = 0; i < count; i++ )
            {
                switch( elem_type )
                {
                case CV_8U:
                    ptr = icv_itoa( *(uchar*)data, buf, 10 );
                    data++;
                    break;
                case CV_8S:
                    ptr = icv_itoa( *(char*)data, buf, 10 );
                    data++;
                    break;
                case CV_16U:
                    ptr = icv_itoa( *(ushort*)data, buf, 10 );
                    data += sizeof(ushort);
                    break;
                case CV_16S:
                    ptr = icv_itoa( *(short*)data, buf, 10 );
                    data += sizeof(short);
                    break;
                case CV_32S:
                    ptr = icv_itoa( *(int*)data, buf, 10 );
                    data += sizeof(int);
                    break;
                case CV_32F:
                    ptr = icvFloatToString( buf, *(float*)data );
                    data += sizeof(float);
                    break;
                case CV_64F:
                    ptr = icvDoubleToString( buf, *(double*)data );
                    data += sizeof(double);
                    break;
                case CV_USRTYPE1: /* reference */
                    ptr = icv_itoa( (int)*(size_t*)data, buf, 10 );
                    data += sizeof(size_t);
                    break;
                default:
                    assert(0);
                    return;
                }

                if( fs->is_xml )
                {
                    int buf_len = (int)strlen(ptr);
                    icvXMLWriteScalar( fs, 0, ptr, buf_len );
                }
                else
                    icvYMLWrite( fs, 0, ptr );
            }

            offset = (int)(data - data0);
        }
    }
}


CV_IMPL void
cvStartReadRawData( const CvFileStorage* fs, const CvFileNode* src, CvSeqReader* reader )
{
    int node_type;
    CV_CHECK_FILE_STORAGE( fs );

    if( !src || !reader )
        CV_Error( CV_StsNullPtr, "Null pointer to source file node or reader" );

    node_type = CV_NODE_TYPE(src->tag);
    if( node_type == CV_NODE_INT || node_type == CV_NODE_REAL )
    {
        // emulate reading from 1-element sequence
        reader->ptr = (schar*)src;
        reader->block_max = reader->ptr + sizeof(*src)*2;
        reader->block_min = reader->ptr;
        reader->seq = 0;
    }
    else if( node_type == CV_NODE_SEQ )
    {
        cvStartReadSeq( src->data.seq, reader, 0 );
    }
    else if( node_type == CV_NODE_NONE )
    {
        memset( reader, 0, sizeof(*reader) );
    }
    else
        CV_Error( CV_StsBadArg, "The file node should be a numerical scalar or a sequence" );
}


CV_IMPL void
cvReadRawDataSlice( const CvFileStorage* fs, CvSeqReader* reader,
                    int len, void* _data, const char* dt )
{
    char* data0 = (char*)_data;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS*2], k = 0, fmt_pair_count;
    int i = 0, offset = 0, count = 0;

    CV_CHECK_FILE_STORAGE( fs );

    if( !reader || !data0 )
        CV_Error( CV_StsNullPtr, "Null pointer to reader or destination array" );

    if( !reader->seq && len != 1 )
        CV_Error( CV_StsBadSize, "The readed sequence is a scalar, thus len must be 1" );

    fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );

    for(;;)
    {
        for( k = 0; k < fmt_pair_count; k++ )
        {
            int elem_type = fmt_pairs[k*2+1];
            int elem_size = CV_ELEM_SIZE(elem_type);
            char* data;

            count = fmt_pairs[k*2];
            offset = cvAlign( offset, elem_size );
            data = data0 + offset;

            for( i = 0; i < count; i++ )
            {
                CvFileNode* node = (CvFileNode*)reader->ptr;
                if( CV_NODE_IS_INT(node->tag) )
                {
                    int ival = node->data.i;

                    switch( elem_type )
                    {
                    case CV_8U:
                        *(uchar*)data = CV_CAST_8U(ival);
                        data++;
                        break;
                    case CV_8S:
                        *(char*)data = CV_CAST_8S(ival);
                        data++;
                        break;
                    case CV_16U:
                        *(ushort*)data = CV_CAST_16U(ival);
                        data += sizeof(ushort);
                        break;
                    case CV_16S:
                        *(short*)data = CV_CAST_16S(ival);
                        data += sizeof(short);
                        break;
                    case CV_32S:
                        *(int*)data = ival;
                        data += sizeof(int);
                        break;
                    case CV_32F:
                        *(float*)data = (float)ival;
                        data += sizeof(float);
                        break;
                    case CV_64F:
                        *(double*)data = (double)ival;
                        data += sizeof(double);
                        break;
                    case CV_USRTYPE1: /* reference */
                        *(size_t*)data = ival;
                        data += sizeof(size_t);
                        break;
                    default:
                        assert(0);
                        return;
                    }
                }
                else if( CV_NODE_IS_REAL(node->tag) )
                {
                    double fval = node->data.f;
                    int ival;

                    switch( elem_type )
                    {
                    case CV_8U:
                        ival = cvRound(fval);
                        *(uchar*)data = CV_CAST_8U(ival);
                        data++;
                        break;
                    case CV_8S:
                        ival = cvRound(fval);
                        *(char*)data = CV_CAST_8S(ival);
                        data++;
                        break;
                    case CV_16U:
                        ival = cvRound(fval);
                        *(ushort*)data = CV_CAST_16U(ival);
                        data += sizeof(ushort);
                        break;
                    case CV_16S:
                        ival = cvRound(fval);
                        *(short*)data = CV_CAST_16S(ival);
                        data += sizeof(short);
                        break;
                    case CV_32S:
                        ival = cvRound(fval);
                        *(int*)data = ival;
                        data += sizeof(int);
                        break;
                    case CV_32F:
                        *(float*)data = (float)fval;
                        data += sizeof(float);
                        break;
                    case CV_64F:
                        *(double*)data = fval;
                        data += sizeof(double);
                        break;
                    case CV_USRTYPE1: /* reference */
                        ival = cvRound(fval);
                        *(size_t*)data = ival;
                        data += sizeof(size_t);
                        break;
                    default:
                        assert(0);
                        return;
                    }
                }
                else
                    CV_Error( CV_StsError,
                    "The sequence element is not a numerical scalar" );

                CV_NEXT_SEQ_ELEM( sizeof(CvFileNode), *reader );
                if( !--len )
                    goto end_loop;
            }

            offset = (int)(data - data0);
        }
    }

end_loop:
    if( i != count - 1 || k != fmt_pair_count - 1 )
        CV_Error( CV_StsBadSize,
        "The sequence slice does not fit an integer number of records" );

    if( !reader->seq )
        reader->ptr -= sizeof(CvFileNode);
}


CV_IMPL void
cvReadRawData( const CvFileStorage* fs, const CvFileNode* src,
               void* data, const char* dt )
{
    CvSeqReader reader;

    if( !src || !data )
        CV_Error( CV_StsNullPtr, "Null pointers to source file node or destination array" );

    cvStartReadRawData( fs, src, &reader );
    cvReadRawDataSlice( fs, &reader, CV_NODE_IS_SEQ(src->tag) ?
                        src->data.seq->total : 1, data, dt );
}


static void
icvWriteFileNode( CvFileStorage* fs, const char* name, const CvFileNode* node );

static void
icvWriteCollection( CvFileStorage* fs, const CvFileNode* node )
{
    int i, total = node->data.seq->total;
    int elem_size = node->data.seq->elem_size;
    int is_map = CV_NODE_IS_MAP(node->tag);
    CvSeqReader reader;

    cvStartReadSeq( node->data.seq, &reader, 0 );

    for( i = 0; i < total; i++ )
    {
        CvFileMapNode* elem = (CvFileMapNode*)reader.ptr;
        if( !is_map || CV_IS_SET_ELEM(elem) )
        {
            const char* name = is_map ? elem->key->str.ptr : 0;
            icvWriteFileNode( fs, name, &elem->value );
        }
        CV_NEXT_SEQ_ELEM( elem_size, reader );
    }
}

static void
icvWriteFileNode( CvFileStorage* fs, const char* name, const CvFileNode* node )
{
    switch( CV_NODE_TYPE(node->tag) )
    {
    case CV_NODE_INT:
        fs->write_int( fs, name, node->data.i );
        break;
    case CV_NODE_REAL:
        fs->write_real( fs, name, node->data.f );
        break;
    case CV_NODE_STR:
        fs->write_string( fs, name, node->data.str.ptr, 0 );
        break;
    case CV_NODE_SEQ:
    case CV_NODE_MAP:
        fs->start_write_struct( fs, name, CV_NODE_TYPE(node->tag) +
                (CV_NODE_SEQ_IS_SIMPLE(node->data.seq) ? CV_NODE_FLOW : 0),
                node->info ? node->info->type_name : 0 );
        icvWriteCollection( fs, node );
        fs->end_write_struct( fs );
        break;
    case CV_NODE_NONE:
        fs->start_write_struct( fs, name, CV_NODE_SEQ, 0 );
        fs->end_write_struct( fs );
        break;
    default:
        CV_Error( CV_StsBadFlag, "Unknown type of file node" );
    }
}


CV_IMPL void
cvWriteFileNode( CvFileStorage* fs, const char* new_node_name,
                 const CvFileNode* node, int embed )
{
    CvFileStorage* dst = 0;
    CV_CHECK_OUTPUT_FILE_STORAGE(fs);

    if( !node )
        return;

    if( CV_NODE_IS_COLLECTION(node->tag) && embed )
    {
        icvWriteCollection( fs, node );
    }
    else
    {
        icvWriteFileNode( fs, new_node_name, node );
    }
    /*
    int i, stream_count;
    stream_count = fs->roots->total;
    for( i = 0; i < stream_count; i++ )
    {
        CvFileNode* node = (CvFileNode*)cvGetSeqElem( fs->roots, i, 0 );
        icvDumpCollection( dst, node );
        if( i < stream_count - 1 )
            dst->start_next_stream( dst );
    }*/
    cvReleaseFileStorage( &dst );
}


CV_IMPL const char*
cvGetFileNodeName( const CvFileNode* file_node )
{
    return file_node && CV_NODE_HAS_NAME(file_node->tag) ?
        ((CvFileMapNode*)file_node)->key->str.ptr : 0;
}

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

static int
icvIsMat( const void* ptr )
{
    return CV_IS_MAT_HDR(ptr);
}

static void
icvWriteMat( CvFileStorage* fs, const char* name,
             const void* struct_ptr, CvAttrList /*attr*/ )
{
    const CvMat* mat = (const CvMat*)struct_ptr;
    char dt[16];
    CvSize size;
    int y;

    assert( CV_IS_MAT(mat) );

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_MAT );
    cvWriteInt( fs, "rows", mat->rows );
    cvWriteInt( fs, "cols", mat->cols );
    cvWriteString( fs, "dt", icvEncodeFormat( CV_MAT_TYPE(mat->type), dt ), 0 );
    cvStartWriteStruct( fs, "data", CV_NODE_SEQ + CV_NODE_FLOW );

    size = cvGetSize(mat);
    if( CV_IS_MAT_CONT(mat->type) )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( y = 0; y < size.height; y++ )
        cvWriteRawData( fs, mat->data.ptr + y*mat->step, size.width, dt );
    cvEndWriteStruct( fs );
    cvEndWriteStruct( fs );
}


static int
icvFileNodeSeqLen( CvFileNode* node )
{
    return CV_NODE_IS_COLLECTION(node->tag) ? node->data.seq->total :
           CV_NODE_TYPE(node->tag) != CV_NODE_NONE;
}


static void*
icvReadMat( CvFileStorage* fs, CvFileNode* node )
{
    void* ptr = 0;
    CvMat* mat;
    const char* dt;
    CvFileNode* data;
    int rows, cols, elem_type;

    rows = cvReadIntByName( fs, node, "rows", 0 );
    cols = cvReadIntByName( fs, node, "cols", 0 );
    dt = cvReadStringByName( fs, node, "dt", 0 );

    if( rows == 0 || cols == 0 || dt == 0 )
        CV_Error( CV_StsError, "Some of essential matrix attributes are absent" );

    elem_type = icvDecodeSimpleFormat( dt );

    data = cvGetFileNodeByName( fs, node, "data" );
    if( !data )
        CV_Error( CV_StsError, "The matrix data is not found in file storage" );

    if( icvFileNodeSeqLen( data ) != rows*cols*CV_MAT_CN(elem_type) )
        CV_Error( CV_StsUnmatchedSizes,
        "The matrix size does not match to the number of stored elements" );

    mat = cvCreateMat( rows, cols, elem_type );
    cvReadRawData( fs, data, mat->data.ptr, dt );

    ptr = mat;
    return ptr;
}


/******************************* CvMatND ******************************/

static int
icvIsMatND( const void* ptr )
{
    return CV_IS_MATND(ptr);
}


static void
icvWriteMatND( CvFileStorage* fs, const char* name,
               const void* struct_ptr, CvAttrList /*attr*/ )
{
    void* mat = (void*)struct_ptr;
    CvMatND stub;
    CvNArrayIterator iterator;
    int dims, sizes[CV_MAX_DIM];
    char dt[16];

    assert( CV_IS_MATND(mat) );

    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_MATND );
    dims = cvGetDims( mat, sizes );
    cvStartWriteStruct( fs, "sizes", CV_NODE_SEQ + CV_NODE_FLOW );
    cvWriteRawData( fs, sizes, dims, "i" );
    cvEndWriteStruct( fs );
    cvWriteString( fs, "dt", icvEncodeFormat( cvGetElemType(mat), dt ), 0 );
    cvStartWriteStruct( fs, "data", CV_NODE_SEQ + CV_NODE_FLOW );

    cvInitNArrayIterator( 1, &mat, 0, &stub, &iterator );

    do
        cvWriteRawData( fs, iterator.ptr[0], iterator.size.width, dt );
    while( cvNextNArraySlice( &iterator ));
    cvEndWriteStruct( fs );
    cvEndWriteStruct( fs );
}


static void*
icvReadMatND( CvFileStorage* fs, CvFileNode* node )
{
    void* ptr = 0;
    CvMatND* mat;
    const char* dt;
    CvFileNode* data;
    CvFileNode* sizes_node;
    int sizes[CV_MAX_DIM], dims, elem_type;
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
        total_size *= sizes[i];

    if( icvFileNodeSeqLen( data ) != total_size )
        CV_Error( CV_StsUnmatchedSizes,
        "The matrix size does not match to the number of stored elements" );

    mat = cvCreateMatND( dims, sizes, elem_type );
    cvReadRawData( fs, data, mat->data.ptr, dt );

    ptr = mat;
    return ptr;
}


/******************************* CvSparseMat ******************************/

static int
icvIsSparseMat( const void* ptr )
{
    return CV_IS_SPARSE_MAT(ptr);
}


static int
icvSortIdxCmpFunc( const void* _a, const void* _b, void* userdata )
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


static void
icvWriteSparseMat( CvFileStorage* fs, const char* name,
                   const void* struct_ptr, CvAttrList /*attr*/ )
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


static void*
icvReadSparseMat( CvFileStorage* fs, CvFileNode* node )
{
    void* ptr = 0;
    CvSparseMat* mat;
    const char* dt;
    CvFileNode* data;
    CvFileNode* sizes_node;
    CvSeqReader reader;
    CvSeq* elements;
    int* idx;
    int* sizes = 0, dims, elem_type, cn;
    int i;

    sizes_node = cvGetFileNodeByName( fs, node, "sizes" );
    dt = cvReadStringByName( fs, node, "dt", 0 );

    if( !sizes_node || !dt )
        CV_Error( CV_StsError, "Some of essential matrix attributes are absent" );

    dims = CV_NODE_IS_SEQ(sizes_node->tag) ? sizes_node->data.seq->total :
           CV_NODE_IS_INT(sizes_node->tag) ? 1 : -1;

    if( dims <= 0 || dims > CV_MAX_DIM_HEAP )
        CV_Error( CV_StsParseError, "Could not determine sparse matrix dimensionality" );

    sizes = (int*)cvStackAlloc( dims*sizeof(sizes[0]));
    cvReadRawData( fs, sizes_node, sizes, "i" );
    elem_type = icvDecodeSimpleFormat( dt );

    data = cvGetFileNodeByName( fs, node, "data" );
    if( !data || !CV_NODE_IS_SEQ(data->tag) )
        CV_Error( CV_StsError, "The matrix data is not found in file storage" );

    mat = cvCreateSparseMat( dims, sizes, elem_type );

    cn = CV_MAT_CN(elem_type);
    idx = (int*)alloca( dims*sizeof(idx[0]) );
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

static int
icvIsImage( const void* ptr )
{
    return CV_IS_IMAGE_HDR(ptr);
}

static void
icvWriteImage( CvFileStorage* fs, const char* name,
               const void* struct_ptr, CvAttrList /*attr*/ )
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
    sprintf( dt_buf, "%d%c", image->nChannels, icvTypeSymbol[depth] );
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


static void*
icvReadImage( CvFileStorage* fs, CvFileNode* node )
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
    if( strcmp( data_order, "interleaved" ) != 0 )
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

static int
icvIsSeq( const void* ptr )
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


static void*
icvCloneSeq( const void* ptr )
{
    return cvSeqSlice( (CvSeq*)ptr, CV_WHOLE_SEQ,
                       0 /* use the same storage as for the original sequence */, 1 );
}


static void
icvWriteHeaderData( CvFileStorage* fs, const CvSeq* seq,
                    CvAttrList* attr, int initial_header_size )
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


static char*
icvGetFormat( const CvSeq* seq, const char* dt_key, CvAttrList* attr,
              int initial_elem_size, char* dt_buf )
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


static void
icvWriteSeq( CvFileStorage* fs, const char* name,
             const void* struct_ptr,
             CvAttrList attr, int level )
{
    const CvSeq* seq = (CvSeq*)struct_ptr;
    CvSeqBlock* block;
    char buf[128];
    char dt_buf[128], *dt;

    assert( CV_IS_SEQ( seq ));
    cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_SEQ );

    if( level >= 0 )
        cvWriteInt( fs, "level", level );

    sprintf( buf, "%08x", seq->flags );
    cvWriteString( fs, "flags", buf, 1 );
    cvWriteInt( fs, "count", seq->total );
    dt = icvGetFormat( seq, "dt", &attr, 0, dt_buf );
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


static void
icvWriteSeqTree( CvFileStorage* fs, const char* name,
                 const void* struct_ptr, CvAttrList attr )
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


static void*
icvReadSeq( CvFileStorage* fs, CvFileNode* node )
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

    flags = (int)strtol( flags_str, &endptr, 16 );
    if( endptr == flags_str || (flags & CV_MAGIC_MASK) != CV_SEQ_MAGIC_VAL )
        CV_Error( CV_StsError, "The sequence flags are invalid" );

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


static void*
icvReadSeqTree( CvFileStorage* fs, CvFileNode* node )
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

static int
icvIsGraph( const void* ptr )
{
    return CV_IS_GRAPH(ptr);
}


static void
icvReleaseGraph( void** ptr )
{
    if( !ptr )
        CV_Error( CV_StsNullPtr, "NULL double pointer" );

    *ptr = 0; // it's impossible now to release graph, so just clear the pointer
}


static void*
icvCloneGraph( const void* ptr )
{
    return cvCloneGraph( (const CvGraph*)ptr, 0 );
}


static void
icvWriteGraph( CvFileStorage* fs, const char* name,
               const void* struct_ptr, CvAttrList attr )
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

    sprintf( buf, "%08x", graph->flags );
    cvWriteString( fs, "flags", buf, 1 );

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
                if( fmt_pair_count > 2 || CV_ELEM_SIZE(fmt_pairs[2*2+1]) >= (int)sizeof(double))
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


static void*
icvReadGraph( CvFileStorage* fs, CvFileNode* node )
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
        CV_Error( CV_StsError, "Some of essential sequence attributes are absent" );

    flags = (int)strtol( flags_str, &endptr, 16 );
    if( endptr == flags_str ||
        (flags & (CV_SEQ_KIND_MASK|CV_MAGIC_MASK)) != (CV_GRAPH|CV_SET_MAGIC_VAL))
        CV_Error( CV_StsError, "Invalid graph signature" );

    header_dt = cvReadStringByName( fs, node, "header_dt", 0 );
    header_node = cvGetFileNodeByName( fs, node, "header_user_data" );

    if( (header_dt != 0) ^ (header_node != 0) )
        CV_Error( CV_StsError,
        "One of \"header_dt\" and \"header_user_data\" is there, while the other is not" );

    if( header_dt )
        header_size = icvCalcElemSize( header_dt, header_size );

    if( vtx_dt > 0 )
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
            dst_edge_dt = edge_dt + 3 + isdigit(edge_dt[2]);
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
        cvReadRawData( fs, header_node, (char*)graph + sizeof(CvGraph), header_dt );

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
                    CV_Error( CV_StsBadArg, "Duplicated edge has occured" );

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

CV_IMPL  void
cvRegisterType( const CvTypeInfo* _info )
{
    CvTypeInfo* info = 0;
    int i, len;
    char c;

    //if( !CvType::first )
    //    icvCreateStandardTypes();

    if( !_info || _info->header_size != sizeof(CvTypeInfo) )
        CV_Error( CV_StsBadSize, "Invalid type info" );

    if( !_info->is_instance || !_info->release ||
        !_info->read || !_info->write )
        CV_Error( CV_StsNullPtr,
        "Some of required function pointers "
        "(is_instance, release, read or write) are NULL");

    c = _info->type_name[0];
    if( !isalpha(c) && c != '_' )
        CV_Error( CV_StsBadArg, "Type name should start with a letter or _" );

    len = (int)strlen(_info->type_name);

    for( i = 0; i < len; i++ )
    {
        c = _info->type_name[i];
        if( !isalnum(c) && c != '-' && c != '_' )
            CV_Error( CV_StsBadArg,
            "Type name should contain only letters, digits, - and _" );
    }

    info = (CvTypeInfo*)malloc( sizeof(*info) + len + 1 );

    *info = *_info;
    info->type_name = (char*)(info + 1);
    memcpy( (char*)info->type_name, _info->type_name, len + 1 );

    info->flags = 0;
    info->next = CvType::first;
    info->prev = 0;
    if( CvType::first )
        CvType::first->prev = info;
    else
        CvType::last = info;
    CvType::first = info;
}


CV_IMPL void
cvUnregisterType( const char* type_name )
{
    CvTypeInfo* info;

    info = cvFindType( type_name );
    if( info )
    {
        if( info->prev )
            info->prev->next = info->next;
        else
            CvType::first = info->next;

        if( info->next )
            info->next->prev = info->prev;
        else
            CvType::last = info->prev;

        if( !CvType::first || !CvType::last )
            CvType::first = CvType::last = 0;

        free( info );
    }
}


CV_IMPL CvTypeInfo*
cvFirstType( void )
{
    return CvType::first;
}


CV_IMPL CvTypeInfo*
cvFindType( const char* type_name )
{
    CvTypeInfo* info = 0;

    if (type_name)
      for( info = CvType::first; info != 0; info = info->next )
        if( strcmp( info->type_name, type_name ) == 0 )
	  break;
    
    return info;
}


CV_IMPL CvTypeInfo*
cvTypeOf( const void* struct_ptr )
{
    CvTypeInfo* info = 0;

    for( info = CvType::first; info != 0; info = info->next )
        if( info->is_instance( struct_ptr ))
            break;

    return info;
}


/* universal functions */
CV_IMPL void
cvRelease( void** struct_ptr )
{
    CvTypeInfo* info;

    if( !struct_ptr )
        CV_Error( CV_StsNullPtr, "NULL double pointer" );

    if( *struct_ptr )
    {
        info = cvTypeOf( *struct_ptr );
        if( !info )
            CV_Error( CV_StsError, "Unknown object type" );
        if( !info->release )
            CV_Error( CV_StsError, "release function pointer is NULL" );

        info->release( struct_ptr );
        *struct_ptr = 0;
    }
}


void* cvClone( const void* struct_ptr )
{
    void* struct_copy = 0;
    CvTypeInfo* info;

    if( !struct_ptr )
        CV_Error( CV_StsNullPtr, "NULL structure pointer" );

    info = cvTypeOf( struct_ptr );
    if( !info )
        CV_Error( CV_StsError, "Unknown object type" );
    if( !info->clone )
        CV_Error( CV_StsError, "clone function pointer is NULL" );

    struct_copy = info->clone( struct_ptr );
    return struct_copy;
}


/* reads matrix, image, sequence, graph etc. */
CV_IMPL void*
cvRead( CvFileStorage* fs, CvFileNode* node, CvAttrList* list )
{
    void* obj = 0;
    CV_CHECK_FILE_STORAGE( fs );

    if( !node )
        return 0;

    if( !CV_NODE_IS_USER(node->tag) || !node->info )
        CV_Error( CV_StsError, "The node does not represent a user object (unknown type?)" );

    obj = node->info->read( fs, node );
    if( list )
        *list = cvAttrList(0,0);

    return obj;
}


/* writes matrix, image, sequence, graph etc. */
CV_IMPL void
cvWrite( CvFileStorage* fs, const char* name,
         const void* ptr, CvAttrList attributes )
{
    CvTypeInfo* info;

    CV_CHECK_OUTPUT_FILE_STORAGE( fs );

    if( !ptr )
        CV_Error( CV_StsNullPtr, "Null pointer to the written object" );

    info = cvTypeOf( ptr );
    if( !info )
        CV_Error( CV_StsBadArg, "Unknown object" );

    if( !info->write )
        CV_Error( CV_StsBadArg, "The object does not have write function" );

    info->write( fs, name, ptr, attributes );
}


/* simple API for reading/writing data */
CV_IMPL void
cvSave( const char* filename, const void* struct_ptr,
        const char* _name, const char* comment, CvAttrList attributes )
{
    CvFileStorage* fs = 0;

    if( !struct_ptr )
        CV_Error( CV_StsNullPtr, "NULL object pointer" );

    fs = cvOpenFileStorage( filename, 0, CV_STORAGE_WRITE );
    if( !fs )
        CV_Error( CV_StsError, "Could not open the file storage. Check the path and permissions" );

    cv::string name = _name ? cv::string(_name) : cv::FileStorage::getDefaultObjectName(filename);

    if( comment )
        cvWriteComment( fs, comment, 0 );
    cvWrite( fs, name.c_str(), struct_ptr, attributes );
    cvReleaseFileStorage( &fs );
}

CV_IMPL void*
cvLoad( const char* filename, CvMemStorage* memstorage,
        const char* name, const char** _real_name )
{
    void* ptr = 0;
    const char* real_name = 0;
    cv::FileStorage fs(cvOpenFileStorage(filename, memstorage, CV_STORAGE_READ));

    CvFileNode* node = 0;

    if( !fs.isOpened() )
        return 0;

    if( name )
    {
        node = cvGetFileNodeByName( *fs, 0, name );
    }
    else
    {
        int i, k;
        for( k = 0; k < (*fs)->roots->total; k++ )
        {
            CvSeq* seq;
            CvSeqReader reader;

            node = (CvFileNode*)cvGetSeqElem( (*fs)->roots, k );
            if( !CV_NODE_IS_MAP( node->tag ))
                return 0;
            seq = node->data.seq;
            node = 0;

            cvStartReadSeq( seq, &reader, 0 );

            // find the first element in the map
            for( i = 0; i < seq->total; i++ )
            {
                if( CV_IS_SET_ELEM( reader.ptr ))
                {
                    node = (CvFileNode*)reader.ptr;
                    goto stop_search;
                }
                CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
            }
        }

stop_search:
        ;
    }

    if( !node )
        CV_Error( CV_StsObjectNotFound, "Could not find the/an object in file storage" );

    real_name = cvGetFileNodeName( node );
    ptr = cvRead( *fs, node, 0 );

    // sanity check
    if( !memstorage && (CV_IS_SEQ( ptr ) || CV_IS_SET( ptr )) )
        CV_Error( CV_StsNullPtr,
        "NULL memory storage is passed - the loaded dynamic structure can not be stored" );

    if( cvGetErrStatus() < 0 )
    {
        cvRelease( (void**)&ptr );
        real_name = 0;
    }

    if( _real_name)
    {
	if (real_name)
	{
	    *_real_name = (const char*)cvAlloc(strlen(real_name));
    	    memcpy((void*)*_real_name, real_name, strlen(real_name));
	} else {
	    *_real_name = 0;
	}
    }

    return ptr;
}


///////////////////////// new C++ interface for CvFileStorage ///////////////////////////

namespace cv
{

static void getElemSize( const string& fmt, size_t& elemSize, size_t& cn )
{
    const char* dt = fmt.c_str();
    cn = 1;
    if( isdigit(dt[0]) )
    {
        cn = dt[0] - '0';
        dt++;
    }
    char c = dt[0];
    elemSize = cn*(c == 'u' || c == 'c' ? sizeof(uchar) : c == 'w' || c == 's' ? sizeof(ushort) :
        c == 'i' ? sizeof(int) : c == 'f' ? sizeof(float) : c == 'd' ? sizeof(double) :
        c == 'r' ? sizeof(void*) : (size_t)0);
}

FileStorage::FileStorage()
{
    state = UNDEFINED;
}

FileStorage::FileStorage(const string& filename, int flags)
{
    state = UNDEFINED;
    open( filename, flags );
}

FileStorage::FileStorage(CvFileStorage* _fs)
{
    fs = Ptr<CvFileStorage>(_fs);
    state = _fs ? NAME_EXPECTED + INSIDE_MAP : UNDEFINED;
}

FileStorage::~FileStorage()
{
    while( structs.size() > 0 )
    {
        cvEndWriteStruct(fs);
        structs.pop_back();
    }
}

bool FileStorage::open(const string& filename, int flags)
{
    release();
    fs = Ptr<CvFileStorage>(cvOpenFileStorage( filename.c_str(), 0, flags ));
    bool ok = isOpened();
    state = ok ? NAME_EXPECTED + INSIDE_MAP : UNDEFINED;
    return ok;
}

bool FileStorage::isOpened() const
{
    return !fs.empty();
}

void FileStorage::release()
{
    fs.release();
    structs.clear();
    state = UNDEFINED;
}

FileNode FileStorage::root(int streamidx) const
{
    return isOpened() ? FileNode(fs, cvGetRootFileNode(fs, streamidx)) : FileNode();
}

FileStorage& operator << (FileStorage& fs, const string& str)
{
    enum { NAME_EXPECTED = FileStorage::NAME_EXPECTED,
        VALUE_EXPECTED = FileStorage::VALUE_EXPECTED,
        INSIDE_MAP = FileStorage::INSIDE_MAP };
    const char* _str = str.c_str();
    if( !fs.isOpened() || !_str )
        return fs;
    if( *_str == '}' || *_str == ']' )
    {
        if( fs.structs.empty() )
            CV_Error_( CV_StsError, ("Extra closing '%c'", *_str) );
        if( (*_str == ']' ? '[' : '{') != fs.structs.back() )
            CV_Error_( CV_StsError,
            ("The closing '%c' does not match the opening '%c'", *_str, fs.structs.back()));
        fs.structs.pop_back();
        fs.state = fs.structs.empty() || fs.structs.back() == '{' ?
            INSIDE_MAP + NAME_EXPECTED : VALUE_EXPECTED;
        cvEndWriteStruct( *fs );
        fs.elname = string();
    }
    else if( fs.state == NAME_EXPECTED + INSIDE_MAP )
    {
        if( !isalpha(*_str) )
            CV_Error_( CV_StsError, ("Incorrect element name %s", _str) );
        fs.elname = str;
        fs.state = VALUE_EXPECTED + INSIDE_MAP;
    }
    else if( (fs.state & 3) == VALUE_EXPECTED )
    {
        if( *_str == '{' || *_str == '[' )
        {
            fs.structs.push_back(*_str);
            int flags = *_str++ == '{' ? CV_NODE_MAP : CV_NODE_SEQ;
            fs.state = flags == CV_NODE_MAP ? INSIDE_MAP +
                NAME_EXPECTED : VALUE_EXPECTED;
            if( *_str == ':' )
            {
                flags |= CV_NODE_FLOW;
                _str++;
            }
            cvStartWriteStruct( *fs, fs.elname.size() > 0 ? fs.elname.c_str() : 0,
                flags, *_str ? _str : 0 );
            fs.elname = string();
        }
        else
        {
            write( fs, fs.elname, (_str[0] == '\\' && (_str[1] == '{' || _str[1] == '}' ||
                _str[1] == '[' || _str[1] == ']')) ? string(_str+1) : str );
            if( fs.state == INSIDE_MAP + VALUE_EXPECTED )
                fs.state = INSIDE_MAP + NAME_EXPECTED;
        }
    }
    else
        CV_Error( CV_StsError, "Invalid fs.state" );
    return fs;
}


void FileStorage::writeRaw( const string& fmt, const uchar* vec, size_t len )
{
    if( !isOpened() )
        return;
    size_t elemSize, cn;
    getElemSize( fmt, elemSize, cn );
    CV_Assert( len % elemSize == 0 );
    cvWriteRawData( fs, vec, (int)(len/elemSize), fmt.c_str());
}


void FileStorage::writeObj( const string& name, const void* obj )
{
    if( !isOpened() )
        return;
    cvWrite( fs, name.size() > 0 ? name.c_str() : 0, obj );
}


FileNode FileStorage::operator[](const string& nodename) const
{
    return FileNode(fs, cvGetFileNodeByName(fs, 0, nodename.c_str()));
}

FileNode FileStorage::operator[](const char* nodename) const
{
    return FileNode(fs, cvGetFileNodeByName(fs, 0, nodename));
}    

FileNode FileNode::operator[](const string& nodename) const
{
    return FileNode(fs, cvGetFileNodeByName(fs, node, nodename.c_str()));
}

FileNode FileNode::operator[](const char* nodename) const
{
    return FileNode(fs, cvGetFileNodeByName(fs, node, nodename));
}

FileNode FileNode::operator[](int i) const
{
    return isSeq() ? FileNode(fs, (CvFileNode*)cvGetSeqElem(node->data.seq, i)) :
        i == 0 ? *this : FileNode();
}
    
string FileNode::name() const
{
    const char* str;
    return !node || (str = cvGetFileNodeName(node)) == 0 ? string() : string(str);
}    
    
void* FileNode::readObj() const
{
    if( !fs || !node )
        return 0;
    return cvRead( (CvFileStorage*)fs, (CvFileNode*)node );
}


FileNodeIterator::FileNodeIterator()
{
    fs = 0;
    container = 0;
    reader.ptr = 0;
    remaining = 0;
}

FileNodeIterator::FileNodeIterator(const CvFileStorage* _fs,
                                   const CvFileNode* _node, size_t _ofs)
{
    if( _fs && _node )
    {
        int node_type = _node->tag & FileNode::TYPE_MASK;
        fs = _fs;
        container = _node;
        if( !(_node->tag & FileNode::USER) && (node_type == FileNode::SEQ || node_type == FileNode::MAP) )
        {
            cvStartReadSeq( _node->data.seq, &reader );
            remaining = FileNode(_fs, _node).size();
        }
        else
        {
            reader.ptr = (schar*)_node;
            reader.seq = 0;
            remaining = 1;
        }
        (*this) += (int)_ofs;
    }
    else
    {
        fs = 0;
        container = 0;
        reader.ptr = 0;
        remaining = 0;
    }
}

FileNodeIterator::FileNodeIterator(const FileNodeIterator& it)
{
    fs = it.fs;
    container = it.container;
    reader = it.reader;
    remaining = it.remaining;
}

FileNodeIterator& FileNodeIterator::operator ++()
{
    if( remaining > 0 )
    {
        if( reader.seq )
            CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        remaining--;
    }
    return *this;
}

FileNodeIterator FileNodeIterator::operator ++(int)
{
    FileNodeIterator it = *this;
    ++(*this);
    return it;
}

FileNodeIterator& FileNodeIterator::operator --()
{
    if( remaining < FileNode(fs, container).size() )
    {
        if( reader.seq )
            CV_PREV_SEQ_ELEM( reader.seq->elem_size, reader );
        remaining++;
    }
    return *this;
}

FileNodeIterator FileNodeIterator::operator --(int)
{
    FileNodeIterator it = *this;
    --(*this);
    return it;
}

FileNodeIterator& FileNodeIterator::operator += (int ofs)
{
    if( ofs == 0 )
        return *this;
    if( ofs > 0 )
        ofs = std::min(ofs, (int)remaining);
    else
    {
        size_t count = FileNode(fs, container).size();
        ofs = (int)(remaining - std::min(remaining - ofs, count));
    }
    remaining -= ofs;
    if( reader.seq )
        cvSetSeqReaderPos( &reader, ofs, 1 );
    return *this;
}

FileNodeIterator& FileNodeIterator::operator -= (int ofs)
{
    return operator += (-ofs);
}


FileNodeIterator& FileNodeIterator::readRaw( const string& fmt, uchar* vec, size_t maxCount )
{
    if( fs && container && remaining > 0 )
    {
        size_t elem_size, cn;
        getElemSize( fmt, elem_size, cn );
        CV_Assert( elem_size > 0 );
        size_t count = std::min(remaining, maxCount);
        
        if( reader.seq )
        {
            cvReadRawDataSlice( fs, &reader, (int)count, vec, fmt.c_str() );
            remaining -= count*cn;
        }
        else
        {
            cvReadRawData( fs, container, vec, fmt.c_str() );
            remaining = 0;
        }
    }
    return *this;
}

    
void write( FileStorage& fs, const string& name, int value )
{ cvWriteInt( *fs, name.size() ? name.c_str() : 0, value ); }

void write( FileStorage& fs, const string& name, float value )
{ cvWriteReal( *fs, name.size() ? name.c_str() : 0, value ); }

void write( FileStorage& fs, const string& name, double value )
{ cvWriteReal( *fs, name.size() ? name.c_str() : 0, value ); }

void write( FileStorage& fs, const string& name, const string& value )
{ cvWriteString( *fs, name.size() ? name.c_str() : 0, value.c_str() ); }

void writeScalar(FileStorage& fs, int value )
{ cvWriteInt( *fs, 0, value ); }

void writeScalar(FileStorage& fs, float value )
{ cvWriteReal( *fs, 0, value ); }

void writeScalar(FileStorage& fs, double value )
{ cvWriteReal( *fs, 0, value ); }

void writeScalar(FileStorage& fs, const string& value )
{ cvWriteString( *fs, 0, value.c_str() ); }    

    
void write( FileStorage& fs, const string& name, const Mat& value )
{
    CvMat mat = value;
    cvWrite( *fs, name.size() ? name.c_str() : 0, &mat );
}
    
void write( FileStorage& fs, const string& name, const MatND& value )
{
    CvMatND mat = value;
    cvWrite( *fs, name.size() ? name.c_str() : 0, &mat );
}

// TODO: the 4 functions below need to be implemented more efficiently 
void write( FileStorage& fs, const string& name, const SparseMat& value )
{
    Ptr<CvSparseMat> mat = (CvSparseMat*)value;
    cvWrite( *fs, name.size() ? name.c_str() : 0, mat );
}
    
    
WriteStructContext::WriteStructContext(FileStorage& _fs, const string& name,
                   int flags, const string& typeName) : fs(&_fs)
{
    cvStartWriteStruct(**fs, !name.empty() ? name.c_str() : 0, flags,
                       !typeName.empty() ? typeName.c_str() : 0);
}
    
WriteStructContext::~WriteStructContext() { cvEndWriteStruct(**fs); }    

    
void read( const FileNode& node, Mat& mat, const Mat& default_mat )
{
    if( node.empty() )
    {
        default_mat.copyTo(mat);
        return;
    }
    Ptr<CvMat> m = (CvMat*)cvRead((CvFileStorage*)node.fs, (CvFileNode*)*node);
    CV_Assert(CV_IS_MAT(m));
    Mat(m).copyTo(mat);
}
    
void read( const FileNode& node, MatND& mat, const MatND& default_mat )
{
    if( node.empty() )
    {
        default_mat.copyTo(mat);
        return;
    }
    Ptr<CvMatND> m = (CvMatND*)cvRead((CvFileStorage*)node.fs, (CvFileNode*)*node);
    CV_Assert(CV_IS_MATND(m));
    MatND(m).copyTo(mat);
}
        
void read( const FileNode& node, SparseMat& mat, const SparseMat& default_mat )
{
    if( node.empty() )
    {
        default_mat.copyTo(mat);
        return;
    }
    Ptr<CvSparseMat> m = (CvSparseMat*)cvRead((CvFileStorage*)node.fs, (CvFileNode*)*node);
    CV_Assert(CV_IS_SPARSE_MAT(m));
    SparseMat(m).copyTo(mat);
}
    
}

/* End of file. */
