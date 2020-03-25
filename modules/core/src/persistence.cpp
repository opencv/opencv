// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "persistence.hpp"

char* icv_itoa( int _val, char* buffer, int /*radix*/ )
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

void icvPuts( CvFileStorage* fs, const char* str )
{
    if( fs->outbuf )
        std::copy(str, str + strlen(str), std::back_inserter(*fs->outbuf));
    else if( fs->file )
        fputs( str, fs->file );
#if USE_ZLIB
    else if( fs->gzfile )
        gzputs( fs->gzfile, str );
#endif
    else
        CV_Error( CV_StsError, "The storage is not opened" );
}

char* icvGets( CvFileStorage* fs, char* str, int maxCount )
{
    if( fs->strbuf )
    {
        size_t i = fs->strbufpos, len = fs->strbufsize;
        int j = 0;
        const char* instr = fs->strbuf;
        while( i < len && j < maxCount-1 )
        {
            char c = instr[i++];
            if( c == '\0' )
                break;
            str[j++] = c;
            if( c == '\n' )
                break;
        }
        str[j++] = '\0';
        fs->strbufpos = i;
        if (maxCount > 256 && !(fs->flags & cv::FileStorage::BASE64))
            CV_Assert(j < maxCount - 1 && "OpenCV persistence doesn't support very long lines");
        return j > 1 ? str : 0;
    }
    if( fs->file )
    {
        char* ptr = fgets( str, maxCount, fs->file );
        if (ptr && maxCount > 256 && !(fs->flags & cv::FileStorage::BASE64))
        {
            size_t sz = strnlen(ptr, maxCount);
            CV_Assert(sz < (size_t)(maxCount - 1) && "OpenCV persistence doesn't support very long lines");
        }
        return ptr;
    }
#if USE_ZLIB
    if( fs->gzfile )
    {
        char* ptr = gzgets( fs->gzfile, str, maxCount );
        if (ptr && maxCount > 256 && !(fs->flags & cv::FileStorage::BASE64))
        {
            size_t sz = strnlen(ptr, maxCount);
            CV_Assert(sz < (size_t)(maxCount - 1) && "OpenCV persistence doesn't support very long lines");
        }
        return ptr;
    }
#endif
    CV_Error(CV_StsError, "The storage is not opened");
}

int icvEof( CvFileStorage* fs )
{
    if( fs->strbuf )
        return fs->strbufpos >= fs->strbufsize;
    if( fs->file )
        return feof(fs->file);
#if USE_ZLIB
    if( fs->gzfile )
        return gzeof(fs->gzfile);
#endif
    return false;
}

void icvCloseFile( CvFileStorage* fs )
{
    if( fs->file )
        fclose( fs->file );
#if USE_ZLIB
    else if( fs->gzfile )
        gzclose( fs->gzfile );
#endif
    fs->file = 0;
    fs->gzfile = 0;
    fs->strbuf = 0;
    fs->strbufpos = 0;
    fs->is_opened = false;
}

void icvRewind( CvFileStorage* fs )
{
    if( fs->file )
        rewind(fs->file);
#if USE_ZLIB
    else if( fs->gzfile )
        gzrewind(fs->gzfile);
#endif
    fs->strbufpos = 0;
}

CvGenericHash* cvCreateMap( int flags, int header_size, int elem_size, CvMemStorage* storage, int start_tab_size )
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

void icvParseError(const CvFileStorage* fs, const char* func_name,
               const char* err_msg, const char* source_file, int source_line )
{
    cv::String msg = cv::format("%s(%d): %s", fs->filename, fs->lineno, err_msg);
    cv::errorNoReturn(cv::Error::StsParseError, func_name, msg.c_str(), source_file, source_line );
}

void icvFSCreateCollection( CvFileStorage* fs, int tag, CvFileNode* collection )
{
    if( CV_NODE_IS_MAP(tag) )
    {
        if( collection->tag != CV_NODE_NONE )
        {
            assert( fs->fmt == CV_STORAGE_FORMAT_XML );
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

static char* icvFSDoResize( CvFileStorage* fs, char* ptr, int len )
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

char* icvFSResizeWriteBuffer( CvFileStorage* fs, char* ptr, int len )
{
    return ptr + len < fs->buffer_end ? ptr : icvFSDoResize( fs, ptr, len );
}

char* icvFSFlush( CvFileStorage* fs )
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
        memset( fs->buffer_start, ' ', indent );
        fs->space = indent;
    }

    ptr = fs->buffer = fs->buffer_start + fs->space;

    return ptr;
}

void icvClose( CvFileStorage* fs, cv::String* out )
{
    if( out )
        out->clear();

    if( !fs )
        CV_Error( CV_StsNullPtr, "NULL double pointer to file storage" );

    if( fs->is_opened )
    {
        if( fs->write_mode && (fs->file || fs->gzfile || fs->outbuf) )
        {
            if( fs->write_stack )
            {
                while( fs->write_stack->total > 0 )
                    cvEndWriteStruct(fs);
            }
            icvFSFlush(fs);
            if( fs->fmt == CV_STORAGE_FORMAT_XML )
                icvPuts( fs, "</opencv_storage>\n" );
            else if ( fs->fmt == CV_STORAGE_FORMAT_JSON )
                icvPuts( fs, "}\n" );
        }
    }

    icvCloseFile(fs);

    if( fs->outbuf && out )
    {
        *out = cv::String(fs->outbuf->begin(), fs->outbuf->end());
    }
}

char* icvDoubleToString( char* buf, double value )
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
            for( ; cv_isdigit(*ptr); ptr++ )
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

char* icvFloatToString( char* buf, float value )
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
            for( ; cv_isdigit(*ptr); ptr++ )
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

static void icvProcessSpecialDouble( CvFileStorage* fs, char* buf, double* value, char** endptr )
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

    union{double d; uint64 i;} v;
    v.d = 0.;
    if( toupper(buf[1]) == 'I' && toupper(buf[2]) == 'N' && toupper(buf[3]) == 'F' )
        v.i = (uint64)inf_hi << 32;
    else if( toupper(buf[1]) == 'N' && toupper(buf[2]) == 'A' && toupper(buf[3]) == 'N' )
        v.i = (uint64)-1;
    else
        CV_PARSE_ERROR( "Bad format of floating-point constant" );
    *value = v.d;

    *endptr = buf + 4;
}


double icv_strtod( CvFileStorage* fs, char* ptr, char** endptr )
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

    if( *endptr == ptr || cv_isalpha(**endptr) )
        icvProcessSpecialDouble( fs, ptr, &fval, endptr );

    return fval;
}

void switch_to_Base64_state( CvFileStorage* fs, base64::fs::State state )
{
    const char * err_unkonwn_state = "Unexpected error, unable to determine the Base64 state.";
    const char * err_unable_to_switch = "Unexpected error, unable to switch to this state.";

    /* like a finite state machine */
    switch (fs->state_of_writing_base64)
    {
    case base64::fs::Uncertain:
        switch (state)
        {
        case base64::fs::InUse:
            CV_DbgAssert( fs->base64_writer == 0 );
            fs->base64_writer = new base64::Base64Writer( fs );
            break;
        case base64::fs::Uncertain:
            break;
        case base64::fs::NotUse:
            break;
        default:
            CV_Error( CV_StsError, err_unkonwn_state );
            break;
        }
        break;
    case base64::fs::InUse:
        switch (state)
        {
        case base64::fs::InUse:
        case base64::fs::NotUse:
            CV_Error( CV_StsError, err_unable_to_switch );
            break;
        case base64::fs::Uncertain:
            delete fs->base64_writer;
            fs->base64_writer = 0;
            break;
        default:
            CV_Error( CV_StsError, err_unkonwn_state );
            break;
        }
        break;
    case base64::fs::NotUse:
        switch (state)
        {
        case base64::fs::InUse:
        case base64::fs::NotUse:
            CV_Error( CV_StsError, err_unable_to_switch );
            break;
        case base64::fs::Uncertain:
            break;
        default:
            CV_Error( CV_StsError, err_unkonwn_state );
            break;
        }
        break;
    default:
        CV_Error( CV_StsError, err_unkonwn_state );
        break;
    }

    fs->state_of_writing_base64 = state;
}

void check_if_write_struct_is_delayed( CvFileStorage* fs, bool change_type_to_base64 )
{
    if ( fs->is_write_struct_delayed )
    {
        /* save data to prevent recursive call errors */
        std::string struct_key;
        std::string type_name;
        int struct_flags = fs->delayed_struct_flags;

        if ( fs->delayed_struct_key != 0 && *fs->delayed_struct_key != '\0' )
        {
            struct_key.assign(fs->delayed_struct_key);
        }
        if ( fs->delayed_type_name != 0 && *fs->delayed_type_name != '\0' )
        {
            type_name.assign(fs->delayed_type_name);
        }

        /* reset */
        delete[] fs->delayed_struct_key;
        delete[] fs->delayed_type_name;
        fs->delayed_struct_key   = 0;
        fs->delayed_struct_flags = 0;
        fs->delayed_type_name    = 0;

        fs->is_write_struct_delayed = false;

        /* call */
        if ( change_type_to_base64 )
        {
            fs->start_write_struct( fs, struct_key.c_str(), struct_flags, "binary");
            if ( fs->state_of_writing_base64 != base64::fs::Uncertain )
                switch_to_Base64_state( fs, base64::fs::Uncertain );
            switch_to_Base64_state( fs, base64::fs::InUse );
        }
        else
        {
            fs->start_write_struct( fs, struct_key.c_str(), struct_flags, type_name.c_str());
            if ( fs->state_of_writing_base64 != base64::fs::Uncertain )
                switch_to_Base64_state( fs, base64::fs::Uncertain );
            switch_to_Base64_state( fs, base64::fs::NotUse );
        }
    }
}

void make_write_struct_delayed( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name )
{
    CV_Assert( fs->is_write_struct_delayed == false );
    CV_DbgAssert( fs->delayed_struct_key   == 0 );
    CV_DbgAssert( fs->delayed_struct_flags == 0 );
    CV_DbgAssert( fs->delayed_type_name    == 0 );

    fs->delayed_struct_flags = struct_flags;

    if ( key != 0 )
    {
        fs->delayed_struct_key = new char[strlen(key) + 1U];
        strcpy(fs->delayed_struct_key, key);
    }

    if ( type_name != 0 )
    {
        fs->delayed_type_name = new char[strlen(type_name) + 1U];
        strcpy(fs->delayed_type_name, type_name);
    }

    fs->is_write_struct_delayed = true;
}

static const char symbols[9] = "ucwsifdr";

char icvTypeSymbol(int depth)
{
    CV_Assert(depth >=0 && depth < 9);
    return symbols[depth];
}

static int icvSymbolToType(char c)
{
    const char* pos = strchr( symbols, c );
    if( !pos )
        CV_Error( CV_StsBadArg, "Invalid data type specification" );
    return static_cast<int>(pos - symbols);
}

char* icvEncodeFormat( int elem_type, char* dt )
{
    sprintf( dt, "%d%c", CV_MAT_CN(elem_type), icvTypeSymbol(CV_MAT_DEPTH(elem_type)) );
    return dt + ( dt[2] == '\0' && dt[0] == '1' );
}

int icvDecodeFormat( const char* dt, int* fmt_pairs, int max_len )
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

        if( cv_isdigit(c) )
        {
            int count = c - '0';
            if( cv_isdigit(dt[k+1]) )
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
            int depth = icvSymbolToType(c);
            if( fmt_pairs[i] == 0 )
                fmt_pairs[i] = 1;
            fmt_pairs[i+1] = depth;
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


int icvCalcElemSize( const char* dt, int initial_size )
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


int icvCalcStructSize( const char* dt, int initial_size )
{
    int size = icvCalcElemSize( dt, initial_size );
    size_t elem_max_size = 0;
    for ( const char * type = dt; *type != '\0'; type++ ) {
        switch ( *type )
        {
        case 'u': { elem_max_size = std::max( elem_max_size, sizeof(uchar ) ); break; }
        case 'c': { elem_max_size = std::max( elem_max_size, sizeof(schar ) ); break; }
        case 'w': { elem_max_size = std::max( elem_max_size, sizeof(ushort) ); break; }
        case 's': { elem_max_size = std::max( elem_max_size, sizeof(short ) ); break; }
        case 'i': { elem_max_size = std::max( elem_max_size, sizeof(int   ) ); break; }
        case 'f': { elem_max_size = std::max( elem_max_size, sizeof(float ) ); break; }
        case 'd': { elem_max_size = std::max( elem_max_size, sizeof(double) ); break; }
        default: break;
        }
    }
    size = cvAlign( size, static_cast<int>(elem_max_size) );
    return size;
}

int icvDecodeSimpleFormat( const char* dt )
{
    int elem_type = -1;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS], fmt_pair_count;

    fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
    if( fmt_pair_count != 1 || fmt_pairs[0] >= CV_CN_MAX)
        CV_Error( CV_StsError, "Too complex format for the matrix" );

    elem_type = CV_MAKETYPE( fmt_pairs[1], fmt_pairs[0] );

    return elem_type;
}

void icvWriteCollection( CvFileStorage* fs, const CvFileNode* node )
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

void icvWriteFileNode( CvFileStorage* fs, const char* name, const CvFileNode* node )
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
        cvStartWriteStruct( fs, name, CV_NODE_TYPE(node->tag) +
                (CV_NODE_SEQ_IS_SIMPLE(node->data.seq) ? CV_NODE_FLOW : 0),
                node->info ? node->info->type_name : 0 );
        icvWriteCollection( fs, node );
        cvEndWriteStruct( fs );
        break;
    case CV_NODE_NONE:
        cvStartWriteStruct( fs, name, CV_NODE_SEQ, 0 );
        cvEndWriteStruct( fs );
        break;
    default:
        CV_Error( CV_StsBadFlag, "Unknown type of file node" );
    }
}
