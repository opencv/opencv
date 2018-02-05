// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "persistence.hpp"

#define CV_YML_INDENT  3
#define CV_YML_INDENT_FLOW  1

/****************************************************************************************\
*                                       YAML Parser                                      *
\****************************************************************************************/

static char* icvYMLSkipSpaces( CvFileStorage* fs, char* ptr, int min_indent, int max_comment_indent )
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


static void icvYMLGetMultilineStringContent(CvFileStorage* fs, char* ptr, int indent, char* &beg, char* &end)
{
    ptr = icvYMLSkipSpaces(fs, ptr, 0, INT_MAX);
    beg = ptr;
    end = ptr;
    if (fs->dummy_eof)
        return ; /* end of file */

    if (ptr - fs->buffer_start != indent)
        return ; /* end of string */

    /* find end */
    while(cv_isprint(*ptr)) /* no check for base64 string */
        ++ ptr;
    if (*ptr == '\0')
        CV_PARSE_ERROR("Unexpected end of line");

    end = ptr;
}

static char* icvYMLParseBase64(CvFileStorage* fs, char* ptr, int indent, CvFileNode * node)
{
    char * beg = 0;
    char * end = 0;

    icvYMLGetMultilineStringContent(fs, ptr, indent, beg, end);
    if (beg >= end)
        return end; // CV_PARSE_ERROR("Empty Binary Data");

    /* calc (decoded) total_byte_size from header */
    std::string dt;
    {
        if (end - beg < static_cast<int>(base64::ENCODED_HEADER_SIZE))
            CV_PARSE_ERROR("Unrecognized Base64 header");

        std::vector<char> header(base64::HEADER_SIZE + 1, ' ');
        base64::base64_decode(beg, header.data(), 0U, base64::ENCODED_HEADER_SIZE);
        if ( !base64::read_base64_header(header, dt) || dt.empty() )
            CV_PARSE_ERROR("Invalid `dt` in Base64 header");

        beg += base64::ENCODED_HEADER_SIZE;
    }

    /* get all Base64 data */
    std::string base64_buffer;
    base64_buffer.reserve( PARSER_BASE64_BUFFER_SIZE );
    while( beg < end )
    {
        base64_buffer.append( beg, end );
        beg = end;
        icvYMLGetMultilineStringContent( fs, beg, indent, beg, end );
    }
    if ( base64_buffer.empty() ||
         !base64::base64_valid(base64_buffer.data(), 0U, base64_buffer.size()) )
        CV_PARSE_ERROR( "Invalid Base64 data." );

    /* buffer for decoded data(exclude header) */
    std::vector<uchar> binary_buffer( base64::base64_decode_buffer_size(base64_buffer.size()) );
    int total_byte_size = static_cast<int>(
        base64::base64_decode_buffer_size( base64_buffer.size(), base64_buffer.data(), false )
        );
    {
        base64::Base64ContextParser parser(binary_buffer.data(), binary_buffer.size() );
        const uchar * buffer_beg = reinterpret_cast<const uchar *>( base64_buffer.data() );
        const uchar * buffer_end = buffer_beg + base64_buffer.size();
        parser.read( buffer_beg, buffer_end );
        parser.flush();
    }

    /* save as CvSeq */
    int elem_size = ::icvCalcStructSize(dt.c_str(), 0);
    if (total_byte_size % elem_size != 0)
        CV_PARSE_ERROR("Byte size not match elememt size");
    int elem_cnt = total_byte_size / elem_size;

    node->tag = CV_NODE_NONE;
    int struct_flags = CV_NODE_FLOW | CV_NODE_SEQ;
    /* after icvFSCreateCollection, node->tag == struct_flags */
    icvFSCreateCollection(fs, struct_flags, node);
    base64::make_seq(binary_buffer.data(), elem_cnt, dt.c_str(), *node->data.seq);

    if (fs->dummy_eof) {
        /* end of file */
        return fs->buffer_start;
    } else {
        /* end of line */
        return end;
    }
}


static char* icvYMLParseKey( CvFileStorage* fs, char* ptr, CvFileNode* map_node, CvFileNode** value_placeholder )
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
    char buf[CV_FS_MAX_LEN + 1024] = {0};
    char* endptr = 0;
    char c = ptr[0], d = ptr[1];
    int is_parent_flow = CV_NODE_IS_FLOW(parent_flags);
    int value_type = CV_NODE_NONE;
    int len;
    bool is_binary_string = false;

    memset( node, 0, sizeof(*node) );

    if( c == '!' ) // handle explicit type specification
    {
        if( d == '!' || d == '^' )
        {
            ptr++;
            value_type |= CV_NODE_USER;
        }
        if ( d == '<') //support of full type heading from YAML 1.2
        {
            const char* yamlTypeHeading = "<tag:yaml.org,2002:";
            const size_t headingLenght = strlen(yamlTypeHeading);

            char* typeEndPtr = ++ptr;

            do d = *++typeEndPtr;
            while( cv_isprint(d) && d != ' ' && d != '>' );

            if ( d == '>' && (size_t)(typeEndPtr - ptr) > headingLenght )
            {
                if ( memcmp(ptr, yamlTypeHeading, headingLenght) == 0 )
                {
                    value_type |= CV_NODE_USER;
                    *typeEndPtr = ' ';
                    ptr += headingLenght - 1;
                }
            }
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
        else if (len == 6 && CV_NODE_IS_USER(value_type))
        {
            if( memcmp( ptr, "binary", 6 ) == 0 ) {
                value_type = CV_NODE_SEQ;
                is_binary_string = true;

                /* for ignore '|' */

                /**** operation with endptr ****/
                *endptr = d;

                do {
                    d = *++endptr;
                    if (d == '|')
                        break;
                } while (d == ' ');

                d = *++endptr;
                *endptr = '\0';
            }
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
            if (value_type == CV_NODE_STRING && c != '\'' && c != '\"')
                goto force_string;
            if( value_type == CV_NODE_INT )
                goto force_int;
            if( value_type == CV_NODE_REAL )
                goto force_real;
        }
    }

    if (is_binary_string)
    {
        /* for base64 string */
        int indent = static_cast<int>(ptr - fs->buffer_start);
        ptr = icvYMLParseBase64(fs, ptr, indent, node);
    }
    else if( cv_isdigit(c) ||
        ((c == '-' || c == '+') && (cv_isdigit(d) || d == '.')) ||
        (c == '.' && cv_isalnum(d))) // a number
    {
        double fval;
        int ival;
        endptr = ptr + (c == '-' || c == '+');
        while( cv_isdigit(*endptr) )
            endptr++;
        if( *endptr == '.' || *endptr == 'e' )
        {
force_real:
            fval = icv_strtod( fs, ptr, &endptr );
            /*if( endptr == ptr || cv_isalpha(*endptr) )
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
                if( cv_isalnum(c) || (c != '\'' && cv_isprint(c)))
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
                if( cv_isalnum(c) || (c != '\\' && c != '\"' && cv_isprint(c)))
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
                    else if( d == 'x' || (cv_isdigit(d) && d < '8') )
                    {
                        int val, is_hex = d == 'x';
                        c = ptr[3];
                        ptr[3] = '\0';
                        val = (int)strtol( ptr + is_hex, &endptr, is_hex ? 8 : 16 );
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
        bool is_simple = true;

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
            CV_Assert(elem);
            ptr = icvYMLParseValue( fs, ptr, elem, struct_flags, new_min_indent );
            if( CV_NODE_IS_MAP(struct_flags) )
                elem->tag |= CV_NODE_NAMED;
            is_simple = is_simple && !CV_NODE_IS_COLLECTION(elem->tag);
        }
        node->data.seq->flags |= is_simple ? CV_NODE_SEQ_SIMPLE : 0;
    }
    else
    {
        int indent, struct_flags;
        bool is_simple;

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
        is_simple = true;

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
            CV_Assert(elem);
            ptr = icvYMLSkipSpaces( fs, ptr, indent + 1, INT_MAX );
            ptr = icvYMLParseValue( fs, ptr, elem, struct_flags, indent + 1 );
            if( CV_NODE_IS_MAP(struct_flags) )
                elem->tag |= CV_NODE_NAMED;
            is_simple = is_simple && !CV_NODE_IS_COLLECTION(elem->tag);

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


void icvYMLParse( CvFileStorage* fs )
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
                if( memcmp( ptr, "%YAML", 5 ) == 0 &&
                    memcmp( ptr, "%YAML:1.", 8 ) != 0 &&
                    memcmp( ptr, "%YAML 1.", 8 ) != 0)
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
            else if( cv_isalnum(*ptr) || *ptr=='_')
            {
                if( !is_first )
                    CV_PARSE_ERROR( "The YAML streams must start with '---', except the first one" );
                break;
            }
            else if( fs->dummy_eof )
                break;
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

void icvYMLWrite( CvFileStorage* fs, const char* key, const char* data )
{
    check_if_write_struct_is_delayed( fs );
    if ( fs->state_of_writing_base64 == base64::fs::Uncertain )
    {
        switch_to_Base64_state( fs, base64::fs::NotUse );
    }
    else if ( fs->state_of_writing_base64 == base64::fs::InUse )
    {
        CV_Error( CV_StsError, "At present, output Base64 data only." );
    }

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
        if( !cv_isalpha(key[0]) && key[0] != '_' )
            CV_Error( CV_StsBadArg, "Key must start with a letter or _" );

        ptr = icvFSResizeWriteBuffer( fs, ptr, keylen );

        for( i = 0; i < keylen; i++ )
        {
            char c = key[i];

            ptr[i] = c;
            if( !cv_isalnum(c) && c != '-' && c != '_' && c != ' ' )
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


void icvYMLStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name)
{
    int parent_flags;
    char buf[CV_FS_MAX_LEN + 1024];
    const char* data = 0;

    if ( type_name && *type_name == '\0' )
        type_name = 0;

    struct_flags = (struct_flags & (CV_NODE_TYPE_MASK|CV_NODE_FLOW)) | CV_NODE_EMPTY;
    if( !CV_NODE_IS_COLLECTION(struct_flags))
        CV_Error( CV_StsBadArg,
        "Some collection type - CV_NODE_SEQ or CV_NODE_MAP, must be specified" );

    if (type_name && memcmp(type_name, "binary", 6) == 0)
    {
        /* reset struct flag. in order not to print ']' */
        struct_flags = CV_NODE_SEQ;
        sprintf(buf, "!!binary |");
        data = buf;
    }
    else if( CV_NODE_IS_FLOW(struct_flags))
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


void icvYMLEndWriteStruct( CvFileStorage* fs )
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


void icvYMLStartNextStream( CvFileStorage* fs )
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


void icvYMLWriteInt( CvFileStorage* fs, const char* key, int value )
{
    char buf[128];
    icvYMLWrite( fs, key, icv_itoa( value, buf, 10 ));
}


void icvYMLWriteReal( CvFileStorage* fs, const char* key, double value )
{
    char buf[128];
    icvYMLWrite( fs, key, icvDoubleToString( buf, value ));
}


void icvYMLWriteString( CvFileStorage* fs, const char* key, const char* str, int quote)
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
        int need_quote = quote || len == 0 || str[0] == ' ';
        data = buf;
        *data++ = '\"';
        for( i = 0; i < len; i++ )
        {
            char c = str[i];

            if( !need_quote && !cv_isalnum(c) && c != '_' && c != ' ' && c != '-' &&
                c != '(' && c != ')' && c != '/' && c != '+' && c != ';' )
                need_quote = 1;

            if( !cv_isalnum(c) && (!cv_isprint(c) || c == '\\' || c == '\'' || c == '\"') )
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
        if( !need_quote && (cv_isdigit(str[0]) ||
            str[0] == '+' || str[0] == '-' || str[0] == '.' ))
            need_quote = 1;

        if( need_quote )
            *data++ = '\"';
        *data++ = '\0';
        data = buf + !need_quote;
    }

    icvYMLWrite( fs, key, data );
}


void icvYMLWriteComment( CvFileStorage* fs, const char* comment, int eol_comment )
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
