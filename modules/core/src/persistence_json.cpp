// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "persistence.hpp"

/****************************************************************************************\
*                                       JSON Parser                                      *
\****************************************************************************************/

static char*
icvJSONSkipSpaces( CvFileStorage* fs, char* ptr )
{
    bool is_eof = false;
    bool is_completed = false;

    while ( is_eof == false && is_completed == false )
    {
        switch ( *ptr )
        {
        /* comment */
        case '/' : {
            ptr++;
            if ( *ptr == '\0' )
            {
                ptr = icvGets( fs, fs->buffer_start, static_cast<int>(fs->buffer_end - fs->buffer_start) );
                if ( !ptr ) { is_eof = true; break; }
            }

            if ( *ptr == '/' )
            {
                while ( *ptr != '\n' && *ptr != '\r' )
                {
                    if ( *ptr == '\0' )
                    {
                        ptr = icvGets( fs, fs->buffer_start, static_cast<int>(fs->buffer_end - fs->buffer_start) );
                        if ( !ptr ) { is_eof = true; break; }
                    }
                    else
                    {
                        ptr++;
                    }
                }
            }
            else if ( *ptr == '*' )
            {
                ptr++;
                for (;;)
                {
                    if ( *ptr == '\0' )
                    {
                        ptr = icvGets( fs, fs->buffer_start, static_cast<int>(fs->buffer_end - fs->buffer_start) );
                        if ( !ptr ) { is_eof = true; break; }
                    }
                    else if ( *ptr == '*' )
                    {
                        ptr++;
                        if ( *ptr == '\0' )
                        {
                            ptr = icvGets( fs, fs->buffer_start, static_cast<int>(fs->buffer_end - fs->buffer_start) );
                            if ( !ptr ) { is_eof = true; break; }
                        }
                        if ( *ptr == '/' )
                        {
                            ptr++;
                            break;
                        }
                    }
                    else
                    {
                        ptr++;
                    }
                }
            }
            else
            {
                CV_PARSE_ERROR( "Not supported escape character" );
            }
        } break;
        /* whitespace */
        case '\t':
        case ' ' : {
            ptr++;
        } break;
        /* newline || end mark */
        case '\0':
        case '\n':
        case '\r': {
            ptr = icvGets( fs, fs->buffer_start, static_cast<int>(fs->buffer_end - fs->buffer_start) );
            if ( !ptr ) { is_eof = true; break; }
        } break;
        /* other character */
        default: {
            if ( !cv_isprint(*ptr) )
                CV_PARSE_ERROR( "Invalid character in the stream" );
            is_completed = true;
        } break;
        }
    }

    if ( is_eof )
    {
        ptr = fs->buffer_start;
        *ptr = '\0';
        fs->dummy_eof = 1;
    }
    else if ( !is_completed )
    {
        /* should not be executed */
        ptr = 0;
        fs->dummy_eof = 1;
        CV_PARSE_ERROR( "Abort at parse time" );
    }
    return ptr;
}


static char* icvJSONParseKey( CvFileStorage* fs, char* ptr, CvFileNode* map, CvFileNode** value_placeholder )
{
    if( *ptr != '"' )
        CV_PARSE_ERROR( "Key must start with \'\"\'" );

    char * beg = ptr + 1;

    do {
        ++ptr;
        CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG();
    } while( cv_isprint(*ptr) && *ptr != '"' );

    if( *ptr != '"' )
        CV_PARSE_ERROR( "Key must end with \'\"\'" );

    const char * end = ptr;
    ptr++;
    ptr = icvJSONSkipSpaces( fs, ptr );
    if ( ptr == 0 || fs->dummy_eof )
        return 0;

    if( *ptr != ':' )
        CV_PARSE_ERROR( "Missing \':\' between key and value" );

    /* [beg, end) */
    if( end <= beg )
        CV_PARSE_ERROR( "Key is empty" );

    if ( end - beg == 7u && memcmp(beg, "type_id", 7u) == 0 )
    {
        *value_placeholder = 0;
    }
    else
    {
        CvStringHashNode* str_hash_node = cvGetHashedKey( fs, beg, static_cast<int>(end - beg), 1 );
        *value_placeholder = cvGetFileNode( fs, map, str_hash_node, 1 );
    }

    ptr++;
    return ptr;
}

static char* icvJSONParseValue( CvFileStorage* fs, char* ptr, CvFileNode* node )
{
    ptr = icvJSONSkipSpaces( fs, ptr );
    if ( ptr == 0 || fs->dummy_eof )
        CV_PARSE_ERROR( "Unexpected End-Of-File" );

    memset( node, 0, sizeof(*node) );

    if ( *ptr == '"' )
    {   /* must be string or Base64 string */
        ptr++;
        char * beg = ptr;
        size_t len = 0u;
        for ( ; (cv_isalnum(*ptr) || *ptr == '$' ) && len <= 9u; ptr++ )
            len++;

        if ( len >= 8u && memcmp( beg, "$base64$", 8u ) == 0 )
        {   /**************** Base64 string ****************/
            ptr = beg += 8;

            std::string base64_buffer;
            base64_buffer.reserve( PARSER_BASE64_BUFFER_SIZE );

            bool is_matching = false;
            while ( !is_matching )
            {
                switch ( *ptr )
                {
                case '\0':
                {
                    base64_buffer.append( beg, ptr );

                    ptr = icvGets( fs, fs->buffer_start, static_cast<int>(fs->buffer_end - fs->buffer_start) );
                    if ( !ptr )
                        CV_PARSE_ERROR( "'\"' - right-quote of string is missing" );

                    beg = ptr;
                    break;
                }
                case '\"':
                {
                    base64_buffer.append( beg, ptr );
                    beg = ptr;
                    is_matching = true;
                    break;
                }
                case '\n':
                case '\r':
                {
                    CV_PARSE_ERROR( "'\"' - right-quote of string is missing" );
                    break;
                }
                default:
                {
                    ptr++;
                    break;
                }
                }
            }

            if ( *ptr != '\"' )
                CV_PARSE_ERROR( "'\"' - right-quote of string is missing" );
            else
                ptr++;

            if ( base64_buffer.size() >= base64::ENCODED_HEADER_SIZE )
            {
                const char * base64_beg = base64_buffer.data();
                const char * base64_end = base64_beg + base64_buffer.size();

                /* get dt from header */
                std::string dt;
                {
                    std::vector<char> header(base64::HEADER_SIZE + 1, ' ');
                    base64::base64_decode(base64_beg, header.data(), 0U, base64::ENCODED_HEADER_SIZE);
                    if ( !base64::read_base64_header(header, dt) || dt.empty() )
                        CV_PARSE_ERROR("Invalid `dt` in Base64 header");
                }


                if ( base64_buffer.size() > base64::ENCODED_HEADER_SIZE )
                {
                    /* set base64_beg to beginning of base64 data */
                    base64_beg = &base64_buffer.at( base64::ENCODED_HEADER_SIZE );
                    if ( !base64::base64_valid( base64_beg, 0U, base64_end - base64_beg ) )
                        CV_PARSE_ERROR( "Invalid Base64 data." );

                    /* buffer for decoded data(exclude header) */
                    std::vector<uchar> binary_buffer( base64::base64_decode_buffer_size(base64_end - base64_beg) );
                    int total_byte_size = static_cast<int>(
                        base64::base64_decode_buffer_size( base64_end - base64_beg, base64_beg, false )
                        );
                    {
                        base64::Base64ContextParser parser(binary_buffer.data(), binary_buffer.size() );
                        const uchar * binary_beg = reinterpret_cast<const uchar *>( base64_beg );
                        const uchar * binary_end = binary_beg + (base64_end - base64_beg);
                        parser.read( binary_beg, binary_end );
                        parser.flush();
                    }

                    /* after icvFSCreateCollection, node->tag == struct_flags */
                    icvFSCreateCollection(fs, CV_NODE_FLOW | CV_NODE_SEQ, node);
                    base64::make_seq(fs, binary_buffer.data(), total_byte_size, dt.c_str(), *node->data.seq);
                }
                else
                {
                    /* empty */
                    icvFSCreateCollection(fs, CV_NODE_FLOW | CV_NODE_SEQ, node);
                }
            }
            else if ( base64_buffer.empty() )
            {
                /* empty */
                icvFSCreateCollection(fs, CV_NODE_FLOW | CV_NODE_SEQ, node);
            }
            else
            {
                CV_PARSE_ERROR("Unrecognized Base64 header");
            }
        }
        else
        {   /**************** normal string ****************/
            std::string string_buffer;
            string_buffer.reserve( PARSER_BASE64_BUFFER_SIZE );

            ptr = beg;
            bool is_matching = false;
            while ( !is_matching )
            {
                switch ( *ptr )
                {
                case '\\':
                {
                    string_buffer.append( beg, ptr );
                    ptr++;
                    switch ( *ptr )
                    {
                    case '\\':
                    case '\"':
                    case '\'': { string_buffer.append( 1u, *ptr ); break; }
                    case 'n' : { string_buffer.append( 1u, '\n' ); break; }
                    case 'r' : { string_buffer.append( 1u, '\r' ); break; }
                    case 't' : { string_buffer.append( 1u, '\t' ); break; }
                    case 'b' : { string_buffer.append( 1u, '\b' ); break; }
                    case 'f' : { string_buffer.append( 1u, '\f' ); break; }
                    case 'u' : { CV_PARSE_ERROR( "'\\uXXXX' currently not supported" ); }
                    default  : { CV_PARSE_ERROR( "Invalid escape character" ); }
                        break;
                    }
                    ptr++;
                    beg = ptr;
                    break;
                }
                case '\0':
                {
                    string_buffer.append( beg, ptr );

                    ptr = icvGets( fs, fs->buffer_start, static_cast<int>(fs->buffer_end - fs->buffer_start) );
                    if ( !ptr )
                        CV_PARSE_ERROR( "'\"' - right-quote of string is missing" );

                    beg = ptr;
                    break;
                }
                case '\"':
                {
                    string_buffer.append( beg, ptr );
                    beg = ptr;
                    is_matching = true;
                    break;
                }
                case '\n':
                case '\r':
                {
                    CV_PARSE_ERROR( "'\"' - right-quote of string is missing" );
                    break;
                }
                default:
                {
                    ptr++;
                    break;
                }
                }
            }

            if ( *ptr != '\"' )
                CV_PARSE_ERROR( "'\"' - right-quote of string is missing" );
            else
                ptr++;

            node->data.str = cvMemStorageAllocString
            (
                fs->memstorage,
                string_buffer.c_str(),
                static_cast<int>(string_buffer.size())
            );
            node->tag = CV_NODE_STRING;
        }
    }
    else if ( cv_isdigit(*ptr) || *ptr == '-' || *ptr == '+' || *ptr == '.' )
    {    /**************** number ****************/
        char * beg = ptr;
        if ( *ptr == '+' || *ptr == '-' )
        {
            ptr++;
            CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG();
        }
        while( cv_isdigit(*ptr) )
        {
            ptr++;
            CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG();
        }
        if (*ptr == '.' || *ptr == 'e')
        {
            node->data.f = icv_strtod( fs, beg, &ptr );
            CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG();
            node->tag = CV_NODE_REAL;
        }
        else
        {
            node->data.i = static_cast<int>(strtol( beg, &ptr, 0 ));
            CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG();
            node->tag = CV_NODE_INT;
        }

        if ( beg >= ptr )
            CV_PARSE_ERROR( "Invalid numeric value (inconsistent explicit type specification?)" );
    }
    else
    {    /**************** other data ****************/
        const char * beg = ptr;
        size_t len = 0u;
        for ( ; cv_isalpha(*ptr) && len <= 6u; )
        {
            len++;
            ptr++;
            CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG();
        }

        if ( len >= 4u && memcmp( beg, "null", 4u ) == 0 )
        {
            CV_PARSE_ERROR( "Value 'null' is not supported by this parser" );
        }
        else if ( len >= 4u && memcmp( beg, "true", 4u ) == 0 )
        {
            node->data.i = 1;
            node->tag = CV_NODE_INT;
        }
        else if ( len >= 5u && memcmp( beg, "false", 5u ) == 0 )
        {
            node->data.i = 0;
            node->tag = CV_NODE_INT;
        }
        else
        {
            CV_PARSE_ERROR( "Unrecognized value" );
        }
    }

    return ptr;
}

static char* icvJSONParseSeq( CvFileStorage* fs, char* ptr, CvFileNode* node );
static char* icvJSONParseMap( CvFileStorage* fs, char* ptr, CvFileNode* node );

static char* icvJSONParseSeq( CvFileStorage* fs, char* ptr, CvFileNode* node )
{
    if (!ptr)
        CV_PARSE_ERROR( "ptr is NULL" );

    if ( *ptr != '[' )
        CV_PARSE_ERROR( "'[' - left-brace of seq is missing" );
    else
        ptr++;

    memset( node, 0, sizeof(*node) );
    icvFSCreateCollection( fs, CV_NODE_SEQ, node );

    for (;;)
    {
        ptr = icvJSONSkipSpaces( fs, ptr );
        if ( ptr == 0 || fs->dummy_eof )
            break;

        if ( *ptr != ']' )
        {
            CvFileNode* child = (CvFileNode*)cvSeqPush( node->data.seq, 0 );

            if ( *ptr == '[' )
                ptr = icvJSONParseSeq( fs, ptr, child );
            else if ( *ptr == '{' )
                ptr = icvJSONParseMap( fs, ptr, child );
            else
                ptr = icvJSONParseValue( fs, ptr, child );
        }

        ptr = icvJSONSkipSpaces( fs, ptr );
        if ( ptr == 0 || fs->dummy_eof )
            break;

        if ( *ptr == ',' )
            ptr++;
        else if ( *ptr == ']' )
            break;
        else
            CV_PARSE_ERROR( "Unexpected character" );
    }

    if (!ptr)
        CV_PARSE_ERROR("ptr is NULL");

    if ( *ptr != ']' )
        CV_PARSE_ERROR( "']' - right-brace of seq is missing" );
    else
        ptr++;

    return ptr;
}

static char* icvJSONParseMap( CvFileStorage* fs, char* ptr, CvFileNode* node )
{
    if (!ptr)
        CV_PARSE_ERROR("ptr is NULL");

    if ( *ptr != '{' )
        CV_PARSE_ERROR( "'{' - left-brace of map is missing" );
    else
        ptr++;

    memset( node, 0, sizeof(*node) );
    icvFSCreateCollection( fs, CV_NODE_MAP, node );

    for ( ;; )
    {
        ptr = icvJSONSkipSpaces( fs, ptr );
        if ( ptr == 0 || fs->dummy_eof )
            break;

        if ( *ptr == '"' )
        {
            CvFileNode* child = 0;
            ptr = icvJSONParseKey( fs, ptr, node, &child );
            if ( ptr == 0 || fs->dummy_eof )
                break;
            ptr = icvJSONSkipSpaces( fs, ptr );
            if ( ptr == 0 || fs->dummy_eof )
                break;

            if ( child == 0 )
            {   /* type_id */
                CvFileNode tmp;
                ptr = icvJSONParseValue( fs, ptr, &tmp );
                if ( CV_NODE_IS_STRING(tmp.tag) )
                {
                    node->info = cvFindType( tmp.data.str.ptr );
                    if ( node->info )
                        node->tag |= CV_NODE_USER;
                    // delete tmp.data.str
                }
                else
                {
                    CV_PARSE_ERROR( "\"type_id\" should be of type string" );
                }
            }
            else
            {   /* normal */
                if ( *ptr == '[' )
                    ptr = icvJSONParseSeq( fs, ptr, child );
                else if ( *ptr == '{' )
                    ptr = icvJSONParseMap( fs, ptr, child );
                else
                    ptr = icvJSONParseValue( fs, ptr, child );
                child->tag |= CV_NODE_NAMED;
            }
        }

        ptr = icvJSONSkipSpaces( fs, ptr );
        if ( ptr == 0 || fs->dummy_eof )
            break;

        if ( *ptr == ',' )
            ptr++;
        else if ( *ptr == '}' )
            break;
        else
            CV_PARSE_ERROR( "Unexpected character" );
    }

    if (!ptr)
        CV_PARSE_ERROR("ptr is NULL");

    if ( *ptr != '}' )
        CV_PARSE_ERROR( "'}' - right-brace of map is missing" );
    else
        ptr++;

    return ptr;
}


void icvJSONParse( CvFileStorage* fs )
{
    char* ptr = fs->buffer_start;
    ptr = icvJSONSkipSpaces( fs, ptr );
    if ( ptr == 0 || fs->dummy_eof )
        return;

    if ( *ptr == '{' )
    {
        CvFileNode* root_node = (CvFileNode*)cvSeqPush( fs->roots, 0 );
        icvJSONParseMap( fs, ptr, root_node );
    }
    else if ( *ptr == '[' )
    {
        CvFileNode* root_node = (CvFileNode*)cvSeqPush( fs->roots, 0 );
        icvJSONParseSeq( fs, ptr, root_node );
    }
    else
    {
        CV_PARSE_ERROR( "left-brace of top level is missing" );
    }

    if ( fs->dummy_eof != 0 )
        CV_PARSE_ERROR( "Unexpected End-Of-File" );
}


/****************************************************************************************\
*                                       JSON Emitter                                     *
\****************************************************************************************/

void icvJSONWrite( CvFileStorage* fs, const char* key, const char* data )
{
    /* check write_struct */

    check_if_write_struct_is_delayed( fs );
    if ( fs->state_of_writing_base64 == base64::fs::Uncertain )
    {
        switch_to_Base64_state( fs, base64::fs::NotUse );
    }
    else if ( fs->state_of_writing_base64 == base64::fs::InUse )
    {
        CV_Error( CV_StsError, "At present, output Base64 data only." );
    }

    /* check parameters */

    size_t key_len = 0u;
    if( key && *key == '\0' )
        key = 0;
    if ( key )
    {
        key_len = strlen(key);
        if ( key_len == 0u )
            CV_Error( CV_StsBadArg, "The key is an empty" );
        else if ( static_cast<int>(key_len) > CV_FS_MAX_LEN )
            CV_Error( CV_StsBadArg, "The key is too long" );
    }

    size_t data_len = 0u;
    if ( data )
        data_len = strlen(data);

    int struct_flags = fs->struct_flags;
    if( CV_NODE_IS_COLLECTION(struct_flags) )
    {
        if ( (CV_NODE_IS_MAP(struct_flags) ^ (key != 0)) )
            CV_Error( CV_StsBadArg, "An attempt to add element without a key to a map, "
                                    "or add element with key to sequence" );
    } else {
        fs->is_first = 0;
        struct_flags = CV_NODE_EMPTY | (key ? CV_NODE_MAP : CV_NODE_SEQ);
    }

    /* start to write */

    char* ptr = 0;

    if( CV_NODE_IS_FLOW(struct_flags) )
    {
        int new_offset;
        ptr = fs->buffer;
        if( !CV_NODE_IS_EMPTY(struct_flags) )
            *ptr++ = ',';
        new_offset = static_cast<int>(ptr - fs->buffer_start + key_len + data_len);
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
        if ( !CV_NODE_IS_EMPTY(struct_flags) )
        {
            ptr = fs->buffer;
            *ptr++ = ',';
            *ptr++ = '\n';
            *ptr++ = '\0';
            ::icvPuts( fs, fs->buffer_start );
            fs->buffer = fs->buffer_start;
        }
        ptr = icvFSFlush(fs);
    }

    if( key )
    {
        if( !cv_isalpha(key[0]) && key[0] != '_' )
            CV_Error( CV_StsBadArg, "Key must start with a letter or _" );

        ptr = icvFSResizeWriteBuffer( fs, ptr, static_cast<int>(key_len) );
        *ptr++ = '\"';

        for( size_t i = 0u; i < key_len; i++ )
        {
            char c = key[i];

            ptr[i] = c;
            if( !cv_isalnum(c) && c != '-' && c != '_' && c != ' ' )
                CV_Error( CV_StsBadArg, "Key names may only contain alphanumeric characters [a-zA-Z0-9], '-', '_' and ' '" );
        }

        ptr += key_len;
        *ptr++ = '\"';
        *ptr++ = ':';
        *ptr++ = ' ';
    }

    if( data )
    {
        ptr = icvFSResizeWriteBuffer( fs, ptr, static_cast<int>(data_len) );
        memcpy( ptr, data, data_len );
        ptr += data_len;
    }

    fs->buffer = ptr;
    fs->struct_flags = struct_flags & ~CV_NODE_EMPTY;
}


void icvJSONStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name)
{
    int parent_flags;
    char data[CV_FS_MAX_LEN + 1024];

    struct_flags = (struct_flags & (CV_NODE_TYPE_MASK|CV_NODE_FLOW)) | CV_NODE_EMPTY;
    if( !CV_NODE_IS_COLLECTION(struct_flags))
        CV_Error( CV_StsBadArg,
        "Some collection type - CV_NODE_SEQ or CV_NODE_MAP, must be specified" );

    if ( type_name && *type_name == '\0' )
        type_name = 0;

    bool has_type_id = false;
    bool is_real_collection = true;
    if (type_name && memcmp(type_name, "binary", 6) == 0)
    {
        struct_flags = CV_NODE_STR;
        data[0] = '\0';
        is_real_collection = false;
    }
    else if( type_name )
    {
        has_type_id = true;
    }

    if ( is_real_collection )
    {
        char c = CV_NODE_IS_MAP(struct_flags) ? '{' : '[';
        data[0] = c;
        data[1] = '\0';
    }

    icvJSONWrite( fs, key, data );

    parent_flags = fs->struct_flags;
    cvSeqPush( fs->write_stack, &parent_flags );
    fs->struct_flags = struct_flags;
    fs->struct_indent += 4;

    if ( has_type_id )
        fs->write_string( fs, "type_id", type_name, 1 );
}


void icvJSONEndWriteStruct( CvFileStorage* fs )
{
    if( fs->write_stack->total == 0 )
        CV_Error( CV_StsError, "EndWriteStruct w/o matching StartWriteStruct" );

    int parent_flags = 0;
    int struct_flags = fs->struct_flags;
    cvSeqPop( fs->write_stack, &parent_flags );
    fs->struct_indent -= 4;
    fs->struct_flags = parent_flags & ~CV_NODE_EMPTY;
    assert( fs->struct_indent >= 0 );

    if ( CV_NODE_IS_COLLECTION(struct_flags) )
    {
        if ( !CV_NODE_IS_FLOW(struct_flags) )
        {
            if ( fs->buffer <= fs->buffer_start + fs->space )
            {
                /* some bad code for base64_writer... */
                *fs->buffer++ = '\n';
                *fs->buffer++ = '\0';
                icvPuts( fs, fs->buffer_start );
                fs->buffer = fs->buffer_start;
            }
            icvFSFlush(fs);
        }

        char* ptr = fs->buffer;
        if( ptr > fs->buffer_start + fs->struct_indent && !CV_NODE_IS_EMPTY(struct_flags) )
            *ptr++ = ' ';
        *ptr++ = CV_NODE_IS_MAP(struct_flags) ? '}' : ']';
        fs->buffer = ptr;
    }
}


void icvJSONStartNextStream( CvFileStorage* fs )
{
    if( !fs->is_first )
    {
        while( fs->write_stack->total > 0 )
            icvJSONEndWriteStruct(fs);

        fs->struct_indent = 4;
        icvFSFlush(fs);
        fs->buffer = fs->buffer_start;
    }
}


void icvJSONWriteInt( CvFileStorage* fs, const char* key, int value )
{
    char buf[128];
    icvJSONWrite( fs, key, icv_itoa( value, buf, 10 ));
}


void icvJSONWriteReal( CvFileStorage* fs, const char* key, double value )
{
    char buf[128];
    size_t len = strlen( icvDoubleToString( buf, value ) );
    if( len > 0 && buf[len-1] == '.' )
    {
        // append zero if string ends with decimal place to match JSON standard
        buf[len] = '0';
        buf[len+1] = '\0';
    }
    icvJSONWrite( fs, key, buf );
}


void icvJSONWriteString( CvFileStorage* fs, const char* key, const char* str, int quote)
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
        int need_quote = 1;
        data = buf;
        *data++ = '\"';
        for( i = 0; i < len; i++ )
        {
            char c = str[i];

            switch ( c )
            {
            case '\\':
            case '\"':
            case '\'': { *data++ = '\\'; *data++ = c;   break; }
            case '\n': { *data++ = '\\'; *data++ = 'n'; break; }
            case '\r': { *data++ = '\\'; *data++ = 'r'; break; }
            case '\t': { *data++ = '\\'; *data++ = 't'; break; }
            case '\b': { *data++ = '\\'; *data++ = 'b'; break; }
            case '\f': { *data++ = '\\'; *data++ = 'f'; break; }
            default  : { *data++ = c; }
                break;
            }
        }

        *data++ = '\"';
        *data++ = '\0';
        data = buf + !need_quote;
    }

    icvJSONWrite( fs, key, data );
}


void icvJSONWriteComment( CvFileStorage* fs, const char* comment, int eol_comment )
{
    if( !comment )
        CV_Error( CV_StsNullPtr, "Null comment" );

    int         len = static_cast<int>(strlen(comment));
    char*       ptr = fs->buffer;
    const char* eol = strchr(comment, '\n');
    bool  multiline = eol != 0;

    if( !eol_comment || multiline || fs->buffer_end - ptr < len || ptr == fs->buffer_start )
        ptr = icvFSFlush( fs );
    else
        *ptr++ = ' ';

    while( comment )
    {
        *ptr++ = '/';
        *ptr++ = '/';
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
