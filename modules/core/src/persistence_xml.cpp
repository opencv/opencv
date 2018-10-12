// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "persistence.hpp"

#define CV_XML_INDENT  2
#define CV_XML_INSIDE_COMMENT 1
#define CV_XML_INSIDE_TAG 2
#define CV_XML_INSIDE_DIRECTIVE 3
#define CV_XML_OPENING_TAG 1
#define CV_XML_CLOSING_TAG 2
#define CV_XML_EMPTY_TAG 3
#define CV_XML_HEADER_TAG 4
#define CV_XML_DIRECTIVE_TAG 5

/****************************************************************************************\
*                                       XML Parser                                       *
\****************************************************************************************/

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
                ptr = fs->buffer_start;  // FIXIT Why do we need this hack? What is about other parsers JSON/YAML?
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
            fs->lineno++;  // FIXIT doesn't really work with long lines. It must be counted via '\n' or '\r' symbols, not the number of icvGets() calls.
        }
    }
    return ptr;
}


static void icvXMLGetMultilineStringContent(CvFileStorage* fs,
    char* ptr, char* &beg, char* &end)
{
    ptr = icvXMLSkipSpaces(fs, ptr, CV_XML_INSIDE_TAG);
    beg = ptr;
    end = ptr;
    if ( fs->dummy_eof )
        return ; /* end of file */

    if ( *beg == '<' )
        return; /* end of string */

    /* find end */
    while( cv_isprint(*ptr) ) /* no check for base64 string */
        ++ ptr;
    if ( *ptr == '\0' )
        CV_PARSE_ERROR( "Unexpected end of line" );

    end = ptr;
}


static char* icvXMLParseBase64(CvFileStorage* fs, char* ptr, CvFileNode * node)
{
    char * beg = 0;
    char * end = 0;

    icvXMLGetMultilineStringContent(fs, ptr, beg, end);
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
    std::string base64_buffer; // not an efficient way.
    base64_buffer.reserve( PARSER_BASE64_BUFFER_SIZE );
    while( beg < end )
    {
        base64_buffer.append( beg, end );
        beg = end;
        icvXMLGetMultilineStringContent( fs, beg, beg, end );
    }
    if ( base64_buffer.empty() ||
         !base64::base64_valid(base64_buffer.data(), 0U, base64_buffer.size()) )
        CV_PARSE_ERROR( "Invalid Base64 data." );

    /* alloc buffer for all decoded data(include header) */
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
        CV_PARSE_ERROR("data size not matches elememt size");
    int elem_cnt = total_byte_size / elem_size;

    node->tag = CV_NODE_NONE;
    int struct_flags = CV_NODE_SEQ;
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


static char*
icvXMLParseTag( CvFileStorage* fs, char* ptr, CvStringHashNode** _tag,
                CvAttrList** _list, int* _tag_type );

static char*
icvXMLParseValue( CvFileStorage* fs, char* ptr, CvFileNode* node,
                  int value_type CV_DEFAULT(CV_NODE_NONE))
{
    CvFileNode *elem = node;
    bool have_space = true, is_simple = true;
    int is_user_type = CV_NODE_IS_USER(value_type);
    memset( node, 0, sizeof(*node) );

    value_type = CV_NODE_TYPE(value_type);

    for(;;)
    {
        char c = *ptr, d;
        char* endptr;

        if( cv_isspace(c) || c == '\0' || (c == '<' && ptr[1] == '!' && ptr[2] == '-') )  // FIXIT ptr[1], ptr[2] - out of bounds read without check or data fetch (#11061)
        {
            ptr = icvXMLSkipSpaces( fs, ptr, 0 );
            have_space = true;
            c = *ptr;
        }

        d = ptr[1];  // FIXIT ptr[1] - out of bounds read without check or data fetch (#11061)

        if( c =='<' || c == '\0' )
        {
            CvStringHashNode *key = 0, *key2 = 0;
            CvAttrList* list = 0;
            CvTypeInfo* info = 0;
            int tag_type = 0;
            int is_noname = 0;
            const char* type_name = 0;
            int elem_type = CV_NODE_NONE;

            if( d == '/' || c == '\0' )
                break;

            ptr = icvXMLParseTag( fs, ptr, &key, &list, &tag_type );

            if( tag_type == CV_XML_DIRECTIVE_TAG )
                CV_PARSE_ERROR( "Directive tags are not allowed here" );
            if( tag_type == CV_XML_EMPTY_TAG )
                CV_PARSE_ERROR( "Empty tags are not supported" );

            CV_Assert(tag_type == CV_XML_OPENING_TAG);

            /* for base64 string */
            bool is_binary_string = false;

            type_name = list ? cvAttrValue( list, "type_id" ) : 0;
            if( type_name )
            {
                if( strcmp( type_name, "str" ) == 0 )
                    elem_type = CV_NODE_STRING;
                else if( strcmp( type_name, "map" ) == 0 )
                    elem_type = CV_NODE_MAP;
                else if( strcmp( type_name, "seq" ) == 0 )
                    elem_type = CV_NODE_SEQ;
                else if (strcmp(type_name, "binary") == 0)
                {
                    elem_type = CV_NODE_NONE;
                    is_binary_string = true;
                }
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
            CV_Assert(elem);
            if (!is_binary_string)
                ptr = icvXMLParseValue( fs, ptr, elem, elem_type);
            else {
                /* for base64 string */
                ptr = icvXMLParseBase64( fs, ptr, elem);
                ptr = icvXMLSkipSpaces( fs, ptr, 0 );
            }

            if( !is_noname )
                elem->tag |= CV_NODE_NAMED;
            is_simple = is_simple && !CV_NODE_IS_COLLECTION(elem->tag);
            elem->info = info;
            ptr = icvXMLParseTag( fs, ptr, &key2, &list, &tag_type );
            if( tag_type != CV_XML_CLOSING_TAG || key2 != key )
                CV_PARSE_ERROR( "Mismatched closing tag" );
            have_space = true;
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
                (cv_isdigit(c) || ((c == '-' || c == '+') &&
                (cv_isdigit(d) || d == '.')) || (c == '.' && cv_isalnum(d))) ) // a number
            {
                double fval;
                int ival;
                endptr = ptr + (c == '-' || c == '+');
                while( cv_isdigit(*endptr) )
                    endptr++;
                if( *endptr == '.' || *endptr == 'e' )
                {
                    fval = icv_strtod( fs, ptr, &endptr );
                    /*if( endptr == ptr || cv_isalpha(*endptr) )
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
                CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG();
            }
            else
            {
                // string
                char buf[CV_FS_MAX_LEN+16] = {0};
                int i = 0, len, is_quoted = 0;
                elem->tag = CV_NODE_STRING;
                if( c == '\"' )
                    is_quoted = 1;
                else
                    --ptr;

                for( ;; )
                {
                    c = *++ptr;
                    CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG();
                    if( !cv_isalnum(c) )
                    {
                        if( c == '\"' )
                        {
                            if( !is_quoted )
                                CV_PARSE_ERROR( "Literal \" is not allowed within a string. Use &quot;" );
                            ++ptr;
                            break;
                        }
                        else if( !cv_isprint(c) || c == '<' || (!is_quoted && cv_isspace(c)))
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
                            if( *++ptr == '#' )
                            {
                                int val, base = 10;
                                ptr++;
                                if( *ptr == 'x' )
                                {
                                    base = 16;
                                    ptr++;
                                }
                                val = (int)strtol( ptr, &endptr, base );
                                if( (unsigned)val > (unsigned)255 ||
                                    !endptr || *endptr != ';' )
                                    CV_PARSE_ERROR( "Invalid numeric value in the string" );
                                c = (char)val;
                            }
                            else
                            {
                                endptr = ptr;
                                do c = *++endptr;
                                while( cv_isalnum(c) );
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
                            CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG();
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
            have_space = false;
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

    if( *ptr == '\0' )
        CV_PARSE_ERROR( "Preliminary end of the stream" );

    if( *ptr != '<' )
        CV_PARSE_ERROR( "Tag should start with \'<\'" );

    ptr++;
    CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG();
    if( cv_isalnum(*ptr) || *ptr == '_' )
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

        if( !cv_isalpha(*ptr) && *ptr != '_' )
            CV_PARSE_ERROR( "Name should start with a letter or underscore" );

        endptr = ptr - 1;
        do c = *++endptr;
        while( cv_isalnum(c) || c == '_' || c == '-' );

        attrname = cvGetHashedKey( fs, ptr, (int)(endptr - ptr), 1 );
        CV_Assert(attrname);
        ptr = endptr;
        CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG();

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
        have_space = cv_isspace(c) || c == '\0';

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
            if( ptr[1] != '>'  )  // FIXIT ptr[1] - out of bounds read without check
                CV_PARSE_ERROR( "Invalid closing tag for <?xml ..." );
            ptr += 2;
            break;
        }
        else if( c == '/' && ptr[1] == '>' && tag_type == CV_XML_OPENING_TAG )  // FIXIT ptr[1] - out of bounds read without check
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


void icvXMLParse( CvFileStorage* fs )
{
    char* ptr = fs->buffer_start;
    CvStringHashNode *key = 0, *key2 = 0;
    CvAttrList* list = 0;
    int tag_type = 0;

    // CV_XML_INSIDE_TAG is used to prohibit leading comments
    ptr = icvXMLSkipSpaces( fs, ptr, CV_XML_INSIDE_TAG );

    if( memcmp( ptr, "<?xml", 5 ) != 0 )  // FIXIT ptr[1..] - out of bounds read without check
        CV_PARSE_ERROR( "Valid XML should start with \'<?xml ...?>\'" );

    ptr = icvXMLParseTag( fs, ptr, &key, &list, &tag_type );

    /*{
        const char* version = cvAttrValue( list, "version" );
        if( version && strncmp( version, "1.", 2 ) != 0 )
            CV_Error( CV_StsParseError, "Unsupported version of XML" );
    }*/
    // we support any 8-bit encoding, so we do not need to check the actual encoding.
    // we do not support utf-16, but in the case of utf-16 we will not get here anyway.
    /*{
        const char* encoding = cvAttrValue( list, "encoding" );
        if( encoding && strcmp( encoding, "ASCII" ) != 0 &&
            strcmp( encoding, "UTF-8" ) != 0 &&
            strcmp( encoding, "utf-8" ) != 0 )
            CV_PARSE_ERROR( "Unsupported encoding" );
    }*/

    while( *ptr != '\0' )
    {
        ptr = icvXMLSkipSpaces( fs, ptr, 0 );

        if( *ptr != '\0' )
        {
            CvFileNode* root_node;
            ptr = icvXMLParseTag( fs, ptr, &key, &list, &tag_type );
            if( tag_type != CV_XML_OPENING_TAG ||
                !key ||
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
    CV_Assert( fs->dummy_eof != 0 );
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

    if( !cv_isalpha(key[0]) && key[0] != '_' )
        CV_Error( CV_StsBadArg, "Key should start with a letter or _" );

    ptr = icvFSResizeWriteBuffer( fs, ptr, len );
    for( i = 0; i < len; i++ )
    {
        char c = key[i];
        if( !cv_isalnum(c) && c != '_' && c != '-' )
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


void icvXMLStartWriteStruct( CvFileStorage* fs, const char* key, int struct_flags, const char* type_name)
{
    CvXMLStackRecord parent;
    const char* attr[10];
    int idx = 0;

    struct_flags = (struct_flags & (CV_NODE_TYPE_MASK|CV_NODE_FLOW)) | CV_NODE_EMPTY;
    if( !CV_NODE_IS_COLLECTION(struct_flags))
        CV_Error( CV_StsBadArg,
        "Some collection type: CV_NODE_SEQ or CV_NODE_MAP must be specified" );

    if ( type_name && *type_name == '\0' )
        type_name = 0;

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


void icvXMLEndWriteStruct( CvFileStorage* fs )
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


void icvXMLStartNextStream( CvFileStorage* fs )
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


void icvXMLWriteScalar( CvFileStorage* fs, const char* key, const char* data, int len )
{
    check_if_write_struct_is_delayed( fs );
    if ( fs->state_of_writing_base64 == base64::fs::Uncertain )
    {
        switch_to_Base64_state( fs, base64::fs::NotUse );
    }
    else if ( fs->state_of_writing_base64 == base64::fs::InUse )
    {
        CV_Error( CV_StsError, "Currently only Base64 data is allowed." );
    }

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


void icvXMLWriteInt( CvFileStorage* fs, const char* key, int value )
{
    char buf[128], *ptr = icv_itoa( value, buf, 10 );
    int len = (int)strlen(ptr);
    icvXMLWriteScalar( fs, key, ptr, len );
}


void icvXMLWriteReal( CvFileStorage* fs, const char* key, double value )
{
    char buf[128];
    int len = (int)strlen( icvDoubleToString( buf, value ));
    icvXMLWriteScalar( fs, key, buf, len );
}


void icvXMLWriteString( CvFileStorage* fs, const char* key, const char* str, int quote )
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

            if( (uchar)c >= 128 || c == ' ' )
            {
                *data++ = c;
                need_quote = 1;
            }
            else if( !cv_isprint(c) || c == '<' || c == '>' || c == '&' || c == '\'' || c == '\"' )
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
                    sprintf( data, "#x%02x", (uchar)c );
                    data += 4;
                }
                *data++ = ';';
                need_quote = 1;
            }
            else
                *data++ = c;
        }
        if( !need_quote && (cv_isdigit(str[0]) ||
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


void icvXMLWriteComment( CvFileStorage* fs, const char* comment, int eol_comment )
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
