// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "persistence.hpp"

enum
{
    CV_XML_INDENT = 2,
    CV_XML_INSIDE_COMMENT = 1,
    CV_XML_INSIDE_TAG = 2,
    CV_XML_INSIDE_DIRECTIVE = 3,
    CV_XML_OPENING_TAG = 1,
    CV_XML_CLOSING_TAG = 2,
    CV_XML_EMPTY_TAG = 3,
    CV_XML_HEADER_TAG = 4,
    CV_XML_DIRECTIVE_TAG = 5
};

namespace cv
{

class XMLEmitter : public FileStorageEmitter
{
public:
    XMLEmitter(FileStorage_API* _fs) : fs(_fs)
    {
    }
    virtual ~XMLEmitter() {}

    void writeTag( const char* key, int tag_type, const std::vector<std::string>& attrlist=std::vector<std::string>() )
    {
        char* ptr = fs->bufferPtr();
        int i, len = 0;
        FStructData& current_struct = fs->getCurrentStruct();
        int struct_flags = current_struct.flags;

        if( key && key[0] == '\0' )
            key = 0;

        if( tag_type == CV_XML_OPENING_TAG || tag_type == CV_XML_EMPTY_TAG )
        {
            if( FileNode::isCollection(struct_flags) )
            {
                if( FileNode::isMap(struct_flags) ^ (key != 0) )
                    CV_Error( cv::Error::StsBadArg, "An attempt to add element without a key to a map, "
                             "or add element with key to sequence" );
            }
            else
            {
                struct_flags = FileNode::EMPTY + (key ? FileNode::MAP : FileNode::SEQ);
                //fs->is_first = 0;
            }

            if( !FileNode::isEmptyCollection(struct_flags) )
                ptr = fs->flush();
        }

        if( !key )
            key = "_";
        else if( key[0] == '_' && key[1] == '\0' )
            CV_Error( cv::Error::StsBadArg, "A single _ is a reserved tag name" );

        len = (int)strlen( key );
        *ptr++ = '<';
        if( tag_type == CV_XML_CLOSING_TAG )
        {
            if( !attrlist.empty() )
                CV_Error( cv::Error::StsBadArg, "Closing tag should not include any attributes" );
            *ptr++ = '/';
        }

        if( !cv_isalpha(key[0]) && key[0] != '_' )
            CV_Error( cv::Error::StsBadArg, "Key should start with a letter or _" );

        ptr = fs->resizeWriteBuffer( ptr, len );
        for( i = 0; i < len; i++ )
        {
            char c = key[i];
            if( !cv_isalnum(c) && c != '_' && c != '-' )
                CV_Error( cv::Error::StsBadArg, "Key name may only contain alphanumeric characters [a-zA-Z0-9], '-' and '_'" );
            ptr[i] = c;
        }
        ptr += len;

        int nattr = (int)attrlist.size();
        CV_Assert( nattr % 2 == 0 );

        for( i = 0; i < nattr; i += 2 )
        {
            size_t len0 = attrlist[i].size();
            size_t len1 = attrlist[i+1].size();
            CV_Assert( len0 > 0 );

            ptr = fs->resizeWriteBuffer( ptr, (int)(len0 + len1 + 4) );
            *ptr++ = ' ';

            memcpy( ptr, attrlist[i].c_str(), len0 );
            ptr += len0;
            *ptr++ = '=';
            *ptr++ = '\"';
            if( len1 > 0 )
                memcpy( ptr, attrlist[i+1].c_str(), len1 );
            ptr += len1;
            *ptr++ = '\"';
        }

        if( tag_type == CV_XML_EMPTY_TAG )
            *ptr++ = '/';
        *ptr++ = '>';
        fs->setBufferPtr(ptr);
        current_struct.flags = struct_flags & ~FileNode::EMPTY;
    }

    FStructData startWriteStruct(const FStructData& parent, const char* key,
                                 int struct_flags, const char* type_name=0)
    {
        std::vector<std::string> attrlist;
        if( type_name && *type_name )
        {
            attrlist.push_back("type_id");
            attrlist.push_back(type_name);
        }

        writeTag( key, CV_XML_OPENING_TAG, attrlist );

        FStructData current_struct;
        current_struct.tag = key ? std::string(key) : std::string();
        current_struct.flags = struct_flags;
        current_struct.indent = parent.indent + CV_XML_INDENT;

        return current_struct;
    }

    void endWriteStruct(const FStructData& current_struct)
    {
        writeTag( current_struct.tag.c_str(), CV_XML_CLOSING_TAG );
    }

    void write(const char* key, int value)
    {
        char buf[128], *ptr = fs::itoa( value, buf, 10 );
        writeScalar( key, ptr);
    }

    void write( const char* key, double value )
    {
        char buf[128];
        writeScalar( key, fs::doubleToString( buf, sizeof(buf), value, false ) );
    }

    void write(const char* key, const char* str, bool quote)
    {
        char buf[CV_FS_MAX_LEN*6+16];
        char* data = (char*)str;
        int i, len;

        if( !str )
            CV_Error( cv::Error::StsNullPtr, "Null string pointer" );

        len = (int)strlen(str);
        if( len > CV_FS_MAX_LEN )
            CV_Error( cv::Error::StsBadArg, "The written string is too long" );

        if( quote || len == 0 || str[0] != '\"' || str[0] != str[len-1] )
        {
            bool need_quote = quote || len == 0;
            data = buf;
            *data++ = '\"';
            for( i = 0; i < len; i++ )
            {
                char c = str[i];

                if( (uchar)c >= 128 || c == ' ' )
                {
                    *data++ = c;
                    need_quote = true;
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
                need_quote = true;

            if( need_quote )
                *data++ = '\"';
            len = (int)(data - buf) - !need_quote;
            *data++ = '\0';
            data = buf + (int)!need_quote;
        }

        writeScalar( key, data );
    }

    void writeScalar(const char* key, const char* data)
    {
        fs->check_if_write_struct_is_delayed(false);
        if ( fs->get_state_of_writing_base64() == FileStorage_API::Uncertain )
        {
            fs->switch_to_Base64_state( FileStorage_API::NotUse );
        }
        else if ( fs->get_state_of_writing_base64() == FileStorage_API::InUse )
        {
            CV_Error( cv::Error::StsError, "At present, output Base64 data only." );
        }

        int len = (int)strlen(data);
        if( key && *key == '\0' )
            key = 0;

        FStructData& current_struct = fs->getCurrentStruct();
        int struct_flags = current_struct.flags;

        if( FileNode::isMap(struct_flags) ||
           (!FileNode::isCollection(struct_flags) && key) )
        {
            writeTag( key, CV_XML_OPENING_TAG );
            char* ptr = fs->resizeWriteBuffer( fs->bufferPtr(), len );
            memcpy( ptr, data, len );
            fs->setBufferPtr( ptr + len );
            writeTag( key, CV_XML_CLOSING_TAG );
        }
        else
        {
            char* ptr = fs->bufferPtr();
            int new_offset = (int)(ptr - fs->bufferStart()) + len;

            if( key )
                CV_Error( cv::Error::StsBadArg, "elements with keys can not be written to sequence" );

            current_struct.flags = FileNode::SEQ;

            if( (new_offset > fs->wrapMargin() && new_offset - current_struct.indent > 10) ||
               (ptr > fs->bufferStart() && ptr[-1] == '>') )
            {
                ptr = fs->flush();
            }
            else if( ptr > fs->bufferStart() + current_struct.indent && ptr[-1] != '>' )
                *ptr++ = ' ';

            memcpy( ptr, data, len );
            fs->setBufferPtr(ptr + len);
        }
    }

    void writeComment(const char* comment, bool eol_comment)
    {
        FStructData& current_struct = fs->getCurrentStruct();
        int len;
        int multiline;
        const char* eol;
        char* ptr;

        if( !comment )
            CV_Error( cv::Error::StsNullPtr, "Null comment" );

        if( strstr(comment, "--") != 0 )
            CV_Error( cv::Error::StsBadArg, "Double hyphen \'--\' is not allowed in the comments" );

        len = (int)strlen(comment);
        eol = strchr(comment, '\n');
        multiline = eol != 0;
        ptr = fs->bufferPtr();

        if( multiline || !eol_comment || fs->bufferEnd() - ptr < len + 5 )
            ptr = fs->flush();
        else if( ptr > fs->bufferStart() + current_struct.indent )
            *ptr++ = ' ';

        if( !multiline )
        {
            ptr = fs->resizeWriteBuffer( ptr, len + 9 );
            sprintf( ptr, "<!-- %s -->", comment );
            len = (int)strlen(ptr);
        }
        else
        {
            strcpy( ptr, "<!--" );
            len = 4;
        }

        fs->setBufferPtr(ptr + len);
        ptr = fs->flush();

        if( multiline )
        {
            while( comment )
            {
                if( eol )
                {
                    ptr = fs->resizeWriteBuffer( ptr, (int)(eol - comment) + 1 );
                    memcpy( ptr, comment, eol - comment + 1 );
                    ptr += eol - comment;
                    comment = eol + 1;
                    eol = strchr( comment, '\n' );
                }
                else
                {
                    len = (int)strlen(comment);
                    ptr = fs->resizeWriteBuffer( ptr, len );
                    memcpy( ptr, comment, len );
                    ptr += len;
                    comment = 0;
                }
                fs->setBufferPtr(ptr);
                ptr = fs->flush();
            }
            sprintf( ptr, "-->" );
            fs->setBufferPtr(ptr + 3);
            fs->flush();
        }
    }

    void startNextStream()
    {
        fs->puts( "\n<!-- next stream -->\n" );
    }

protected:
    FileStorage_API* fs;
};

class XMLParser : public FileStorageParser
{
public:
    XMLParser(FileStorage_API* _fs) : fs(_fs)
    {
    }

    virtual ~XMLParser() {}

    char* skipSpaces( char* ptr, int mode )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid input");

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
                    CV_Assert( ptr[1] == '-' && ptr[2] == '>' );
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
                        CV_PARSE_ERROR_CPP( "Comments are not allowed here" );
                    mode = CV_XML_INSIDE_COMMENT;
                    ptr += 4;
                }
                else if( cv_isprint(c) )
                    break;
            }

            if( !cv_isprint(*ptr) )
            {
                if( *ptr != '\0' && *ptr != '\n' && *ptr != '\r' )
                    CV_PARSE_ERROR_CPP( "Invalid character in the stream" );
                ptr = fs->gets();
                if( !ptr || *ptr == '\0' )
                    break;
            }
        }
        return ptr;
    }

    bool getBase64Row(char* ptr, int /*indent*/, char* &beg, char* &end)
    {
        beg = end = ptr = skipSpaces(ptr, CV_XML_INSIDE_TAG);
        if( !ptr || !*ptr )
            return false;

        // closing XML tag
        if ( *beg == '<' )
            return false;

        // find end of the row
        while( cv_isprint(*ptr) )
            ++ptr;
        if ( *ptr == '\0' )
            CV_PARSE_ERROR_CPP( "Unexpected end of line" );

        end = ptr;
        return true;
    }

    char* parseValue( char* ptr, FileNode& node )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid input");

        FileNode new_elem;
        bool have_space = true;
        int value_type = node.type();
        std::string key, key2, type_name;

        for(;;)
        {
            char c = *ptr, d;
            char* endptr;

            // FIXIT ptr[1], ptr[2] - out of bounds read without check or data fetch (#11061)
            if( cv_isspace(c) || c == '\0' ||
                (c == '<' && ptr[1] == '!' && ptr[2] == '-') )
            {
                ptr = skipSpaces( ptr, 0 );
                if (!ptr)
                    CV_PARSE_ERROR_CPP("Invalid input");
                have_space = true;
                c = *ptr;
            }

            d = ptr[1];  // FIXIT ptr[1] - out of bounds read without check or data fetch (#11061)

            if( c =='<' || c == '\0' )
            {
                int tag_type = 0;
                int elem_type = FileNode::NONE;

                if( d == '/' || c == '\0' )
                    break;

                ptr = parseTag( ptr, key, type_name, tag_type );

                if( tag_type == CV_XML_DIRECTIVE_TAG )
                    CV_PARSE_ERROR_CPP( "Directive tags are not allowed here" );
                if( tag_type == CV_XML_EMPTY_TAG )
                    CV_PARSE_ERROR_CPP( "Empty tags are not supported" );

                CV_Assert(tag_type == CV_XML_OPENING_TAG);

                /* for base64 string */
                bool binary_string = false;

                if( !type_name.empty() )
                {
                    const char* tn = type_name.c_str();
                    if( strcmp(tn, "str") == 0 )
                        elem_type = FileNode::STRING;
                    else if( strcmp( tn, "map" ) == 0 )
                        elem_type = FileNode::MAP;
                    else if( strcmp( tn, "seq" ) == 0 )
                        elem_type = FileNode::SEQ;
                    else if( strcmp( tn, "binary") == 0)
                        binary_string = true;
                }

                new_elem = fs->addNode(node, key, elem_type, 0);
                if (!binary_string)
                    ptr = parseValue(ptr, new_elem);
                else
                {
                    ptr = fs->parseBase64( ptr, 0, new_elem);
                    ptr = skipSpaces( ptr, 0 );
                    if (!ptr)
                        CV_PARSE_ERROR_CPP("Invalid input");
                }

                ptr = parseTag( ptr, key2, type_name, tag_type );
                if( tag_type != CV_XML_CLOSING_TAG || key2 != key )
                    CV_PARSE_ERROR_CPP( "Mismatched closing tag" );
                have_space = true;
            }
            else
            {
                if( !have_space )
                    CV_PARSE_ERROR_CPP( "There should be space between literals" );

                FileNode* elem = &node;
                if( node.type() != FileNode::NONE )
                {
                    fs->convertToCollection( FileNode::SEQ, node );
                    new_elem = fs->addNode(node, std::string(), FileNode::NONE, 0);
                    elem = &new_elem;
                }

                if( value_type != FileNode::STRING &&
                   (cv_isdigit(c) || ((c == '-' || c == '+') &&
                   (cv_isdigit(d) || d == '.')) || (c == '.' && cv_isalnum(d))) ) // a number
                {
                    endptr = ptr + (c == '-' || c == '+');
                    while( cv_isdigit(*endptr) )
                        endptr++;
                    if( *endptr == '.' || *endptr == 'e' )
                    {
                        double fval = fs->strtod( ptr, &endptr );
                        elem->setValue(FileNode::REAL, &fval);
                    }
                    else
                    {
                        int ival = (int)strtol( ptr, &endptr, 0 );
                        elem->setValue(FileNode::INT, &ival);
                    }

                    if( endptr == ptr )
                        CV_PARSE_ERROR_CPP( "Invalid numeric value (inconsistent explicit type specification?)" );

                    ptr = endptr;
                    CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();
                }
                else
                {
                    // string
                    int i = 0, len, is_quoted = 0;
                    if( c == '\"' )
                        is_quoted = 1;
                    else
                        --ptr;
                    strbuf[0] = '\0';

                    for( ;; )
                    {
                        c = *++ptr;
                        CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();

                        if( !cv_isalnum(c) )
                        {
                            if( c == '\"' )
                            {
                                if( !is_quoted )
                                    CV_PARSE_ERROR_CPP( "Literal \" is not allowed within a string. Use &quot;" );
                                ++ptr;
                                break;
                            }
                            else if( !cv_isprint(c) || c == '<' || (!is_quoted && cv_isspace(c)))
                            {
                                if( is_quoted )
                                    CV_PARSE_ERROR_CPP( "Closing \" is expected" );
                                break;
                            }
                            else if( c == '\'' || c == '>' )
                            {
                                CV_PARSE_ERROR_CPP( "Literal \' or > are not allowed. Use &apos; or &gt;" );
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
                                        CV_PARSE_ERROR_CPP( "Invalid numeric value in the string" );
                                    c = (char)val;
                                }
                                else
                                {
                                    endptr = ptr;
                                    do c = *++endptr;
                                    while( cv_isalnum(c) );
                                    if( c != ';' )
                                        CV_PARSE_ERROR_CPP( "Invalid character in the symbol entity name" );
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
                                        if (len + 2 + i >= CV_FS_MAX_LEN)
                                            CV_PARSE_ERROR_CPP("string is too long");
                                        memcpy( strbuf + i, ptr-1, len + 2 );
                                        i += len + 2;
                                    }
                                }
                                ptr = endptr;
                                CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();
                            }
                        }
                        if (i + 1 >= CV_FS_MAX_LEN)
                            CV_PARSE_ERROR_CPP("Too long string literal");
                        strbuf[i++] = c;
                    }
                    elem->setValue(FileNode::STRING, strbuf, i);
                }

                if( value_type != FileNode::NONE && value_type != FileNode::SEQ && value_type != FileNode::MAP )
                    break;
                have_space = false;
            }
        }
        fs->finalizeCollection(node);

        return ptr;
    }

    char* parseTag( char* ptr, std::string& tag_name,
                    std::string& type_name, int& tag_type )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid tag input");

        if( *ptr == '\0' )
            CV_PARSE_ERROR_CPP( "Unexpected end of the stream" );

        if( *ptr != '<' )
            CV_PARSE_ERROR_CPP( "Tag should start with \'<\'" );

        ptr++;
        CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();

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
            CV_Assert( ptr[1] != '-' || ptr[2] != '-' );
            ptr++;
        }
        else
            CV_PARSE_ERROR_CPP( "Unknown tag type" );

        tag_name.clear();
        type_name.clear();

        for(;;)
        {
            char c, *endptr;
            if( !cv_isalpha(*ptr) && *ptr != '_' )
                CV_PARSE_ERROR_CPP( "Name should start with a letter or underscore" );

            endptr = ptr - 1;
            do c = *++endptr;
            while( cv_isalnum(c) || c == '_' || c == '-' );

            std::string attrname(ptr, (size_t)(endptr - ptr));
            ptr = endptr;
            CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();

            if( tag_name.empty() )
                tag_name = attrname;
            else
            {
                if( tag_type == CV_XML_CLOSING_TAG )
                    CV_PARSE_ERROR_CPP( "Closing tag should not contain any attributes" );

                if( *ptr != '=' )
                {
                    ptr = skipSpaces( ptr, CV_XML_INSIDE_TAG );
                    if (!ptr)
                        CV_PARSE_ERROR_CPP("Invalid attribute");
                    if( *ptr != '=' )
                        CV_PARSE_ERROR_CPP( "Attribute name should be followed by \'=\'" );
                }

                c = *++ptr;
                if( c != '\"' && c != '\'' )
                {
                    ptr = skipSpaces( ptr, CV_XML_INSIDE_TAG );
                    if( *ptr != '\"' && *ptr != '\'' )
                        CV_PARSE_ERROR_CPP( "Attribute value should be put into single or double quotes" );
                }

                char quote = *ptr++;
                endptr = ptr;
                for(;;)
                {
                    c = *endptr++;
                    if( c == quote )
                        break;
                    if( c == '\0' )
                        CV_PARSE_ERROR_CPP( "Unexpected end of line" );
                }

                if( attrname == "type_id" )
                {
                    CV_Assert( type_name.empty() );
                    type_name = std::string(ptr, (size_t)(endptr - 1 - ptr));
                }

                ptr = endptr;
            }

            c = *ptr;
            bool have_space = cv_isspace(c) || c == '\0';

            if( c != '>' )
            {
                ptr = skipSpaces( ptr, CV_XML_INSIDE_TAG );
                if (!ptr)
                    CV_PARSE_ERROR_CPP("Invalid input");
                c = *ptr;
            }

            if( c == '>' )
            {
                if( tag_type == CV_XML_HEADER_TAG )
                    CV_PARSE_ERROR_CPP( "Invalid closing tag for <?xml ..." );
                ptr++;
                break;
            }
            else if( c == '?' && tag_type == CV_XML_HEADER_TAG )
            {
                if( ptr[1] != '>'  )  // FIXIT ptr[1] - out of bounds read without check
                    CV_PARSE_ERROR_CPP( "Invalid closing tag for <?xml ..." );
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
                CV_PARSE_ERROR_CPP( "There should be space between attributes" );
        }

        return ptr;
    }

    bool parse(char* ptr)
    {
        CV_Assert( fs != 0 );

        std::string key, key2, type_name;
        int tag_type = 0;
        bool ok = false;

        // CV_XML_INSIDE_TAG is used to prohibit leading comments
        ptr = skipSpaces( ptr, CV_XML_INSIDE_TAG );
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid input");

        if( memcmp( ptr, "<?xml", 5 ) != 0 )  // FIXIT ptr[1..] - out of bounds read without check
            CV_PARSE_ERROR_CPP( "Valid XML should start with \'<?xml ...?>\'" );

        ptr = parseTag( ptr, key, type_name, tag_type );
        FileNode root_collection(fs->getFS(), 0, 0);

        while( ptr && *ptr != '\0' )
        {
            ptr = skipSpaces( ptr, 0 );
            if (!ptr)
                CV_PARSE_ERROR_CPP("Invalid input");

            if( *ptr != '\0' )
            {
                ptr = parseTag( ptr, key, type_name, tag_type );
                if( tag_type != CV_XML_OPENING_TAG || key != "opencv_storage" )
                    CV_PARSE_ERROR_CPP( "<opencv_storage> tag is missing" );
                FileNode root = fs->addNode(root_collection, std::string(), FileNode::MAP, 0);
                ptr = parseValue( ptr, root );
                ptr = parseTag( ptr, key2, type_name, tag_type );
                if( tag_type != CV_XML_CLOSING_TAG || key != key2 )
                    CV_PARSE_ERROR_CPP( "</opencv_storage> tag is missing" );
                ptr = skipSpaces( ptr, 0 );
                ok = true;
            }
        }
        CV_Assert( fs->eof() );
        return ok;
    }

    FileStorage_API* fs;
    char strbuf[CV_FS_MAX_LEN+16];
};

Ptr<FileStorageEmitter> createXMLEmitter(FileStorage_API* fs)
{
    return makePtr<XMLEmitter>(fs);
}

Ptr<FileStorageParser> createXMLParser(FileStorage_API* fs)
{
    return makePtr<XMLParser>(fs);
}

}
