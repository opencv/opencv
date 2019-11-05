// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "persistence.hpp"

enum
{
    CV_YML_INDENT = 3,
    CV_YML_INDENT_FLOW = 1
};

namespace cv
{

class YAMLEmitter : public FileStorageEmitter
{
public:
    YAMLEmitter(FileStorage_API* _fs) : fs(_fs)
    {
    }
    virtual ~YAMLEmitter() {}

    FStructData startWriteStruct(const FStructData& parent, const char* key,
                                 int struct_flags, const char* type_name=0)
    {
        char buf[CV_FS_MAX_LEN + 1024];
        const char* data = 0;

        if ( type_name && *type_name == '\0' )
            type_name = 0;

        struct_flags = (struct_flags & (FileNode::TYPE_MASK|FileNode::FLOW)) | FileNode::EMPTY;
        if( !FileNode::isCollection(struct_flags))
            CV_Error( CV_StsBadArg,
                     "Some collection type - FileNode::SEQ or FileNode::MAP, must be specified" );

        if (type_name && memcmp(type_name, "binary", 6) == 0)
        {
            /* reset struct flag. in order not to print ']' */
            struct_flags = FileNode::SEQ;
            sprintf(buf, "!!binary |");
            data = buf;
        }
        else if( FileNode::isFlow(struct_flags))
        {
            char c = FileNode::isMap(struct_flags) ? '{' : '[';
            struct_flags |= FileNode::FLOW;

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

        writeScalar( key, data );

        FStructData fsd;
        fsd.indent = parent.indent;
        fsd.flags = struct_flags;

        if( !FileNode::isFlow(parent.flags) )
            fsd.indent += CV_YML_INDENT + FileNode::isFlow(struct_flags);

        return fsd;
    }

    void endWriteStruct(const FStructData& current_struct)
    {
        char* ptr;

        int struct_flags = current_struct.flags;

        if( FileNode::isFlow(struct_flags) )
        {
            ptr = fs->bufferPtr();
            if( ptr > fs->bufferStart() + current_struct.indent && !FileNode::isEmptyCollection(struct_flags) )
                *ptr++ = ' ';
            *ptr++ = FileNode::isMap(struct_flags) ? '}' : ']';
            fs->setBufferPtr(ptr);
        }
        else if( FileNode::isEmptyCollection(struct_flags) )
        {
            ptr = fs->flush();
            memcpy( ptr, FileNode::isMap(struct_flags) ? "{}" : "[]", 2 );
            fs->setBufferPtr(ptr + 2);
        }
        /*
        if( !FileNode::isFlow(parent_flags) )
            fs->struct_indent -= CV_YML_INDENT + FileNode::isFlow(struct_flags);
        assert( fs->struct_indent >= 0 );*/
    }

    void write(const char* key, int value)
    {
        char buf[128];
        writeScalar( key, fs::itoa( value, buf, 10 ));
    }

    void write( const char* key, double value )
    {
        char buf[128];
        writeScalar( key, fs::doubleToString( buf, value, false ));
    }

    void write(const char* key, const char* str, bool quote)
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

        writeScalar( key, data);
    }

    void writeScalar(const char* key, const char* data)
    {
        int i, keylen = 0;
        int datalen = 0;
        char* ptr;

        FStructData& current_struct = fs->getCurrentStruct();

        int struct_flags = current_struct.flags;

        if( key && key[0] == '\0' )
            key = 0;

        if( FileNode::isCollection(struct_flags) )
        {
            if( (FileNode::isMap(struct_flags) ^ (key != 0)) )
                CV_Error( CV_StsBadArg, "An attempt to add element without a key to a map, "
                         "or add element with key to sequence" );
        }
        else
        {
            fs->setNonEmpty();
            struct_flags = FileNode::EMPTY | (key ? FileNode::MAP : FileNode::SEQ);
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

        if( FileNode::isFlow(struct_flags) )
        {
            ptr = fs->bufferPtr();
            if( !FileNode::isEmptyCollection(struct_flags) )
                *ptr++ = ',';
            int new_offset = (int)(ptr - fs->bufferStart()) + keylen + datalen;
            if( new_offset > fs->wrapMargin() && new_offset - current_struct.indent > 10 )
            {
                fs->setBufferPtr(ptr);
                ptr = fs->flush();
            }
            else
                *ptr++ = ' ';
        }
        else
        {
            ptr = fs->flush();
            if( !FileNode::isMap(struct_flags) )
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

            ptr = fs->resizeWriteBuffer( ptr, keylen );

            for( i = 0; i < keylen; i++ )
            {
                char c = key[i];

                ptr[i] = c;
                if( !cv_isalnum(c) && c != '-' && c != '_' && c != ' ' )
                    CV_Error( CV_StsBadArg, "Key names may only contain alphanumeric characters [a-zA-Z0-9], '-', '_' and ' '" );
            }

            ptr += keylen;
            *ptr++ = ':';
            if( !FileNode::isFlow(struct_flags) && data )
                *ptr++ = ' ';
        }

        if( data )
        {
            ptr = fs->resizeWriteBuffer( ptr, datalen );
            memcpy( ptr, data, datalen );
            ptr += datalen;
        }

        fs->setBufferPtr(ptr);
        current_struct.flags &= ~FileNode::EMPTY;
    }

    void writeComment(const char* comment, bool eol_comment)
    {
        if( !comment )
            CV_Error( CV_StsNullPtr, "Null comment" );

        int len = (int)strlen(comment);
        const char* eol = strchr(comment, '\n');
        bool multiline = eol != 0;
        char* ptr = fs->bufferPtr();

        if( !eol_comment || multiline ||
            fs->bufferEnd() - ptr < len || ptr == fs->bufferStart() )
            ptr = fs->flush();
        else
            *ptr++ = ' ';

        while( comment )
        {
            *ptr++ = '#';
            *ptr++ = ' ';
            if( eol )
            {
                ptr = fs->resizeWriteBuffer( ptr, (int)(eol - comment) + 1 );
                memcpy( ptr, comment, eol - comment + 1 );
                fs->setBufferPtr(ptr + (eol - comment));
                comment = eol + 1;
                eol = strchr( comment, '\n' );
            }
            else
            {
                len = (int)strlen(comment);
                ptr = fs->resizeWriteBuffer( ptr, len );
                memcpy( ptr, comment, len );
                fs->setBufferPtr(ptr + len);
                comment = 0;
            }
            ptr = fs->flush();
        }
    }

    void startNextStream()
    {
        fs->puts( "...\n" );
        fs->puts( "---\n" );
    }

protected:
    FileStorage_API* fs;
};


class YAMLParser : public FileStorageParser
{
public:
    YAMLParser(FileStorage_API* _fs) : fs(_fs)
    {
    }

    virtual ~YAMLParser() {}

    char* skipSpaces( char* ptr, int min_indent, int max_comment_indent )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid input");

        for(;;)
        {
            while( *ptr == ' ' )
                ptr++;
            if( *ptr == '#' )
            {
                if( ptr - fs->bufferStart() > max_comment_indent )
                    return ptr;
                *ptr = '\0';
            }
            else if( cv_isprint(*ptr) )
            {
                if( ptr - fs->bufferStart() < min_indent )
                    CV_PARSE_ERROR_CPP( "Incorrect indentation" );
                break;
            }
            else if( *ptr == '\0' || *ptr == '\n' || *ptr == '\r' )
            {
                ptr = fs->gets();
                if( !ptr )
                {
                    // emulate end of stream
                    ptr = fs->bufferStart();
                    ptr[0] = ptr[1] = ptr[2] = '.';
                    ptr[3] = '\0';
                    fs->setEof();
                    break;
                }
                else
                {
                    int l = (int)strlen(ptr);
                    if( ptr[l-1] != '\n' && ptr[l-1] != '\r' && !fs->eof() )
                        CV_PARSE_ERROR_CPP( "Too long string or a last string w/o newline" );
                }
            }
            else
                CV_PARSE_ERROR_CPP( *ptr == '\t' ? "Tabs are prohibited in YAML!" : "Invalid character" );
        }

        return ptr;
    }

    bool getBase64Row(char* ptr, int indent, char* &beg, char* &end)
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid input");

        beg = end = ptr = skipSpaces(ptr, 0, INT_MAX);
        if (!ptr || !*ptr)
            return false; // end of file

        if (ptr - fs->bufferStart() != indent)
            return false; // end of base64 data

        /* find end */
        while(cv_isprint(*ptr)) /* no check for base64 string */
            ++ptr;
        if (*ptr == '\0')
            CV_PARSE_ERROR_CPP("Unexpected end of line");

        end = ptr;
        return true;
    }


    char* parseKey( char* ptr, FileNode& map_node, FileNode& value_placeholder )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid input");

        char c;
        char *endptr = ptr - 1, *saveptr;

        if( *ptr == '-' )
            CV_PARSE_ERROR_CPP( "Key may not start with \'-\'" );

        do c = *++endptr;
        while( cv_isprint(c) && c != ':' );

        if( c != ':' )
            CV_PARSE_ERROR_CPP( "Missing \':\'" );

        saveptr = endptr + 1;
        do c = *--endptr;
        while( c == ' ' );

        ++endptr;
        if( endptr == ptr )
            CV_PARSE_ERROR_CPP( "An empty key" );

        value_placeholder = fs->addNode(map_node, std::string(ptr, endptr - ptr), FileNode::NONE);
        ptr = saveptr;

        return ptr;
    }

    char* parseValue( char* ptr, FileNode& node, int min_indent, bool is_parent_flow )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid input");

        char* endptr = 0;
        char c = ptr[0], d = ptr[1];
        int value_type = FileNode::NONE;
        int len;
        bool is_binary_string = false;
        bool is_user = false;

        if( c == '!' ) // handle explicit type specification
        {
            if( d == '!' || d == '^' )
            {
                ptr++;
                is_user = true;
                //value_type |= FileNode::USER;
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
                        *typeEndPtr = ' ';
                        ptr += headingLenght - 1;
                        is_user = true;
                        //value_type |= FileNode::USER;
                    }
                }
            }

            endptr = ptr++;
            do d = *++endptr;
            while( cv_isprint(d) && d != ' ' );
            len = (int)(endptr - ptr);
            if( len == 0 )
                CV_PARSE_ERROR_CPP( "Empty type name" );
            d = *endptr;
            *endptr = '\0';

            if( len == 3 && !is_user )
            {
                if( memcmp( ptr, "str", 3 ) == 0 )
                    value_type = FileNode::STRING;
                else if( memcmp( ptr, "int", 3 ) == 0 )
                    value_type = FileNode::INT;
                else if( memcmp( ptr, "seq", 3 ) == 0 )
                    value_type = FileNode::SEQ;
                else if( memcmp( ptr, "map", 3 ) == 0 )
                    value_type = FileNode::MAP;
            }
            else if( len == 5 && !is_user )
            {
                if( memcmp( ptr, "float", 5 ) == 0 )
                    value_type = FileNode::REAL;
            }
            else if (len == 6 && is_user)
            {
                if( memcmp( ptr, "binary", 6 ) == 0 ) {
                    value_type = FileNode::SEQ;
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

            *endptr = d;
            ptr = skipSpaces( endptr, min_indent, INT_MAX );
            if (!ptr)
                CV_PARSE_ERROR_CPP("Invalid input");

            c = *ptr;

            if( !is_user )
            {
                if (value_type == FileNode::STRING && c != '\'' && c != '\"')
                    goto force_string;
                if( value_type == FileNode::INT )
                    goto force_int;
                if( value_type == FileNode::REAL )
                    goto force_real;
            }
        }

        if (is_binary_string)
        {
            int indent = static_cast<int>(ptr - fs->bufferStart());
            ptr = fs->parseBase64(ptr, indent, node);
        }
        else if( cv_isdigit(c) ||
                ((c == '-' || c == '+') && (cv_isdigit(d) || d == '.')) ||
                (c == '.' && cv_isalnum(d))) // a number
        {
            endptr = ptr + (c == '-' || c == '+');
            while( cv_isdigit(*endptr) )
                endptr++;
            if( *endptr == '.' || *endptr == 'e' )
            {
            force_real:
                double fval = fs->strtod( ptr, &endptr );
                node.setValue(FileNode::REAL, &fval);
            }
            else
            {
            force_int:
                int ival = (int)strtol( ptr, &endptr, 0 );
                node.setValue(FileNode::INT, &ival);
            }

            if( !endptr || endptr == ptr )
                CV_PARSE_ERROR_CPP( "Invalid numeric value (inconsistent explicit type specification?)" );

            ptr = endptr;
            CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();
        }
        else if( c == '\'' || c == '\"' ) // an explicit string
        {
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
                        CV_PARSE_ERROR_CPP( "Invalid character" );
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
                        CV_PARSE_ERROR_CPP( "Invalid character" );
                }

            if( len >= CV_FS_MAX_LEN )
                CV_PARSE_ERROR_CPP( "Too long string literal" );

            node.setValue(FileNode::STRING, buf, len);
        }
        else if( c == '[' || c == '{' ) // collection as a flow
        {
            int new_min_indent = min_indent + !is_parent_flow;
            int struct_type = c == '{' ? FileNode::MAP : FileNode::SEQ;
            int nelems = 0;

            fs->convertToCollection(struct_type, node);
            d = c == '[' ? ']' : '}';

            for( ++ptr ;; nelems++ )
            {
                FileNode elem;

                ptr = skipSpaces( ptr, new_min_indent, INT_MAX );
                if (!ptr)
                    CV_PARSE_ERROR_CPP("Invalid input");
                if( *ptr == '}' || *ptr == ']' )
                {
                    if( *ptr != d )
                        CV_PARSE_ERROR_CPP( "The wrong closing bracket" );
                    ptr++;
                    break;
                }

                if( nelems != 0 )
                {
                    if( *ptr != ',' )
                        CV_PARSE_ERROR_CPP( "Missing , between the elements" );
                    ptr = skipSpaces( ptr + 1, new_min_indent, INT_MAX );
                    if (!ptr)
                        CV_PARSE_ERROR_CPP("Invalid input");
                }

                if( struct_type == FileNode::MAP )
                {
                    ptr = parseKey( ptr, node, elem );
                    ptr = skipSpaces( ptr, new_min_indent, INT_MAX );
                }
                else
                {
                    if( *ptr == ']' )
                        break;
                    elem = fs->addNode(node, std::string(), FileNode::NONE);
                }
                ptr = parseValue( ptr, elem, new_min_indent, true );
            }
            fs->finalizeCollection(node);
        }
        else
        {
            int indent, struct_type;

            if( is_parent_flow || c != '-' )
            {
                // implicit (one-line) string or nested block-style collection
                if( !is_parent_flow )
                {
                    if( c == '?' )
                        CV_PARSE_ERROR_CPP( "Complex keys are not supported" );
                    if( c == '|' || c == '>' )
                        CV_PARSE_ERROR_CPP( "Multi-line text literals are not supported" );
                }

            force_string:
                endptr = ptr - 1;

                do c = *++endptr;
                while( cv_isprint(c) &&
                      (!is_parent_flow || (c != ',' && c != '}' && c != ']')) &&
                      (is_parent_flow || c != ':' || value_type == FileNode::STRING));

                if( endptr == ptr )
                    CV_PARSE_ERROR_CPP( "Invalid character" );

                if( is_parent_flow || c != ':' )
                {
                    char* str_end = endptr;
                    // strip spaces in the end of string
                    do c = *--str_end;
                    while( str_end > ptr && c == ' ' );
                    str_end++;
                    node.setValue(FileNode::STRING, ptr, (int)(str_end - ptr));
                    ptr = endptr;
                    return ptr;
                }
                struct_type = FileNode::MAP;
            }
            else
                struct_type = FileNode::SEQ;

            fs->convertToCollection( struct_type, node );
            indent = (int)(ptr - fs->bufferStart());

            for(;;)
            {
                FileNode elem;

                if( struct_type == FileNode::MAP )
                {
                    ptr = parseKey( ptr, node, elem );
                }
                else
                {
                    c = *ptr++;
                    if( c != '-' )
                        CV_PARSE_ERROR_CPP( "Block sequence elements must be preceded with \'-\'" );

                    elem = fs->addNode(node, std::string(), FileNode::NONE);
                }
                ptr = skipSpaces( ptr, indent + 1, INT_MAX );
                ptr = parseValue( ptr, elem, indent + 1, false );
                ptr = skipSpaces( ptr, 0, INT_MAX );
                if( ptr - fs->bufferStart() != indent )
                {
                    if( ptr - fs->bufferStart() < indent )
                        break;
                    else
                        CV_PARSE_ERROR_CPP( "Incorrect indentation" );
                }
                if( memcmp( ptr, "...", 3 ) == 0 )
                    break;
            }
            fs->finalizeCollection(node);
        }

        return ptr;
    }

    bool parse( char* ptr )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid input");

        bool first = true;
        bool ok = true;
        FileNode root_collection(fs->getFS(), 0, 0);

        for(;;)
        {
            // 0. skip leading comments and directives  and ...
            // 1. reach the first item
            for(;;)
            {
                ptr = skipSpaces( ptr, 0, INT_MAX );
                if( !ptr || *ptr == '\0' )
                {
                    ok = !first;
                    break;
                }

                if( *ptr == '%' )
                {
                    if( memcmp( ptr, "%YAML", 5 ) == 0 &&
                        memcmp( ptr, "%YAML:1.", 8 ) != 0 &&
                        memcmp( ptr, "%YAML 1.", 8 ) != 0)
                        CV_PARSE_ERROR_CPP( "Unsupported YAML version (it must be 1.x)" );
                    *ptr = '\0';
                }
                else if( *ptr == '-' )
                {
                    if( memcmp(ptr, "---", 3) == 0 )
                    {
                        ptr += 3;
                        break;
                    }
                    else if( first )
                        break;
                }
                else if( cv_isalnum(*ptr) || *ptr=='_')
                {
                    if( !first )
                        CV_PARSE_ERROR_CPP( "The YAML streams must start with '---', except the first one" );
                    break;
                }
                else if( fs->eof() )
                    break;
                else
                    CV_PARSE_ERROR_CPP( "Invalid or unsupported syntax" );
            }

            if( ptr )
                ptr = skipSpaces( ptr, 0, INT_MAX );
            if( !ptr || !ptr[0] )
                break;
            if( memcmp( ptr, "...", 3 ) != 0 )
            {
                // 2. parse the collection
                FileNode root_node = fs->addNode(root_collection, std::string(), FileNode::NONE);

                ptr = parseValue( ptr, root_node, 0, false );
                if( !root_node.isMap() && !root_node.isSeq() )
                    CV_PARSE_ERROR_CPP( "Only collections as YAML streams are supported by this parser" );

                // 3. parse until the end of file or next collection
                ptr = skipSpaces( ptr, 0, INT_MAX );
                if( !ptr )
                    break;
            }

            if( fs->eof() )
                break;
            ptr += 3;
            first = false;
        }

        return ok;
    }

    FileStorage_API* fs;
    char buf[CV_FS_MAX_LEN+1024];
};

Ptr<FileStorageEmitter> createYAMLEmitter(FileStorage_API* fs)
{
    return makePtr<YAMLEmitter>(fs);
}

Ptr<FileStorageParser> createYAMLParser(FileStorage_API* fs)
{
    return makePtr<YAMLParser>(fs);
}

}
