// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "persistence.hpp"

namespace cv
{

class JSONEmitter : public FileStorageEmitter
{
public:
    JSONEmitter(FileStorage_API* _fs) : fs(_fs)
    {
    }
    virtual ~JSONEmitter() {}

    FStructData startWriteStruct( const FStructData& parent, const char* key,
                                  int struct_flags, const char* type_name=0 )
    {
        char data[CV_FS_MAX_LEN + 1024];

        struct_flags = (struct_flags & (FileNode::TYPE_MASK|FileNode::FLOW)) | FileNode::EMPTY;
        if( !FileNode::isCollection(struct_flags))
            CV_Error( cv::Error::StsBadArg,
                     "Some collection type - FileNode::SEQ or FileNode::MAP, must be specified" );

        if( type_name && *type_name == '\0' )
            type_name = 0;

        bool is_real_collection = true;
        if (type_name && memcmp(type_name, "binary", 6) == 0)
        {
            struct_flags = FileNode::STR;
            data[0] = '\0';
            is_real_collection = false;
        }

        if ( is_real_collection )
        {
            char c = FileNode::isMap(struct_flags) ? '{' : '[';
            data[0] = c;
            data[1] = '\0';
        }

        writeScalar( key, data );
        FStructData current_struct("", struct_flags, parent.indent + 4);

        return current_struct;
    }

    void endWriteStruct(const FStructData& current_struct)
    {
        int struct_flags = current_struct.flags;

        if (FileNode::isCollection(struct_flags)) {
            if (!FileNode::isFlow(struct_flags)) {
                if (fs->bufferPtr() <= fs->bufferStart() + fs->get_space()) {
                    /* some bad code for base64_writer... */
                    char *ptr = fs->bufferPtr();
                    *ptr++ = '\n';
                    *ptr++ = '\0';
                    fs->puts(fs->bufferStart());
                    fs->setBufferPtr(fs->bufferStart());
                }
                fs->flush();
            }

            char *ptr = fs->bufferPtr();
            if (ptr > fs->bufferStart() + current_struct.indent && !FileNode::isEmptyCollection(struct_flags))
                *ptr++ = ' ';
            *ptr++ = FileNode::isMap(struct_flags) ? '}' : ']';
            fs->setBufferPtr(ptr);
        }
    }

    void write(const char* key, int value)
    {
        char buf[128];
        writeScalar( key, fs::itoa( value, buf, 10 ));
    }

    void write(const char* key, int64_t value)
    {
        char buf[128];
        writeScalar( key, fs::itoa( value, buf, 10, true ));
    }

    void write( const char* key, double value )
    {
        char buf[128];
        writeScalar( key, fs::doubleToString( buf, sizeof(buf), value, true ));
    }

    void write(const char* key, const char* str, bool quote)
    {
        char buf[CV_FS_MAX_LEN*4+16];
        char* data = (char*)str;
        int i, len;

        if( !str )
            CV_Error( cv::Error::StsNullPtr, "Null string pointer" );

        len = (int)strlen(str);
        if( len > CV_FS_MAX_LEN )
            CV_Error( cv::Error::StsBadArg, "The written string is too long" );

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
                }
            }

            *data++ = '\"';
            *data++ = '\0';
            data = buf + !need_quote;
        }

        writeScalar( key, data);
    }

    void writeScalar(const char* key, const char* data)
    {
        /* check write_struct */

        fs->check_if_write_struct_is_delayed(false);
        if ( fs->get_state_of_writing_base64() == FileStorage_API::Uncertain )
        {
            fs->switch_to_Base64_state( FileStorage_API::NotUse );
        }
        else if ( fs->get_state_of_writing_base64() == FileStorage_API::InUse )
        {
            CV_Error( cv::Error::StsError, "At present, output Base64 data only." );
        }

        /* check parameters */

        size_t key_len = 0u;
        if( key && *key == '\0' )
            key = 0;
        if ( key )
        {
            key_len = strlen(key);
            if ( key_len == 0u )
                CV_Error( cv::Error::StsBadArg, "The key is an empty" );
            else if ( static_cast<int>(key_len) > CV_FS_MAX_LEN )
                CV_Error( cv::Error::StsBadArg, "The key is too long" );
        }

        size_t data_len = 0u;
        if ( data )
            data_len = strlen(data);

        FStructData& current_struct = fs->getCurrentStruct();
        int struct_flags = current_struct.flags;
        if( FileNode::isCollection(struct_flags) )
        {
            if ( (FileNode::isMap(struct_flags) ^ (key != 0)) )
                CV_Error( cv::Error::StsBadArg, "An attempt to add element without a key to a map, "
                         "or add element with key to sequence" );
        } else {
            fs->setNonEmpty();
            struct_flags = FileNode::EMPTY | (key ? FileNode::MAP : FileNode::SEQ);
        }

        // start to write
        char* ptr = 0;

        if( FileNode::isFlow(struct_flags) )
        {
            int new_offset;
            ptr = fs->bufferPtr();
            if( !FileNode::isEmptyCollection(struct_flags) )
                *ptr++ = ',';
            new_offset = static_cast<int>(ptr - fs->bufferStart() + key_len + data_len);
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
            if ( !FileNode::isEmptyCollection(struct_flags) )
            {
                ptr = fs->bufferPtr();
                *ptr++ = ',';
                *ptr++ = '\n';
                *ptr++ = '\0';
                fs->puts( fs->bufferStart() );
                fs->setBufferPtr(fs->bufferStart());
            }
            ptr = fs->flush();
        }

        if( key )
        {
            if( !cv_isalpha(key[0]) && key[0] != '_' )
                CV_Error( cv::Error::StsBadArg, "Key must start with a letter or _" );

            ptr = fs->resizeWriteBuffer( ptr, static_cast<int>(key_len) );
            *ptr++ = '\"';

            for( size_t i = 0u; i < key_len; i++ )
            {
                char c = key[i];

                ptr[i] = c;
                if( !cv_isalnum(c) && c != '-' && c != '_' && c != ' ' )
                    CV_Error( cv::Error::StsBadArg, "Key names may only contain alphanumeric characters [a-zA-Z0-9], '-', '_' and ' '" );
            }

            ptr += key_len;
            *ptr++ = '\"';
            *ptr++ = ':';
            *ptr++ = ' ';
        }

        if( data )
        {
            ptr = fs->resizeWriteBuffer( ptr, static_cast<int>(data_len) );
            memcpy( ptr, data, data_len );
            ptr += data_len;
        }

        fs->setBufferPtr(ptr);
        current_struct.flags &= ~FileNode::EMPTY;
    }

    void writeComment(const char* comment, bool eol_comment)
    {
        if( !comment )
            CV_Error( cv::Error::StsNullPtr, "Null comment" );

        int len = static_cast<int>(strlen(comment));
        char* ptr = fs->bufferPtr();
        const char* eol = strchr(comment, '\n');
        bool multiline = eol != 0;

        if( !eol_comment || multiline || fs->bufferEnd() - ptr < len || ptr == fs->bufferStart() )
            ptr = fs->flush();
        else
            *ptr++ = ' ';

        while( comment )
        {
            *ptr++ = '/';
            *ptr++ = '/';
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

class JSONParser : public FileStorageParser
{
public:
    JSONParser(FileStorage_API* _fs) : fs(_fs)
    {
    }

    virtual ~JSONParser() {}

    char* skipSpaces( char* ptr )
    {
        bool is_eof = false;
        bool is_completed = false;

        while ( is_eof == false && is_completed == false )
        {
            if (!ptr)
                CV_PARSE_ERROR_CPP("Invalid input");
            switch ( *ptr )
            {
                /* comment */
                case '/' : {
                    ptr++;
                    if ( *ptr == '\0' )
                    {
                        ptr = fs->gets();
                        if( !ptr || !*ptr ) { is_eof = true; break; }
                    }

                    if ( *ptr == '/' )
                    {
                        while ( *ptr != '\n' && *ptr != '\r' )
                        {
                            if ( *ptr == '\0' )
                            {
                                ptr = fs->gets();
                                if( !ptr || !*ptr ) { is_eof = true; break; }
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
                                ptr = fs->gets();
                                if( !ptr || !*ptr ) { is_eof = true; break; }
                            }
                            else if ( *ptr == '*' )
                            {
                                ptr++;
                                if ( *ptr == '\0' )
                                {
                                    ptr = fs->gets();
                                    if( !ptr || !*ptr ) { is_eof = true; break; }
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
                        CV_PARSE_ERROR_CPP( "Not supported escape character" );
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
                    ptr = fs->gets();
                    if( !ptr || !*ptr ) { is_eof = true; break; }
                } break;
                    /* other character */
                default: {
                    if( !cv_isprint(*ptr) )
                        CV_PARSE_ERROR_CPP( "Invalid character in the stream" );
                    is_completed = true;
                } break;
            }
        }

        if ( is_eof || !is_completed )
        {
            ptr = fs->bufferStart();
            CV_Assert(ptr);
            *ptr = '\0';
            fs->setEof();
            if( !is_completed )
                CV_PARSE_ERROR_CPP( "Abort at parse time" );
        }

        return ptr;
    }

    char* parseKey( char* ptr, FileNode& collection, FileNode& value_placeholder )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid input");

        if( *ptr != '"' )
            CV_PARSE_ERROR_CPP( "Key must start with \'\"\'" );

        char * beg = ptr + 1;
        do {
            if (*ptr == '\\') { // skip the next character if current is back slash
                ++ptr;
                CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();
                ++ptr;
                CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();
            } else {
                ++ptr;
                CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();
            }
        } while( cv_isprint(*ptr) && *ptr != '"' );

        if( *ptr != '"' )
            CV_PARSE_ERROR_CPP( "Key must end with \'\"\'" );

        if( ptr == beg )
            CV_PARSE_ERROR_CPP( "Key is empty" );
        value_placeholder = fs->addNode(collection, std::string(beg, (size_t)(ptr - beg)), FileNode::NONE);

        ptr++;
        ptr = skipSpaces( ptr );
        if( !ptr || !*ptr )
            return 0;

        if( *ptr != ':' )
            CV_PARSE_ERROR_CPP( "Missing \':\' between key and value" );

        return ++ptr;
    }

    bool getBase64Row(char* ptr, int /*indent*/, char* &beg, char* &end)
    {
        beg = end = ptr;
        if( !ptr || !*ptr )
            return false;

        // find end of the row
        while( cv_isprint(*ptr) && (*ptr != ',') && (*ptr != '"'))
            ++ptr;
        if ( *ptr == '\0' )
            CV_PARSE_ERROR_CPP( "Unexpected end of line" );

        end = ptr;
        return true;
    }

    char* parseValue( char* ptr, FileNode& node )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid value input");

        ptr = skipSpaces( ptr );
        if( !ptr || !*ptr )
            CV_PARSE_ERROR_CPP( "Unexpected End-Of-File" );

        if( *ptr == '"' )
        {   /* must be string or Base64 string */
            ptr++;
            char * beg = ptr;
            size_t len = 0u;
            for ( ; (cv_isalnum(*ptr) || *ptr == '$' ) && len <= 9u; ptr++ )
                len++;

            if ((len >= 8u) && (memcmp( beg, "$base64$", 8u ) == 0) )
            {   /**************** Base64 string ****************/
                ptr = beg + 8;
                ptr = fs->parseBase64(ptr, 0, node);

                if ( *ptr != '\"' )
                    CV_PARSE_ERROR_CPP( "'\"' - right-quote of string is missing" );
                else
                    ptr++;
            }
            else
            {   /**************** normal string ****************/
                int i = 0, sz;

                ptr = beg;
                bool is_matching = false;
                while ( !is_matching )
                {
                    switch ( *ptr )
                    {
                        case '\\':
                        {
                            sz = (int)(ptr - beg);
                            if( sz > 0 )
                            {
                                if (i + sz >= CV_FS_MAX_LEN)
                                    CV_PARSE_ERROR_CPP("string is too long");
                                memcpy(buf + i, beg, sz);
                                i += sz;
                            }
                            ptr++;
                            if (i + 1 >= CV_FS_MAX_LEN)
                                CV_PARSE_ERROR_CPP("string is too long");
                            switch ( *ptr )
                            {
                            case '\\':
                            case '\"':
                            case '\'': { buf[i++] = *ptr; break; }
                            case 'n' : { buf[i++] = '\n'; break; }
                            case 'r' : { buf[i++] = '\r'; break; }
                            case 't' : { buf[i++] = '\t'; break; }
                            case 'b' : { buf[i++] = '\b'; break; }
                            case 'f' : { buf[i++] = '\f'; break; }
                            case 'u' : { CV_PARSE_ERROR_CPP( "'\\uXXXX' currently not supported" ); break; }
                            default  : { CV_PARSE_ERROR_CPP( "Invalid escape character" ); }
                            break;
                            }
                            ptr++;
                            beg = ptr;
                            break;
                        }
                        case '\0':
                        {
                            sz = (int)(ptr - beg);
                            if( sz > 0 )
                            {
                                if (i + sz >= CV_FS_MAX_LEN)
                                    CV_PARSE_ERROR_CPP("string is too long");
                                memcpy(buf + i, beg, sz);
                                i += sz;
                            }
                            ptr = fs->gets();
                            if ( !ptr || !*ptr )
                                CV_PARSE_ERROR_CPP( "'\"' - right-quote of string is missing" );

                            beg = ptr;
                            break;
                        }
                        case '\"':
                        {
                            sz = (int)(ptr - beg);
                            if( sz > 0 )
                            {
                                if (i + sz >= CV_FS_MAX_LEN)
                                    CV_PARSE_ERROR_CPP("string is too long");
                                memcpy(buf + i, beg, sz);
                                i += sz;
                            }
                            beg = ptr;
                            is_matching = true;
                            break;
                        }
                        case '\n':
                        case '\r':
                        {
                            CV_PARSE_ERROR_CPP( "'\"' - right-quote of string is missing" );
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
                    CV_PARSE_ERROR_CPP( "'\"' - right-quote of string is missing" );
                else
                    ptr++;

                node.setValue(FileNode::STRING, buf, i);
            }
        }
        else if ( cv_isdigit(*ptr) || *ptr == '-' || *ptr == '+' || *ptr == '.' )
        {    /**************** number ****************/
            char * beg = ptr;
            if ( *ptr == '+' || *ptr == '-' )
            {
                ptr++;
                CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();
            }
            while( cv_isdigit(*ptr) )
            {
                ptr++;
                CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();
            }
            if (*ptr == '.' || *ptr == 'e')
            {
                double fval = fs->strtod( beg, &ptr );
                CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();

                node.setValue(FileNode::REAL, &fval);
            }
            else
            {
                int64_t ival = strtoll( beg, &ptr, 0 );
                CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();

                node.setValue(FileNode::INT, &ival);
            }

            if ( beg >= ptr )
                CV_PARSE_ERROR_CPP( "Invalid numeric value (inconsistent explicit type specification?)" );
        }
        else
        {    /**************** other data ****************/
            const char* beg = ptr;
            int len = 0;
            for ( ; cv_isalpha(*ptr) && len <= 6; )
            {
                len++;
                ptr++;
                CV_PERSISTENCE_CHECK_END_OF_BUFFER_BUG_CPP();
            }

            if( len == 4 && memcmp( beg, "null", 4 ) == 0 )
                ;
            else if( (len == 4 && memcmp( beg, "true", 4 ) == 0) ||
                     (len == 5 && memcmp( beg, "false", 5 ) == 0) )
            {
                int64_t ival = *beg == 't' ? 1 : 0;
                node.setValue(FileNode::INT, &ival);
            }
            else
            {
                CV_PARSE_ERROR_CPP( "Unrecognized value" );
            }
        }

        return ptr;
    }

    char* parseSeq( char* ptr, FileNode& node )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP( "ptr is NULL" );

        if ( *ptr != '[' )
            CV_PARSE_ERROR_CPP( "'[' - left-brace of seq is missing" );
        else
            ptr++;

        fs->convertToCollection(FileNode::SEQ, node);

        for (;;)
        {
            ptr = skipSpaces( ptr );
            if( !ptr || !*ptr )
                break;

            if ( *ptr != ']' )
            {
                FileNode child = fs->addNode(node, std::string(), FileNode::NONE );

                if ( *ptr == '[' )
                    ptr = parseSeq( ptr, child );
                else if ( *ptr == '{' )
                    ptr = parseMap( ptr, child );
                else
                    ptr = parseValue( ptr, child );
            }

            ptr = skipSpaces( ptr );
            if( !ptr || !*ptr )
                break;

            if ( *ptr == ',' )
                ptr++;
            else if ( *ptr == ']' )
                break;
            else
                CV_PARSE_ERROR_CPP( "Unexpected character" );
        }

        if (!ptr)
            CV_PARSE_ERROR_CPP("ptr is NULL");

        if ( *ptr != ']' )
            CV_PARSE_ERROR_CPP( "']' - right-brace of seq is missing" );
        else
            ptr++;

        fs->finalizeCollection(node);
        return ptr;
    }

    char* parseMap( char* ptr, FileNode& node )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("ptr is NULL");

        if ( *ptr != '{' )
            CV_PARSE_ERROR_CPP( "'{' - left-brace of map is missing" );
        else
            ptr++;

        fs->convertToCollection(FileNode::MAP, node);

        for( ;; )
        {
            ptr = skipSpaces( ptr );
            if( !ptr || !*ptr )
                break;

            if ( *ptr == '"' )
            {
                FileNode child;
                ptr = parseKey( ptr, node, child );
                if( !ptr || !*ptr )
                    break;
                ptr = skipSpaces( ptr );
                if( !ptr || !*ptr )
                    break;

                if ( *ptr == '[' )
                    ptr = parseSeq( ptr, child );
                else if ( *ptr == '{' )
                    ptr = parseMap( ptr, child );
                else
                    ptr = parseValue( ptr, child );
            }

            ptr = skipSpaces( ptr );
            if( !ptr || !*ptr )
                break;

            if ( *ptr == ',' )
                ptr++;
            else if ( *ptr == '}' )
                break;
            else
            {
                CV_PARSE_ERROR_CPP( "Unexpected character" );
            }
        }

        if (!ptr)
            CV_PARSE_ERROR_CPP("ptr is NULL");

        if ( *ptr != '}' )
            CV_PARSE_ERROR_CPP( "'}' - right-brace of map is missing" );
        else
            ptr++;

        fs->finalizeCollection(node);
        return ptr;
    }

    bool parse( char* ptr )
    {
        if (!ptr)
            CV_PARSE_ERROR_CPP("Invalid input");

        ptr = skipSpaces( ptr );
        if ( !ptr || !*ptr )
            return false;

        FileNode root_collection(fs->getFS(), 0, 0);

        if( *ptr == '{' )
        {
            FileNode root_node = fs->addNode(root_collection, std::string(), FileNode::MAP);
            parseMap( ptr, root_node );
        }
        else if ( *ptr == '[' )
        {
            FileNode root_node = fs->addNode(root_collection, std::string(), FileNode::SEQ);
            parseSeq( ptr, root_node );
        }
        else
        {
            CV_PARSE_ERROR_CPP( "left-brace of top level is missing" );
        }

        return true;
    }

    FileStorage_API* fs;
    char buf[CV_FS_MAX_LEN+1024];
};

Ptr<FileStorageEmitter> createJSONEmitter(FileStorage_API* fs)
{
    return makePtr<JSONEmitter>(fs);
}

Ptr<FileStorageParser> createJSONParser(FileStorage_API* fs)
{
    return makePtr<JSONParser>(fs);
}

}
