// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "persistence.hpp"
#include <unordered_map>
#include <iterator>

namespace cv
{

namespace fs
{

int strcasecmp(const char* s1, const char* s2)
{
    const char* dummy="";
    if(!s1) s1=dummy;
    if(!s2) s2=dummy;

    size_t len1 = strlen(s1);
    size_t len2 = strlen(s2);
    size_t i, len = std::min(len1, len2);
    for( i = 0; i < len; i++ )
    {
        int d = tolower((int)s1[i]) - tolower((int)s2[i]);
        if( d != 0 )
            return d;
    }
    return len1 < len2 ? -1 : len1 > len2 ? 1 : 0;
}

char* itoa( int _val, char* buffer, int /*radix*/ )
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

char* doubleToString( char* buf, double value, bool explicitZero )
{
    Cv64suf val;
    unsigned ieee754_hi;

    val.f = value;
    ieee754_hi = (unsigned)(val.u >> 32);

    if( (ieee754_hi & 0x7ff00000) != 0x7ff00000 )
    {
        int ivalue = cvRound(value);
        if( ivalue == value )
        {
            if( explicitZero )
                sprintf( buf, "%d.0", ivalue );
            else
                sprintf( buf, "%d.", ivalue );
        }
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

char* floatToString( char* buf, float value, bool halfprecision, bool explicitZero )
{
    Cv32suf val;
    unsigned ieee754;
    val.f = value;
    ieee754 = val.u;

    if( (ieee754 & 0x7f800000) != 0x7f800000 )
    {
        int ivalue = cvRound(value);
        if( ivalue == value )
        {
            if( explicitZero )
                sprintf( buf, "%d.0", ivalue );
            else
                sprintf( buf, "%d.", ivalue );
        }
        else
        {
            char* ptr = buf;
            if (halfprecision)
                sprintf(buf, "%.4e", value);
            else
                sprintf(buf, "%.8e", value);
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

static const char symbols[9] = "ucwsifdh";

static char typeSymbol(int depth)
{
    CV_StaticAssert(CV_64F == 6, "");
    CV_Assert(depth >=0 && depth <= CV_64F);
    return symbols[depth];
}

static int symbolToType(char c)
{
    const char* pos = strchr( symbols, c );
    if( !pos )
        CV_Error( CV_StsBadArg, "Invalid data type specification" );
    if (c == 'r')
        return CV_SEQ_ELTYPE_PTR;
    return static_cast<int>(pos - symbols);
}

char* encodeFormat(int elem_type, char* dt)
{
    int cn = (elem_type == CV_SEQ_ELTYPE_PTR/*CV_USRTYPE1*/) ? 1 : CV_MAT_CN(elem_type);
    char symbol = (elem_type == CV_SEQ_ELTYPE_PTR/*CV_USRTYPE1*/) ? 'r' : typeSymbol(CV_MAT_DEPTH(elem_type));
    sprintf(dt, "%d%c", cn, symbol);
    return dt + (cn == 1 ? 1 : 0);
}

int decodeFormat( const char* dt, int* fmt_pairs, int max_len )
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
            int depth = symbolToType(c);
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

int calcElemSize( const char* dt, int initial_size )
{
    int size = 0;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS], i, fmt_pair_count;
    int comp_size;

    fmt_pair_count = decodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
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


int calcStructSize( const char* dt, int initial_size )
{
    int size = calcElemSize( dt, initial_size );
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

int decodeSimpleFormat( const char* dt )
{
    int elem_type = -1;
    int fmt_pairs[CV_FS_MAX_FMT_PAIRS], fmt_pair_count;

    fmt_pair_count = decodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
    if( fmt_pair_count != 1 || fmt_pairs[0] >= CV_CN_MAX)
        CV_Error( CV_StsError, "Too complex format for the matrix" );

    elem_type = CV_MAKETYPE( fmt_pairs[1], fmt_pairs[0] );

    return elem_type;
}

}

#if defined __i386__ || defined(_M_IX86) || defined __x86_64__ || defined(_M_X64)
#define CV_UNALIGNED_LITTLE_ENDIAN_MEM_ACCESS 1
#else
#define CV_UNALIGNED_LITTLE_ENDIAN_MEM_ACCESS 0
#endif

static inline int readInt(const uchar* p)
{
#if CV_UNALIGNED_LITTLE_ENDIAN_MEM_ACCESS
    return *(const int*)p;
#else
    int val = (int)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
    return val;
#endif
}

static inline double readReal(const uchar* p)
{
#if CV_UNALIGNED_LITTLE_ENDIAN_MEM_ACCESS
    return *(const double*)p;
#else
    unsigned val0 = (unsigned)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
    unsigned val1 = (unsigned)(p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24));
    Cv64suf val;
    val.u = val0 | ((uint64)val1 << 32);
    return val.f;
#endif
}

static inline void writeInt(uchar* p, int ival)
{
#if CV_UNALIGNED_LITTLE_ENDIAN_MEM_ACCESS
    int* ip = (int*)p;
    *ip = ival;
#else
    p[0] = (uchar)ival;
    p[1] = (uchar)(ival >> 8);
    p[2] = (uchar)(ival >> 16);
    p[3] = (uchar)(ival >> 24);
#endif
}

static inline void writeReal(uchar* p, double fval)
{
#if CV_UNALIGNED_LITTLE_ENDIAN_MEM_ACCESS
    double* fp = (double*)p;
    *fp = fval;
#else
    Cv64suf v;
    v.f = fval;
    p[0] = (uchar)v.u;
    p[1] = (uchar)(v.u >> 8);
    p[2] = (uchar)(v.u >> 16);
    p[3] = (uchar)(v.u >> 24);
    p[4] = (uchar)(v.u >> 32);
    p[5] = (uchar)(v.u >> 40);
    p[6] = (uchar)(v.u >> 48);
    p[7] = (uchar)(v.u >> 56);
#endif
}

class FileStorage::Impl : public FileStorage_API
{
public:
    void init()
    {
        flags = 0;
        buffer.clear();
        bufofs = 0;
        state = UNDEFINED;
        is_opened = false;
        is_changed = false;
        dummy_eof = false;
        write_mode = false;
        rw_mode = false;
        save_mode = false;
        mem_mode = false;
        write_raw_base64 = false;
        space = 0;
        wrap_margin = 71;
        fmt = 0;
        file = 0;
        gzfile = 0;
        empty_stream = true;

        strbufv.clear();
        strbuf = 0;
        strbufsize = strbufpos = 0;
        roots.clear();

        fs_data.clear();
        fs_data_ptrs.clear();
        fs_data_blksz.clear();
        fs_data_insert_marks.clear();
        end_block = makePtr<FileNode>(fs_ext, 0, 0);
        current_block = end_block;

        str_hash.clear();
        str_hash_data.clear();
        str_hash_data.resize(1);
        str_hash_data[0] = '\0';

        filename.clear();
        lineno = 0;
    }

    Impl(FileStorage* _fs)
    {
        fs_ext = _fs;
        init();
    }

    virtual ~Impl()
    {
        release();
    }

    void release(String* out=0)
    {
        if( is_opened )
        {
            if(out)
                out->clear();
            if( write_mode )
            {
                while( write_stack.size() > 1 )
                {
                    endWriteStruct();
                }
                flush();
                if( fmt == FileStorage::FORMAT_XML )
                    puts( "</opencv_storage>\n" );
                else if ( fmt == FileStorage::FORMAT_JSON )
                    puts( "}\n" );
            }
            if( mem_mode && out )
            {
                *out = cv::String(outbuf.begin(), outbuf.end());
            }
            write_stack.clear();
        }
        closeFile();
        if ( !write_mode && is_changed )
            fs_ext->save();
        if ( !save_mode )
            init();
    }

    void analyze_file_name( const std::string& file_name, std::vector<std::string>& params )
    {
        params.clear();
        static const char not_file_name       = '\n';
        static const char parameter_begin     = '?';
        static const char parameter_separator = '&';

        if( file_name.find(not_file_name, (size_t)0) != std::string::npos )
            return;

        size_t beg = file_name.find_last_of(parameter_begin);
        params.push_back(file_name.substr((size_t)0, beg));

        if( beg != std::string::npos )
        {
            size_t end = file_name.size();
            beg++;
            for( size_t param_beg = beg, param_end = beg;
                 param_end < end;
                 param_beg = param_end + 1 )
            {
                param_end = file_name.find_first_of( parameter_separator, param_beg );
                if( (param_end == std::string::npos || param_end != param_beg) && param_beg + 1 < end )
                {
                    params.push_back( file_name.substr( param_beg, param_end - param_beg ) );
                }
            }
        }
    }

    bool open( const char* filename_or_buf, int _flags, const char* encoding )
    {
        write_raw_base64 = (_flags & FileStorage::BASE64) != 0;

        bool ok = true;
        if (!save_mode)
            release();

        mem_mode = (_flags & FileStorage::MEMORY) != 0;

        const int rw_flags = (_flags & (FileStorage::WRITE | FileStorage::APPEND));
        bool append = rw_flags == FileStorage::APPEND;
        write_mode = rw_flags == FileStorage::WRITE || append;
        rw_mode = rw_flags == FileStorage::READWRITE;

        bool isGZ = false;
        size_t fnamelen = 0;

        std::vector<std::string> params;
        //if ( !mem_mode )
        {
            analyze_file_name( filename_or_buf, params );
            if( !params.empty() )
                filename = params[0];

            /*if( !write_base64 && params.size() >= 2 &&
                std::find(params.begin()+1, params.end(), std::string("base64")) != params.end())
                write_base64 = (write_mode || append);*/
        }

        if( filename.size() == 0 && !mem_mode && !write_mode )
            CV_Error( CV_StsNullPtr, "NULL or empty filename" );

        if( mem_mode && append )
            CV_Error( CV_StsBadFlag, "FileStorage::APPEND and FileStorage::MEMORY are not currently compatible" );

        flags = _flags;

        if( !mem_mode )
        {
            char* dot_pos = strrchr((char*)filename.c_str(), '.');
            char compression = '\0';

            if( dot_pos && dot_pos[1] == 'g' && dot_pos[2] == 'z' &&
               (dot_pos[3] == '\0' || (cv_isdigit(dot_pos[3]) && dot_pos[4] == '\0')) )
            {
                if( append )
                {
                    CV_Error(CV_StsNotImplemented, "Appending data to compressed file is not implemented" );
                }
                isGZ = true;
                compression = dot_pos[3];
                if( compression )
                    dot_pos[3] = '\0', fnamelen--;
            }

            if( !isGZ )
            {
                String fopen_mode;
                if( append )
                    fopen_mode = "a+t";
                else if( write_mode )
                    fopen_mode = "wt";
                else // READ or READWRITE mode
                    fopen_mode = "rt";
                file = fopen(filename.c_str(), fopen_mode.c_str());
                if( !file && !rw_mode )
                    return false;
            }
            else
            {
#if USE_ZLIB
                char mode[] = { write_mode ? 'w' : 'r', 'b', compression ? compression : '3', '\0' };
                gzfile = gzopen(filename.c_str(), mode);
                if( !gzfile )
                    return false;
#else
                CV_Error(CV_StsNotImplemented, "There is no compressed file storage support in this configuration");
#endif
            }
        }

        if(!save_mode)
        {
            roots.clear();
            fs_data.clear();
        }
        wrap_margin = 71;

        fmt = flags & FileStorage::FORMAT_MASK;
        if( fmt == FileStorage::FORMAT_AUTO && !filename.empty() )
        {
            const char* dot_pos = NULL;
            const char* dot_pos2 = NULL;
            // like strrchr() implementation, but save two last positions simultaneously
            for (const char* pos = &filename[0]; pos[0] != 0; pos++)
            {
                if( pos[0] == '.' )
                {
                    dot_pos2 = dot_pos;
                    dot_pos = pos;
                }
            }
            if (fs::strcasecmp(dot_pos, ".gz") == 0 && dot_pos2 != NULL)
            {
                dot_pos = dot_pos2;
            }
            if( fs::strcasecmp(dot_pos, ".xml") == 0 || fs::strcasecmp(dot_pos, ".xml.gz") == 0 )
                fmt = FileStorage::FORMAT_XML;
            else if ( fs::strcasecmp(dot_pos, ".json") == 0 || fs::strcasecmp(dot_pos, ".json.gz") == 0 )
                fmt = FileStorage::FORMAT_JSON;
            else if ( fs::strcasecmp(dot_pos, ".yml") == 0 || fs::strcasecmp(dot_pos, ".yml.gz") == 0
                    || fs::strcasecmp(dot_pos, ".yaml") == 0 || fs::strcasecmp(dot_pos, ".yaml.gz") == 0 )
            {
                fmt = FileStorage::FORMAT_YAML;
            }
            // Write default for unknown filenames
            if( write_mode && fmt == FORMAT_AUTO )
                fmt = FileStorage::FORMAT_YAML;
        }
        else if( write_mode && fmt == FORMAT_AUTO )
        {
            // Write default for empty filename
            fmt = FileStorage::FORMAT_XML;
        }

        if( write_mode )
        {
            if( mem_mode )
                outbuf.clear();

            // we use factor=6 for XML (the longest characters (' and ") are encoded with 6 bytes (&apos; and &quot;)
            // and factor=4 for YAML ( as we use 4 bytes for non ASCII characters (e.g. \xAB))
            int buf_size = CV_FS_MAX_LEN*(fmt == FileStorage::FORMAT_XML ? 6 : 4) + 1024;

            if (append)
            {
                fseek( file, 0, SEEK_END );
                if (ftell(file) == 0)
                    append = false;
            }

            write_stack.clear();
            empty_stream = true;
            write_stack.push_back(FStructData("", FileNode::MAP | FileNode::EMPTY, 0));
            buffer.reserve(buf_size + 1024);
            buffer.resize(buf_size);
            bufofs = 0;

            if( fmt == FileStorage::FORMAT_XML )
            {
                size_t file_size = file ? (size_t)ftell(file) : (size_t)0;
                if( !append || file_size == 0 )
                {
                    if( encoding && *encoding != '\0' )
                    {
                        if( fs::strcasecmp(encoding, "UTF-16" ) == 0 )
                        {
                            release();
                            CV_Error( CV_StsBadArg, "UTF-16 XML encoding is not supported! Use 8-bit encoding\n");
                        }

                        CV_Assert( strlen(encoding) < 1000 );
                        char buf[1100];
                        sprintf(buf, "<?xml version=\"1.0\" encoding=\"%s\"?>\n", encoding);
                        puts( buf );
                    }
                    else
                        puts( "<?xml version=\"1.0\"?>\n" );
                    puts( "<opencv_storage>\n" );
                }
                else
                {
                    int xml_buf_size = 1 << 10;
                    char substr[] = "</opencv_storage>";
                    int last_occurrence = -1;
                    xml_buf_size = MIN(xml_buf_size, int(file_size));
                    fseek( file, -xml_buf_size, SEEK_END );
                    // find the last occurrence of </opencv_storage>
                    for(;;)
                    {
                        int line_offset = (int)ftell( file );
                        const char* ptr0 = this->gets(xml_buf_size);
                        const char* ptr = NULL;
                        if( !ptr0 )
                            break;
                        ptr = ptr0;
                        for(;;)
                        {
                            ptr = strstr( ptr, substr );
                            if( !ptr )
                                break;
                            last_occurrence = line_offset + (int)(ptr - ptr0);
                            ptr += strlen(substr);
                        }
                    }
                    if( last_occurrence < 0 )
                    {
                        release();
                        CV_Error( CV_StsError, "Could not find </opencv_storage> in the end of file.\n" );
                    }
                    closeFile();
                    file = fopen( filename.c_str(), "r+t" );
                    CV_Assert(file != 0);
                    fseek( file, last_occurrence, SEEK_SET );
                    // replace the last "</opencv_storage>" with " <!-- resumed -->", which has the same length
                    puts( " <!-- resumed -->" );
                    fseek( file, 0, SEEK_END );
                    puts( "\n" );
                }

                emitter = createXMLEmitter(this);
            }
            else if( fmt == FileStorage::FORMAT_YAML )
            {
                if( !append)
                    puts( "%YAML:1.0\n---\n" );
                else
                    puts( "...\n---\n" );

                emitter = createYAMLEmitter(this);
            }
            else
            {
                CV_Assert( fmt == FileStorage::FORMAT_JSON );
                if( !append )
                    puts( "{\n" );
                else
                {
                    bool valid = false;
                    long roffset = 0;
                    for ( ;
                         fseek( file, roffset, SEEK_END ) == 0;
                         roffset -= 1 )
                    {
                        const char end_mark = '}';
                        if ( fgetc( file ) == end_mark )
                        {
                            fseek( file, roffset, SEEK_END );
                            valid = true;
                            break;
                        }
                    }

                    if ( valid )
                    {
                        closeFile();
                        file = fopen( filename.c_str(), "r+t" );
                        CV_Assert(file != 0);
                        fseek( file, roffset, SEEK_END );
                        fputs( ",", file );
                    }
                    else
                    {
                        CV_Error( CV_StsError, "Could not find '}' in the end of file.\n" );
                    }
                }
                write_stack.back().indent = 4;
                emitter = createJSONEmitter(this);
            }
            is_opened = true;
        }
        else
        {
            emitter = createFileNodeEmitter(this);
            state = NAME_EXPECTED;
            const size_t buf_size0 = 40;
            buffer.resize(buf_size0);
            if( mem_mode )
            {
                strbuf = (char*)filename_or_buf;
                strbufsize = strlen(strbuf);
            }

            if( rw_mode && !file && !gzfile && fmt == FileStorage::FORMAT_AUTO )
            {
                fmt = FileStorage::FORMAT_XML;
            }
            else
            {
                char* buf = this->gets(16);
                CV_Assert(buf);
                char* bufPtr = cv_skip_BOM(buf);
                size_t bufOffset = bufPtr - buf;

                if( fmt == FileStorage::FORMAT_AUTO )
                {
                    const char* yaml_signature = "%YAML";
                    const char* json_signature = "{";
                    const char* xml_signature  = "<?xml";
                    if(strncmp( bufPtr, yaml_signature, strlen(yaml_signature) ) == 0)
                        fmt = FileStorage::FORMAT_YAML;
                    else if(strncmp( bufPtr, json_signature, strlen(json_signature) ) == 0)
                        fmt = FileStorage::FORMAT_JSON;
                    else if(strncmp( bufPtr, xml_signature, strlen(xml_signature) ) == 0)
                        fmt = FileStorage::FORMAT_XML;
                    else if(strbufsize  == bufOffset)
                        CV_Error(CV_BADARG_ERR, "Input file is invalid");
                    else
                        CV_Error(CV_BADARG_ERR, "Unsupported file storage format");
                }
                rewind();
                strbufpos = bufOffset;
            }
            bufofs = 0;

            try
            {
                char* ptr = bufferStart();
                ptr[0] = ptr[1] = ptr[2] = '\0';
                FileNode root_nodes(fs_ext, 0, 0);

                uchar* rptr = reserveNodeSpace(root_nodes, 9);
                *rptr = FileNode::SEQ;
                writeInt(rptr + 1, 4);
                writeInt(rptr + 5, 0);
                FStructData fsd;
                fsd.startNode = root_nodes;
                write_stack.push_back(fsd);

                roots.clear();

                switch (fmt)
                {
                    case FileStorage::FORMAT_XML: parser = createXMLParser(this); break;
                    case FileStorage::FORMAT_YAML: parser = createYAMLParser(this); break;
                    case FileStorage::FORMAT_JSON: parser = createJSONParser(this); break;
                    default: parser = Ptr<FileStorageParser>();
                }

                if( !parser.empty() )
                {
                    if( rw_mode && !file && !gzfile )
                    {
                        FileNode top_level_map = addNode(root_nodes, std::string(), FileNode::MAP, nullptr, 0);
                        finalizeCollection(top_level_map);
                    }
                    else
                    {
                        ok = parser->parse(ptr);
                    }
                    if( ok )
                    {
                        finalizeCollection(root_nodes);
                        write_stack.clear();
                        current_block.reset();

                        CV_Assert( !fs_data_ptrs.empty() );
                        FileNode roots_node(fs_ext, 0, 0);
                        size_t i, nroots = roots_node.size();
                        FileNodeIterator it = roots_node.begin();

                        for( i = 0; i < nroots; i++, ++it )
                            roots.push_back(*it);
                    }
                }
            }
            catch(...)
            {
                is_opened = true;
                release();
                throw;
            }

            // release resources that we do not need anymore
            closeFile();
            is_opened = true;
            std::vector<char> tmpbuf;
            std::swap(buffer, tmpbuf);
            bufofs = 0;
        }
        return ok;
    }

    void puts( const char* str )
    {
        CV_Assert( write_mode );
        if( mem_mode )
            std::copy(str, str + strlen(str), std::back_inserter(outbuf));
        else if( file )
            fputs( str, file );
#if USE_ZLIB
        else if( gzfile )
            gzputs( gzfile, str );
#endif
        else
            CV_Error( CV_StsError, "The storage is not opened" );
    }

    char* getsFromFile( char* buf, int count )
    {
        if( file )
            return fgets( buf, count, file );
    #if USE_ZLIB
        if( gzfile )
            return gzgets( gzfile, buf, count );
    #endif
        CV_Error(CV_StsError, "The storage is not opened");
    }

    char* gets( size_t maxCount )
    {
        if( strbuf )
        {
            size_t i = strbufpos, len = strbufsize;
            const char* instr = strbuf;
            for( ; i < len; i++ )
            {
                char c = instr[i];
                if( c == '\0' || c == '\n' )
                {
                    if( c == '\n' )
                        i++;
                    break;
                }
            }
            size_t count = i - strbufpos;
            if( maxCount == 0 || maxCount > count )
                maxCount = count;
            buffer.resize(std::max(buffer.size(), maxCount + 8));
            memcpy(&buffer[0], instr + strbufpos, maxCount);
            buffer[maxCount] = '\0';
            strbufpos = i;
            return maxCount > 0 ? &buffer[0] : 0;
        }

        const size_t MAX_BLOCK_SIZE = INT_MAX/2; // hopefully, that will be enough
        if( maxCount == 0 )
            maxCount = MAX_BLOCK_SIZE;
        else
            CV_Assert(maxCount < MAX_BLOCK_SIZE);
        size_t ofs = 0;

        for(;;)
        {
            int count = (int)std::min(buffer.size() - ofs - 16, maxCount);
            char* ptr = getsFromFile( &buffer[ofs], count+1 );
            if( !ptr )
                break;
            int delta = (int)strlen(ptr);
            ofs += delta;
            maxCount -= delta;
            if( ptr[delta-1] == '\n' || maxCount == 0 )
                break;
            if( delta == count )
                buffer.resize((size_t)(buffer.size()*1.5));
        }
        return ofs > 0 ? &buffer[0] : 0;
    }

    char* gets()
    {
        char* ptr = this->gets(0);
        if( !ptr )
        {
            ptr = bufferStart();  // FIXIT Why do we need this hack? What is about other parsers JSON/YAML?
            *ptr = '\0';
            setEof();
            return 0;
        }
        else
        {
            size_t l = strlen(ptr);
            if( l > 0 && ptr[l-1] != '\n' && ptr[l-1] != '\r' && !eof() )
            {
                ptr[l] = '\n';
                ptr[l+1] = '\0';
            }
        }
        lineno++;
        return ptr;
    }

    bool eof()
    {
        if( dummy_eof )
            return true;
        if( strbuf )
            return strbufpos >= strbufsize;
        if( file )
            return feof(file) != 0;
#if USE_ZLIB
        if( gzfile )
            return gzeof(gzfile) != 0;
#endif
        return false;
    }

    void setEof()
    {
        dummy_eof = true;
    }

    void closeFile()
    {
        if( file )
            fclose( file );
#if USE_ZLIB
        else if( gzfile )
            gzclose( gzfile );
#endif
        file = 0;
        gzfile = 0;
        strbuf = 0;
        strbufpos = 0;
        is_opened = false;
    }

    void rewind()
    {
        if( file )
            ::rewind(file);
#if USE_ZLIB
        else if( gzfile )
            gzrewind(gzfile);
#endif
        strbufpos = 0;
    }

    char* resizeWriteBuffer( char* ptr, int len )
    {
        const char* buffer_end = &buffer[0] + buffer.size();
        if( ptr + len < buffer_end )
            return ptr;

        const char* buffer_start = &buffer[0];
        int written_len = (int)(ptr - buffer_start);

        CV_Assert(written_len <= (int)buffer.size());
        int new_size = (int)((buffer_end - buffer_start)*3/2);
        new_size = MAX( written_len + len, new_size );
        buffer.reserve( new_size + 256 );
        buffer.resize( new_size );
        bufofs = written_len;
        return &buffer[0] + bufofs;
    }

    char* flush()
    {
        char* buffer_start = &buffer[0];
        char* ptr = buffer_start + bufofs;

        if( ptr > buffer_start + space )
        {
            ptr[0] = '\n';
            ptr[1] = '\0';
            puts( buffer_start );
            bufofs = 0;
        }

        int indent = ( write_stack.empty() ? 0 : write_stack.back().indent );

        if( space != indent )
        {
            memset( buffer_start, ' ', indent );
            space = indent;
        }
        bufofs = space;
        ptr = buffer_start + bufofs;

        return ptr;
    }

    void endWriteStruct()
    {
        CV_Assert( write_mode || rw_mode );
        CV_Assert( !write_stack.empty() );

        FStructData& current_struct = write_stack.back();
        if( fmt == FileStorage::FORMAT_JSON && !FileNode::isFlow(current_struct.flags) && write_stack.size() > 1 )
            current_struct.indent = write_stack[write_stack.size() - 2].indent;

        const FileNode &collection = current_struct.startNode;
        size_t orig_capacity = current_struct.startNode.capacity();
        emitter->endWriteStruct(current_struct);

        write_stack.pop_back();
        const int sz_change = static_cast<int>(collection.capacity() - orig_capacity);
        if( !write_stack.empty() )
        {
            write_stack.back().flags &= ~FileNode::EMPTY;
        }
        else
        {
            if( sz_change != 0 )
            {
                Ptr<FileNode> n = collection.parent;
                while ( n )
                {
                    CV_Assert( n->isCollection() );
                    uchar *ptr = n->ptr();
                    ptr += ( n->isNamed() ? 5 : 1 );
                    orig_capacity = readInt(ptr);
                    writeInt(ptr, static_cast<int>(orig_capacity) + sz_change);
                    n = n->parent;
                }
            }
            current_block.reset();
        }
    }

    void startWriteStruct( const char* key, int struct_flags,
                           const char* type_name )
    {
        CV_Assert( write_mode || rw_mode );
        struct_flags = (struct_flags & (FileNode::TYPE_MASK|FileNode::FLOW)) | FileNode::EMPTY;
        if( !FileNode::isCollection(struct_flags))
            CV_Error( CV_StsBadArg,
                     "Some collection type: FileNode::SEQ or FileNode::MAP must be specified" );

        if( type_name && type_name[0] == '\0' )
            type_name = 0;

        FStructData s = emitter->startWriteStruct( write_stack.empty() ? FStructData() : write_stack.back(),
                key, struct_flags, type_name );
        write_stack.push_back(s);
        size_t write_stack_size = write_stack.size();
        if( write_stack_size > 1 )
            write_stack[write_stack_size-2].flags &= ~FileNode::EMPTY;

        if( !FileNode::isFlow(s.flags) )
            flush();

        if( fmt == FileStorage::FORMAT_JSON && type_name && type_name[0] && FileNode::isMap(struct_flags))
        {
            emitter->write("type_id", type_name, false);
        }
    }

    void writeComment( const char* comment, bool eol_comment )
    {
        CV_Assert(write_mode);
        emitter->writeComment( comment, eol_comment );
    }

    void startNextStream()
    {
        CV_Assert(write_mode);
        if( !empty_stream )
        {
            while( !write_stack.empty() )
                endWriteStruct();
            flush();
            emitter->startNextStream();
            empty_stream = true;
            write_stack.push_back(FStructData("", FileNode::MAP | FileNode::EMPTY, 0));
            bufofs = 0;
        }
    }

    void write( const String& key, int value )
    {
        CV_Assert( write_mode || rw_mode );
        emitter->write(key.c_str(), value);
    }

    void write( const String& key, double value )
    {
        CV_Assert( write_mode || rw_mode );
        emitter->write(key.c_str(), value);
    }

    void write( const String& key, const String& value )
    {
        CV_Assert( write_mode || rw_mode );
        emitter->write(key.c_str(), value.c_str(), false);
    }

    void writeRawData( const std::string& dt, const void* _data, size_t len )
    {
        CV_Assert( write_mode || rw_mode );
        size_t elemSize = fs::calcStructSize(dt.c_str(), 0);
        CV_Assert( len % elemSize == 0 );
        len /= elemSize;

        bool explicitZero = fmt == FileStorage::FORMAT_JSON;
        const uchar* data0 = (const uchar*)_data;
        int fmt_pairs[CV_FS_MAX_FMT_PAIRS*2], k, fmt_pair_count;
        char buf[256] = "";

        fmt_pair_count = fs::decodeFormat( dt.c_str(), fmt_pairs, CV_FS_MAX_FMT_PAIRS );

        if( !len )
            return;

        if( !data0 )
            CV_Error( CV_StsNullPtr, "Null data pointer" );

        if( fmt_pair_count == 1 )
        {
            fmt_pairs[0] *= (int)len;
            len = 1;
        }

        for(;len--; data0 += elemSize)
        {
            int offset = 0;
            for( k = 0; k < fmt_pair_count; k++ )
            {
                int i, count = fmt_pairs[k*2];
                int elem_type = fmt_pairs[k*2+1];
                int elem_size = CV_ELEM_SIZE(elem_type);
                const char *ptr;

                offset = cvAlign( offset, elem_size );
                const uchar* data = data0 + offset;

                for( i = 0; i < count; i++ )
                {
                    if (write_mode)
                    {
                        switch( elem_type )
                        {
                        case CV_8U:
                            ptr = fs::itoa( *(uchar*)data, buf, 10 );
                            data++;
                            break;
                        case CV_8S:
                            ptr = fs::itoa( *(char*)data, buf, 10 );
                            data++;
                            break;
                        case CV_16U:
                            ptr = fs::itoa( *(ushort*)data, buf, 10 );
                            data += sizeof(ushort);
                            break;
                        case CV_16S:
                            ptr = fs::itoa( *(short*)data, buf, 10 );
                            data += sizeof(short);
                            break;
                        case CV_32S:
                            ptr = fs::itoa( *(int*)data, buf, 10 );
                            data += sizeof(int);
                            break;
                        case CV_32F:
                            ptr = fs::floatToString( buf, *(float*)data, false, explicitZero );
                            data += sizeof(float);
                            break;
                        case CV_64F:
                            ptr = fs::doubleToString( buf, *(double*)data, explicitZero );
                            data += sizeof(double);
                            break;
                        case CV_16F: /* reference */
                            ptr = fs::floatToString( buf, (float)*(float16_t*)data, true, explicitZero );
                            data += sizeof(float16_t);
                            break;
                        default:
                            CV_Error( CV_StsUnsupportedFormat, "Unsupported type" );
                            return;
                        }

                        emitter->writeScalar(0, ptr);
                    }
                    else
                    {
                        int int_value;
                        double real_value;
                        int node_type;
                        switch( elem_type )
                        {
                        case CV_8U:
                            int_value = *(uchar*)data;
                            node_type = FileNode::INT;
                            data++;
                            break;
                        case CV_8S:
                            int_value = *(char*)data;
                            node_type = FileNode::INT;
                            data++;
                            break;
                        case CV_16U:
                            int_value = *(ushort*)data;
                            node_type = FileNode::INT;
                            data += sizeof(ushort);
                            break;
                        case CV_16S:
                            int_value = *(short*)data;
                            node_type = FileNode::INT;
                            data += sizeof(short);
                            break;
                        case CV_32S:
                            int_value = *data;
                            node_type = FileNode::INT;
                            data += sizeof(int);
                            break;
                        case CV_32F:
                            real_value = *(float*)data;
                            node_type = FileNode::REAL;
                            data += sizeof(float);
                            break;
                        case CV_64F:
                            real_value = *(double*)data;
                            node_type = FileNode::REAL;
                            data += sizeof(double);
                            break;
                        case CV_16F: /* reference */
                            real_value = (float)*(float16_t*)data;
                            node_type = FileNode::REAL;
                            data += sizeof(float16_t);
                            break;
                        default:
                            CV_Error( CV_StsUnsupportedFormat, "Unsupported type" );
                            return;
                        }
                        if ( isBusy() )
                        {
                            if ( node_type == FileNode::INT )
                                addNode(write_stack.back().startNode, "", node_type, &int_value);
                            else
                                addNode(write_stack.back().startNode, "", node_type, &real_value);
                        }
                        else
                        {
                            if ( node_type == FileNode::INT )
                                current_block->setValue(node_type, &int_value);
                            else
                                current_block->setValue(node_type, &real_value);
                        }
                    }
                }

                offset = (int)(data - data0);
            }
        }
    }

    void writeRawDataBase64(const void* /*data*/, int /*len*/, const char* /*dt*/ )
    {

    }

    String releaseAndGetString();

    FileNode getFirstTopLevelNode() const
    {
        return roots.empty() ? FileNode() : roots[0];
    }

    FileNode root(int streamIdx=0) const
    {
        return streamIdx >= 0 && streamIdx < (int)roots.size() ? roots[streamIdx] : FileNode();
    }

    int getFormat() const { return fmt; }

    char* bufferPtr() const { return (char*)(&buffer[0] + bufofs); }
    char* bufferStart() const { return (char*)&buffer[0]; }
    char* bufferEnd() const { return (char*)(&buffer[0] + buffer.size()); }
    void setBufferPtr(char* ptr)
    {
        char* bufferstart = bufferStart();
        CV_Assert( ptr >= bufferstart && ptr <= bufferEnd() );
        bufofs = ptr - bufferstart;
    }
    int wrapMargin() const { return wrap_margin; }

    bool isBusy() const { return !write_stack.empty(); }

    FStructData& getCurrentStruct()
    {
        CV_Assert(!write_stack.empty());
        return write_stack.back();
    }

    void setNonEmpty()
    {
        empty_stream = false;
    }

    void processSpecialDouble( char* buf, double* value, char** endptr )
    {
        FileStorage_API* fs = this;
        char c = buf[0];
        int inf_hi = 0x7ff00000;

        if( c == '-' || c == '+' )
        {
            inf_hi = c == '-' ? 0xfff00000 : 0x7ff00000;
            c = *++buf;
        }

        if( c != '.' )
            CV_PARSE_ERROR_CPP( "Bad format of floating-point constant" );

        Cv64suf v;
        v.f = 0.;
        if( toupper(buf[1]) == 'I' && toupper(buf[2]) == 'N' && toupper(buf[3]) == 'F' )
            v.u = (uint64)inf_hi << 32;
        else if( toupper(buf[1]) == 'N' && toupper(buf[2]) == 'A' && toupper(buf[3]) == 'N' )
            v.u = (uint64)-1;
        else
            CV_PARSE_ERROR_CPP( "Bad format of floating-point constant" );
        *value = v.f;
        *endptr = buf + 4;
    }

    double strtod( char* ptr, char** endptr )
    {
        double fval = ::strtod( ptr, endptr );
        if( **endptr == '.' )
        {
            char* dot_pos = *endptr;
            *dot_pos = ',';
            double fval2 = ::strtod( ptr, endptr );
            *dot_pos = '.';
            if( *endptr > dot_pos )
                fval = fval2;
            else
                *endptr = dot_pos;
        }

        if( *endptr == ptr || cv_isalpha(**endptr) )
            processSpecialDouble( ptr, &fval, endptr );

        return fval;
    }

    void convertToCollection(int type, FileNode& node)
    {
        const int plain_type = type & FileNode::TYPE_MASK;
        CV_Assert( plain_type == FileNode::SEQ || plain_type == FileNode::MAP );

        int node_type = node.type();
        if( node_type == plain_type )
        {
            *( node.ptr() ) |= type;
            return;
        }

        bool named = node.isNamed();
        uchar* ptr = node.ptr() + 1 + (named ? 4 : 0);

        int ival = 0;
        double fval = 0;
        std::string sval;
        bool add_first_scalar = false;

        if( node_type != FileNode::NONE )
        {
            // scalar nodes can only be converted to sequences, e.g. in XML:
            // <a>5[parser_position]... => create 5 with name "a"
            // <a>5 6[parser_position]... => 5 is converted to [5] and then 6 is added to it
            //
            // otherwise we don't know where to get the element names from
            CV_Assert( type == FileNode::SEQ );
            if( node_type == FileNode::INT )
            {
                ival = readInt(ptr);
                add_first_scalar = true;
            }
            else if( node_type == FileNode::REAL )
            {
                fval = readReal(ptr);
                add_first_scalar = true;
            }
            else if( node_type == FileNode::STRING )
            {
                sval = std::string(node);
                add_first_scalar = true;
            }
            else
                CV_Error_(Error::StsError, ("The node of type %d cannot be converted to collection", node_type));
        }

        ptr = reserveNodeSpace(node, 1 + (named ? 4 : 0) + 4 + 4);
        *ptr++ = (uchar)(type | (named ? FileNode::NAMED : 0));
        // name has been copied automatically
        if( named )
            ptr += 4;
        // set raw_size(collection)=4, nelems(collection)=0
        writeInt(ptr, 4);
        writeInt(ptr + 4, 0);

        if( add_first_scalar )
            addNode(node, std::string(), node_type,
                    node_type == FileNode::INT ? (const void*)&ival :
                    node_type == FileNode::REAL ? (const void*)&fval :
                    node_type == FileNode::STRING ? (const void*)sval.c_str() : 0,
                    -1);
    }

    // update capacity of parent collections
    void updateCollectionCapacities(const int& sz_change)
    {
        CV_Assert( isBusy() );
        if( !sz_change )
            return;
        FileNode *collection;
        uchar* capacity_ptr;
        int capacity;
        unsigned write_stack_hierarchy = 0;
        while( write_stack.size() > write_stack_hierarchy )
        {
            write_stack_hierarchy++;
            const int write_stack_idx = static_cast<int>(write_stack.size()) - write_stack_hierarchy;
            collection = &write_stack[write_stack_idx].startNode;
            capacity_ptr = collection->ptr() + ( collection->isNamed() ? 5 : 1 );
            capacity = readInt(capacity_ptr);
            capacity += sz_change;
            writeInt(capacity_ptr, capacity);
        }
        collection = write_stack.front().startNode.parent.get();
        while ( collection )
        {
            capacity_ptr = collection->ptr() + ( collection->isNamed() ? 5 : 1 );
            capacity = readInt(capacity_ptr);
            capacity += sz_change;
            writeInt(capacity_ptr, capacity);
            collection = collection->parent.get();
        }
    }

    // a) allocates new FileNode (for that just set blockIdx and ofs to the current block values) or
    // b) reallocates just created new node (blockIdx and ofs must be taken from FileNode).
    //    If there is no enough space in the current block (it should be the last block added so far),
    //    the last block is shrunk so that it ends immediately before the reallocated node. Then,
    //    a new block of sufficient size is allocated and the FileNode is placed in the beginning of it.
    // The case (a) can be used to allocate the very first node by setting blockIdx == ofs == 0.
    // In the case (b) the existing tag and the name are copied automatically.
    uchar* reserveNodeSpace(FileNode& node, size_t sz)
    {
        const bool is_reassign = ( !current_block || current_block != end_block );
        bool shrinkBlock = false;
        size_t shrinkBlockIdx = 0, shrinkSize = 0;

        uchar *ptr = 0, *blockEnd = 0;
        size_t blockIdx = node.blockIdx;

        if( !fs_data_ptrs.empty() )
        {
            size_t ofs = node.ofs;
            CV_Assert( blockIdx == current_block->blockIdx );
            CV_Assert( ofs <= fs_data_blksz[blockIdx] );

            ptr = fs_data_ptrs[blockIdx] + ofs;

            if( is_reassign )
            {
                int sz_change = 0;
                const size_t next_block_idx = nextBlockIdx(blockIdx);
                const int node_type = node.type();
                const bool is_pseudocollection = ( !isBusy() && node_type == FileNode::STRING );
                FileNode node_end = node;
                node_end.ofs += sz;
                normalizeNodeOfs(node_end.blockIdx, node_end.ofs);

                if( !isBusy() && !is_pseudocollection )
                {
                    CV_Assert( node.rawSize() == sz );
                }
                else
                {
                    if( is_pseudocollection )
                    {
                        FStructData fsd;
                        fsd.startNode = node;
                        fsd.startNode.nodeName = "<temp-string-struct>";
                        write_stack.push_back(fsd);
                    }
                    FileNode &collection = write_stack.back().startNode;
                    const size_t orig_collection_sz = collection.rawSize();
                    FileNode orig_collection_end = collection;
                    orig_collection_end.ofs += orig_collection_sz;

                    normalizeNodeOfs(orig_collection_end.blockIdx, orig_collection_end.ofs, node_end.blockIdx);

                    //fits already in allocated space
                    if( node_end.blockIdx == orig_collection_end.blockIdx && node_end.ofs <= orig_collection_end.ofs )
                    {
                        normalizeNodeOfs(orig_collection_end.blockIdx, orig_collection_end.ofs);
                        // ...and in current block
                        if( blockIdx == node_end.blockIdx )
                        {
                            current_block->ofs += sz;
                            if( is_pseudocollection )
                            {
                                sz_change -= static_cast<int>(orig_collection_sz - sz);
                                finalizeCollection(collection);
                                updateCollectionCapacities(sz_change);
                                write_stack.pop_back();
                            }
                            return ptr;
                        }
                        // ...or in already allocated space of next block
                        if( sz < fs_data_blksz[next_block_idx]
                                && ( orig_collection_end.blockIdx != next_block_idx || orig_collection_end.ofs > sz ) )
                        {
                            CV_Assert( !is_pseudocollection );
                            // allocate node in next block & shrink current block
                            node.blockIdx = next_block_idx;
                            node.ofs = 0;
                            current_block->blockIdx = next_block_idx;
                            current_block->ofs = sz;
                            sz_change += static_cast<int>(ofs - fs_data_blksz[blockIdx]);
                            fs_data[blockIdx]->resize(ofs);
                            fs_data_blksz[blockIdx] = ofs;
                            updateCollectionCapacities(sz_change);
                            return node.ptr();
                        }
                    }

                    size_t new_idx, blockSize;
                    const size_t orig_blksz = fs_data_blksz[blockIdx];
                    if( blockIdx == end_block->blockIdx && orig_blksz - end_block->ofs >= sz )
                    {
                        new_idx = blockIdx;
                        blockSize = orig_blksz;
                    }
                    else if( fs_data_blksz[blockIdx] + sz < CV_FS_MAX_LEN*4 )
                    {
                        node_end.blockIdx = blockIdx;
                        node_end.ofs = ofs + sz;
                        blockSize = CV_FS_MAX_LEN*4;
                        fs_data[blockIdx]->resize(blockSize);
                        fs_data_ptrs[blockIdx] = &fs_data[blockIdx]->at(0);
                        sz_change += static_cast<int>(blockSize - orig_blksz);
                        fs_data_blksz[blockIdx] = blockSize;
                        new_idx = blockIdx;
                    }
                    else
                    {
                        // otherwise insert new block
                        new_idx = fs_data_ptrs.size();
                        markDataInsert(blockIdx, new_idx);
                        blockSize = std::max((size_t)CV_FS_MAX_LEN*4 - 256, sz) + 256;
                        Ptr<std::vector<uchar> > pv = makePtr<std::vector<uchar> >(blockSize);
                        fs_data.push_back(pv);
                        ptr = &pv->at(0);
                        fs_data_ptrs.push_back(ptr);
                        fs_data_blksz.push_back(blockSize);
                        sz_change += static_cast<int>(blockSize);
                    }

                    if( blockIdx == orig_collection_end.blockIdx )
                    {
                        //move data after collection
                        uchar *orig_data_begin = orig_collection_end.ptr();
                        const size_t orig_data_end_ofs = ( blockIdx == end_block->blockIdx ? end_block->ofs : orig_blksz );
                        sz_change += static_cast<int>(orig_blksz - orig_data_end_ofs);
                        const size_t data_length = orig_data_end_ofs - orig_collection_end.ofs;
                        const size_t free_space = fs_data_blksz[new_idx] - ( node_end.blockIdx != blockIdx ? sz : 0 );
                        if( free_space < data_length )
                        {
                            // insert new block for trailing data
                            const size_t next_idx = fs_data_ptrs.size();
                            markDataInsert(new_idx, next_idx);
                            new_idx = next_idx;
                            blockSize = std::max((size_t)CV_FS_MAX_LEN*4 - 256, data_length) + 256;
                            Ptr<std::vector<uchar> > pv = makePtr<std::vector<uchar> >(blockSize);
                            fs_data.push_back(pv);
                            fs_data_ptrs.push_back(&pv->at(0));
                            fs_data_blksz.push_back(blockSize);
                            sz_change += static_cast<int>(blockSize);
                        }
                        uchar* data_begin = fs_data_ptrs[new_idx] + blockSize - data_length;
                        std::memmove(data_begin, orig_data_begin, data_length);
                    }
                    if( node_end.blockIdx != blockIdx )
                    {
                        if( is_pseudocollection )
                        {
                            const bool is_named = ( (*node.ptr() & FileNode::NAMED) != 0 );
                            collection.blockIdx = next_block_idx;
                            collection.ofs = 0;
                            std::memcpy(collection.ptr(), node.ptr(), ( is_named ? 9 : 5 ));
                        }
                        sz_change -= static_cast<int>(orig_blksz - ofs);
                        fs_data[blockIdx]->resize(ofs);
                        fs_data_blksz[blockIdx] = ofs;
                        node.blockIdx = next_block_idx;
                        node.ofs = 0;
                    }
                    if( orig_collection_end.blockIdx == end_block->blockIdx )
                    {
                        end_block->ofs = blockSize;
                        end_block->blockIdx = new_idx;
                    }
                    updateCollectionCapacities(sz_change);
                }
                current_block->blockIdx = node.blockIdx;
                current_block->ofs = node.ofs + sz;
                normalizeNodeOfs(current_block->blockIdx, current_block->ofs);
                if( is_pseudocollection )
                {
                    CV_Assert( node.parent );
                    const size_t collection_sz = node.rawSize();
                    finalizeCollection(node);
                    //updateCollectionCapacities(collection.rawSize()-collection_sz);
                    sz_change = static_cast<int>(node.rawSize() - collection_sz);
                    FileNode *n = node.parent.get();
                    uchar *parent_ptr;
                    while (n)
                    {
                        parent_ptr = n->ptr();
                        if (*parent_ptr & FileNode::NAMED)
                            parent_ptr += 4;
                        parent_ptr++;
                        const int original_buffer_size = readInt(parent_ptr);
                        writeInt(parent_ptr, original_buffer_size + sz_change);
                        n = n->parent.get();
                    }
                    write_stack.clear();
                }
                return node.ptr();
            }

            blockEnd = fs_data_ptrs[blockIdx] + fs_data_blksz[blockIdx];

            CV_Assert(ptr >= fs_data_ptrs[blockIdx] && ptr <= blockEnd);
            // FileNode fits into current block
            if( ptr + sz <= blockEnd )
            {
                current_block->ofs = ofs + sz;
                return ptr;
            }
            // FileNode is first component of current block, but too big to fit.
            // Resize current block instead of allocation of a new one.
            else if (ofs == 0)
            {
                fs_data[blockIdx]->resize(sz);
                ptr = &fs_data[blockIdx]->at(0);
                fs_data_ptrs[blockIdx] = ptr;
                fs_data_blksz[blockIdx] = sz;
                current_block->ofs = sz;
                return ptr;
            }

            shrinkBlock = true;
            shrinkBlockIdx = blockIdx;
            shrinkSize = ofs;
        }

        // Create new block
        const size_t new_idx = fs_data_ptrs.size();
        if( new_idx && blockIdx != new_idx - 1)
            markDataInsert(blockIdx, new_idx);
        size_t blockSize = std::max((size_t)CV_FS_MAX_LEN*4 - 256, sz) + 256;
        Ptr<std::vector<uchar> > pv = makePtr<std::vector<uchar> >(blockSize);
        fs_data.push_back(pv);
        uchar* new_ptr = &pv->at(0);
        fs_data_ptrs.push_back(new_ptr);
        fs_data_blksz.push_back(blockSize);
        node.blockIdx = new_idx;
        node.ofs = 0;
        current_block->blockIdx = new_idx;
        current_block->ofs = sz;

        if( ptr && ptr + 5 <= blockEnd )
        {
            new_ptr[0] = ptr[0];
            if( ptr[0] & FileNode::NAMED )
            {
                new_ptr[1] = ptr[1];
                new_ptr[2] = ptr[2];
                new_ptr[3] = ptr[3];
                new_ptr[4] = ptr[4];
            }
        }

        if (shrinkBlock)
        {
            fs_data[shrinkBlockIdx]->resize(shrinkSize);
            fs_data_blksz[shrinkBlockIdx] = shrinkSize;
        }

        return new_ptr;
    }

    unsigned getStringOfs( const std::string& key ) const
    {
        str_hash_t::const_iterator it = str_hash.find(key);
        return it != str_hash.end() ? it->second : 0;
    }

    unsigned addKey( const String& key )
    {
        unsigned strofs = getStringOfs(key);
        if( !strofs )
        {
            strofs = (unsigned)str_hash_data.size();
            size_t keysize = key.size() + 1;
            str_hash_data.resize(strofs + keysize);
            strcpy(&str_hash_data[0] + strofs, key.c_str());
            str_hash.insert(std::make_pair(key, strofs));
        }
        return strofs;
    }

    FileNode addNode( FileNode& collection, const std::string& key,
                       int elem_type, const void* value, int len = -1 )
    {
        const bool is_reassign = ( !current_block || current_block != end_block );
        if( is_reassign && !isBusy() )
        {
            FileNode node = *current_block.get();
            CV_Assert( node.isCollection() );
            current_block->ofs += ( current_block->isNamed() ? 13 : 9 );
            normalizeNodeOfs(current_block->blockIdx, current_block->ofs);
            return node;
        }
        FileStorage_API* fs = this;
        bool noname = key.empty() || (fmt == FileStorage::FORMAT_XML && strcmp(key.c_str(), "_") == 0);
        convertToCollection( noname ? FileNode::SEQ : FileNode::MAP, collection );

        bool isseq = collection.empty() ? false : collection.isSeq();
        if( noname != isseq )
            CV_PARSE_ERROR_CPP( noname ? "Map element should have a name" :
                                "Sequence element should not have name (use <_></_>)" );
        uchar* cp = collection.ptr();

        FileNode node = *current_block.get();
        node.nodeName = key;

        size_t sz0 = 1 + (noname ? 0 : 4) + 8;
        uchar* ptr = reserveNodeSpace(node, sz0);

        *ptr++ = (uchar)(elem_type | (noname ? 0 : FileNode::NAMED));
        if( elem_type == FileNode::NONE )
            current_block->ofs -= 8;

        if( !noname )
        {
            node.setName(key);
            ptr += 4;
        }

        if( (elem_type & FileNode::TYPE_MASK) == FileNode::SEQ || (elem_type & FileNode::TYPE_MASK) == FileNode::MAP )
        {
            if( !is_reassign )
            {
                writeInt(ptr, 4);
            }
            else
            {
                FileNode collection_end(this, collection.blockIdx,
                        collection.ofs + collection.rawSize());
                normalizeNodeOfs(collection_end.blockIdx, collection_end.ofs,
                        current_block->blockIdx);
                const size_t remaining_capacity = collection_end.ofs - current_block->ofs + 4;
                writeInt(ptr, static_cast<int>(remaining_capacity));
            }
            writeInt(ptr + 4, 0);
        }

        if( value )
            node.setValue(elem_type, value, len);

        if( collection.isNamed() )
            cp += 4;
        int nelems = readInt(cp + 5);
        writeInt(cp + 5, nelems + 1);

        return node;
    }

    void finalizeCollection( FileNode& collection )
    {
        const bool is_pseudocollection = ( collection.type() == FileNode::STRING );
        if( !is_pseudocollection && !collection.isSeq() && !collection.isMap() )
            return;
        uchar* ptr0 = collection.ptr();
        size_t header_size = ( collection.isNamed() ? 5 : 1 );
        uchar* ptr = ptr0 + header_size;
        size_t blockIdx = collection.blockIdx;
        size_t ofs, collection_capacity;
        if( is_pseudocollection )
        {
            ofs = collection.ofs + header_size + 4;
            collection_capacity = 0;
        }
        else
        {
            ofs = collection.ofs + header_size + 8;
            collection_capacity = 4;
        }
        if( is_pseudocollection || (unsigned)readInt(ptr + 4) > 0 )
        {
            size_t lastBlockIdx = current_block->blockIdx;

            for( ; blockIdx < lastBlockIdx; blockIdx = nextBlockIdx(blockIdx) )
            {
                collection_capacity += fs_data_blksz[blockIdx] - ofs;
                ofs = 0;
            }
        }
        collection_capacity += current_block->ofs - ofs;
        size_t orig_capacity = readInt(ptr);
        if ( write_stack.size() == 1 && collection == write_stack.back().startNode && orig_capacity > collection_capacity )
            collection.resize(header_size + 4 + collection_capacity);
        writeInt(ptr, (int)collection_capacity);
    }

    FileNode getCurrentNode()
    {
        return ( current_block ? *current_block : FileNode() );
    }

    void normalizeNodeOfs(size_t& blockIdx, size_t& ofs,
            size_t ref_block_idx = std::numeric_limits<size_t>::max()) const
    {
        while( ofs >= offset(blockIdx) && blockIdx != ref_block_idx
                && blockIdx != end_block->blockIdx )
        {
            ofs -= offset(blockIdx);
            blockIdx = nextBlockIdx(blockIdx);
        }
    }

    class Base64Decoder
    {
    public:
        Base64Decoder() { ofs = 0; ptr = 0; indent = 0; totalchars = 0; eos = true; }
        void init(Ptr<FileStorageParser>& _parser, char* _ptr, int _indent)
        {
            parser = _parser;
            ptr = _ptr;
            indent = _indent;
            encoded.clear();
            decoded.clear();
            ofs = 0;
            totalchars = 0;
            eos = false;
        }

        bool readMore(int needed)
        {
            static const uchar base64tab[] =
            {
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 62,  0,  0,  0, 63,
               52, 53, 54, 55, 56, 57, 58, 59, 60, 61,  0,  0,  0,  0,  0,  0,
                0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  0,  0,  0,  0,  0,
                0, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
               41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
            };

            if( eos )
                return false;

            size_t sz = decoded.size();
            CV_Assert( ofs <= sz );
            sz -= ofs;
            for( size_t i = 0; i < sz; i++ )
                decoded[i] = decoded[ofs + i];

            decoded.resize(sz);
            ofs = 0;

            CV_Assert( !parser.empty() && ptr );
            char *beg = 0, *end = 0;
            bool ok = parser->getBase64Row(ptr, indent, beg, end);
            ptr = end;
            std::copy(beg, end, std::back_inserter(encoded));
            totalchars += end - beg;

            if( !ok || beg == end )
            {
                // in the end of base64 sequence pad it with '=' characters so that
                // its total length is multiple of
                eos = true;
                size_t tc = totalchars;
                for( ; tc % 4 != 0; tc++ )
                    encoded.push_back('=');
            }

            int i = 0, j, n = (int)encoded.size();
            if( n > 0 )
            {
                const uchar* tab = base64tab;
                char* src = &encoded[0];

                for( ; i <= n - 4; i += 4 )
                {
                    // dddddd cccccc bbbbbb aaaaaa => ddddddcc ccccbbbb bbaaaaaa
                    uchar d = tab[(int)(uchar)src[i]], c = tab[(int)(uchar)src[i+1]];
                    uchar b = tab[(int)(uchar)src[i+2]], a = tab[(int)(uchar)src[i+3]];

                    decoded.push_back((uchar)((d << 2) | (c >> 4)));
                    decoded.push_back((uchar)((c << 4) | (b >> 2)));
                    decoded.push_back((uchar)((b << 6) | a));
                }
            }

            if( i > 0 && encoded[i-1] == '=' )
            {
                if( i > 1 && encoded[i-2] == '=' && !decoded.empty() )
                    decoded.pop_back();
                if( !decoded.empty() )
                    decoded.pop_back();
            }

            n -= i;
            for( j = 0; j < n; j++ )
                encoded[j] = encoded[i + j];
            encoded.resize(n);

            return (int)decoded.size() >= needed;
        }

        uchar getUInt8()
        {
            size_t sz = decoded.size();
            if( ofs >= sz && !readMore(1) )
                return (uchar)0;
            return decoded[ofs++];
        }

        ushort getUInt16()
        {
            size_t sz = decoded.size();
            if( ofs + 2 > sz && !readMore(2) )
                return (ushort)0;
            ushort val = (decoded[ofs] + (decoded[ofs + 1] << 8));
            ofs += 2;
            return val;
        }

        int getInt32()
        {
            size_t sz = decoded.size();
            if( ofs + 4 > sz && !readMore(4) )
                return 0;
            int ival = readInt(&decoded[ofs]);
            ofs += 4;
            return ival;
        }

        double getFloat64()
        {
            size_t sz = decoded.size();
            if( ofs + 8 > sz && !readMore(8) )
                return 0;
            double fval = readReal(&decoded[ofs]);
            ofs += 8;
            return fval;
        }

        bool endOfStream() const { return eos; }
        char* getPtr() const { return ptr; }
    protected:

        Ptr<FileStorageParser> parser;
        char* ptr;
        int indent;
        std::vector<char> encoded;
        std::vector<uchar> decoded;
        size_t ofs;
        size_t totalchars;
        bool eos;
    };

    char* parseBase64(char* ptr, int indent, FileNode& collection)
    {
        const int BASE64_HDR_SIZE = 24;
        char dt[BASE64_HDR_SIZE+1] = {0};
        base64decoder.init(parser, ptr, indent);

        int i, k;

        for( i = 0; i < BASE64_HDR_SIZE; i++ )
            dt[i] = (char)base64decoder.getUInt8();
        for( i = 0; i < BASE64_HDR_SIZE; i++ )
            if( isspace(dt[i]))
                break;
        dt[i] = '\0';

        CV_Assert( !base64decoder.endOfStream() );

        int fmt_pairs[CV_FS_MAX_FMT_PAIRS*2];
        int fmt_pair_count = fs::decodeFormat( dt, fmt_pairs, CV_FS_MAX_FMT_PAIRS );
        int ival = 0;
        double fval = 0;

        for(;;)
        {
            for( k = 0; k < fmt_pair_count; k++ )
            {
                int elem_type = fmt_pairs[k*2+1];
                int count = fmt_pairs[k*2];

                for( i = 0; i < count; i++ )
                {
                    int node_type = FileNode::INT;
                    switch( elem_type )
                    {
                    case CV_8U:
                        ival = base64decoder.getUInt8();
                        break;
                    case CV_8S:
                        ival = (char)base64decoder.getUInt8();
                        break;
                    case CV_16U:
                        ival = base64decoder.getUInt16();
                        break;
                    case CV_16S:
                        ival = (short)base64decoder.getUInt16();
                        break;
                    case CV_32S:
                        ival = base64decoder.getInt32();
                        break;
                    case CV_32F:
                        {
                        Cv32suf v;
                        v.i = base64decoder.getInt32();
                        fval = v.f;
                        node_type = FileNode::REAL;
                        }
                        break;
                    case CV_64F:
                        fval = base64decoder.getFloat64();
                        node_type = FileNode::REAL;
                        break;
                    case CV_16F:
                        fval = (float)float16_t::fromBits(base64decoder.getUInt16());
                        node_type = FileNode::REAL;
                        break;
                    default:
                        CV_Error( Error::StsUnsupportedFormat, "Unsupported type" );
                    }

                    if( base64decoder.endOfStream() )
                        break;
                    addNode(collection, std::string(), node_type,
                            node_type == FileNode::INT ? (void*)&ival : (void*)&fval, -1);
                }
            }
            if( base64decoder.endOfStream() )
                break;
        }

        finalizeCollection(collection);
        return base64decoder.getPtr();
    }

    void parseError( const char* func_name, const std::string& err_msg, const char* source_file, int source_line )
    {
        std::string msg = format("%s(%d): %s", filename.c_str(), lineno, err_msg.c_str());
        error(Error::StsParseError, func_name, msg.c_str(), source_file, source_line );
    }

    const uchar* getNodePtr(size_t blockIdx, size_t ofs) const
    {
        CV_Assert( blockIdx < fs_data_ptrs.size());
        CV_Assert( ofs <= fs_data_blksz[blockIdx]);

        return fs_data_ptrs[blockIdx] + ofs;
    }

    std::string getName( size_t nameofs ) const
    {
        CV_Assert( nameofs < str_hash_data.size() );
        return std::string(&str_hash_data[nameofs]);
    }

    FileStorage* getFS() { return fs_ext; }

    FileStorage* fs_ext;

    size_t nextBlockIdx( const size_t& idx ) const
    {
        auto it = fs_data_insert_marks.find(idx);
        return ( it != fs_data_insert_marks.end() ) ? it->second : idx + 1;
    }

    size_t offset( const size_t& idx ) const
    {
        CV_Assert( idx < fs_data_blksz.size() );
        if ( idx == end_block->blockIdx )
            return end_block->ofs;
        else
            return fs_data_blksz[idx];
    }

    size_t markDataInsert( const size_t& precursor_idx, const size_t& new_idx )
    {
        size_t old_next = nextBlockIdx(precursor_idx);
        fs_data_insert_marks[precursor_idx] = new_idx;
        fs_data_insert_marks[new_idx] = old_next;
        return new_idx;
    }

    std::string filename;
    int flags;
    bool empty_stream;

    FILE* file;
    gzFile gzfile;

    bool is_opened;
    bool is_changed;
    bool dummy_eof;
    bool write_mode;
    bool mem_mode;
    bool rw_mode;
    bool save_mode;
    bool write_raw_base64;
    int fmt;

    State state; //!< current state of the FileStorage (used only for writing)
    int space, wrap_margin;
    std::deque<FStructData> write_stack;
    std::vector<char> buffer;
    size_t bufofs;

    std::deque<char> outbuf;

    Ptr<FileStorageEmitter> emitter;
    Ptr<FileStorageParser> parser;
    Base64Decoder base64decoder;

    std::vector<FileNode> roots;
    std::vector<Ptr<std::vector<uchar> > > fs_data;
    std::unordered_map<size_t, size_t> fs_data_insert_marks;
    std::vector<uchar*> fs_data_ptrs;
    std::vector<size_t> fs_data_blksz;
    Ptr<FileNode> end_block;
    Ptr<FileNode> current_block;
    typedef std::unordered_map<std::string, unsigned> str_hash_t;
    str_hash_t str_hash;
    std::vector<char> str_hash_data;

    std::vector<char> strbufv;
    char* strbuf;
    size_t strbufsize;
    size_t strbufpos;
    int lineno;
};

FileStorage::FileStorage()
    : state(0)
{
    p = makePtr<FileStorage::Impl>(this);
}

FileStorage::FileStorage(const String& filename, int flags, const String& encoding)
    : state(0)
{
    p = makePtr<FileStorage::Impl>(this);
    bool ok = p->open(filename.c_str(), flags, encoding.c_str());
    if(ok)
        state = FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP;
}

void FileStorage::startWriteStruct(const String& name, int struct_flags, const String& typeName)
{
    setActive();
    p->startWriteStruct(name.c_str(), struct_flags, typeName.c_str());
    elname = String();
    if ((struct_flags & FileNode::TYPE_MASK) == FileNode::SEQ)
        state = FileStorage::VALUE_EXPECTED;
    else
        state = FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP;
}

void FileStorage::endWriteStruct()
{
    p->endWriteStruct();
    state = p->write_stack.empty() || FileNode::isMap(p->write_stack.back().flags) ?
        FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP :
        FileStorage::VALUE_EXPECTED;
    elname = String();
}

void FileStorage::endAllWriteStructs()
{
    while( !p->write_stack.empty() )
        p->endWriteStruct();
    state = FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP;
    elname = String();
}

FileStorage::~FileStorage()
{
}

bool FileStorage::open(const String& filename, int flags, const String& encoding)
{
    try
    {
        bool ok = p->open(filename.c_str(), flags, encoding.c_str());
        if(ok)
            state = FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP;
        return ok;
    }
    catch (...)
    {
        release();
        throw;  // re-throw
    }
}

bool FileStorage::isOpened() const { return p->is_opened; }

void FileStorage::release()
{
    p->release();
}

void FileStorage::releaseWithoutSave()
{
    p->is_changed = false;
    p->release();
}

FileNode FileStorage::root(int i) const
{
    if( p.empty() || p->roots.empty() || i < 0 || i >= (int)p->roots.size() )
        return FileNode();

    return p->roots[i];
}

FileNode FileStorage::getCurrentNode() const
{
    return ( p->current_block ? *p->current_block : FileNode() );
}

FileNode FileStorage::getFirstTopLevelNode() const
{
    FileNode r = root();
    FileNodeIterator it = r.begin();
    return it != r.end() ? *it : FileNode();
}

std::string FileStorage::getDefaultObjectName(const std::string& _filename)
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

    char* name = name_buf.data();

    // name must start with letter or '_'
    if( !cv_isalpha(*ptr) && *ptr!= '_' ){
        *name++ = '_';
    }

    while( ptr < ptr2 )
    {
        char c = *ptr++;
        if( !cv_isalnum(c) && c != '-' && c != '_' )
            c = '_';
        *name++ = c;
    }
    *name = '\0';
    name = name_buf.data();
    if( strcmp( name, "_" ) == 0 )
        strcpy( name, stubname );
    return name;
}


int FileStorage::getFormat() const
{
    return p->fmt;
}

FileNode FileStorage::operator [](const char* key)
{
    return this->operator[](std::string(key));
}

FileNode FileStorage::operator [](const char* key) const
{
    return this->operator[](std::string(key));
}

FileNode FileStorage::operator [](const std::string& key)
{
    setActive();
    if(p->write_mode) {
        CV_Assert( state & FileStorage::NAME_EXPECTED );
        *this << key;
        return FileNode(p.get(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max());
    }
    else if( p->write_stack.empty() ) {
        elname = key;
        state = FileStorage::INSIDE_MAP + FileStorage::VALUE_EXPECTED;
    }
    return ((const FileStorage&)*this)[key];
}

FileNode FileStorage::operator [](const std::string& key) const
{
    CV_Assert( p->state != FileStorage::UNDEFINED );
    CV_Assert( !p->write_mode );
    FileNode res;
    for (size_t i = 0; i < p->roots.size(); i++)
    {
        res = p->roots[i][key];
        if (res.exists())
        {
            res.parent->nodeName = std::to_string(i);
            break;
        }
    }
    if( !res.exists() ) {
        res.fs = p.get();
        res.parent = makePtr<FileNode>(p->roots[p->roots.size() - 1]);
        res.parent->nodeName = std::to_string(p->roots.size() - 1);
    }
    res.nodeName = key;
    res.parent->parent = makePtr<FileNode>(this, 0, 0);
    return res;
}

String FileStorage::releaseAndGetString()
{
    String buf;
    p->release(&buf);
    return buf;
}

void FileStorage::touch()
{
    p->is_changed = true;
}

bool FileStorage::isChanged() const
{
    return p->is_changed;
}

void FileStorage::setActiveNode( FileNode& node )
{
    if( node.parent && node.parent->parent )
        node.updateDataRef();
    if( isBusy() )
    {
        CV_Assert( p && p->current_block && node != *p->current_block );
        endAllWriteStructs();
    }
    elname = node.name();
    state = FileStorage::VALUE_EXPECTED + FileStorage::INSIDE_MAP;
    if( !node.exists() && !p->fs_data.empty() && !p->isBusy() )
    {
        FileNode *parent = node.parent.get();
        CV_Assert( parent );
        CV_Assert( parent->exists() );
        CV_Assert( parent->isCollection() );
        FStructData fsd;
        fsd.startNode = *parent;
        p->write_stack.push_back(fsd);
        FileNode parent_end = *parent;
        parent_end.ofs += parent->rawSize();
        p->normalizeNodeOfs(parent_end.blockIdx, parent_end.ofs);
        p->current_block = makePtr<FileNode>(parent_end);
        return;
    }
    if( !p->current_block || *(p->current_block) != node )
        p->current_block = makePtr<FileNode>(node);
}

void FileStorage::setActive()
{
    p->fs_ext = this;
}

bool FileStorage::isBusy() const
{
    return p->isBusy();
}

void FileStorage::writeRaw( const String& fmt, const void* vec, size_t len )
{
    p->writeRawData(fmt, (const uchar*)vec, len);
}

void FileStorage::writeComment( const String& comment, bool eol_comment )
{
    p->writeComment(comment.c_str(), eol_comment);
}

void writeScalar( FileStorage& fs, int value )
{
    fs.p->write(String(), value);
}

void writeScalar( FileStorage& fs, float value )
{
    fs.p->write(String(), (double)value);
}

void writeScalar( FileStorage& fs, double value )
{
    fs.p->write(String(), value);
}

void writeScalar( FileStorage& fs, const String& value )
{
    fs.p->write(String(), value);
}

void writeNode( FileStorage& fs, const FileNode& node )
{
    const uchar *p0 = node.ptr();
    const uchar *p = p0 + 1;
    const uchar tag = *p0;
    int type = node.type();
    const bool is_named = ( (tag & FileNode::NAMED) != 0 );
    if( is_named ) {
        if( fs.p->write_mode || fs.isBusy() )
            fs << node.name();
        p += 4;
    }
    size_t raw_sz = node.rawSize();
    switch(type) {
    case FileNode::SEQ:
        fs << (( tag & FileNode::FLOW ) ? "[:" : "[");
        p += 8;
        while (static_cast<unsigned>(p - p0) < raw_sz) {
            FileNode sub_node(&fs, node.blockIdx, node.ofs + (p - p0));
            fs.p->normalizeNodeOfs(sub_node.blockIdx, sub_node.ofs);
            fs << sub_node;
            p += sub_node.rawSize();
        }
        fs << "]";
        break;
    case FileNode::MAP:
        fs << (( tag & FileNode::FLOW ) ? "{:" : "{");
        //skip size + nr of elems tags
        p += 8;
        while (static_cast<unsigned>(p - p0) < raw_sz) {
            FileNode sub_node(&fs, node.blockIdx, node.ofs + (p - p0));
            fs.p->normalizeNodeOfs(sub_node.blockIdx, sub_node.ofs);
            fs << sub_node;
            p += sub_node.rawSize();
        }
        fs << "}";
        break;
    case FileNode::INT:
        fs << (int)node;
        break;
    case FileNode::REAL:
        fs << (double)node;
        break;
    case FileNode::STRING:
        fs << (std::string)node;
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Unsopported FileNode type");
        break;
    }
}

void write( FileStorage& fs, const String& name, int value )
{
    fs.p->write(name, value);
}

void write( FileStorage& fs, const String& name, float value )
{
    fs.p->write(name, (double)value);
}

void write( FileStorage& fs, const String& name, double value )
{
    fs.p->write(name, value);
}

void write( FileStorage& fs, const String& name, const String& value )
{
    fs.p->write(name, value);
}

void FileStorage::write(const String& name, int val) { p->write(name, val); }
void FileStorage::write(const String& name, double val) { p->write(name, val); }
void FileStorage::write(const String& name, const String& val) { p->write(name, val); }
void FileStorage::write(const String& name, const Mat& val) { cv::write(*this, name, val); }
void FileStorage::write(const String& name, const std::vector<String>& val) { cv::write(*this, name, val); }

FileStorage& operator << (FileStorage& fs, const String& str)
{
    enum { NAME_EXPECTED = FileStorage::NAME_EXPECTED,
        VALUE_EXPECTED = FileStorage::VALUE_EXPECTED,
        INSIDE_MAP = FileStorage::INSIDE_MAP };
    const char* _str = str.c_str();
    if( (fs.isWrite() && !fs.isOpened()) || !_str )
        return fs;
    Ptr<FileStorage::Impl>& fs_impl = fs.p;
    char c = *_str;

    if( c == '}' || c == ']' )
    {
        if( fs_impl->write_stack.empty() )
            CV_Error_( CV_StsError, ("Extra closing '%c'", *_str) );

        int struct_flags = fs_impl->write_stack.back().flags;
        char expected_bracket = FileNode::isMap(struct_flags) ? '}' : ']';
        if( c != expected_bracket )
            CV_Error_( CV_StsError, ("The closing '%c' does not match the opening '%c'", c, expected_bracket));
        fs_impl->endWriteStruct();
        if( !fs_impl->write_stack.empty() )
        {
            struct_flags = fs_impl->write_stack.back().flags;
            fs.state = FileNode::isMap(struct_flags) ? INSIDE_MAP + NAME_EXPECTED : VALUE_EXPECTED;
            fs.elname = String();
        }
    }
    else if( fs.state == NAME_EXPECTED + INSIDE_MAP )
    {
        if (!cv_isalpha(c) && c != '_')
            CV_Error_( CV_StsError, ("Incorrect element name %s; should start with a letter or '_'", _str) );
        fs.elname = str;
        fs.state = VALUE_EXPECTED + INSIDE_MAP;
    }
    else if( (fs.state & 3) == VALUE_EXPECTED )
    {
        if( c == '{' || c == '[' )
        {
            int struct_flags = c == '{' ? FileNode::MAP : FileNode::SEQ;
            fs.state = struct_flags == FileNode::MAP ? INSIDE_MAP + NAME_EXPECTED : VALUE_EXPECTED;
            _str++;
            if( *_str == ':' )
            {
                _str++;
                if( !*_str )
                    struct_flags |= FileNode::FLOW;
            }
            fs_impl->startWriteStruct(!fs.elname.empty() ? fs.elname.c_str() : 0, struct_flags, *_str ? _str : 0 );
            fs.elname = String();
        }
        else
        {
            write( fs, fs.elname, (c == '\\' && (_str[1] == '{' || _str[1] == '}' ||
                                _str[1] == '[' || _str[1] == ']')) ? String(_str+1) : str );
            if( fs.state == INSIDE_MAP + VALUE_EXPECTED )
                fs.state = INSIDE_MAP + NAME_EXPECTED;
        }
    }
    else
        CV_Error( CV_StsError, "Invalid fs.state" );
    return fs;
}

bool FileStorage::isWrite() const { return p->write_mode; }

void FileStorage::save(const String& filename, int save_flags)
{
    CV_Assert( p->rw_mode );
    CV_Assert( save_flags );
    CV_Assert( (save_flags & FileStorage::READWRITE) != FileStorage::READWRITE );
    String orig_filename = p->filename;
    if( p->write_mode )
    {
        p->flush();
        return;
    }
    if( isBusy() )
        endAllWriteStructs();
    if( filename != p->filename )
        touch();
    if( !p->is_changed )
        return;
    p->save_mode = true;
    open((filename.empty() ? p->filename : filename), save_flags );
    for (size_t i = 0; i < p->roots.size(); i++)
    {
        if( i > 0 )
        {
            p->setNonEmpty();
            p->startNextStream();
        }
        FileNode root = p->roots[i];
        for(auto it = root.begin(); it != root.end(); it++)
            *this << *it;
    }
    p->release();
    p->filename = orig_filename;
    p->emitter = createFileNodeEmitter(p.get());
    p->is_changed = p->save_mode = p->write_mode = false;
    p->rw_mode = true;
}


FileNode::FileNode()
    : fs(NULL)
{
    blockIdx = ofs = std::numeric_limits<size_t>::max();
}

FileNode::FileNode(FileStorage::Impl* _fs, size_t _blockIdx, size_t _ofs)
    : fs(_fs)
{
    blockIdx = _blockIdx;
    ofs = _ofs;
}

FileNode::FileNode(const FileStorage* _fs, size_t _blockIdx, size_t _ofs)
    : FileNode(_fs->p.get(), _blockIdx, _ofs)
{
    // nothing
}

FileNode::FileNode(const FileNode& node)
{
    fs = node.fs;
    blockIdx = node.blockIdx;
    ofs = node.ofs;
    parent = node.parent;
    nodeName = node.nodeName;
}

FileNode& FileNode::operator=(const FileNode& node)
{
    fs = node.fs;
    blockIdx = node.blockIdx;
    ofs = node.ofs;
    parent = node.parent;
    nodeName = node.nodeName;
    return *this;
}

FileNode FileNode::operator[](const std::string& nodename) const
{
    if(!fs || fs->write_mode)
        return FileNode();

    FileNode res;
    res.fs = fs;
    res.parent = makePtr<FileNode>(*this);
    res.nodeName = nodename;

    if( exists() )
    {
        CV_Assert( isMap() );
        unsigned key = fs->getStringOfs(nodename);
        if( key )
        {
            size_t i, sz = size();
            FileNodeIterator it = begin();

            for( i = 0; i < sz; i++, ++it )
            {
                const uchar* p = (*it).ptr();
                unsigned key2 = (unsigned)readInt(p + 1);
                CV_Assert( key2 < fs->str_hash_data.size() );
                if( key == key2 )
                {
                    res.fs = (*it).fs;
                    res.blockIdx = (*it).blockIdx;
                    res.ofs = (*it).ofs;
                    break;
                }
            }
        }
        if( fs->write_stack.empty() )
        {
            FileStorage *fs_ext = getFS();
            fs_ext->elname = nodename;
            fs_ext->state = FileStorage::INSIDE_MAP + FileStorage::VALUE_EXPECTED;
        }
    }
    return res;
}

FileNode FileNode::operator[](const char* nodename) const
{
    return this->operator[](std::string(nodename));
}

FileNode FileNode::operator[](int i) const
{
    if(!fs)
        return FileNode();

    CV_Assert( isSeq() );

    int sz = (int)size();
    CV_Assert( 0 <= i && i < sz );

    FileNodeIterator it = begin();
    it += i;

    return *it;
}

std::vector<String> FileNode::keys() const
{
    CV_Assert(isMap());

    std::vector<String> res;
    res.reserve(size());
    for (FileNodeIterator it = begin(); it != end(); ++it)
    {
        res.push_back((*it).name());
    }
    return res;
}

int FileNode::type() const
{
    const uchar* p = ptr();
    if(!p)
        return NONE;
    return (*p & TYPE_MASK);
}

bool FileNode::isMap(int flags) { return (flags & TYPE_MASK) == MAP; }
bool FileNode::isSeq(int flags) { return (flags & TYPE_MASK) == SEQ; }
bool FileNode::isCollection(int flags) { return isMap(flags) || isSeq(flags); }
bool FileNode::isFlow(int flags) { return (flags & FLOW) != 0; }
bool FileNode::isEmptyCollection(int flags) { return (flags & EMPTY) != 0; }

bool FileNode::empty() const   { return !exists(); }
bool FileNode::exists() const  { return fs != 0 && blockIdx < fs->fs_data_ptrs.size(); }
bool FileNode::isNone() const  { return type() == NONE; }
bool FileNode::isSeq() const   { return type() == SEQ; }
bool FileNode::isMap() const   { return type() == MAP; }
bool FileNode::isCollection() const   { return isSeq() || isMap(); }
bool FileNode::isInt() const   { return type() == INT;  }
bool FileNode::isReal() const  { return type() == REAL; }
bool FileNode::isString() const { return type() == STRING;  }
bool FileNode::isNamed() const
{
    if( !exists() )
        return !nodeName.empty();
    const uchar* p = ptr();
    if(!p)
        return false;
    return (*p & NAMED) != 0;
}

std::string FileNode::name() const
{
    if( !exists() )
        return nodeName;
    if( !isNamed() )
        return std::string();
    const uchar* p = ptr();
    if(!p)
        return std::string();
    size_t nameofs = p[1] | (p[2]<<8) | (p[3]<<16) | (p[4]<<24);
    return fs->getName(nameofs);
}

FileNode::operator int() const
{
    const uchar* p = ptr();
    if(!p)
        return 0;
    int tag = *p;
    int type = (tag & TYPE_MASK);
    p += (tag & NAMED) ? 5 : 1;

    if( type == INT )
    {
        return readInt(p);
    }
    else if( type == REAL )
    {
        return cvRound(readReal(p));
    }
    else
        return 0x7fffffff;
}

FileNode::operator float() const
{
    const uchar* p = ptr();
    if(!p)
        return 0.f;
    int tag = *p;
    int type = (tag & TYPE_MASK);
    p += (tag & NAMED) ? 5 : 1;

    if( type == INT )
    {
        return (float)readInt(p);
    }
    else if( type == REAL )
    {
        return (float)readReal(p);
    }
    else
        return FLT_MAX;
}

FileNode::operator double() const
{
    const uchar* p = ptr();
    if(!p)
        return 0.f;
    int tag = *p;
    int type = (tag & TYPE_MASK);
    p += (tag & NAMED) ? 5 : 1;

    if( type == INT )
    {
        return (double)readInt(p);
    }
    else if( type == REAL )
    {
        return readReal(p);
    }
    else
        return DBL_MAX;
}

double FileNode::real() const  { return double(*this); }
std::string FileNode::string() const
{
    const uchar* p = ptr();
    if( !p || (*p & TYPE_MASK) != STRING )
        return std::string();
    p += (*p & NAMED) ? 5 : 1;
    size_t sz = (size_t)(unsigned)readInt(p);
    return std::string((const char*)(p + 4), sz - 1);
}
Mat FileNode::mat() const { Mat value; read(*this, value, Mat()); return value; }

bool FileNode::operator == (const FileNode& node) const
{
    if( exists() )
    {
        return fs == node.fs
            && blockIdx == node.blockIdx
            && ofs == node.ofs;
    }
    return nodeName == node.nodeName && (
        ( !parent && !node.parent )
        || ( parent && node.parent && *parent == *(node.parent) )
    );
}

FileStorage* FileNode::getFS() const
{
    return fs->getFS();
}

FileNodeIterator FileNode::begin() const { return FileNodeIterator(*this, false); }
FileNodeIterator FileNode::end() const   { return FileNodeIterator(*this, true); }

void FileNode::readRaw( const std::string& fmt, void* vec, size_t len ) const
{
    FileNodeIterator it = begin();
    it.readRaw( fmt, vec, len );
}

size_t FileNode::size() const
{
    const uchar* p = ptr();
    if( !p )
        return 0;
    int tag = *p;
    int tp = tag & TYPE_MASK;
    if( tp == MAP || tp == SEQ )
    {
        if( tag & NAMED )
            p += 4;
        return (size_t)(unsigned)readInt(p + 5);
    }
    return tp != NONE;
}

size_t FileNode::rawSize() const
{
    const uchar* p0 = ptr(), *p = p0;
    if( !p )
        return 0;
    int tag = *p++;
    int tp = tag & TYPE_MASK;
    if( tag & NAMED )
        p += 4;
    size_t sz0 = (size_t)(p - p0);
    if( tp == INT )
        return sz0 + 4;
    if( tp == REAL )
        return sz0 + 8;
    if( tp == NONE )
        return sz0;
    CV_Assert( tp == STRING || tp == SEQ || tp == MAP );
    return sz0 + 4 + readInt(p);
}

size_t FileNode::capacity() const
{
    const uchar* p0 = ptr(), *p = p0;
    if( !p )
        return 0;
    int tag = *p++;
    int tp = tag & TYPE_MASK;
    if( tp == INT )
        return 4;
    if( tp == REAL )
        return 8;
    if( tp == NONE )
        return 0;
    if( tag & NAMED )
        p += 4;
    CV_Assert( tp == STRING || tp == SEQ || tp == MAP );
    return readInt(p);
}

uchar* FileNode::ptr()
{
    return (!exists() || (fs->write_mode && !fs->save_mode) ) ? 0 : (uchar*)fs->getNodePtr(blockIdx, ofs);
}

const uchar* FileNode::ptr() const
{
    return (!exists() || (fs->write_mode && !fs->save_mode) ) ? 0 : fs->getNodePtr(blockIdx, ofs);
}

void FileNode::setName( const String& name )
{
    // if( !fs->isBusy() && parent && parent->parent )
    //     updateDataRef();
    //nodeName = name;
    uchar* p = ptr();
    if( exists() && (*p & FileNode::NAMED) != 0 )
    {
        unsigned strofs = fs->addKey(name);
        writeInt(p + 1, (int)strofs);
    }
    else
        CV_Error(Error::StsNotImplemented, "Cannot rename unnamed FileNode.");
}

size_t FileNode::setValue( int type, const void* value, int len )
{
    FileStorage& fs_ext = *getFS();
    const bool is_reassign = ( !fs->current_block || *(fs->current_block) != *(fs->end_block) );
    if ( is_reassign )
    {
        fs->is_changed = true;
        if( !fs_ext.isBusy() )
            fs_ext.setActiveNode(*this);
        else
            fs->current_block = makePtr<FileNode>(*this);
    }
    uchar *p = ptr();
    CV_Assert(p != 0);

    int tag = *p;
    int current_type = tag & TYPE_MASK;
    CV_Assert( fs->isBusy() || current_type == NONE || current_type == type );

    const bool is_named = ( (tag & NAMED) != 0 );
    size_t sz = 1;

    if( is_named )
        sz += 4;

    if( type == INT )
        sz += 4;
    else if( type == REAL )
        sz += 8;
    else if( type == STRING )
    {
        if( len < 0 )
            len = (int)strlen((const char*)value);
        sz += 4 + len + 1; // besides the string content,
                           // take the size (4 bytes) and the final '\0' into account
    }
    else if( isCollection(type) )
    {
        if (len < 0)
            len = readInt((uchar*)value + sz);
        sz += 4;
        sz += len;
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "Unsupported node type for dynamical assignment");

    p = fs->reserveNodeSpace(*this, sz);
    *p++ = (uchar)(type | (tag & NAMED));
    if( is_named )
        p += 4;

    if( type == INT )
    {
        int ival = *(const int*)value;
        writeInt(p, ival);
    }
    else if( type == REAL )
    {
        double dbval = *(const double*)value;
        writeReal(p, dbval);
    }
    else if( type == STRING )
    {
        const char* str = (const char*)value;
        writeInt(p, len + 1);
        memcpy(p + 4, str, len);
        p[4 + len] = (uchar)'\0';
    }
    else if ( isCollection(type) )
    {
        writeInt(p, len);
        p += 4;
        memcpy(p, (uchar*)value + sz - len, len);
    }
    if( !fs->isBusy() )
        fs->current_block.reset();
    return sz;
}

void FileNode::startWriteStruct( int flags, const String& typeName )
{
    const bool is_reassign = *this != *(fs->end_block);
    getFS()->setActiveNode(*this);
    CV_Assert( isCollection(flags) );
    if( exists() )
    {
        fs->convertToCollection(flags & TYPE_MASK, *this);
        // undo fs->current_block shift in convertToCollection
        getFS()->setActiveNode(*this);
    }
    if( isNamed() )
        fs->addKey(name());
    fs->startWriteStruct(name().c_str(), flags, typeName.c_str());
    if( !is_reassign )
    {
        *(fs->end_block) = *(fs->current_block);
        fs->current_block = fs->end_block;
    }
}

void FileNode::endWriteStruct()
{
    CV_Assert( !fs->write_stack.empty() && *this == fs->write_stack.back().startNode );
    fs->endWriteStruct();
}

FileNode FileNode::addNode( const std::string& key,
    int elem_type, const void* value, int len )
{
    return fs->addNode(*this, key, elem_type, value, len);
}

void FileNode::resize(size_t sz)
{
    CV_Assert( isCollection() || type() == FileNode::STRING );
    const size_t raw_size = rawSize();
    if( raw_size == sz )
        return;
    if( raw_size < sz )
        CV_Error(Error::StsNotImplemented, "Only shrinking of FileNode supported (i.e. rawSize() < sz)");
    FileNode orig_collection_end = *this;
    orig_collection_end.ofs += raw_size;
    fs->normalizeNodeOfs(orig_collection_end.blockIdx, orig_collection_end.ofs);
    FileNode new_collection_end = *this;
    new_collection_end.ofs += sz;
    fs->normalizeNodeOfs(new_collection_end.blockIdx, new_collection_end.ofs);
    const size_t end_data_length = fs->fs_data_blksz[orig_collection_end.blockIdx] - orig_collection_end.ofs;
    const size_t free_space = fs->fs_data_blksz[new_collection_end.blockIdx] - new_collection_end.ofs;
    if( end_data_length <= free_space )
    {
        std::memmove(new_collection_end.ptr(), orig_collection_end.ptr(), end_data_length);
        const size_t new_size = new_collection_end.ofs + end_data_length;
        fs->fs_data[new_collection_end.blockIdx]->resize(new_size);
        fs->fs_data_blksz[new_collection_end.blockIdx] = new_size;
        if( new_collection_end.blockIdx != orig_collection_end.blockIdx )
        {
            fs->fs_data[orig_collection_end.blockIdx]->resize(0);
            fs->fs_data_blksz[orig_collection_end.blockIdx] = 0;
        }
        if( orig_collection_end.blockIdx == fs->end_block->blockIdx )
        {
            fs->end_block->blockIdx = new_collection_end.blockIdx;
            fs->end_block->ofs = fs->end_block->ofs - orig_collection_end.ofs + new_collection_end.ofs;
        }
    }
    else
    {
        fs->fs_data[new_collection_end.blockIdx]->resize(new_collection_end.ofs);
        fs->fs_data_blksz[new_collection_end.blockIdx] = new_collection_end.ofs;
        std::memmove(fs->fs_data_ptrs[orig_collection_end.blockIdx], fs->fs_data_ptrs[orig_collection_end.blockIdx] + orig_collection_end.ofs, end_data_length);
        if( orig_collection_end.blockIdx != fs->end_block->blockIdx )
        {
            fs->fs_data[orig_collection_end.blockIdx]->resize(end_data_length);
            fs->fs_data_blksz[orig_collection_end.blockIdx] = end_data_length;
        }
        else
        {
            fs->end_block->ofs -= orig_collection_end.ofs;
        }
    }

    if ( new_collection_end.blockIdx != orig_collection_end.blockIdx )
    {
        //std::vector<size_t> empty_blocks;
        size_t next_block = fs->nextBlockIdx(new_collection_end.blockIdx);
        while( next_block != orig_collection_end.blockIdx && next_block < fs->fs_data.size() )
        {
            //empty_blocks.push_back(next_block);
            fs->fs_data[next_block]->resize(0);
            fs->fs_data_blksz[next_block] = 0;
            next_block = fs->nextBlockIdx(next_block);
        }
        // fs->fs_data.erase(std::remove_if(fs->fs_data.begin(), fs->fs_data.end(), [empty_blocks](const size_t elem) { return std::find(empty_blocks.begin(), empty_blocks.end(), elem) != empty_blocks.end(); }), fs->fs_data.end());
        // fs->fs_data_ptrs.erase(std::remove_if(fs->fs_data_ptrs.begin(), fs->fs_data_ptrs.end(), [empty_blocks](const size_t elem) { return std::find(empty_blocks.begin(), empty_blocks.end(), elem) != empty_blocks.end(); }), fs->fs_data_ptrs.end());
        // fs->fs_data_blksz.erase(std::remove_if(fs->fs_data_blksz.begin(), fs->fs_data_blksz.end(), [empty_blocks](const size_t elem) { return std::find(empty_blocks.begin(), empty_blocks.end(), elem) != empty_blocks.end(); }), fs->fs_data_blksz.end());
        // fs->markDataInsert(new_collection_end.blockIdx, fs->nextBlockIdx(next_block));
    }
}

void FileNode::updateDataRef()
{
    Ptr<FileNode> p = parent;
    CV_Assert( p && p->parent && ( !nodeName.empty() || parent->isSeq() ) );
    // node is not base node, i.e. has more than 2 parent nodes
    if( p->parent->parent )
    {
        p->updateDataRef();
    }
    else
    {
        FileStorage& fs_ext = *getFS();
        *this = fs_ext[nodeName];
        return;
    }
    *this = (*p)[nodeName];
}


FileNodeIterator::FileNodeIterator()
{
    fs = 0;
    blockIdx = 0;
    ofs = 0;
    blockSize = 0;
    nodeNElems = 0;
    idx = 0;
}

FileNodeIterator::FileNodeIterator( const FileNode& node, bool seekEnd )
{
    fs = node.fs;
    idx = 0;
    if( !node.exists() )
        blockIdx = ofs = blockSize = nodeNElems = 0;
    else
    {
        blockIdx = node.blockIdx;
        ofs = node.ofs;

        bool collection = node.isSeq() || node.isMap();
        if( node.isNone() )
        {
            nodeNElems = 0;
        }
        else if( !collection )
        {
            nodeNElems = 1;
            if( seekEnd )
            {
                idx = 1;
                ofs += node.rawSize();
            }
        }
        else
        {
            nodeNElems = node.size();
            const uchar* p0 = node.ptr(), *p = p0 + 1;
            if(*p0 & FileNode::NAMED )
                p += 4;
            if( !seekEnd )
                ofs += (p - p0) + 8;
            else
            {
                size_t rawsz = (size_t)(unsigned)readInt(p);
                ofs += (p - p0) + 4 + rawsz;
                idx = nodeNElems;
            }
        }
        fs->normalizeNodeOfs(blockIdx, ofs);
        blockSize = fs->fs_data_blksz[blockIdx];
    }
}

FileNodeIterator::FileNodeIterator(const FileNodeIterator& it)
{
    fs = it.fs;
    blockIdx = it.blockIdx;
    ofs = it.ofs;
    blockSize = it.blockSize;
    nodeNElems = it.nodeNElems;
    idx = it.idx;
}

FileNodeIterator& FileNodeIterator::operator=(const FileNodeIterator& it)
{
    fs = it.fs;
    blockIdx = it.blockIdx;
    ofs = it.ofs;
    blockSize = it.blockSize;
    nodeNElems = it.nodeNElems;
    idx = it.idx;
    return *this;
}

FileNode FileNodeIterator::operator *() const
{
    return FileNode(idx < nodeNElems ? fs : NULL, blockIdx, ofs);
}

FileNodeIterator& FileNodeIterator::operator ++ ()
{
    if( idx == nodeNElems || !fs )
        return *this;
    idx++;
    FileNode n(fs, blockIdx, ofs);
    ofs += n.rawSize();
    if( ofs >= blockSize )
    {
        fs->normalizeNodeOfs(blockIdx, ofs);
        blockSize = fs->fs_data_blksz[blockIdx];
    }
    return *this;
}

FileNodeIterator FileNodeIterator::operator ++ (int)
{
    FileNodeIterator it = *this;
    ++(*this);
    return it;
}

FileNodeIterator& FileNodeIterator::operator += (int _ofs)
{
    CV_Assert( _ofs >= 0 );
    for( ; _ofs > 0; _ofs-- )
        this->operator ++();
    return *this;
}

FileNodeIterator& FileNodeIterator::readRaw( const String& fmt, void* _data0, size_t maxsz)
{
    if( fs && idx < nodeNElems )
    {
        uchar* data0 = (uchar*)_data0;
        int fmt_pairs[CV_FS_MAX_FMT_PAIRS*2];
        int fmt_pair_count = fs::decodeFormat( fmt.c_str(), fmt_pairs, CV_FS_MAX_FMT_PAIRS );
        size_t esz = fs::calcStructSize( fmt.c_str(), 0 );

        CV_Assert( maxsz % esz == 0 );
        maxsz /= esz;

        for( ; maxsz > 0; maxsz--, data0 += esz )
        {
            size_t offset = 0;
            for( int k = 0; k < fmt_pair_count; k++ )
            {
                int elem_type = fmt_pairs[k*2+1];
                int elem_size = CV_ELEM_SIZE(elem_type);

                int count = fmt_pairs[k*2];
                offset = alignSize( offset, elem_size );
                uchar* data = data0 + offset;

                for( int i = 0; i < count; i++, ++(*this) )
                {
                    FileNode node = *(*this);
                    if( node.isInt() )
                    {
                        int ival = (int)node;
                        switch( elem_type )
                        {
                        case CV_8U:
                            *(uchar*)data = saturate_cast<uchar>(ival);
                            data++;
                            break;
                        case CV_8S:
                            *(char*)data = saturate_cast<schar>(ival);
                            data++;
                            break;
                        case CV_16U:
                            *(ushort*)data = saturate_cast<ushort>(ival);
                            data += sizeof(ushort);
                            break;
                        case CV_16S:
                            *(short*)data = saturate_cast<short>(ival);
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
                        case CV_16F:
                            *(float16_t*)data = float16_t((float)ival);
                            data += sizeof(float16_t);
                            break;
                        default:
                            CV_Error( Error::StsUnsupportedFormat, "Unsupported type" );
                        }
                    }
                    else if( node.isReal() )
                    {
                        double fval = (double)node;

                        switch( elem_type )
                        {
                        case CV_8U:
                            *(uchar*)data = saturate_cast<uchar>(fval);
                            data++;
                            break;
                        case CV_8S:
                            *(char*)data = saturate_cast<schar>(fval);
                            data++;
                            break;
                        case CV_16U:
                            *(ushort*)data = saturate_cast<ushort>(fval);
                            data += sizeof(ushort);
                            break;
                        case CV_16S:
                            *(short*)data = saturate_cast<short>(fval);
                            data += sizeof(short);
                            break;
                        case CV_32S:
                            *(int*)data = saturate_cast<int>(fval);
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
                        case CV_16F:
                            *(float16_t*)data = float16_t((float)fval);
                            data += sizeof(float16_t);
                            break;
                        default:
                            CV_Error( Error::StsUnsupportedFormat, "Unsupported type" );
                        }
                    }
                    else
                        CV_Error( Error::StsError, "readRawData can only be used to read plain sequences of numbers" );
                }
                offset = (int)(data - data0);
            }
        }
    }

    return *this;
}

bool FileNodeIterator::equalTo(const FileNodeIterator& it) const
{
    return fs == it.fs && blockIdx == it.blockIdx && ofs == it.ofs &&
           idx == it.idx && nodeNElems == it.nodeNElems;
}

size_t FileNodeIterator::remaining() const
{
    return nodeNElems - idx;
}

bool operator == ( const FileNodeIterator& it1, const FileNodeIterator& it2 )
{
    return it1.equalTo(it2);
}

bool operator != ( const FileNodeIterator& it1, const FileNodeIterator& it2 )
{
    return !it1.equalTo(it2);
}

void read(const FileNode& node, int& val, int default_val)
{
    val = default_val;
    if( !node.empty() )
    {
        val = (int)node;
    }
}

void read(const FileNode& node, double& val, double default_val)
{
    val = default_val;
    if( !node.empty() )
    {
        val = (double)node;
    }
}

void read(const FileNode& node, float& val, float default_val)
{
    val = default_val;
    if( !node.empty() )
    {
        val = (float)node;
    }
}

void read(const FileNode& node, std::string& val, const std::string& default_val)
{
    val = default_val;
    if( !node.empty() )
    {
        val = (std::string)node;
    }
}

FileStorage_API::~FileStorage_API() {}

namespace internal
{

WriteStructContext::WriteStructContext(FileStorage& _fs, const std::string& name,
                                       int flags, const std::string& typeName)
{
    fs = &_fs;
    fs->startWriteStruct(name, flags, typeName);
}

WriteStructContext::~WriteStructContext()
{
    fs->endWriteStruct();
}

}

}
