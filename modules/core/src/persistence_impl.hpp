// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_CORE_PERSISTENCE_IMPL_HPP
#define OPENCV_CORE_PERSISTENCE_IMPL_HPP

#include "persistence.hpp"
#include "persistence_base64_encoding.hpp"
#include <unordered_map>
#include <iterator>


namespace cv
{

enum Base64State{
    Uncertain,
    NotUse,
    InUse,
};

class cv::FileStorage::Impl : public FileStorage_API
{
public:
    void init();

    Impl(FileStorage* _fs);

    virtual ~Impl();

    void release(String* out=0);

    void analyze_file_name( const std::string& file_name, std::vector<std::string>& params );

    bool open( const char* filename_or_buf, int _flags, const char* encoding );

    void puts( const char* str );

    char* getsFromFile( char* buf, int count );

    char* gets( size_t maxCount );

    char* gets();

    bool eof();

    void setEof();

    void closeFile();

    void rewind();

    char* resizeWriteBuffer( char* ptr, int len );

    char* flush();

    void endWriteStruct();

    void startWriteStruct_helper( const char* key, int struct_flags,
                                  const char* type_name );

    void startWriteStruct( const char* key, int struct_flags,
                           const char* type_name );

    void writeComment( const char* comment, bool eol_comment );

    void startNextStream();

    void write( const String& key, int value );

    void write( const String& key, int64_t value );

    void write( const String& key, double value );

    void write( const String& key, const String& value );

    void writeRawData( const std::string& dt, const void* _data, size_t len );

    void workaround();

    void switch_to_Base64_state( FileStorage_API::Base64State new_state);

    void make_write_struct_delayed( const char* key, int struct_flags, const char* type_name );

    void check_if_write_struct_is_delayed( bool change_type_to_base64 );

    void writeRawDataBase64(const void* _data, size_t len, const char* dt );

    String releaseAndGetString();

    FileNode getFirstTopLevelNode() const;

    FileNode root(int streamIdx=0) const;

    FileNode operator[](const String& nodename) const;

    FileNode operator[](const char* /*nodename*/) const;

    int getFormat() const;

    char* bufferPtr() const;
    char* bufferStart() const;
    char* bufferEnd() const;
    void setBufferPtr(char* ptr);
    int wrapMargin() const;

    FStructData& getCurrentStruct();

    void setNonEmpty();

    void processSpecialDouble( char* buf, double* value, char** endptr );

    double strtod( char* ptr, char** endptr );

    void convertToCollection(int type, FileNode& node);

    // a) allocates new FileNode (for that just set blockIdx to the last block and ofs to freeSpaceOfs) or
    // b) reallocates just created new node (blockIdx and ofs must be taken from FileNode).
    //    If there is no enough space in the current block (it should be the last block added so far),
    //    the last block is shrunk so that it ends immediately before the reallocated node. Then,
    //    a new block of sufficient size is allocated and the FileNode is placed in the beginning of it.
    // The case (a) can be used to allocate the very first node by setting blockIdx == ofs == 0.
    // In the case (b) the existing tag and the name are copied automatically.
    uchar* reserveNodeSpace(FileNode& node, size_t sz);

    unsigned getStringOfs( const std::string& key ) const;

    FileNode addNode( FileNode& collection, const std::string& key,
                      int elem_type, const void* value, int len );

    void finalizeCollection( FileNode& collection );

    void normalizeNodeOfs(size_t& blockIdx, size_t& ofs) const;

    Base64State get_state_of_writing_base64();

    int get_space();

    class Base64Decoder
    {
    public:
        Base64Decoder();
        void init(const Ptr<FileStorageParser>& _parser, char* _ptr, int _indent);

        bool readMore(int needed);

        uchar getUInt8();

        ushort getUInt16();

        int getInt32();

        double getFloat64();

        bool endOfStream() const;
        char* getPtr() const;
    protected:

        Ptr<FileStorageParser> parser_do_not_use_direct_dereference;
        FileStorageParser& getParser() const
        {
            if (!parser_do_not_use_direct_dereference)
                CV_Error(Error::StsNullPtr, "Parser is not available");
            return *parser_do_not_use_direct_dereference;
        }
        char* ptr;
        int indent;
        std::vector<char> encoded;
        std::vector<uchar> decoded;
        size_t ofs;
        size_t totalchars;
        bool eos;
    };

    char* parseBase64(char* ptr, int indent, FileNode& collection);

    void parseError( const char* func_name, const std::string& err_msg, const char* source_file, int source_line );

    const uchar* getNodePtr(size_t blockIdx, size_t ofs) const;

    std::string getName( size_t nameofs ) const;

    FileStorage* getFS();

    FileStorage* fs_ext;

    std::string filename;
    int flags;
    bool empty_stream;

    FILE* file;
    gzFile gzfile;

    bool is_opened;
    bool dummy_eof;
    bool write_mode;
    bool mem_mode;
    int fmt;

    State state; //!< current state of the FileStorage (used only for writing)
    bool is_using_base64;
    bool is_write_struct_delayed;
    char* delayed_struct_key;
    int   delayed_struct_flags;
    char* delayed_type_name;
    FileStorage_API::Base64State state_of_writing_base64;

    int space, wrap_margin;
    std::deque<FStructData> write_stack;
    std::vector<char> buffer;
    size_t bufofs;

    std::deque<char> outbuf;

    Ptr<FileStorageEmitter> emitter_do_not_use_direct_dereference;
    FileStorageEmitter& getEmitter()
    {
        if (!emitter_do_not_use_direct_dereference)
            CV_Error(Error::StsNullPtr, "Emitter is not available");
        return *emitter_do_not_use_direct_dereference;
    }
    Ptr<FileStorageParser> parser_do_not_use_direct_dereference;
    FileStorageParser& getParser() const
    {
        if (!parser_do_not_use_direct_dereference)
            CV_Error(Error::StsNullPtr, "Parser is not available");
        return *parser_do_not_use_direct_dereference;
    }
    Base64Decoder base64decoder;
    base64::Base64Writer* base64_writer;

    std::vector<FileNode> roots;
    std::vector<Ptr<std::vector<uchar> > > fs_data;
    std::vector<uchar*> fs_data_ptrs;
    std::vector<size_t> fs_data_blksz;
    size_t freeSpaceOfs;
    typedef std::unordered_map<std::string, unsigned> str_hash_t;
    str_hash_t str_hash;
    std::vector<char> str_hash_data;

    std::vector<char> strbufv;
    char* strbuf;
    size_t strbufsize;
    size_t strbufpos;
    int lineno;
};

}

#endif
