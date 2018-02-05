// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "persistence.hpp"

///////////////////////// new C++ interface for CvFileStorage ///////////////////////////

namespace cv
{

static void getElemSize( const String& fmt, size_t& elemSize, size_t& cn )
{
    const char* dt = fmt.c_str();
    cn = 1;
    if( cv_isdigit(dt[0]) )
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

FileStorage::FileStorage(const String& filename, int flags, const String& encoding)
{
    state = UNDEFINED;
    open( filename, flags, encoding );
}

FileStorage::FileStorage(CvFileStorage* _fs, bool owning)
{
    if (owning) fs.reset(_fs);
    else fs = Ptr<CvFileStorage>(Ptr<CvFileStorage>(), _fs);

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

bool FileStorage::open(const String& filename, int flags, const String& encoding)
{
    CV_INSTRUMENT_REGION()

    release();
    fs.reset(cvOpenFileStorage( filename.c_str(), 0, flags,
                                !encoding.empty() ? encoding.c_str() : 0));
    bool ok = isOpened();
    state = ok ? NAME_EXPECTED + INSIDE_MAP : UNDEFINED;
    return ok;
}

bool FileStorage::isOpened() const
{
    return fs && fs->is_opened;
}

void FileStorage::release()
{
    fs.release();
    structs.clear();
    state = UNDEFINED;
}

String FileStorage::releaseAndGetString()
{
    String buf;
    if( fs && fs->outbuf )
        icvClose(fs, &buf);

    release();
    return buf;
}

FileNode FileStorage::root(int streamidx) const
{
    return isOpened() ? FileNode(fs, cvGetRootFileNode(fs, streamidx)) : FileNode();
}

int FileStorage::getFormat() const
{
    CV_Assert(!fs.empty());
    return fs->fmt & FORMAT_MASK;
}

FileStorage& operator << (FileStorage& fs, const String& str)
{
    CV_TRACE_REGION_VERBOSE();

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
        fs.elname = String();
    }
    else if( fs.state == NAME_EXPECTED + INSIDE_MAP )
    {
        if (!cv_isalpha(*_str) && *_str != '_')
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
            fs.elname = String();
        }
        else
        {
            write( fs, fs.elname, (_str[0] == '\\' && (_str[1] == '{' || _str[1] == '}' ||
                _str[1] == '[' || _str[1] == ']')) ? String(_str+1) : str );
            if( fs.state == INSIDE_MAP + VALUE_EXPECTED )
                fs.state = INSIDE_MAP + NAME_EXPECTED;
        }
    }
    else
        CV_Error( CV_StsError, "Invalid fs.state" );
    return fs;
}


void FileStorage::writeRaw( const String& fmt, const uchar* vec, size_t len )
{
    if( !isOpened() )
        return;
    size_t elemSize, cn;
    getElemSize( fmt, elemSize, cn );
    CV_Assert( len % elemSize == 0 );
    cvWriteRawData( fs, vec, (int)(len/elemSize), fmt.c_str());
}


void FileStorage::writeObj( const String& name, const void* obj )
{
    if( !isOpened() )
        return;
    cvWrite( fs, name.size() > 0 ? name.c_str() : 0, obj );
}

void FileStorage::write( const String& name, double val )
{
    *this << name << val;
}

void FileStorage::write( const String& name, const String& val )
{
    *this << name << val;
}

void FileStorage::write( const String& name, InputArray val )
{
    *this << name << val.getMat();
}

void FileStorage::writeComment( const String& comment, bool append )
{
    cvWriteComment(fs, comment.c_str(), append ? 1 : 0);
}

String FileStorage::getDefaultObjectName(const String& _filename)
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
    name = name_buf;
    if( strcmp( name, "_" ) == 0 )
        strcpy( name, stubname );
    return String(name);
}

FileNode FileStorage::operator[](const String& nodename) const
{
    return FileNode(fs, cvGetFileNodeByName(fs, 0, nodename.c_str()));
}

FileNode FileStorage::operator[](const char* nodename) const
{
    return FileNode(fs, cvGetFileNodeByName(fs, 0, nodename));
}

FileNode FileNode::operator[](const String& nodename) const
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

String FileNode::name() const
{
    const char* str;
    return !node || (str = cvGetFileNodeName(node)) == 0 ? String() : String(str);
}

void* FileNode::readObj() const
{
    if( !fs || !node )
        return 0;
    return cvRead( (CvFileStorage*)fs, (CvFileNode*)node );
}

static const FileNodeIterator::SeqReader emptyReader = {0, 0, 0, 0, 0, 0, 0, 0};

FileNodeIterator::FileNodeIterator()
{
    fs = 0;
    container = 0;
    reader = emptyReader;
    remaining = 0;
}

FileNodeIterator::FileNodeIterator(const CvFileStorage* _fs,
                                   const CvFileNode* _node, size_t _ofs)
{
    reader = emptyReader;
    if( _fs && _node && CV_NODE_TYPE(_node->tag) != CV_NODE_NONE )
    {
        int node_type = _node->tag & FileNode::TYPE_MASK;
        fs = _fs;
        container = _node;
        if( !(_node->tag & FileNode::USER) && (node_type == FileNode::SEQ || node_type == FileNode::MAP) )
        {
            cvStartReadSeq( _node->data.seq, (CvSeqReader*)&reader );
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
        {
            if( ((reader).ptr += (((CvSeq*)reader.seq)->elem_size)) >= (reader).block_max )
            {
                cvChangeSeqBlock( (CvSeqReader*)&(reader), 1 );
            }
        }
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
        {
            if( ((reader).ptr -= (((CvSeq*)reader.seq)->elem_size)) < (reader).block_min )
            {
                cvChangeSeqBlock( (CvSeqReader*)&(reader), -1 );
            }
        }
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
        cvSetSeqReaderPos( (CvSeqReader*)&reader, ofs, 1 );
    return *this;
}

FileNodeIterator& FileNodeIterator::operator -= (int ofs)
{
    return operator += (-ofs);
}


FileNodeIterator& FileNodeIterator::readRaw( const String& fmt, uchar* vec, size_t maxCount )
{
    if( fs && container && remaining > 0 )
    {
        size_t elem_size, cn;
        getElemSize( fmt, elem_size, cn );
        CV_Assert( elem_size > 0 );
        size_t count = std::min(remaining, maxCount);

        if( reader.seq )
        {
            cvReadRawDataSlice( fs, (CvSeqReader*)&reader, (int)count, vec, fmt.c_str() );
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


void write( FileStorage& fs, const String& name, int value )
{ cvWriteInt( *fs, name.size() ? name.c_str() : 0, value ); }

void write( FileStorage& fs, const String& name, float value )
{ cvWriteReal( *fs, name.size() ? name.c_str() : 0, value ); }

void write( FileStorage& fs, const String& name, double value )
{ cvWriteReal( *fs, name.size() ? name.c_str() : 0, value ); }

void write( FileStorage& fs, const String& name, const String& value )
{ cvWriteString( *fs, name.size() ? name.c_str() : 0, value.c_str() ); }

void writeScalar(FileStorage& fs, int value )
{ cvWriteInt( *fs, 0, value ); }

void writeScalar(FileStorage& fs, float value )
{ cvWriteReal( *fs, 0, value ); }

void writeScalar(FileStorage& fs, double value )
{ cvWriteReal( *fs, 0, value ); }

void writeScalar(FileStorage& fs, const String& value )
{ cvWriteString( *fs, 0, value.c_str() ); }


void write( FileStorage& fs, const String& name, const Mat& value )
{
    if( value.dims <= 2 )
    {
        CvMat mat = value;
        cvWrite( *fs, name.size() ? name.c_str() : 0, &mat );
    }
    else
    {
        CvMatND mat = value;
        cvWrite( *fs, name.size() ? name.c_str() : 0, &mat );
    }
}

// TODO: the 4 functions below need to be implemented more efficiently
void write( FileStorage& fs, const String& name, const SparseMat& value )
{
    Ptr<CvSparseMat> mat(cvCreateSparseMat(value));
    cvWrite( *fs, name.size() ? name.c_str() : 0, mat );
}


internal::WriteStructContext::WriteStructContext(FileStorage& _fs,
    const String& name, int flags, const String& typeName) : fs(&_fs)
{
    cvStartWriteStruct(**fs, !name.empty() ? name.c_str() : 0, flags,
                       !typeName.empty() ? typeName.c_str() : 0);
    fs->elname = String();
    if ((flags & FileNode::TYPE_MASK) == FileNode::SEQ)
    {
        fs->state = FileStorage::VALUE_EXPECTED;
        fs->structs.push_back('[');
    }
    else
    {
        fs->state = FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP;
        fs->structs.push_back('{');
    }
}

internal::WriteStructContext::~WriteStructContext()
{
    cvEndWriteStruct(**fs);
    fs->structs.pop_back();
    fs->state = fs->structs.empty() || fs->structs.back() == '{' ?
        FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP :
        FileStorage::VALUE_EXPECTED;
    fs->elname = String();
}


void read( const FileNode& node, Mat& mat, const Mat& default_mat )
{
    if( node.empty() )
    {
        default_mat.copyTo(mat);
        return;
    }
    void* obj = cvRead((CvFileStorage*)node.fs, (CvFileNode*)*node);
    if(CV_IS_MAT_HDR_Z(obj))
    {
        cvarrToMat(obj).copyTo(mat);
        cvReleaseMat((CvMat**)&obj);
    }
    else if(CV_IS_MATND_HDR(obj))
    {
        cvarrToMat(obj).copyTo(mat);
        cvReleaseMatND((CvMatND**)&obj);
    }
    else
    {
        cvRelease(&obj);
        CV_Error(CV_StsBadArg, "Unknown array type");
    }
}

void read( const FileNode& node, SparseMat& mat, const SparseMat& default_mat )
{
    if( node.empty() )
    {
        default_mat.copyTo(mat);
        return;
    }
    Ptr<CvSparseMat> m((CvSparseMat*)cvRead((CvFileStorage*)node.fs, (CvFileNode*)*node));
    CV_Assert(CV_IS_SPARSE_MAT(m));
    m->copyToSparseMat(mat);
}

CV_EXPORTS void read(const FileNode& node, KeyPoint& value, const KeyPoint& default_value)
{
    if( node.empty() )
    {
        value = default_value;
        return;
    }
    node >> value;
}

CV_EXPORTS void read(const FileNode& node, DMatch& value, const DMatch& default_value)
{
    if( node.empty() )
    {
        value = default_value;
        return;
    }
    node >> value;
}

#ifdef CV__LEGACY_PERSISTENCE
void write( FileStorage& fs, const String& name, const std::vector<KeyPoint>& vec)
{
    // from template implementation
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ);
    write(fs, vec);
}

void read(const FileNode& node, std::vector<KeyPoint>& keypoints)
{
    FileNode first_node = *(node.begin());
    if (first_node.isSeq())
    {
        // modern scheme
#ifdef OPENCV_TRAITS_ENABLE_DEPRECATED
        FileNodeIterator it = node.begin();
        size_t total = (size_t)it.remaining;
        keypoints.resize(total);
        for (size_t i = 0; i < total; ++i, ++it)
        {
            (*it) >> keypoints[i];
        }
#else
        FileNodeIterator it = node.begin();
        it >> keypoints;
#endif
        return;
    }
    keypoints.clear();
    FileNodeIterator it = node.begin(), it_end = node.end();
    for( ; it != it_end; )
    {
        KeyPoint kpt;
        it >> kpt.pt.x >> kpt.pt.y >> kpt.size >> kpt.angle >> kpt.response >> kpt.octave >> kpt.class_id;
        keypoints.push_back(kpt);
    }
}

void write( FileStorage& fs, const String& name, const std::vector<DMatch>& vec)
{
    // from template implementation
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ);
    write(fs, vec);
}

void read(const FileNode& node, std::vector<DMatch>& matches)
{
    FileNode first_node = *(node.begin());
    if (first_node.isSeq())
    {
        // modern scheme
#ifdef OPENCV_TRAITS_ENABLE_DEPRECATED
        FileNodeIterator it = node.begin();
        size_t total = (size_t)it.remaining;
        matches.resize(total);
        for (size_t i = 0; i < total; ++i, ++it)
        {
            (*it) >> matches[i];
        }
#else
        FileNodeIterator it = node.begin();
        it >> matches;
#endif
        return;
    }
    matches.clear();
    FileNodeIterator it = node.begin(), it_end = node.end();
    for( ; it != it_end; )
    {
        DMatch m;
        it >> m.queryIdx >> m.trainIdx >> m.imgIdx >> m.distance;
        matches.push_back(m);
    }
}
#endif

int FileNode::type() const { return !node ? NONE : (node->tag & TYPE_MASK); }
bool FileNode::isNamed() const { return !node ? false : (node->tag & NAMED) != 0; }

size_t FileNode::size() const
{
    int t = type();
    return t == MAP ? (size_t)((CvSet*)node->data.map)->active_count :
        t == SEQ ? (size_t)node->data.seq->total : (size_t)!isNone();
}

void read(const FileNode& node, int& value, int default_value)
{
    value = !node.node ? default_value :
    CV_NODE_IS_INT(node.node->tag) ? node.node->data.i : std::numeric_limits<int>::max();
}

void read(const FileNode& node, float& value, float default_value)
{
    value = !node.node ? default_value :
        CV_NODE_IS_INT(node.node->tag) ? (float)node.node->data.i :
        CV_NODE_IS_REAL(node.node->tag) ? saturate_cast<float>(node.node->data.f) : std::numeric_limits<float>::max();
}

void read(const FileNode& node, double& value, double default_value)
{
    value = !node.node ? default_value :
        CV_NODE_IS_INT(node.node->tag) ? (double)node.node->data.i :
        CV_NODE_IS_REAL(node.node->tag) ? node.node->data.f : std::numeric_limits<double>::max();
}

void read(const FileNode& node, String& value, const String& default_value)
{
    value = !node.node ? default_value : CV_NODE_IS_STRING(node.node->tag) ? String(node.node->data.str.ptr) : String();
}

void read(const FileNode& node, std::string& value, const std::string& default_value)
{
    value = !node.node ? default_value : CV_NODE_IS_STRING(node.node->tag) ? std::string(node.node->data.str.ptr) : default_value;
}

} // cv::
