// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "persistence.hpp"


namespace cv
{

class FileNodeEmitter : public FileStorageEmitter
{
public:
    FileNodeEmitter(FileStorage_API* _fs) : fs(_fs)
    {
    }
    virtual ~FileNodeEmitter() {}

    FStructData startWriteStruct(const FStructData& parent, const char* key,
                                 int struct_flags, const char* type_name=0)
    {
        const bool is_named = key && key[0];

        if ( type_name && *type_name == '\0' )
            type_name = 0;

        struct_flags = (struct_flags & (FileNode::TYPE_MASK|FileNode::FLOW)) | FileNode::EMPTY;
        if( is_named )
            struct_flags |= FileNode::NAMED;
        if( !FileNode::isCollection(struct_flags))
            CV_Error( CV_StsBadArg,
                     "Some collection type - FileNode::SEQ or FileNode::MAP, must be specified" );

        if (type_name && memcmp(type_name, "binary", 6) == 0)
        {
            CV_Error(Error::StsNotImplemented, "binary FileNode emitter not implemented");
        }
        else if( FileNode::isFlow(struct_flags) ) {}
        else if( type_name ) {}

        FStructData fsd;
        fsd.flags = struct_flags;
        FileNode parent_node = ( fs->isBusy() ? parent.startNode : *fs->getCurrentNode().parent );
        fsd.startNode = fs->addNode(parent_node, key, struct_flags);
        fsd.startNode.nodeName = key;
        return fsd;
    }

    void endWriteStruct(const FStructData& current_struct)
    {
        CV_Assert( !current_struct.startNode.empty() );
        FileNode collection = current_struct.startNode;
        fs->finalizeCollection(collection);
    }

    void setValue(FileNode&& node, int type, const char* key, const void* value)
    {
        CV_Assert( fs->isBusy() || node.name() == key );
        node.setValue(type, value);
    }

    void write(const char* key, int value)
    {
        if( fs->isBusy() )
            fs->getCurrentStruct().startNode.addNode(key, FileNode::INT, &value);
        else
            setValue(fs->getCurrentNode(), FileNode::INT, key, &value);
    }

    void write( const char* key, double value )
    {
        if( fs->isBusy() )
            fs->getCurrentStruct().startNode.addNode(key, FileNode::REAL, &value);
        else
            setValue(fs->getCurrentNode(), FileNode::REAL, key, &value);
    }

    void write(const char* key, const char* str, bool /* quote */)
    {
        if( fs->isBusy() )
            fs->getCurrentStruct().startNode.addNode(key, FileNode::STRING, str);
        else
            setValue(fs->getCurrentNode(), FileNode::STRING, key, str);
    }

    void writeScalar(const char* /* key */, const char* /* data */)
    {
        CV_Error(Error::StsNotImplemented, "FileNode with unstructured data not implemented");
    }

    void writeComment(const char* /* comment */, bool /* eol_comment */)
    {
        CV_Error(Error::StsNotImplemented, "FileNode of type COMMENT not implemented");
    }

    void startNextStream()
    {
        CV_Error(Error::StsNotImplemented, "startNextStream() not implemented for FileNode emitter");
    }

protected:
    FileStorage_API* fs;
};

Ptr<FileStorageEmitter> createFileNodeEmitter(FileStorage_API* fs)
{
    return makePtr<FileNodeEmitter>(fs);
}

} // namespace cv
