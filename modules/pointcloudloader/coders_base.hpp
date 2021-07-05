#ifndef _CODERS_BASE_H_
#define _CODERS_BASE_H_

#include <vector>
#include <string>
#include <opencv2/core.hpp>


namespace cv 
{
namespace pc
{
    
class BasePointCloudDecoder;
typedef Ptr<BasePointCloudDecoder> PointCloudDecoder;

///////////////////////////////// base class for decoders ////////////////////////
class BasePointCloudDecoder
{
public:
    BasePointCloudDecoder();
    virtual ~BasePointCloudDecoder() {}

    virtual void setSource( const std::string& filename );
    virtual bool readHeader() = 0;
    virtual bool readData( std::vector<Point3f>& points, std::vector<Point3f>& normals ) = 0;

    virtual PointCloudDecoder newDecoder() const;

protected:
    std::string m_filename;
    
    std::vector<Point3f> m_vertices;
    std::vector<Point3f> m_normals;
    //std::vector<int> indices;

    int m_vertices_size;
};

}

}

#endif