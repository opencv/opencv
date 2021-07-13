#ifndef _CODERS_BASE_H_
#define _CODERS_BASE_H_

#include <vector>
#include <string>
#include <memory>

#include <opencv2/core.hpp>


namespace cv 
{
namespace pc
{
    
class BasePointCloudDecoder;
class BasePointCloudEncoder;
typedef std::unique_ptr<BasePointCloudDecoder> PointCloudDecoder;
typedef std::unique_ptr<BasePointCloudEncoder> PointCloudEncoder;

///////////////////////////////// base class for decoders ////////////////////////
class BasePointCloudDecoder
{
public:
    BasePointCloudDecoder();
    virtual ~BasePointCloudDecoder() {}

    virtual void setSource( const std::string& filename );
    virtual void readData( std::vector<Point3f>& points, std::vector<Point3f>& normals ) = 0;

    virtual PointCloudDecoder newDecoder() const;

protected:
    std::string m_filename;
    
    std::vector<Point3f> m_vertices;
    std::vector<Point3f> m_normals;
    //std::vector<int> indices;

    int m_vertices_size;
};


///////////////////////////////// base class for encoders ////////////////////////
class BasePointCloudEncoder
{
public:
    BasePointCloudEncoder();
    virtual ~BasePointCloudEncoder() {}

    virtual void setDestination( const std::string& filename );
    virtual void writeData( std::vector<Point3f>& points, std::vector<Point3f>& normals  ) = 0;

    virtual PointCloudEncoder newEncoder() const;

protected:
    std::string m_filename;
    
    std::vector<Point3f> m_vertices;
    std::vector<Point3f> m_normals;
    //std::vector<int> indices;
    
    std::string m_last_error;
};

}

}

#endif