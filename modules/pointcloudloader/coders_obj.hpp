#ifndef _CODERS_OBJ_H_
#define _CODERS_OBJ_H_

#include "coders_base.hpp" 

namespace cv 
{
namespace pc
{

class ObjDecoder CV_FINAL : public BasePointCloudDecoder
{
public:

    ObjDecoder();
    ~ObjDecoder() CV_OVERRIDE;

    void readData( std::vector<Point3f>& points, std::vector<Point3f>& normals ) CV_OVERRIDE;

    PointCloudDecoder newDecoder() const CV_OVERRIDE;

protected:

};

class ObjEncoder CV_FINAL : public BasePointCloudEncoder
{
public:

    ObjEncoder();
    ~ObjEncoder() CV_OVERRIDE;

    void writeData( std::vector<Point3f>& points, std::vector<Point3f>& normals ) CV_OVERRIDE;

    PointCloudEncoder newEncoder() const CV_OVERRIDE;

protected:

};

}

}

#endif