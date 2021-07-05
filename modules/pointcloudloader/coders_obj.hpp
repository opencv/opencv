#ifndef _CODERS_OBJ_H_
#define _CODERS_OBJ_H_

#include "coders_base.hpp"

class ObjDecoder CV_FINAL : public BasePointCloudDecoder
{
public:

    ObjDecoder();
    ~ObjDecoder() CV_OVERRIDE;

    void readData( std::vector<Point3f>& points, std::vector<Point3f>& normals ) CV_OVERRIDE;

    PointCloudDecoder newDecoder() const CV_OVERRIDE;

protected:

};

#endif