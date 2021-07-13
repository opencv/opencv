#ifndef _CODERS_PLY_H_
#define _CODERS_PLY_H_

#include "coders_base.hpp" 

namespace cv 
{
namespace pc
{

class PlyDecoder CV_FINAL : public BasePointCloudDecoder
{
public:

    PlyDecoder();
    ~PlyDecoder() CV_OVERRIDE;

    void readData( std::vector<Point3f>& points, std::vector<Point3f>& normals ) CV_OVERRIDE;

    PointCloudDecoder newDecoder() const CV_OVERRIDE;

protected:

};

}

}

#endif