#include "coders_base.hpp"

namespace cv
{
namespace pc
{

BasePointCloudDecoder::BasePointCloudDecoder()
{
    m_vertices_size = 0;
}

void BasePointCloudDecoder::setSource( const std::string& filename )
{
    m_filename = filename;
}

PointCloudDecoder BasePointCloudDecoder::newDecoder() const
{
    return PointCloudDecoder();
}


BasePointCloudEncoder::BasePointCloudEncoder()
{
    
}


void BasePointCloudEncoder::setDestination( const std::string& filename )
{
    m_filename = filename;
}

PointCloudEncoder BasePointCloudEncoder::newEncoder() const
{
    return PointCloudEncoder();
}

}

}
