#include "load_point_cloud.hpp"

#include "coders_base.hpp"
#include "coders_obj.hpp"
#include "coders_ply.hpp"


namespace cv 
{
namespace pc
{

static PointCloudDecoder findDecoder( const String& filename ) 
{
    int filename_length = filename.length();
    if (filename.substr(filename_length - 4, 4) == ".obj")
    {
        return  std::make_unique<ObjDecoder>();
    }
    if (filename.substr(filename_length - 4, 4) == ".ply")
    {
        //return  std::make_unique<PlyDecoder>();
    }

}

static PointCloudEncoder findEncoder( const String& filename ) 
{
    int filename_length = filename.length();
    if (filename.substr(filename_length - 4, 4) == ".obj")
    {
        return  std::make_unique<ObjEncoder>();
    }
    if (filename.substr(filename_length - 4, 4) == ".ply")
    {
        //return  std::make_unique<PlyDecoder>();
    }

}

void loadPointCloud( const String& filename, OutputArray points, OutputArray normals )
{
    PointCloudDecoder decoder;

    decoder = findDecoder( filename );

    decoder->setSource(filename);

    std::vector<Point3f> vec_points;
    std::vector<Point3f> vec_normals;

    decoder->readData(vec_points, vec_normals);

    if(!vec_points.empty())
        Mat((int)vec_points.size(), 1, CV_32FC3, &vec_points[0]).copyTo(points);

    if(!vec_normals.empty())
        Mat((int)vec_normals.size(), 1, CV_32FC3, &vec_normals[0]).copyTo(normals);
}

void savePointCloud( const String& filename, InputArray vertices, InputArray normals)
{
    
    CV_Assert(!vertices.empty());
    CV_Assert(!normals.empty());
    
    PointCloudEncoder encoder;

    encoder = findEncoder( filename );

    encoder->setDestination(filename);

    Mat mat_vertices = vertices.getMat();
    Mat mat_normals = normals.getMat();
    
    std::vector<Point3f> vec_points;
    std::vector<Point3f> vec_normals;

    for (int i = 0; i < mat_vertices.cols; i++ )
    {
        vec_points.push_back(mat_vertices.at<Point3f>(0, i));
    }
    
    for (int i = 0; i < mat_normals.cols; i++ )
    {
        vec_normals.push_back(mat_normals.at<Point3f>(0, i));
    }

    encoder->writeData(vec_points, vec_normals);
}

}

}
