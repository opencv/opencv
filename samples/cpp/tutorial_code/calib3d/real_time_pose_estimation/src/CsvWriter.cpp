#include <boost/lexical_cast.hpp>
#include "CsvWriter.h"

CsvWriter::CsvWriter(const std::string &path, const std::string &separator){
  _file.open(path.c_str(), std::ofstream::out);
  _isFirstTerm = true;
  _separator = separator;
}

CsvWriter::~CsvWriter() {
  _file.flush();
  _file.close();
}

void CsvWriter::writeXYZ(const std::vector<cv::Point3f> &list_points3d)
{
  std::string x, y, z;
  for(unsigned int i = 0; i < list_points3d.size(); ++i)
  {
    x = boost::lexical_cast< std::string >(list_points3d[i].x);
    y = boost::lexical_cast< std::string >(list_points3d[i].y);
    z = boost::lexical_cast< std::string >(list_points3d[i].z);

    _file << x << _separator << y << _separator << z << std::endl;
  }

}

void CsvWriter::writeUVXYZ(const std::vector<cv::Point3f> &list_points3d, const std::vector<cv::Point2f> &list_points2d, const cv::Mat &descriptors)
{
  std::string u, v, x, y, z, descriptor_str;
  for(int i = 0; i < list_points3d.size(); ++i)
  {
    u = boost::lexical_cast< std::string >(list_points2d[i].x);
    v = boost::lexical_cast< std::string >(list_points2d[i].y);
    x = boost::lexical_cast< std::string >(list_points3d[i].x);
    y = boost::lexical_cast< std::string >(list_points3d[i].y);
    z = boost::lexical_cast< std::string >(list_points3d[i].z);

    _file << u << _separator << v << _separator << x << _separator << y << _separator << z;

    for(int j = 0; j < 32; ++j)
    {
      std::cout << descriptors.at<float>(i,j) << std::endl;
      descriptor_str = boost::lexical_cast< std::string >(descriptors.at<float>(i,j));
      _file << _separator << descriptor_str;
    }
    _file << std::endl;
  }
}
