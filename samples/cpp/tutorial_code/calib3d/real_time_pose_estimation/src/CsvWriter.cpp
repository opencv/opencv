#include "CsvWriter.h"

CsvWriter::CsvWriter(const string &path, const string &separator){
    _file.open(path.c_str(), ofstream::out);
    _isFirstTerm = true;
    _separator = separator;
}

CsvWriter::~CsvWriter() {
    _file.flush();
    _file.close();
}

void CsvWriter::writeXYZ(const vector<Point3f> &list_points3d)
{
    for(size_t i = 0; i < list_points3d.size(); ++i)
    {
        string x = FloatToString(list_points3d[i].x);
        string y = FloatToString(list_points3d[i].y);
        string z = FloatToString(list_points3d[i].z);

        _file << x << _separator << y << _separator << z << std::endl;
    }
}

void CsvWriter::writeUVXYZ(const vector<Point3f> &list_points3d, const vector<Point2f> &list_points2d, const Mat &descriptors)
{
    for(size_t i = 0; i < list_points3d.size(); ++i)
    {
        string u = FloatToString(list_points2d[i].x);
        string v = FloatToString(list_points2d[i].y);
        string x = FloatToString(list_points3d[i].x);
        string y = FloatToString(list_points3d[i].y);
        string z = FloatToString(list_points3d[i].z);

        _file << u << _separator << v << _separator << x << _separator << y << _separator << z;

        for(int j = 0; j < 32; ++j)
        {
            string descriptor_str = FloatToString(descriptors.at<float>((int)i,j));
            _file << _separator << descriptor_str;
        }
        _file << std::endl;
    }
}
