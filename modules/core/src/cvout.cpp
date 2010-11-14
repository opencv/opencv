#include "opencv2/core/core.hpp"

namespace cv
{

namespace
{
template<typename T>
  std::ostream & writevec(std::ostream & out, const std::vector<T> & points)
  {
    typedef T MT_T;
    typedef typename std::vector<T>::const_iterator CIT;
    /* Draw Me:
     plot2( pts1(:,1),pts1(:,2),'r.')  */
    std::streamsize pp = out.precision();
    out.precision(15);
    out << "[";

    CIT it = points.begin();

    for (; it != points.end(); ++it)
    {

      out << *it;
      CIT next = it;
      if (++next != points.end())
      {
        out << " ";
      }
    }
    out << "]";
    out.precision(pp);
    return out;
  }

std::ostream & writeelem(std::ostream & out, const Mat & mat, int i, int j)
{
  if (mat.type() == CV_32F)
    out << mat.at<float> (i, j);
  else if (mat.type() == CV_64F)
    out << mat.at<double> (i, j);
  else if (mat.type() == CV_32S)
    out << mat.at<int> (i, j);
  else if (mat.type() == CV_8U)
    out << int(mat.at<unsigned char> (i, j));
  else if (mat.type() == CV_16U)
    out << int(mat.at<unsigned short> (i, j));
  else
    out << "?";
  return out;
}

}

std::ostream & operator<<(std::ostream & out, const std::vector<Point2f> & points)
{
  return writevec(out, points);
}
std::ostream & operator<<(std::ostream & out, const std::vector<Point3f> & points)
{
  return writevec(out, points);
}

std::ostream & operator<<(std::ostream & out, const Mat & _mat)
{
  std::streamsize pp = out.precision();
  out.precision(15);
  std::vector<Mat> channels;
  split(_mat, channels);
  for (int chn = 0; chn < _mat.channels(); chn++)
  {
    Mat mat = channels[chn];
    out << "[";
    for (int i = 0; i < mat.rows; i++)
    {
      for (int j = 0; j < mat.cols; j++)
      {
        writeelem(out,mat,i,j);
        if (j < mat.cols - 1)
          out << " ";
      }
      if (i < mat.rows - 1)
        out << ";\n";
    }
    out << "]";
  }
  out.precision(pp);
  return out;
}

std::ostream & writeCSV(std::ostream & out, const std::vector<Point3f> & points)
{
  std::streamsize pp = out.precision();
  out.precision(15);
  std::vector<Point3f>::const_iterator it = points.begin();

  for (; it != points.end(); ++it)
  {
    out << it->x << "," << it->y << ","<< it->z << "\n";
  }
  out.precision(pp);
  return out;
}

std::ostream & writeCSV(std::ostream & out, const std::vector<Point2f> & points)
{
  std::streamsize pp = out.precision();
  out.precision(15);
  std::vector<Point2f>::const_iterator it = points.begin();

  for (; it != points.end(); ++it)
  {
    out << it->x << "," << it->y << "\n";
  }
  out.precision(pp);
  return out;
}

std::ostream & writeCSV(std::ostream & out, const Mat & mat)
{
  std::streamsize pp = out.precision();
  out.precision(15);
  for (int i = 0; i < mat.rows; i++)
  {

    for (int j = 0; j < mat.cols; j++)
    {
      writeelem(out,mat,i,j);
      if (j < mat.cols - 1)
        out << ",";
    }
    out << "\n";
  }
  out.precision(pp);
  return out;
}

}

