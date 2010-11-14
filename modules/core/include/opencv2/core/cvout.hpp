#ifndef __OPENCV_CORE_CVOUT_HPP__
#define __OPENCV_CORE_CVOUT_HPP__
#ifdef __cplusplus

#ifndef SKIP_INCLUDES
  #include <iomanip>
  #include <iostream>
  #include <vector>
#endif

namespace cv
{

/** Writes a point to an output stream in Matlab notation
 */
inline std::ostream & operator<<(std::ostream & out, const Point2f & p)
{
  out << "[ " << p.x << "," << p.y << " ]";
  return out;
}

/** Writes a point to an output stream in Matlab notation
 */
inline std::ostream & operator<<(std::ostream & out, const Point3f & p)
{
  out << "[ " << p.x << "," << p.y << "," << p.z << " ]";
  return out;
}

/** \brief	 write points to and output stream
 *  \param out typically cout
 *  \param points the points to be written to the stream
 *  \return	 the stream
 **/
CV_EXPORTS std::ostream & operator<<(std::ostream & out, const std::vector<Point2f> & points);

/** \brief	 write points to and output stream
 *  \param out typically cout
 *  \param points the points to be written to the stream
 *  \return	 the stream
 **/
std::ostream & operator<<(std::ostream & out, const std::vector<Point3f> & points);

/** \brief allows each output of Mat in Matlab for Mat to std::cout
 * use like
 @verbatim
 Mat my_mat = Mat::eye(3,3,CV_32F);
 std::cout << my_mat;
 @endverbatim
 */
CV_EXPORTS std::ostream & operator<<(std::ostream & out, const Mat & mat);

/** \brief write a Mat in csv compatible for Matlab.
 This means that the rows are seperated by newlines and the
 columns by commas ....
 331.413896619595,0,122.365880226491
 0,249.320451610369,122.146722131871
 0,0,1

 *	\param out output stream to write to
 *	\param Mat write a Mat to a csv
 */
CV_EXPORTS std::ostream & writeCSV(std::ostream & out, const Mat & mat);

/** \brief write a vector of points to an
 output stream if possible
 **/
CV_EXPORTS std::ostream & writeCSV(std::ostream & out, const std::vector<Point2f> & points);

/** \brief write a vector of points to an
 output stream if possible
 **/
CV_EXPORTS std::ostream & writeCSV(std::ostream & out, const std::vector<Point3f> & points);

} //namespace cv

#endif

#endif
