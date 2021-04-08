// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

/** This is an implementation of a fast plane detection loosely inspired by
 * Fast Plane Detection and Polygonalization in noisy 3D Range Images
 * Jann Poppinga, Narunas Vaskevicius, Andreas Birk, and Kaustubh Pathak
 * and the follow-up
 * Fast Plane Detection for SLAM from Noisy Range Images in
 * Both Structured and Unstructured Environments
 * Junhao Xiao, Jianhua Zhang and Jianwei Zhang
 * Houxiang Zhang and Hans Petter Hildre
 */

#include "precomp.hpp"

namespace cv
{
namespace rgbd
{
/** Structure defining a plane. The notations are from the second paper */
class PlaneBase
{
public:
  PlaneBase(const Vec3f & m, const Vec3f &n_in, int index)
      :
        index_(index),
        n_(n_in),
        m_sum_(Vec3f(0, 0, 0)),
        m_(m),
        Q_(Matx33f::zeros()),
        mse_(0),
        K_(0)
  {
    UpdateD();
  }

  virtual
  ~PlaneBase()
  {
  }

  /** Compute the distance to the plane. This will be implemented by the children to take into account different
   * sensor models
   * @param p_j
   * @return
   */
  virtual
  float
  distance(const Vec3f& p_j) const = 0;

  /** The d coefficient in the plane equation ax+by+cz+d = 0
   * @return
   */
  inline float
  d() const
  {
    return d_;
  }

  /** The normal to the plane
   * @return the normal to the plane
   */
  const Vec3f &
  n() const
  {
    return n_;
  }

  /** Update the different coefficients of the plane, based on the new statistics
   */
  void
  UpdateParameters()
  {
    if (empty())
      return;
    m_ = m_sum_ / K_;
    // Compute C
    Matx33f C = Q_ - m_sum_ * m_.t();

    // Compute n
    SVD svd(C);
    n_ = Vec3f(svd.vt.at<float>(2, 0), svd.vt.at<float>(2, 1), svd.vt.at<float>(2, 2));
    mse_ = svd.w.at<float>(2) / K_;

    UpdateD();
  }

  /** Update the different sum of point and sum of point*point.t()
   */
  void
  UpdateStatistics(const Vec3f & point, const Matx33f & Q_local)
  {
    m_sum_ += point;
    Q_ += Q_local;
    ++K_;
  }

  inline size_t
  empty() const
  {
    return K_ == 0;
  }

  inline int
  K() const
  {
    return K_;
  }
/** The index of the plane */
  int index_;
protected:
  /** The 4th coefficient in the plane equation ax+by+cz+d = 0 */
  float d_;
  /** Normal of the plane */
  Vec3f n_;
private:
  inline void
  UpdateD()
  {
    d_ = -m_.dot(n_);
  }
  /** The sum of the points */
  Vec3f m_sum_;
  /** The mean of the points */
  Vec3f m_;
  /** The sum of pi * pi^\top */
  Matx33f Q_;
  /** The different matrices we need to update */
  Matx33f C_;
  float mse_;
  /** the number of points that form the plane */
  int K_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Basic planar child, with no sensor error model
 */
class Plane: public PlaneBase
{
public:
  Plane(const Vec3f & m, const Vec3f &n_in, int index)
      :
        PlaneBase(m, n_in, index)
  {
  }

  /** The computed distance is perfect in that case
   * @param p_j the point to compute its distance to
   * @return
   */
  float
  distance(const Vec3f& p_j) const CV_OVERRIDE
  {
    return std::abs(float(p_j.dot(n_) + d_));
  }
};

/** Planar child with a quadratic error model
 */
class PlaneABC: public PlaneBase
{
public:
  PlaneABC(const Vec3f & m, const Vec3f &n_in, int index, float sensor_error_a, float sensor_error_b,
           float sensor_error_c)
      :
        PlaneBase(m, n_in, index),
        sensor_error_a_(sensor_error_a),
        sensor_error_b_(sensor_error_b),
        sensor_error_c_(sensor_error_c)
  {
  }

  /** The distance is now computed by taking the sensor error into account */
  inline
  float
  distance(const Vec3f& p_j) const CV_OVERRIDE
  {
    float cst = p_j.dot(n_) + d_;
    float err = sensor_error_a_ * p_j[2] * p_j[2] + sensor_error_b_ * p_j[2] + sensor_error_c_;
    if (((cst - n_[2] * err <= 0) && (cst + n_[2] * err >= 0)) || ((cst + n_[2] * err <= 0) && (cst - n_[2] * err >= 0)))
      return 0;
    return std::min(std::abs(cst - err), std::abs(cst + err));
  }
private:
  float sensor_error_a_;
  float sensor_error_b_;
  float sensor_error_c_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** The PlaneGrid contains statistic about the individual tiles
 */
class PlaneGrid
{
public:
  PlaneGrid(const Mat_<Vec3f> & points3d, int block_size)
      :
        block_size_(block_size)
  {
    // Figure out some dimensions
    int mini_rows = points3d.rows / block_size;
    if (points3d.rows % block_size != 0)
      ++mini_rows;

    int mini_cols = points3d.cols / block_size;
    if (points3d.cols % block_size != 0)
      ++mini_cols;

    // Compute all the interesting quantities
    m_.create(mini_rows, mini_cols);
    n_.create(mini_rows, mini_cols);
    Q_.create(points3d.rows, points3d.cols);
    mse_.create(mini_rows, mini_cols);
    for (int y = 0; y < mini_rows; ++y)
      for (int x = 0; x < mini_cols; ++x)
      {
        // Update the tiles
        Matx33f Q = Matx33f::zeros();
        Vec3f m = Vec3f(0, 0, 0);
        int K = 0;
        for (int j = y * block_size; j < std::min((y + 1) * block_size, points3d.rows); ++j)
        {
          const Vec3f * vec = points3d.ptr < Vec3f > (j, x * block_size), *vec_end;
          float * pointpointt = reinterpret_cast<float*>(Q_.ptr < Vec<float, 9> > (j, x * block_size));
          if (x == mini_cols - 1)
            vec_end = points3d.ptr < Vec3f > (j, points3d.cols - 1) + 1;
          else
            vec_end = vec + block_size;
          for (; vec != vec_end; ++vec, pointpointt += 9)
          {
            if (cvIsNaN(vec->val[0]))
              continue;
            // Fill point*point.t()
            *pointpointt = vec->val[0] * vec->val[0];
            *(pointpointt + 1) = vec->val[0] * vec->val[1];
            *(pointpointt + 2) = vec->val[0] * vec->val[2];
            *(pointpointt + 3) = *(pointpointt + 1);
            *(pointpointt + 4) = vec->val[1] * vec->val[1];
            *(pointpointt + 5) = vec->val[1] * vec->val[2];
            *(pointpointt + 6) = *(pointpointt + 2);
            *(pointpointt + 7) = *(pointpointt + 5);
            *(pointpointt + 8) = vec->val[2] * vec->val[2];

            Q += *reinterpret_cast<Matx33f*>(pointpointt);
            m += (*vec);
            ++K;
          }
        }
        if (K == 0)
        {
          mse_(y, x) = std::numeric_limits<float>::max();
          continue;
        }

        m /= K;
        m_(y, x) = m;

        // Compute C
        Matx33f C = Q - K * m * m.t();

        // Compute n
        SVD svd(C);
        n_(y, x) = Vec3f(svd.vt.at<float>(2, 0), svd.vt.at<float>(2, 1), svd.vt.at<float>(2, 2));
        mse_(y, x) = svd.w.at<float>(2) / K;
      }
  }

  /** The size of the block */
  int block_size_;
  Mat_<Vec3f> m_;
  Mat_<Vec3f> n_;
  Mat_<Vec<float, 9> > Q_;
  Mat_<float> mse_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TileQueue
{
public:
  struct PlaneTile
  {
    PlaneTile(int x, int y, float mse)
        :
          x_(x),
          y_(y),
          mse_(mse)
    {
    }

    bool
    operator<(const PlaneTile &tile2) const
    {
      return mse_ < tile2.mse_;
    }

    int x_;
    int y_;
    float mse_;
  };

  TileQueue(const PlaneGrid &plane_grid)
  {
    done_tiles_ = Mat_<unsigned char>::zeros(plane_grid.mse_.rows, plane_grid.mse_.cols);
    tiles_.clear();
    for (int y = 0; y < plane_grid.mse_.rows; ++y)
      for (int x = 0; x < plane_grid.mse_.cols; ++x)
        if (plane_grid.mse_(y, x) != std::numeric_limits<float>::max())
          // Update the tiles
          tiles_.push_back(PlaneTile(x, y, plane_grid.mse_(y, x)));
    // Sort tiles by MSE
    tiles_.sort();
  }

  bool
  empty()
  {
    while (!tiles_.empty())
    {
      const PlaneTile & tile = tiles_.front();
      if (done_tiles_(tile.y_, tile.x_))
        tiles_.pop_front();
      else
        break;
    }
    return tiles_.empty();
  }

  const PlaneTile &
  front() const
  {
    return tiles_.front();
  }

  void
  remove(int y, int x)
  {
    done_tiles_(y, x) = 1;
  }
private:
  /** The list of tiles ordered from most planar to least */
  std::list<PlaneTile> tiles_;
  /** contains 1 when the tiles has been studied, 0 otherwise */
  Mat_<unsigned char> done_tiles_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class InlierFinder
{
public:
  InlierFinder(float err, const Mat_<Vec3f> & points3d, const Mat_<Vec3f> & normals,
               unsigned char plane_index, int block_size)
      :
        err_(err),
        points3d_(points3d),
        normals_(normals),
        plane_index_(plane_index),
        block_size_(block_size)
  {
  }

  void
  Find(const PlaneGrid &plane_grid, Ptr<PlaneBase> & plane, TileQueue & tile_queue,
       std::set<TileQueue::PlaneTile> & neighboring_tiles, Mat_<unsigned char> & overall_mask,
       Mat_<unsigned char> & plane_mask)
  {
    // Do not use reference as we pop the from later on
    TileQueue::PlaneTile tile = *(neighboring_tiles.begin());

    // Figure the part of the image to look at
    Range range_x, range_y;
    int x = tile.x_ * block_size_, y = tile.y_ * block_size_;

    if (tile.x_ == plane_mask.cols - 1)
      range_x = Range(x, overall_mask.cols);
    else
      range_x = Range(x, x + block_size_);

    if (tile.y_ == plane_mask.rows - 1)
      range_y = Range(y, overall_mask.rows);
    else
      range_y = Range(y, y + block_size_);

    int n_valid_points = 0;
    for (int yy = range_y.start; yy != range_y.end; ++yy)
    {
      uchar* data = overall_mask.ptr(yy, range_x.start), *data_end = data + range_x.size();
      const Vec3f* point = points3d_.ptr < Vec3f > (yy, range_x.start);
      const Matx33f* Q_local = reinterpret_cast<const Matx33f *>(plane_grid.Q_.ptr < Vec<float, 9>
          > (yy, range_x.start));

      // Depending on whether you have a normal, check it
      if (!normals_.empty())
      {
        const Vec3f* normal = normals_.ptr < Vec3f > (yy, range_x.start);
        for (; data != data_end; ++data, ++point, ++normal, ++Q_local)
        {
          // Don't do anything if the point already belongs to another plane
          if (cvIsNaN(point->val[0]) || ((*data) != 255))
            continue;

          // If the point is close enough to the plane
          if (plane->distance(*point) < err_)
          {
            // make sure the normals are similar to the plane
            if (std::abs(plane->n().dot(*normal)) > 0.3)
            {
              // The point now belongs to the plane
              plane->UpdateStatistics(*point, *Q_local);
              *data = plane_index_;
              ++n_valid_points;
            }
          }
        }
      }
      else
      {
        for (; data != data_end; ++data, ++point, ++Q_local)
        {
          // Don't do anything if the point already belongs to another plane
          if (cvIsNaN(point->val[0]) || ((*data) != 255))
            continue;

          // If the point is close enough to the plane
          if (plane->distance(*point) < err_)
          {
            // The point now belongs to the plane
            plane->UpdateStatistics(*point, *Q_local);
            *data = plane_index_;
            ++n_valid_points;
          }
        }
      }
    }

    plane->UpdateParameters();

    // Mark the front as being done and pop it
    if (n_valid_points > (range_x.size() * range_y.size()) / 2)
      tile_queue.remove(tile.y_, tile.x_);
    plane_mask(tile.y_, tile.x_) = 1;
    neighboring_tiles.erase(neighboring_tiles.begin());

    // Add potential neighbors of the tile
    std::vector<std::pair<int, int> > pairs;
    if (tile.x_ > 0)
      for (unsigned char * val = overall_mask.ptr<unsigned char>(range_y.start, range_x.start), *val_end = val
          + range_y.size() * overall_mask.step; val != val_end; val += overall_mask.step)
        if (*val == plane_index_)
        {
          pairs.push_back(std::pair<int, int>(tile.x_ - 1, tile.y_));
          break;
        }
    if (tile.x_ < plane_mask.cols - 1)
      for (unsigned char * val = overall_mask.ptr<unsigned char>(range_y.start, range_x.end - 1), *val_end = val
          + range_y.size() * overall_mask.step; val != val_end; val += overall_mask.step)
        if (*val == plane_index_)
        {
          pairs.push_back(std::pair<int, int>(tile.x_ + 1, tile.y_));
          break;
        }
    if (tile.y_ > 0)
      for (unsigned char * val = overall_mask.ptr<unsigned char>(range_y.start, range_x.start), *val_end = val
          + range_x.size(); val != val_end; ++val)
        if (*val == plane_index_)
        {
          pairs.push_back(std::pair<int, int>(tile.x_, tile.y_ - 1));
          break;
        }
    if (tile.y_ < plane_mask.rows - 1)
      for (unsigned char * val = overall_mask.ptr<unsigned char>(range_y.end - 1, range_x.start), *val_end = val
          + range_x.size(); val != val_end; ++val)
        if (*val == plane_index_)
        {
          pairs.push_back(std::pair<int, int>(tile.x_, tile.y_ + 1));
          break;
        }

    for (unsigned char i = 0; i < pairs.size(); ++i)
      if (!plane_mask(pairs[i].second, pairs[i].first))
        neighboring_tiles.insert(
            TileQueue::PlaneTile(pairs[i].first, pairs[i].second, plane_grid.mse_(pairs[i].second, pairs[i].first)));
  }

private:
  float err_;
  const Mat_<Vec3f> & points3d_;
  const Mat_<Vec3f> & normals_;
  unsigned char plane_index_;
  /** THe block size as defined in the main algorithm */
  int block_size_;

  const InlierFinder & operator = (const InlierFinder &);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  RgbdPlane::RgbdPlane(int method, int block_size,
                     int min_size, double threshold, double sensor_error_a,
                     double sensor_error_b, double sensor_error_c) :
                     method_(method),
                     block_size_(block_size),
                     min_size_(min_size),
                     threshold_(threshold),
                     sensor_error_a_(sensor_error_a),
                     sensor_error_b_(sensor_error_b),
                     sensor_error_c_(sensor_error_c)
  {}

  Ptr<RgbdPlane> RgbdPlane::create(int method, int block_size, int min_size, double threshold,
                                 double sensor_error_a, double sensor_error_b,
                                 double sensor_error_c ) {
    return makePtr<RgbdPlane>(method, block_size, min_size, threshold,
                              sensor_error_a, sensor_error_b, sensor_error_c);
  }

  RgbdPlane::~RgbdPlane()
  {}

  void
  RgbdPlane::operator()(InputArray points3d_in, OutputArray mask_out, OutputArray plane_coefficients)
  {
    this->operator()(points3d_in, Mat(), mask_out, plane_coefficients);
  }

  void
  RgbdPlane::operator()(InputArray points3d_in, InputArray normals_in, OutputArray mask_out,
                        OutputArray plane_coefficients_out)
  {
    Mat_<Vec3f> points3d, normals;
    if (points3d_in.depth() == CV_32F)
      points3d = points3d_in.getMat();
    else
      points3d_in.getMat().convertTo(points3d, CV_32F);
    if (!normals_in.empty())
    {
      if (normals_in.depth() == CV_32F)
        normals = normals_in.getMat();
      else
        normals_in.getMat().convertTo(normals, CV_32F);
    }

    // Pre-computations
    mask_out.create(points3d.size(), CV_8U);
    Mat mask_out_mat = mask_out.getMat();
    Mat_<unsigned char> mask_out_uc = (Mat_<unsigned char>&) mask_out_mat;
    mask_out_uc.setTo(255);
    PlaneGrid plane_grid(points3d, block_size_);
    TileQueue plane_queue(plane_grid);
    size_t index_plane = 0;

    std::vector<Vec4f> plane_coefficients;
    float mse_min = (float)(threshold_ * threshold_);

    while (!plane_queue.empty())
    {
      // Get the first tile if it's good enough
      const TileQueue::PlaneTile front_tile = plane_queue.front();
      if (front_tile.mse_ > mse_min)
        break;

      InlierFinder inlier_finder((float)threshold_, points3d, normals, (unsigned char)index_plane, block_size_);

      // Construct the plane for the first tile
      int x = front_tile.x_, y = front_tile.y_;
      const Vec3f & n = plane_grid.n_(y, x);
      Ptr<PlaneBase> plane;
      if ((sensor_error_a_ == 0) && (sensor_error_b_ == 0) && (sensor_error_c_ == 0))
        plane = Ptr<PlaneBase>(new Plane(plane_grid.m_(y, x), n, (int)index_plane));
      else
        plane = Ptr<PlaneBase>(new PlaneABC(plane_grid.m_(y, x), n, (int)index_plane,
			(float)sensor_error_a_, (float)sensor_error_b_, (float)sensor_error_c_));

      Mat_<unsigned char> plane_mask = Mat_<unsigned char>::zeros(divUp(points3d.rows, block_size_),
                                                                  divUp(points3d.cols, block_size_));
      std::set<TileQueue::PlaneTile> neighboring_tiles;
      neighboring_tiles.insert(front_tile);
      plane_queue.remove(front_tile.y_, front_tile.x_);

      // Process all the neighboring tiles
      while (!neighboring_tiles.empty())
        inlier_finder.Find(plane_grid, plane, plane_queue, neighboring_tiles, mask_out_uc, plane_mask);

      // Don't record the plane if it's empty
      if (plane->empty())
        continue;
      // Don't record the plane if it's smaller than asked
      if (plane->K() < min_size_) {
        // Reset the plane index in the mask
        for (y = 0; y < plane_mask.rows; ++y)
          for (x = 0; x < plane_mask.cols; ++x) {
            if (!plane_mask(y, x))
              continue;
            // Go over the tile
            for (int yy = y * block_size_;
                yy < std::min((y + 1) * block_size_, mask_out_uc.rows); ++yy) {
              uchar* data = mask_out_uc.ptr(yy, x * block_size_);
              uchar* data_end = data
                  + std::min(block_size_,
                      mask_out_uc.cols - x * block_size_);
              for (; data != data_end; ++data) {
                if (*data == index_plane)
                  *data = 255;
              }
            }
          }
        continue;
      }

      ++index_plane;
      if (index_plane >= 255)
        break;
      Vec4f coeffs(plane->n()[0], plane->n()[1], plane->n()[2], plane->d());
      if (coeffs(2) > 0)
        coeffs = -coeffs;
      plane_coefficients.push_back(coeffs);
    };

    // Fill the plane coefficients
    if (plane_coefficients.empty())
      return;
    plane_coefficients_out.create((int)plane_coefficients.size(), 1, CV_32FC4);
    Mat plane_coefficients_mat = plane_coefficients_out.getMat();
    float* data = plane_coefficients_mat.ptr<float>(0);
    for(size_t i=0; i<plane_coefficients.size(); ++i)
      for(uchar j=0; j<4; ++j, ++data)
        *data = plane_coefficients[i][j];
  }
}
}
