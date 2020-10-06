// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#include "opencv2/flann/miniflann.hpp"
#include <map>

namespace cv { namespace usac {
double Utils::getCalibratedThreshold (double threshold, const Mat &K1, const Mat &K2) {
    return threshold / ((K1.at<double>(0, 0) + K1.at<double>(1, 1) +
                         K2.at<double>(0, 0) + K2.at<double>(1, 1)) / 4.0);
}

/*
 * K1, K2 are 3x3 intrinsics matrices
 * points is matrix of size |N| x 4
 * Assume K = [k11 k12 k13
 *              0  k22 k23
 *              0   0   1]
 */
void Utils::calibratePoints (const Mat &K1, const Mat &K2, const Mat &points, Mat &calib_points) {
    const auto * const points_ = (float *) points.data;
    const auto * const k1 = (double *) K1.data;
    const auto inv1_k11 = float(1 / k1[0]); // 1 / k11
    const auto inv1_k12 = float(-k1[1] / (k1[0]*k1[4])); // -k12 / (k11*k22)
    // (-k13*k22 + k12*k23) / (k11*k22)
    const auto inv1_k13 = float((-k1[2]*k1[4] + k1[1]*k1[5]) / (k1[0]*k1[4]));
    const auto inv1_k22 = float(1 / k1[4]); // 1 / k22
    const auto inv1_k23 = float(-k1[5] / k1[4]); // -k23 / k22

    const auto * const k2 = (double *) K2.data;
    const auto inv2_k11 = float(1 / k2[0]);
    const auto inv2_k12 = float(-k2[1] / (k2[0]*k2[4]));
    const auto inv2_k13 = float((-k2[2]*k2[4] + k2[1]*k2[5]) / (k2[0]*k2[4]));
    const auto inv2_k22 = float(1 / k2[4]);
    const auto inv2_k23 = float(-k2[5] / k2[4]);

    calib_points = Mat ( points.rows, 4, points.type());
    auto * calib_points_ = (float *) calib_points.data;

    for (int i = 0; i <  points.rows; i++) {
        const int idx = 4*i;
        (*calib_points_++) = inv1_k11 * points_[idx  ] + inv1_k12 * points_[idx+1] + inv1_k13;
        (*calib_points_++) =                             inv1_k22 * points_[idx+1] + inv1_k23;
        (*calib_points_++) = inv2_k11 * points_[idx+2] + inv2_k12 * points_[idx+3] + inv2_k13;
        (*calib_points_++) =                             inv2_k22 * points_[idx+3] + inv2_k23;
    }
}

/*
 * K is 3x3 intrinsic matrix
 * points is matrix of size |N| x 5, first two columns are image points [u_i, v_i]
 * calib_norm_pts are  K^-1 [u v 1]^T / ||K^-1 [u v 1]^T||
 */
void Utils::calibrateAndNormalizePointsPnP (const Mat &K, const Mat &pts, Mat &calib_norm_pts) {
    const auto * const points = (float *) pts.data;
    const auto * const k = (double *) K.data;
    const auto inv_k11 = float(1 / k[0]);
    const auto inv_k12 = float(-k[1] / (k[0]*k[4]));
    const auto inv_k13 = float((-k[2]*k[4] + k[1]*k[5]) / (k[0]*k[4]));
    const auto inv_k22 = float(1 / k[4]);
    const auto inv_k23 = float(-k[5] / k[4]);

    calib_norm_pts = Mat (pts.rows, 3, pts.type());
    auto * calib_norm_pts_ = (float *) calib_norm_pts.data;

    for (int i = 0; i < pts.rows; i++) {
        const int idx = 5 * i;
        const float k_inv_u = inv_k11 * points[idx] + inv_k12 * points[idx+1] + inv_k13;
        const float k_inv_v =                         inv_k22 * points[idx+1] + inv_k23;
        const float norm = 1.f / sqrtf(k_inv_u*k_inv_u + k_inv_v*k_inv_v + 1);
        (*calib_norm_pts_++) = k_inv_u * norm;
        (*calib_norm_pts_++) = k_inv_v * norm;
        (*calib_norm_pts_++) =           norm;
    }
}

void Utils::normalizeAndDecalibPointsPnP (const Mat &K_, Mat &pts, Mat &calib_norm_pts) {
    const auto * const K = (double *) K_.data;
    const auto k11 = (float)K[0], k12 = (float)K[1], k13 = (float)K[2],
               k22 = (float)K[4], k23 = (float)K[5];
    calib_norm_pts = Mat (pts.rows, 3, pts.type());
    auto * points = (float *) pts.data;
    auto * calib_norm_pts_ = (float *) calib_norm_pts.data;

    for (int i = 0; i < pts.rows; i++) {
        const int idx = 5 * i;
        const float k_inv_u = points[idx  ];
        const float k_inv_v = points[idx+1];
        const float norm = 1.f / sqrtf(k_inv_u*k_inv_u + k_inv_v*k_inv_v + 1);
        (*calib_norm_pts_++) = k_inv_u * norm;
        (*calib_norm_pts_++) = k_inv_v * norm;
        (*calib_norm_pts_++) =           norm;
        points[idx  ] = k11 * k_inv_u + k12 * k_inv_v + k13;
        points[idx+1] =                 k22 * k_inv_v + k23;
    }
}
/*
 * decompose Projection Matrix to calibration, rotation and translation
 * Assume K = [fx  0   tx
 *             0   fy  ty
 *             0   0   1]
 */
void Utils::decomposeProjection (const Mat &P, Mat &K_, Mat &R, Mat &t, bool same_focal) {
    const Mat M = P.colRange(0,3);
    double scale = norm(M.row(2)); scale *= scale;
    Matx33d K = Matx33d::eye();
    K(1,2) = M.row(1).dot(M.row(2)) / scale;
    K(0,2) = M.row(0).dot(M.row(2)) / scale;
    K(1,1) = sqrt(M.row(1).dot(M.row(1)) / scale - K(1,2)*K(1,2));
    K(0,0) = sqrt(M.row(0).dot(M.row(0)) / scale - K(0,2)*K(0,2));
    if (same_focal)
        K(0,0) = K(1,1) = (K(0,0) + K(1,1)) / 2;
    R = K.inv() * M / sqrt(scale);
    if (determinant(M) < 0) R *= -1;
    t = R * M.inv() * P.col(3);
    K_ = Mat(K);
}

Matx33d Math::getSkewSymmetric(const Vec3d &v) {
     return Matx33d(0,    -v[2], v[1],
                   v[2],  0,    -v[0],
                  -v[1],  v[0], 0);
}

Matx33d Math::rotVec2RotMat (const Vec3d &v) {
    const double phi = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    const double x = v[0] / phi, y = v[1] / phi, z = v[2] / phi;
    const double a = sin(phi), b = cos(phi);
    // R = I + sin(phi) * skew(v) + (1 - cos(phi) * skew(v)^2
    return Matx33d((b - 1)*y*y + (b - 1)*z*z + 1, -a*z - x*y*(b - 1), a*y - x*z*(b - 1),
     a*z - x*y*(b - 1), (b - 1)*x*x + (b - 1)*z*z + 1, -a*x - y*z*(b - 1),
    -a*y - x*z*(b - 1), a*x - y*z*(b - 1), (b - 1)*x*x + (b - 1)*y*y + 1);
}

Vec3d Math::rotMat2RotVec (const Matx33d &R) {
    // https://math.stackexchange.com/questions/83874/efficient-and-accurate-numerical-implementation-of-the-inverse-rodrigues-rotatio?rq=1
    Vec3d rot_vec;
    const double trace = R(0,0)+R(1,1)+R(2,2);
    if (trace >= 3 - FLT_EPSILON) {
        rot_vec = (0.5 * (trace-3)/12)*Vec3d(R(2,1)-R(1,2),
                                             R(0,2)-R(2,0),
                                             R(1,0)-R(0,1));
    } else if (3 - FLT_EPSILON > trace && trace > -1 + FLT_EPSILON) {
        double theta = acos((trace - 1) / 2);
        rot_vec = (theta / (2 * sin(theta))) * Vec3d(R(2,1)-R(1,2),
                                                     R(0,2)-R(2,0),
                                                     R(1,0)-R(0,1));
    } else {
        int a;
        if (R(0,0) > R(1,1))
            a = R(0,0) > R(2,2) ? 0 : 2;
        else
            a = R(1,1) > R(2,2) ? 1 : 2;
        Vec3d v;
        int b = (a + 1) % 3, c = (a + 2) % 3;
        double s = sqrt(R(a,a) - R(b,b) - R(c,c) + 1);
        v[a] = s / 2;
        v[b] = (R(b,a) + R(a,b)) / (2 * s);
        v[c] = (R(c,a) + R(a,c)) / (2 * s);
        rot_vec = M_PI * v / norm(v);
    }
    return rot_vec;
}

/*
 * Eliminate matrix of m rows and n columns to be upper triangular.
 */
bool Math::eliminateUpperTriangular (std::vector<double> &a, int m, int n) {
    for (int r = 0; r < m; r++){
        double pivot = a[r*n+r];
        int row_with_pivot = r;

        // find the maximum pivot value among r-th column
        for (int k = r+1; k < m; k++)
            if (fabs(pivot) < fabs(a[k*n+r])) {
                pivot = a[k*n+r];
                row_with_pivot = k;
            }

        // if pivot value is 0 continue
        if (fabs(pivot) < DBL_EPSILON)
            return false; // matrix is not full rank -> terminate

        // swap row with maximum pivot value with current row
        for (int c = r; c < n; c++)
            std::swap(a[row_with_pivot*n+c], a[r*n+c]);

        // eliminate other rows
        for (int j = r+1; j < m; j++){
            const int row_idx1 = j*n, row_idx2 = r*n;
            const auto fac = a[row_idx1+r] / pivot;
            a[row_idx1+r] = 0; // zero eliminated element
            for (int c = r+1; c < n; c++)
                a[row_idx1+c] -= fac * a[row_idx2+c];
        }
    }
    return true;
}

//////////////////////////////////////// RANDOM GENERATOR /////////////////////////////
class UniformRandomGeneratorImpl : public UniformRandomGenerator {
private:
    int subset_size = 0, max_range = 0;
    std::vector<int> subset;
    RNG rng;
public:
    explicit UniformRandomGeneratorImpl (int state) : rng(state) {}

    // interval is <0; max_range);
    UniformRandomGeneratorImpl (int state, int max_range_, int subset_size_) : rng(state) {
        subset_size = subset_size_;
        max_range = max_range_;
        subset = std::vector<int>(subset_size_);
    }

    int getRandomNumber () override {
        return rng.uniform(0, max_range);
    }

    int getRandomNumber (int max_rng) override {
        return rng.uniform(0, max_rng);
    }

    // closed range
    void resetGenerator (int max_range_) override {
        CV_CheckGE(0, max_range_, "max range must be greater than 0");
        max_range = max_range_;
    }

    void generateUniqueRandomSet (std::vector<int>& sample) override {
        CV_CheckLE(subset_size, max_range, "RandomGenerator. Subset size must be LE than range!");
        int j, num;
        sample[0] = rng.uniform(0, max_range);
        for (int i = 1; i < subset_size;) {
            num = rng.uniform(0, max_range);
            // check if value is in array
            for (j = i - 1; j >= 0; j--)
                if (num == sample[j])
                    // if so, generate again
                    break;
            // success, value is not in array, so it is unique, add to sample.
            if (j == -1) sample[i++] = num;
        }
    }

    // interval is <0; max_range)
    void generateUniqueRandomSet (std::vector<int>& sample, int max_range_) override {
        /*
         * necessary condition:
         * if subset size is bigger than range then array cannot be unique,
         * so function has infinite loop.
         */
        CV_CheckLE(subset_size, max_range_, "RandomGenerator. Subset size must be LE than range!");
        int num, j;
        sample[0] = rng.uniform(0, max_range_);
        for (int i = 1; i < subset_size;) {
            num = rng.uniform(0, max_range_);
            for (j = i - 1; j >= 0; j--)
                if (num == sample[j])
                    break;
            if (j == -1) sample[i++] = num;
        }
    }

    // interval is <0, max_range)
    void generateUniqueRandomSet (std::vector<int>& sample, int subset_size_, int max_range_) override {
        CV_CheckLE(subset_size_, max_range_, "RandomGenerator. Subset size must be LE than range!");
        int num, j;
        sample[0] = rng.uniform(0, max_range_);
        for (int i = 1; i < subset_size_;) {
            num = rng.uniform(0, max_range_);
            for (j = i - 1; j >= 0; j--)
                if (num == sample[j])
                    break;
            if (j == -1) sample[i++] = num;
        }
    }
    const std::vector<int> &generateUniqueRandomSubset (std::vector<int> &array1, int size1) override {
        CV_CheckLE(subset_size, size1, "RandomGenerator. Subset size must be LE than range!");
        int temp_size1 = size1;
        for (int i = 0; i < subset_size; i++) {
            const int idx1 = rng.uniform(0, temp_size1);
            subset[i] = array1[idx1];
            std::swap(array1[idx1], array1[--temp_size1]);
        }
        return subset;
    }

    void setSubsetSize (int subset_size_) override {
        subset_size = subset_size_;
    }
    int getSubsetSize () const override { return subset_size; }
    Ptr<RandomGenerator> clone (int state) const override {
        return makePtr<UniformRandomGeneratorImpl>(state, max_range, subset_size);
    }
};

Ptr<UniformRandomGenerator> UniformRandomGenerator::create (int state) {
    return makePtr<UniformRandomGeneratorImpl>(state);
}
Ptr<UniformRandomGenerator> UniformRandomGenerator::create
        (int state, int max_range, int subset_size_) {
    return makePtr<UniformRandomGeneratorImpl>(state, max_range, subset_size_);
}

// @k_minth - desired k-th minimal element. For median is half of array
// closed working interval of array <@left; @right>
float quicksort_median (std::vector<float> &array, int k_minth, int left, int right);
float quicksort_median (std::vector<float> &array, int k_minth, int left, int right) {
    // length is 0, return single value
    if (right - left == 0) return array[left];

    // get pivot, the rightest value in array
    const auto pivot = array[right];
    int right_ = right - 1; // -1, not including pivot
    // counter of values smaller equal than pivot
    int j = left, values_less_eq_pivot = 1; // 1, inludes pivot already
    for (; j <= right_;) {
        if (array[j] <= pivot) {
            j++;
            values_less_eq_pivot++;
        } else
            // value is bigger than pivot, swap with right_ value
            // swap values in array and decrease interval
            std::swap(array[j], array[right_--]);
    }
    if (values_less_eq_pivot == k_minth) return pivot;
    if (k_minth > values_less_eq_pivot)
        return quicksort_median(array, k_minth - values_less_eq_pivot, j, right-1);
    else
        return quicksort_median(array, k_minth, left, j-1);
}

// find median using quicksort with complexity O(log n)
// Note, function changes order of values in array
float Utils::findMedian (std::vector<float> &array) {
    const int length = static_cast<int>(array.size());
    if (length % 2) {
        // odd number of values
        return quicksort_median (array, length/2+1, 0, length-1);
    } else {
        // even: return average
        return (quicksort_median(array, length/2  , 0, length-1) +
                quicksort_median(array, length/2+1, 0, length-1))/2;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Radius Search Graph /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
class RadiusSearchNeighborhoodGraphImpl : public RadiusSearchNeighborhoodGraph {
private:
    std::vector<std::vector<int>> graph;
public:
    RadiusSearchNeighborhoodGraphImpl (const Mat &container_, int points_size,
                               double radius, int flann_search_params, int num_kd_trees) {
        // Radius search OpenCV works only with float data
        CV_Assert(container_.type() == CV_32F);

        FlannBasedMatcher flann(makePtr<flann::KDTreeIndexParams>(num_kd_trees), makePtr<flann::SearchParams>(flann_search_params));
        std::vector<std::vector<DMatch>> neighbours;
        flann.radiusMatch(container_, container_, neighbours, (float)radius);

        // allocate graph
        graph = std::vector<std::vector<int>> (points_size);

        int pt = 0;
        for (const auto &n : neighbours) {
            auto &graph_row = graph[pt];
            graph_row = std::vector<int>(n.size()-1);
            int j = 0;
            for (const auto &idx : n)
                // skip neighbor which has the same index as requested point
                if (idx.trainIdx != pt)
                    graph_row[j++] = idx.trainIdx;
            pt++;
        }
    }

    inline const std::vector<int> &getNeighbors(int point_idx) const override {
        return graph[point_idx];
    }
};
Ptr<RadiusSearchNeighborhoodGraph> RadiusSearchNeighborhoodGraph::create (const Mat &points,
        int points_size, double radius_, int flann_search_params, int num_kd_trees) {
    return makePtr<RadiusSearchNeighborhoodGraphImpl> (points, points_size, radius_,
            flann_search_params, num_kd_trees);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// FLANN Graph /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
class FlannNeighborhoodGraphImpl : public FlannNeighborhoodGraph {
private:
    std::vector<std::vector<int>> graph;
    std::vector<std::vector<double>> distances;
public:
    FlannNeighborhoodGraphImpl (const Mat &container_, int points_size, int k_nearest_neighbors,
            bool get_distances, int flann_search_params_, int num_kd_trees) {
        CV_Assert(k_nearest_neighbors <= points_size);
        // FLANN works only with float data
        CV_Assert(container_.type() == CV_32F);

        flann::Index flannIndex (container_.reshape(1), flann::KDTreeIndexParams(num_kd_trees));
        Mat dists, nearest_neighbors;

        flannIndex.knnSearch(container_, nearest_neighbors, dists, k_nearest_neighbors+1,
                flann::SearchParams(flann_search_params_));

        // first nearest neighbor of point is this point itself.
        // remove this first column
        nearest_neighbors.colRange(1, k_nearest_neighbors+1).copyTo (nearest_neighbors);

        graph = std::vector<std::vector<int>>(points_size, std::vector<int>(k_nearest_neighbors));
        const auto * const nn = (int *) nearest_neighbors.data;
        const auto * const dists_ptr = (float *) dists.data;

        if (get_distances)
            distances = std::vector<std::vector<double>>(points_size, std::vector<double>(k_nearest_neighbors));

        for (int pt = 0; pt < points_size; pt++) {
            std::copy(nn + k_nearest_neighbors*pt, nn + k_nearest_neighbors*pt + k_nearest_neighbors, &graph[pt][0]);
            if (get_distances)
                std::copy(dists_ptr + k_nearest_neighbors*pt, dists_ptr + k_nearest_neighbors*pt + k_nearest_neighbors,
                          &distances[pt][0]);
        }
    }
    const std::vector<double>& getNeighborsDistances (int idx) const override {
        return distances[idx];
    }
    inline const std::vector<int> &getNeighbors(int point_idx) const override {
        // CV_Assert(point_idx_ < num_vertices);
        return graph[point_idx];
    }
};

Ptr<FlannNeighborhoodGraph> FlannNeighborhoodGraph::create(const Mat &points,
           int points_size, int k_nearest_neighbors_, bool get_distances,
           int flann_search_params_, int num_kd_trees) {
    return makePtr<FlannNeighborhoodGraphImpl>(points, points_size,
        k_nearest_neighbors_, get_distances, flann_search_params_, num_kd_trees);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Grid Neighborhood Graph /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
class GridNeighborhoodGraphImpl : public GridNeighborhoodGraph {
private:
    // This struct is used for the nearest neighbors search by griding two images.
    struct CellCoord {
        int c1x, c1y, c2x, c2y;
        CellCoord (int c1x_, int c1y_, int c2x_, int c2y_) {
            c1x = c1x_; c1y = c1y_; c2x = c2x_; c2y = c2y_;
        }
        bool operator==(const CellCoord &o) const {
            return c1x == o.c1x && c1y == o.c1y && c2x == o.c2x && c2y == o.c2y;
        }
        bool operator<(const CellCoord &o) const {
            if (c1x < o.c1x) return true;
            if (c1x == o.c1x && c1y < o.c1y) return true;
            if (c1x == o.c1x && c1y == o.c1y && c2x < o.c2x) return true;
            return c1x == o.c1x && c1y == o.c1y && c2x == o.c2x && c2y < o.c2y;
        }
    };

    std::map<CellCoord, std::vector<int >> neighbors_map;
    std::vector<std::vector<int>> graph;
public:
    GridNeighborhoodGraphImpl (const Mat &container_, int points_size,
          int cell_size_x_img1, int cell_size_y_img1, int cell_size_x_img2, int cell_size_y_img2,
          int max_neighbors) {

        const auto * const container = (float *) container_.data;
        // <int, int, int, int> -> {neighbors set}
        // Key is cell position. The value is indexes of neighbors.

        const float cell_sz_x1 = 1.f / (float) cell_size_x_img1,
                    cell_sz_y1 = 1.f / (float) cell_size_y_img1,
                    cell_sz_x2 = 1.f / (float) cell_size_x_img2,
                    cell_sz_y2 = 1.f / (float) cell_size_y_img2;
        const int dimension = container_.cols;
        for (int i = 0; i < points_size; i++) {
            const int idx = dimension * i;
            neighbors_map[CellCoord((int)(container[idx  ] * cell_sz_x1),
                                    (int)(container[idx+1] * cell_sz_y1),
                                    (int)(container[idx+2] * cell_sz_x2),
                                    (int)(container[idx+3] * cell_sz_y2))].emplace_back(i);
        }

        //--------- create a graph ----------
        graph = std::vector<std::vector<int>>(points_size);

        // store neighbors cells into graph (2D vector)
        for (const auto &cell : neighbors_map) {
            const int neighbors_in_cell = static_cast<int>(cell.second.size());

            // only one point in cell -> no neighbors
            if (neighbors_in_cell < 2) continue;

            const std::vector<int> &neighbors = cell.second;
            // ---------- fill graph -----
            for (int v_in_cell : neighbors) {
                // there is always at least one neighbor
                auto &graph_row = graph[v_in_cell];
                graph_row = std::vector<int>(std::min(max_neighbors, neighbors_in_cell-1));
                int j = 0;
                for (int n : neighbors)
                    if (n != v_in_cell){
                        graph_row[j++] = n;
                        if (j >= max_neighbors)
                            break;
                    }
            }
        }
    }

    inline const std::vector<int> &getNeighbors(int point_idx) const override {
        // Note, neighbors vector also includes point_idx!
        // return neighbors_map[vertices_to_cells[point_idx]];
        return graph[point_idx];
    }
};

Ptr<GridNeighborhoodGraph> GridNeighborhoodGraph::create(const Mat &points,
     int points_size, int cell_size_x_img1_, int cell_size_y_img1_,
     int cell_size_x_img2_, int cell_size_y_img2_, int max_neighbors) {
    return makePtr<GridNeighborhoodGraphImpl>(points, points_size,
      cell_size_x_img1_, cell_size_y_img1_, cell_size_x_img2_, cell_size_y_img2_, max_neighbors);
}
}}