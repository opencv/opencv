// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"
#include "opencv2/flann/miniflann.hpp"
#include <map>

namespace cv { namespace usac {
/*
SolvePoly is used to find only real roots of N-degree polynomial using Sturm sequence.
It recursively finds interval where a root lies, and the actual root is found using Regula-Falsi method.
*/
class SolvePoly : public SolverPoly {
private:
    static int sgn(double val) {
        return (double(0) < val) - (val < double(0));
    }
    class Poly {
    public:
        Poly () = default;
        Poly (const std::vector<double> &coef_) {
            coef = coef_;
            checkDegree();
        }
        Poly (const Poly &p) { coef = p.coef; }
        // a_n x^n + a_n-1 x^(n-1) + ... + a_1 x + a_0
        // coef[i] = a_i
        std::vector<double> coef = {0};
        inline int degree() const { return (int)coef.size()-1; }
        void multiplyScalar (double s) {
            // multiplies polynom by scalar
            if (fabs(s) < DBL_EPSILON) { // check if scalar is 0
                coef = {0};
                return;
            }
            for (double &c : coef) c *= s;
        }
        void checkDegree() {
            int deg = degree(); // approximate degree
            // check if coefficients of the highest power is non-zero
            while (fabs(coef[deg]) < DBL_EPSILON) {
                coef.pop_back(); // remove last zero element
                if (--deg == 0)
                    break;
            }
        }
        double eval (double x) const {
            // Horner method a0 + x (a1 + x (a2 + x (a3 + ... + x (an-1 + x an))))
            const int d = degree();
            double y = coef[d];
            for (int i = d; i >= 1; i--)
                y = coef[i-1] + x * y;
            return y;
        }
        // at +inf and -inf
        std::pair<int,int> signsAtInf () const {
            // lim x->+-inf p(x) = lim x->+-inf a_n x^n
            const int d = degree();
            const int s = sgn(coef[d]); // sign of the highest coefficient
            // compare even and odd degree
            return std::make_pair(s, d % 2 == 0 ? s : -s);
        }
        Poly derivative () const {
            Poly deriv;
            if (degree() == 0)
                return deriv;
            // derive.degree = poly.degree-1;
            deriv.coef = std::vector<double>(coef.size()-1);
            for (int i = degree(); i > 0; i--)
                // (a_n * x^n)' =  n * a_n * x^(n-1)
                deriv.coef[i-1] = i * coef[i];
            return deriv;
        }
        void copyFrom (const Poly &p) { coef = p.coef; }
    };
    // return remainder
    static void dividePoly (const Poly &p1, const Poly &p2, /*Poly &quotient,*/ Poly &remainder) {
        remainder.copyFrom(p1);
        int p2_degree = p2.degree(), remainder_degree = remainder.degree();
        if (p1.degree() < p2_degree)
            return;
        if (p2_degree == 0) { // special case for dividing polynomial by constant
            remainder.multiplyScalar(1/p2.coef[0]);
            // quotient.coef[0] = p2.coef[0];
            return;
        }
        // quotient.coef = std::vector<double>(p1.degree() - p2_degree + 1, 0);
        const double p2_term = 1/p2.coef[p2_degree];
        while (remainder_degree >= p2_degree) {
            const double temp = remainder.coef[remainder_degree] * p2_term;
            // quotient.coef[remainder_degree-p2_degree] = temp;
            // polynoms now have the same degree, but p2 is shorter than remainder
            for (int i = p2_degree, j = remainder_degree; i >= 0; i--, j--)
                remainder.coef[j] -= temp * p2.coef[i];
            remainder.checkDegree();
            remainder_degree = remainder.degree();
        }
    }

    constexpr static int REGULA_FALSI_MAX_ITERS = 500, MAX_POWER = 10, MAX_LEVEL = 200;
    constexpr static double TOLERANCE = 1e-10, DIFF_TOLERANCE = 1e-7;

    static bool findRootRegulaFalsi (const Poly &poly, double min, double max, double &root) {
        double f_min = poly.eval(min), f_max = poly.eval(max);
        if (f_min * f_max > 0 || min > max) {// conditions are not fulfilled
            return false;
        }
        int sign = 0, iter = 0;
        for (; iter < REGULA_FALSI_MAX_ITERS; iter++) {
            root = (f_min * max - f_max * min) / (f_min - f_max);
            const double f_root = poly.eval(root);
            if (fabs(f_root) < TOLERANCE || fabs(min - max) < DIFF_TOLERANCE) {
                return true; // root is found
            }

            if (f_root * f_max > 0) {
                max = root; f_max = f_root;
                if (sign == -1)
                    f_min *= 0.5;
                sign = -1;
            } else if (f_min * f_root > 0) {
                min = root; f_min = f_root;
                if (sign ==  1)
                    f_max *= 0.5;
                sign =  1;
            }
        }
        return false;
    }

    static int numberOfSignChanges (const std::vector<Poly> &sturm, double x) {
        int prev_sign = 0, sign_changes = 0;
        for (const auto &poly : sturm) {
            const int s = sgn(poly.eval(x));
            if (s != 0 && prev_sign != 0 && s != prev_sign)
                sign_changes++;
            prev_sign = s;
        }
        return sign_changes;
    }

    static void findRootsRecursive (const Poly &poly, const std::vector<Poly> &sturm, double min, double max,
            int sign_changes_at_min, int sign_changes_at_max, std::vector<double> &roots, int level) {
        const int num_roots = sign_changes_at_min - sign_changes_at_max;
        if (level == MAX_LEVEL) {
            // roots are too close
            const double mid = (min + max) * 0.5;
            if (fabs(poly.eval(mid)) < DBL_EPSILON) {
                roots.emplace_back(mid);
            }
        } else if (num_roots == 1) {
            double root;
            if (findRootRegulaFalsi(poly, min, max, root)) {
                roots.emplace_back(root);
            }
        } else if (num_roots > 1) { // at least 2 roots
            const double mid = (min + max) * 0.5;
            const int sign_changes_at_mid = numberOfSignChanges(sturm, mid);
            // try to split interval equally for the roots
            if (sign_changes_at_min - sign_changes_at_mid > 0)
                findRootsRecursive(poly, sturm, min, mid, sign_changes_at_min, sign_changes_at_mid, roots, level+1);
            if (sign_changes_at_mid - sign_changes_at_max > 0)
                findRootsRecursive(poly, sturm, mid, max, sign_changes_at_mid, sign_changes_at_max, roots, level+1);
        }
    }
public:
    int getRealRoots (const std::vector<double> &coeffs, std::vector<double> &real_roots) override {
        if (coeffs.empty())
            return 0;
        Poly input(coeffs);
        if (input.degree() < 1)
            return 0;
        // derivative of input polynomial
        const Poly input_der = input.derivative();
        /////////// build Sturm sequence //////////
        Poly p (input), q (input_der), remainder;
        std::vector<std::pair<int,int>> signs_at_inf; signs_at_inf.reserve(p.degree()); // +inf, -inf pair
        signs_at_inf.emplace_back(p.signsAtInf());
        signs_at_inf.emplace_back(q.signsAtInf());
        std::vector<Poly> sturm_sequence; sturm_sequence.reserve(input.degree());
        sturm_sequence.emplace_back(input);
        sturm_sequence.emplace_back(input_der);
         while (q.degree() > 0) {
            dividePoly(p, q, remainder);
            remainder.multiplyScalar(-1);
            p.copyFrom(q);
            q.copyFrom(remainder);
            sturm_sequence.emplace_back(remainder);
            signs_at_inf.emplace_back(remainder.signsAtInf());
        }
        ////////// find changes in signs of Sturm sequence /////////
        int num_sign_changes_at_pos_inf = 0, num_sign_changes_at_neg_inf = 0;
        int prev_sign_pos_inf = signs_at_inf[0].first, prev_sign_neg_inf = signs_at_inf[0].second;
        for (int i = 1; i < (int)signs_at_inf.size(); i++) {
            const auto s_pos_inf = signs_at_inf[i].first, s_neg_inf = signs_at_inf[i].second;
            // zeros must be ignored
            if (s_pos_inf != 0) {
                if (prev_sign_pos_inf != 0 && prev_sign_pos_inf != s_pos_inf)
                    num_sign_changes_at_pos_inf++;
                prev_sign_pos_inf = s_pos_inf;
            }
            if (s_neg_inf != 0) {
                if (prev_sign_neg_inf != 0 && prev_sign_neg_inf != s_neg_inf)
                    num_sign_changes_at_neg_inf++;
                prev_sign_neg_inf = s_neg_inf;
            }
        }
        ////////// find roots' bounds for numerical method for roots finding /////////
        double root_neg_bound = -0.01, root_pos_bound = 0.01;
        int num_sign_changes_min_x = -1, num_sign_changes_pos_x = -1; // -1 = unknown, trigger next if condition
        for (int i = 0; i < MAX_POWER; i++) {
            if (num_sign_changes_min_x != num_sign_changes_at_neg_inf) {
                root_neg_bound *= 10;
                num_sign_changes_min_x = numberOfSignChanges(sturm_sequence, root_neg_bound);
            }
            if (num_sign_changes_pos_x != num_sign_changes_at_pos_inf) {
                root_pos_bound *= 10;
                num_sign_changes_pos_x = numberOfSignChanges(sturm_sequence, root_pos_bound);
            }
        }
        /////////// get real roots //////////
        real_roots.clear();
        findRootsRecursive(input, sturm_sequence, root_neg_bound, root_pos_bound, num_sign_changes_min_x, num_sign_changes_pos_x, real_roots, 0 /*level*/);
        ///////////////////////////////
        if ((int)real_roots.size() > input.degree())
            real_roots.resize(input.degree()); // must not happen, unless some roots repeat
        return (int) real_roots.size();
    }
};
Ptr<SolverPoly> SolverPoly::create() {
    return makePtr<SolvePoly>();
}

double Utils::getCalibratedThreshold (double threshold, const Mat &K1, const Mat &K2) {
    const auto * const k1 = (double *) K1.data, * const k2 = (double *) K2.data;
    return threshold / ((k1[0] + k1[4] + k2[0] + k2[4]) / 4.0);
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
void Utils::decomposeProjection (const Mat &P, Matx33d &K, Matx33d &R, Vec3d &t, bool same_focal) {
    const Matx33d M = P.colRange(0,3);
    double scale = norm(M.row(2)); scale *= scale;
    K = Matx33d::eye();
    K(1,2) = M.row(1).dot(M.row(2)) / scale;
    K(0,2) = M.row(0).dot(M.row(2)) / scale;
    K(1,1) = sqrt(M.row(1).dot(M.row(1)) / scale - K(1,2)*K(1,2));
    K(0,0) = sqrt(M.row(0).dot(M.row(0)) / scale - K(0,2)*K(0,2));
    if (same_focal)
        K(0,0) = K(1,1) = (K(0,0) + K(1,1)) / 2;
    R = K.inv() * M / sqrt(scale);
    if (determinant(M) < 0) R *= -1;
    t = R * M.inv() * Vec3d(P.col(3));
}

double Utils::getPoissonCDF (double lambda, int inliers) {
    double exp_lamda = exp(-lambda), cdf = exp_lamda, lambda_i_div_fact_i = 1;
    for (int i = 1; i <= inliers; i++) {
        lambda_i_div_fact_i *= (lambda / i);
        cdf += exp_lamda * lambda_i_div_fact_i;
        if (fabs(cdf - 1) < DBL_EPSILON) // cdf is almost 1
            break;
    }
    return cdf;
}

// since F(E) has rank 2 we use cross product to compute epipole,
// since the third column / row is linearly dependent on two first
// this is faster than SVD
// It is recommended to normalize F, such that ||F|| = 1
Vec3d Utils::getLeftEpipole (const Mat &F/*E*/) {
    Vec3d _e = F.col(0).cross(F.col(2)); // F^T e' = 0; e'^T F = 0
    const auto * const e = _e.val;
    if (e[0] <= DBL_EPSILON && e[0] > -DBL_EPSILON &&
        e[1] <= DBL_EPSILON && e[1] > -DBL_EPSILON &&
        e[2] <= DBL_EPSILON && e[2] > -DBL_EPSILON)
        _e = Vec3d(Mat(F.col(1))).cross(F.col(2));  // if e' is zero
    return _e; // e'
}
Vec3d Utils::getRightEpipole (const Mat &F/*E*/) {
    Vec3d _e = F.row(0).cross(F.row(2)); // Fe = 0
    const auto * const e = _e.val;
    if (e[0] <= DBL_EPSILON && e[0] > -DBL_EPSILON &&
        e[1] <= DBL_EPSILON && e[1] > -DBL_EPSILON &&
        e[2] <= DBL_EPSILON && e[2] > -DBL_EPSILON)
        _e = F.row(1).cross(F.row(2));  // if e is zero
    return _e;
}

void Utils::densitySort (const Mat &points, int knn, Mat &sorted_points, std::vector<int> &sorted_mask) {
    // mask of sorted points (array of indexes)
    const int points_size = points.rows, dim = points.cols;
    sorted_mask = std::vector<int >(points_size);
    for (int i = 0; i < points_size; i++)
        sorted_mask[i] = i;

    // get neighbors
    FlannNeighborhoodGraph &graph = *FlannNeighborhoodGraph::create(points, points_size, knn,
            true /*get distances */, 6, 1);

    std::vector<double> sum_knn_distances (points_size, 0);
    for (int p = 0; p < points_size; p++) {
        const std::vector<double> &dists = graph.getNeighborsDistances(p);
        for (int k = 0; k < knn; k++)
            sum_knn_distances[p] += dists[k];
    }

    // compare by sum of distances to k nearest neighbors.
    std::sort(sorted_mask.begin(), sorted_mask.end(), [&](int a, int b) {
        return sum_knn_distances[a] < sum_knn_distances[b];
    });

    // copy array of points to array with sorted points
    // using @sorted_idx mask of sorted points indexes

    sorted_points = Mat(points_size, dim, points.type());
    const auto * const points_ptr = (float *) points.data;
    auto * spoints_ptr = (float *) sorted_points.data;
    for (int i = 0; i < points_size; i++) {
        const int pt2 = sorted_mask[i] * dim;
        for (int j = 0; j < dim; j++)
            (*spoints_ptr++) =  points_ptr[pt2+j];
    }
}

Matx33d Math::getSkewSymmetric(const Vec3d &v) {
     return {0,    -v[2], v[1],
             v[2],  0,   -v[0],
            -v[1],  v[0], 0};
}

Matx33d Math::rotVec2RotMat (const Vec3d &v) {
    const double phi = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    const double x = v[0] / phi, y = v[1] / phi, z = v[2] / phi;
    const double a = sin(phi), b = cos(phi);
    // R = I + sin(phi) * skew(v) + (1 - cos(phi) * skew(v)^2
    return {(b - 1)*y*y + (b - 1)*z*z + 1, -a*z - x*y*(b - 1), a*y - x*z*(b - 1),
     a*z - x*y*(b - 1), (b - 1)*x*x + (b - 1)*z*z + 1, -a*x - y*z*(b - 1),
    -a*y - x*z*(b - 1), a*x - y*z*(b - 1), (b - 1)*x*x + (b - 1)*y*y + 1};
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
            continue;

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

double Utils::intersectionOverUnion (const std::vector<bool> &a, const std::vector<bool> &b) {
    int intersects = 0, unions = 0;
    for (int i = 0; i < (int)a.size(); i++)
        if (a[i] || b[i]) {
            unions++; // one value is true
            if (a[i] && b[i])
                intersects++; // a[i] == b[i] and if they both true
        }
    if (unions == 0) return 0.0;
    return (double) intersects / unions;
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
        if (subset_size < subset_size_)
            subset.resize(subset_size_);
        subset_size = subset_size_;
    }
    int getSubsetSize () const override { return subset_size; }
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
                quicksort_median(array, length/2+1, 0, length-1))*.5f;
    }
}

///////////////////////////////// Radius Search Graph /////////////////////////////////////////////
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
            if (n.size() <= 1)
                continue;
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
    const std::vector<std::vector<int>> &getGraph () const override { return graph; }
    inline const std::vector<int> &getNeighbors(int point_idx) const override {
        return graph[point_idx];
    }
};
Ptr<RadiusSearchNeighborhoodGraph> RadiusSearchNeighborhoodGraph::create (const Mat &points,
        int points_size, double radius_, int flann_search_params, int num_kd_trees) {
    return makePtr<RadiusSearchNeighborhoodGraphImpl> (points, points_size, radius_,
            flann_search_params, num_kd_trees);
}

///////////////////////////////// FLANN Graph /////////////////////////////////////////////
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
    const std::vector<std::vector<int>> &getGraph () const override { return graph; }
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

///////////////////////////////// Grid Neighborhood Graph /////////////////////////////////////////
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

    std::vector<std::vector<int>> graph;
public:
    GridNeighborhoodGraphImpl (const Mat &container_, int points_size,
          int cell_size_x_img1, int cell_size_y_img1, int cell_size_x_img2, int cell_size_y_img2,
          int max_neighbors) {

        std::map<CellCoord, std::vector<int >> neighbors_map;
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
    const std::vector<std::vector<int>> &getGraph () const override { return graph; }
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

class GridNeighborhoodGraph2ImagesImpl : public GridNeighborhoodGraph2Images {
private:
    // This struct is used for the nearest neighbors search by griding two images.
    struct CellCoord {
        int c1x, c1y;
        CellCoord (int c1x_, int c1y_) {
            c1x = c1x_; c1y = c1y_;
        }
        bool operator==(const CellCoord &o) const {
            return c1x == o.c1x && c1y == o.c1y;
        }
        bool operator<(const CellCoord &o) const {
            if (c1x < o.c1x) return true;
            return c1x == o.c1x && c1y < o.c1y;
        }
    };

    std::vector<std::vector<int>> graph;
public:
    GridNeighborhoodGraph2ImagesImpl (const Mat &container_, int points_size,
            float cell_size_x_img1, float cell_size_y_img1, float cell_size_x_img2, float cell_size_y_img2) {

        std::map<CellCoord, std::vector<int >> neighbors_map1, neighbors_map2;
        const auto * const container = (float *) container_.data;
        // Key is cell position. The value is indexes of neighbors.

        const auto cell_sz_x1 = 1.f / cell_size_x_img1,
                   cell_sz_y1 = 1.f / cell_size_y_img1,
                   cell_sz_x2 = 1.f / cell_size_x_img2,
                   cell_sz_y2 = 1.f / cell_size_y_img2;
        const int dimension = container_.cols;
        for (int i = 0; i < points_size; i++) {
            const int idx = dimension * i;
            neighbors_map1[CellCoord((int)(container[idx  ] * cell_sz_x1),
                                    (int)(container[idx+1] * cell_sz_y1))].emplace_back(i);
            neighbors_map2[CellCoord((int)(container[idx+2] * cell_sz_x2),
                                    (int)(container[idx+3] * cell_sz_y2))].emplace_back(i);
        }

        //--------- create a graph ----------
        graph = std::vector<std::vector<int>>(points_size);

        // store neighbors cells into graph (2D vector)
        for (const auto &cell : neighbors_map1) {
            const int neighbors_in_cell = static_cast<int>(cell.second.size());
            // only one point in cell -> no neighbors
            if (neighbors_in_cell < 2) continue;

            const std::vector<int> &neighbors = cell.second;
            // ---------- fill graph -----
            // for speed-up we make no symmetric graph, eg, x has a neighbor y, but y does not have x
            const int v_in_cell = neighbors[0];
            // there is always at least one neighbor
            auto &graph_row = graph[v_in_cell];
            graph_row.reserve(neighbors_in_cell);
            for (int n : neighbors)
                if (n != v_in_cell)
                    graph_row.emplace_back(n);
        }

        // fill neighbors of a second image
        for (const auto &cell : neighbors_map2) {
            if (cell.second.size() < 2) continue;
            const std::vector<int> &neighbors = cell.second;
            const int v_in_cell = neighbors[0];
            auto &graph_row = graph[v_in_cell];
            for (const int &n : neighbors)
                if (n != v_in_cell) {
                    bool has = false;
                    for (const int &nn : graph_row)
                        if (n == nn) {
                            has = true; break;
                        }
                    if (!has) graph_row.emplace_back(n);
                }
        }
    }
    const std::vector<std::vector<int>> &getGraph () const override { return graph; }
    inline const std::vector<int> &getNeighbors(int point_idx) const override {
        // Note, neighbors vector also includes point_idx!
        return graph[point_idx];
    }
};

Ptr<GridNeighborhoodGraph2Images> GridNeighborhoodGraph2Images::create(const Mat &points,
        int points_size, float cell_size_x_img1_, float cell_size_y_img1_, float cell_size_x_img2_, float cell_size_y_img2_) {
    return makePtr<GridNeighborhoodGraph2ImagesImpl>(points, points_size,
            cell_size_x_img1_, cell_size_y_img1_, cell_size_x_img2_, cell_size_y_img2_);
}
}}