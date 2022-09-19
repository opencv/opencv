// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"

namespace cv
{

/** Just compute the norm of a vector
 * @param vec a vector of size 3 and any type T
 * @return
 */
template<typename T>
T inline norm_vec(const Vec<T, 3>& vec)
{
    return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}
template<typename T>
T inline norm_vec(const Vec<T, 4>& vec)
{
    return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

/** Given 3d points, compute their distance to the origin
 * @param points
 * @return
 */
template<typename T>
Mat_<T> computeRadius(const Mat& points)
{
    typedef Vec<T, 4> PointT;

    // Compute the
    Size size(points.cols, points.rows);
    Mat_<T> r(size);
    if (points.isContinuous())
        size = Size(points.cols * points.rows, 1);
    for (int y = 0; y < size.height; ++y)
    {
        const PointT* point = points.ptr < PointT >(y), * point_end = points.ptr < PointT >(y) + size.width;
        T* row = r[y];
        for (; point != point_end; ++point, ++row)
            *row = norm_vec(*point);
    }

    return r;
}

// Compute theta and phi according to equation 3 of
// ``Fast and Accurate Computation of Surface Normals from Range Images``
// by H. Badino, D. Huber, Y. Park and T. Kanade
template<typename T>
void computeThetaPhi(int rows, int cols, const Matx<T, 3, 3>& K, Mat& cos_theta, Mat& sin_theta,
                     Mat& cos_phi, Mat& sin_phi)
{
    // Create some bogus coordinates
    Mat depth_image = K(0, 0) * Mat_<T> ::ones(rows, cols);
    Mat points3d;
    depthTo3d(depth_image, Mat(K), points3d);

    //typedef Vec<T, 3> Vec3T;
    typedef Vec<T, 4> Vec4T;

    cos_theta = Mat_<T>(rows, cols);
    sin_theta = Mat_<T>(rows, cols);
    cos_phi = Mat_<T>(rows, cols);
    sin_phi = Mat_<T>(rows, cols);
    Mat r = computeRadius<T>(points3d);
    for (int y = 0; y < rows; ++y)
    {
        T* row_cos_theta = cos_theta.ptr <T>(y), * row_sin_theta = sin_theta.ptr <T>(y);
        T* row_cos_phi = cos_phi.ptr <T>(y), * row_sin_phi = sin_phi.ptr <T>(y);
        const Vec4T* row_points = points3d.ptr <Vec4T>(y),
               * row_points_end = points3d.ptr <Vec4T>(y) + points3d.cols;
        const T* row_r = r.ptr < T >(y);
        for (; row_points < row_points_end;
            ++row_cos_theta, ++row_sin_theta, ++row_cos_phi, ++row_sin_phi, ++row_points, ++row_r)
        {
            // In the paper, z goes away from the camera, y goes down, x goes right
            // OpenCV has the same conventions
            // Theta goes from z to x (and actually goes from -pi/2 to pi/2, phi goes from z to y
            float theta = (float)std::atan2(row_points->val[0], row_points->val[2]);
            *row_cos_theta = std::cos(theta);
            *row_sin_theta = std::sin(theta);
            float phi = (float)std::asin(row_points->val[1] / (*row_r));
            *row_cos_phi = std::cos(phi);
            *row_sin_phi = std::sin(phi);
        }
    }
}

/** Modify normals to make sure they point towards the camera
 * @param normals
 */
template<typename T>
inline void signNormal(const Vec<T, 3>& normal_in, Vec<T, 3>& normal_out)
{
    Vec<T, 3> res;
    if (normal_in[2] > 0)
        res = -normal_in / norm_vec(normal_in);
    else
        res = normal_in / norm_vec(normal_in);

    normal_out[0] = res[0];
    normal_out[1] = res[1];
    normal_out[2] = res[2];
}
template<typename T>
inline void signNormal(const Vec<T, 3>& normal_in, Vec<T, 4>& normal_out)
{
    Vec<T, 3> res;
    if (normal_in[2] > 0)
        res = -normal_in / norm_vec(normal_in);
    else
        res = normal_in / norm_vec(normal_in);

    normal_out[0] = res[0];
    normal_out[1] = res[1];
    normal_out[2] = res[2];
    normal_out[3] = 0;
}

/** Modify normals to make sure they point towards the camera
 * @param normals
 */
template<typename T>
inline void signNormal(T a, T b, T c, Vec<T, 3>& normal)
{
    T norm = 1 / std::sqrt(a * a + b * b + c * c);
    if (c > 0)
    {
        normal[0] = -a * norm;
        normal[1] = -b * norm;
        normal[2] = -c * norm;
    }
    else
    {
        normal[0] = a * norm;
        normal[1] = b * norm;
        normal[2] = c * norm;
    }
}
template<typename T>
inline void signNormal(T a, T b, T c, Vec<T, 4>& normal)
{
    T norm = 1 / std::sqrt(a * a + b * b + c * c);
    if (c > 0)
    {
        normal[0] = -a * norm;
        normal[1] = -b * norm;
        normal[2] = -c * norm;
        normal[3] = 0;
    }
    else
    {
        normal[0] = a * norm;
        normal[1] = b * norm;
        normal[2] = c * norm;
        normal[3] = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class RgbdNormalsImpl : public RgbdNormals
{
public:
    static const int dtype = cv::traits::Depth<T>::value;

    RgbdNormalsImpl(int _rows, int _cols, int _windowSize, const Mat& _K, RgbdNormals::RgbdNormalsMethod _method) :
        rows(_rows),
        cols(_cols),
        windowSize(_windowSize),
        method(_method),
        cacheIsDirty(true)
    {
        CV_Assert(_K.cols == 3 && _K.rows == 3);

        _K.convertTo(K, dtype);
        _K.copyTo(K_ori);
    }

    virtual ~RgbdNormalsImpl() CV_OVERRIDE
    { }

    virtual int getDepth() const CV_OVERRIDE
    {
        return dtype;
    }
    virtual int getRows() const CV_OVERRIDE
    {
        return rows;
    }
    virtual void setRows(int val) CV_OVERRIDE
    {
        rows = val; cacheIsDirty = true;
    }
    virtual int getCols() const CV_OVERRIDE
    {
        return cols;
    }
    virtual void setCols(int val) CV_OVERRIDE
    {
        cols = val; cacheIsDirty = true;
    }
    virtual int getWindowSize() const CV_OVERRIDE
    {
        return windowSize;
    }
    virtual void setWindowSize(int val) CV_OVERRIDE
    {
        windowSize = val; cacheIsDirty = true;
    }
    virtual void getK(OutputArray val) const CV_OVERRIDE
    {
        K.copyTo(val);
    }
    virtual void setK(InputArray val) CV_OVERRIDE
    {
        K = val.getMat(); cacheIsDirty = true;
    }
    virtual RgbdNormalsMethod getMethod() const CV_OVERRIDE
    {
        return method;
    }

    virtual void compute(const Mat& in, Mat& normals) const = 0;

    /** Given a set of 3d points in a depth image, compute the normals at each point
     * @param points3d_in depth a float depth image. Or it can be rows x cols x 3 is they are 3d points
     * @param normals a rows x cols x 3 matrix
     */
    virtual void apply(InputArray points3d_in, OutputArray normals_out) const CV_OVERRIDE
    {
        Mat points3d_ori = points3d_in.getMat();

        CV_Assert(points3d_ori.dims == 2);

        // Either we have 3d points or a depth image

        bool ptsAre4F = (points3d_ori.channels() == 4) && (points3d_ori.depth() == CV_32F || points3d_ori.depth() == CV_64F);
        bool ptsAreDepth = (points3d_ori.channels() == 1) && (points3d_ori.depth() == CV_16U || points3d_ori.depth() == CV_32F || points3d_ori.depth() == CV_64F);
        if (method == RGBD_NORMALS_METHOD_FALS || method == RGBD_NORMALS_METHOD_SRI || method == RGBD_NORMALS_METHOD_CROSS_PRODUCT)
        {
            if (!ptsAre4F)
            {
                CV_Error(Error::StsBadArg, "Input image should have 4 float-point channels");
            }
        }
        else if (method == RGBD_NORMALS_METHOD_LINEMOD)
        {
            if (!ptsAre4F && !ptsAreDepth)
            {
                CV_Error(Error::StsBadArg, "Input image should have 4 float-point channels or have 1 ushort or float-point channel");
            }
        }
        else
        {
            CV_Error(Error::StsInternal, "Unknown normal computer algorithm");
        }

        // Initialize the pimpl
        cache();

        // Precompute something for RGBD_NORMALS_METHOD_SRI and RGBD_NORMALS_METHOD_FALS
        Mat points3d;
        if (method != RGBD_NORMALS_METHOD_LINEMOD)
        {
            // Make the points have the right depth
            if (points3d_ori.depth() == dtype)
                points3d = points3d_ori;
            else
                points3d_ori.convertTo(points3d, dtype);
        }

        // Get the normals
        normals_out.create(points3d_ori.size(), CV_MAKETYPE(dtype, 4));
        if (points3d_ori.empty())
            return;

        Mat normals = normals_out.getMat();
        if ((method == RGBD_NORMALS_METHOD_FALS) || (method == RGBD_NORMALS_METHOD_SRI))
        {
            // Compute the distance to the points
            Mat radius = computeRadius<T>(points3d);
            compute(radius, normals);
        }
        else if (method == RGBD_NORMALS_METHOD_LINEMOD)
        {
            compute(points3d_ori, normals);
        }
        else if (method == RGBD_NORMALS_METHOD_CROSS_PRODUCT)
        {
            compute(points3d, normals);
        }
        else
        {
            CV_Error(Error::StsInternal, "Unknown normal computer algorithm");
        }
    }

    int rows, cols;
    Mat K, K_ori;
    int windowSize;
    RgbdNormalsMethod method;
    mutable bool cacheIsDirty;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Given a set of 3d points in a depth image, compute the normals at each point
 * using the FALS method described in
 * ``Fast and Accurate Computation of Surface Normals from Range Images``
 * by H. Badino, D. Huber, Y. Park and T. Kanade
 */
template<typename T>
class FALS : public RgbdNormalsImpl<T>
{
public:
    typedef Matx<T, 3, 3> Mat33T;
    typedef Vec<T, 9> Vec9T;
    typedef Vec<T, 4> Vec4T;
    typedef Vec<T, 3> Vec3T;

    FALS(int _rows, int _cols, int _windowSize, const Mat& _K) :
        RgbdNormalsImpl<T>(_rows, _cols, _windowSize, _K, RgbdNormals::RGBD_NORMALS_METHOD_FALS)
    { }
    virtual ~FALS() CV_OVERRIDE
    { }

    /** Compute cached data
     */
    virtual void cache() const CV_OVERRIDE
    {
        if (!this->cacheIsDirty)
            return;

        // Compute theta and phi according to equation 3
        Mat cos_theta, sin_theta, cos_phi, sin_phi;
        computeThetaPhi<T>(this->rows, this->cols, this->K, cos_theta, sin_theta, cos_phi, sin_phi);

        // Compute all the v_i for every points
        std::vector<Mat> channels(3);
        channels[0] = sin_theta.mul(cos_phi);
        channels[1] = sin_phi;
        channels[2] = cos_theta.mul(cos_phi);
        merge(channels, V_);

        // Compute M
        Mat_<Vec9T> M(this->rows, this->cols);
        Mat33T VVt;
        const Vec3T* vec = V_[0];
        Vec9T* M_ptr = M[0], * M_ptr_end = M_ptr + this->rows * this->cols;
        for (; M_ptr != M_ptr_end; ++vec, ++M_ptr)
        {
            VVt = (*vec) * vec->t();
            *M_ptr = Vec9T(VVt.val);
        }

        boxFilter(M, M, M.depth(), Size(this->windowSize, this->windowSize), Point(-1, -1), false);

        // Compute M's inverse
        Mat33T M_inv;
        M_inv_.create(this->rows, this->cols);
        Vec9T* M_inv_ptr = M_inv_[0];
        for (M_ptr = &M(0); M_ptr != M_ptr_end; ++M_inv_ptr, ++M_ptr)
        {
            // We have a semi-definite matrix
            invert(Mat33T(M_ptr->val), M_inv, DECOMP_CHOLESKY);
            *M_inv_ptr = Vec9T(M_inv.val);
        }

        this->cacheIsDirty = false;
    }

    /** Compute the normals
     * @param r
     * @return
     */
    virtual void compute(const Mat& r, Mat& normals) const CV_OVERRIDE
    {
        // Compute B
        Mat_<Vec3T> B(this->rows, this->cols);

        const T* row_r = r.ptr < T >(0), * row_r_end = row_r + this->rows * this->cols;
        const Vec3T* row_V = V_[0];
        Vec3T* row_B = B[0];
        for (; row_r != row_r_end; ++row_r, ++row_B, ++row_V)
        {
            Vec3T val = (*row_V) / (*row_r);
            if (cvIsInf(val[0]) || cvIsNaN(val[0]) ||
                cvIsInf(val[1]) || cvIsNaN(val[1]) ||
                cvIsInf(val[2]) || cvIsNaN(val[2]))
                *row_B = Vec3T();
            else
                *row_B = val;
        }

        // Apply a box filter to B
        boxFilter(B, B, B.depth(), Size(this->windowSize, this->windowSize), Point(-1, -1), false);

        // compute the Minv*B products
        row_r = r.ptr < T >(0);
        const Vec3T* B_vec = B[0];
        const Mat33T* M_inv = reinterpret_cast<const Mat33T*>(M_inv_.ptr(0));
        //Vec3T* normal = normals.ptr<Vec3T>(0);
        Vec4T* normal = normals.ptr<Vec4T>(0);
        for (; row_r != row_r_end; ++row_r, ++B_vec, ++normal, ++M_inv)
            if (cvIsNaN(*row_r))
            {
                (*normal)[0] = *row_r;
                (*normal)[1] = *row_r;
                (*normal)[2] = *row_r;
                (*normal)[3] = 0;
            }
            else
            {
                Mat33T Mr = *M_inv;
                Vec3T Br = *B_vec;
                Vec3T MBr(Mr(0, 0) * Br[0] + Mr(0, 1) * Br[1] + Mr(0, 2) * Br[2],
                          Mr(1, 0) * Br[0] + Mr(1, 1) * Br[1] + Mr(1, 2) * Br[2],
                          Mr(2, 0) * Br[0] + Mr(2, 1) * Br[1] + Mr(2, 2) * Br[2]);
                signNormal(MBr, *normal);
            }
    }

    // Cached data
    mutable Mat_<Vec3T> V_;
    mutable Mat_<Vec9T> M_inv_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Function that multiplies K_inv by a vector. It is just meant to speed up the product as we know
 * that K_inv is upper triangular and K_inv(2,2)=1
 * @param K_inv
 * @param a
 * @param b
 * @param c
 * @param res
 */
template<typename T, typename U>
void multiply_by_K_inv(const Matx<T, 3, 3>& K_inv, U a, U b, U c, Vec<T, 3>& res)
{
    res[0] = (T)(K_inv(0, 0) * a + K_inv(0, 1) * b + K_inv(0, 2) * c);
    res[1] = (T)(K_inv(1, 1) * b + K_inv(1, 2) * c);
    res[2] = (T)c;
}

/** Given a depth image, compute the normals as detailed in the LINEMOD paper
 * ``Gradient Response Maps for Real-Time Detection of Texture-Less Objects``
 * by S. Hinterstoisser, C. Cagniart, S. Ilic, P. Sturm, N. Navab, P. Fua, and V. Lepetit
 */
template<typename T>
class LINEMOD : public RgbdNormalsImpl<T>
{
public:
    typedef Vec<T, 4> Vec4T;
    typedef Vec<T, 3> Vec3T;
    typedef Matx<T, 3, 3> Mat33T;

    LINEMOD(int _rows, int _cols, int _windowSize, const Mat& _K, double _diffThr = 50.0) :
        RgbdNormalsImpl<T>(_rows, _cols, _windowSize, _K, RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD),
        differenceThreshold(_diffThr)
    { }

    /** Compute cached data
     */
    virtual void cache() const CV_OVERRIDE
    {
        this->cacheIsDirty = false;
    }

    /** Compute the normals
     * @param r
     * @param normals the output normals
     */
    virtual void compute(const Mat& points3d, Mat& normals) const CV_OVERRIDE
    {
        // Only focus on the depth image for LINEMOD
        Mat depth_in;
        //if (points3d.channels() == 3)
        if (points3d.channels() == 4)
        {
            std::vector<Mat> channels;
            split(points3d, channels);
            depth_in = channels[2];
        }
        else
            depth_in = points3d;

        switch (depth_in.depth())
        {
        case CV_16U:
        {
            const Mat_<unsigned short>& d(depth_in);
            computeImpl<unsigned short, long>(d, normals);
            break;
        }
        case CV_32F:
        {
            const Mat_<float>& d(depth_in);
            computeImpl<float, float>(d, normals);
            break;
        }
        case CV_64F:
        {
            const Mat_<double>& d(depth_in);
            computeImpl<double, double>(d, normals);
            break;
        }
        }
    }

    /** Compute the normals
     * @param r
     * @return
     */
    template<typename DepthDepth, typename ContainerDepth>
    Mat computeImpl(const Mat_<DepthDepth>& depthIn, Mat& normals) const
    {
        const int r = 5; // used to be 7
        const int sample_step = r;
        const int square_size = ((2 * r / sample_step) + 1);
        long offsets[square_size * square_size];
        long offsets_x[square_size * square_size];
        long offsets_y[square_size * square_size];
        long offsets_x_x[square_size * square_size];
        long offsets_x_y[square_size * square_size];
        long offsets_y_y[square_size * square_size];
        for (int j = -r, index = 0; j <= r; j += sample_step)
            for (int i = -r; i <= r; i += sample_step, ++index)
            {
                offsets_x[index] = i;
                offsets_y[index] = j;
                offsets_x_x[index] = i * i;
                offsets_x_y[index] = i * j;
                offsets_y_y[index] = j * j;
                offsets[index] = j * this->cols + i;
            }

        // Define K_inv by hand, just for higher accuracy
        Mat33T K_inv = Matx<T, 3, 3>::eye(), kmat;
        this->K.copyTo(kmat);
        K_inv(0, 0) = 1.0f / kmat(0, 0);
        K_inv(0, 1) = -kmat(0, 1) / (kmat(0, 0) * kmat(1, 1));
        K_inv(0, 2) = (kmat(0, 1) * kmat(1, 2) - kmat(0, 2) * kmat(1, 1)) / (kmat(0, 0) * kmat(1, 1));
        K_inv(1, 1) = 1 / kmat(1, 1);
        K_inv(1, 2) = -kmat(1, 2) / kmat(1, 1);

        Vec3T X1_minus_X, X2_minus_X;

        ContainerDepth difference_threshold((ContainerDepth)differenceThreshold);
        //TODO: fixit, difference threshold should not depend on input type
        difference_threshold *= (ContainerDepth)(std::is_same<DepthDepth, ushort>::value ? 1000.0 : 1.0);
        normals.setTo(std::numeric_limits<DepthDepth>::quiet_NaN());
        for (int y = r; y < this->rows - r - 1; ++y)
        {
            const DepthDepth* p_line = reinterpret_cast<const DepthDepth*>(depthIn.ptr(y, r));
            Vec4T* normal = normals.ptr<Vec4T>(y, r);

            for (int x = r; x < this->cols - r - 1; ++x)
            {
                DepthDepth d = p_line[0];

                // accum
                long A[4];
                A[0] = A[1] = A[2] = A[3] = 0;
                ContainerDepth b[2];
                b[0] = b[1] = 0;
                for (unsigned int i = 0; i < square_size * square_size; ++i) {
                    // We need to cast to ContainerDepth in case we have unsigned DepthDepth
                    ContainerDepth delta = ContainerDepth(p_line[offsets[i]]) - ContainerDepth(d);
                    if (std::abs(delta) > difference_threshold)
                        continue;

                    A[0] += offsets_x_x[i];
                    A[1] += offsets_x_y[i];
                    A[3] += offsets_y_y[i];
                    b[0] += offsets_x[i] * delta;
                    b[1] += offsets_y[i] * delta;
                }

                // solve for the optimal gradient D of equation (8)
                long det = A[0] * A[3] - A[1] * A[1];
                // We should divide the following two by det, but instead, we multiply
                // X1_minus_X and X2_minus_X by det (which does not matter as we normalize the normals)
                // Therefore, no division is done: this is only for speedup
                ContainerDepth dx = (A[3] * b[0] - A[1] * b[1]);
                ContainerDepth dy = (-A[1] * b[0] + A[0] * b[1]);

                // Compute the dot product
                //Vec3T X = K_inv * Vec3T(x, y, 1) * depth(y, x);
                //Vec3T X1 = K_inv * Vec3T(x + 1, y, 1) * (depth(y, x) + dx);
                //Vec3T X2 = K_inv * Vec3T(x, y + 1, 1) * (depth(y, x) + dy);
                //Vec3T nor = (X1 - X).cross(X2 - X);
                multiply_by_K_inv(K_inv, d * det + (x + 1) * dx, y * dx, dx, X1_minus_X);
                multiply_by_K_inv(K_inv, x * dy, d * det + (y + 1) * dy, dy, X2_minus_X);
                Vec3T nor = X1_minus_X.cross(X2_minus_X);
                signNormal(nor, *normal);

                ++p_line;
                ++normal;
            }
        }

        return normals;
    }

    double differenceThreshold;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Given a set of 3d points in a depth image, compute the normals at each point
 * using the SRI method described in
 * ``Fast and Accurate Computation of Surface Normals from Range Images``
 * by H. Badino, D. Huber, Y. Park and T. Kanade
 */
template<typename T>
class SRI : public RgbdNormalsImpl<T>
{
public:
    typedef Matx<T, 3, 3> Mat33T;
    typedef Vec<T, 9> Vec9T;
    typedef Vec<T, 4> Vec4T;
    typedef Vec<T, 3> Vec3T;

    SRI(int _rows, int _cols, int _windowSize, const Mat& _K) :
        RgbdNormalsImpl<T>(_rows, _cols, _windowSize, _K, RgbdNormals::RGBD_NORMALS_METHOD_SRI),
        phi_step_(0),
        theta_step_(0)
    { }

    /** Compute cached data
     */
    virtual void cache() const CV_OVERRIDE
    {
        if (!this->cacheIsDirty)
            return;

        Mat_<T> cos_theta, sin_theta, cos_phi, sin_phi;
        computeThetaPhi<T>(this->rows, this->cols, this->K, cos_theta, sin_theta, cos_phi, sin_phi);

        // Create the derivative kernels
        getDerivKernels(kx_dx_, ky_dx_, 1, 0, this->windowSize, true, this->dtype);
        getDerivKernels(kx_dy_, ky_dy_, 0, 1, this->windowSize, true, this->dtype);

        // Get the mapping function for SRI
        float min_theta = (float)std::asin(sin_theta(0, 0)), max_theta = (float)std::asin(sin_theta(0, this->cols - 1));
        float min_phi = (float)std::asin(sin_phi(0, this->cols / 2 - 1)), max_phi = (float)std::asin(sin_phi(this->rows - 1, this->cols / 2 - 1));

        std::vector<Point3f> points3d(this->cols * this->rows);
        R_hat_.create(this->rows, this->cols);
        phi_step_ = float(max_phi - min_phi) / (this->rows - 1);
        theta_step_ = float(max_theta - min_theta) / (this->cols - 1);
        for (int phi_int = 0, k = 0; phi_int < this->rows; ++phi_int)
        {
            float phi = min_phi + phi_int * phi_step_;
            float phi_sin = std::sin(phi), phi_cos = std::cos(phi);
            for (int theta_int = 0; theta_int < this->cols; ++theta_int, ++k)
            {
                float theta = min_theta + theta_int * theta_step_;
                float theta_sin = std::sin(theta), theta_cos = std::cos(theta);
                // Store the 3d point to project it later
                Point3f pp(theta_sin * phi_cos, phi_sin, theta_cos * phi_cos);
                points3d[k] = pp;

                // Cache the rotation matrix and negate it
                Matx<T, 3, 3> mat = Matx<T, 3, 3> (0, 1, 0,  0, 0, 1,  1, 0, 0) *
                                    Matx<T, 3, 3> (theta_cos, -theta_sin, 0,  theta_sin, theta_cos, 0,  0, 0, 1) *
                                    Matx<T, 3, 3> (phi_cos, 0, -phi_sin,  0, 1, 0,  phi_sin, 0, phi_cos);

                for (unsigned char i = 0; i < 3; ++i)
                    mat(i, 1) = mat(i, 1) / phi_cos;
                // The second part of the matrix is never explained in the paper ... but look at the wikipedia normal article
                mat(0, 0) = mat(0, 0) - 2 * pp.x;
                mat(1, 0) = mat(1, 0) - 2 * pp.y;
                mat(2, 0) = mat(2, 0) - 2 * pp.z;

                R_hat_(phi_int, theta_int) = Vec9T(mat.val);
            }
        }

        map_.create(this->rows, this->cols);
        projectPoints(points3d, Mat(3, 1, CV_32FC1, Scalar::all(0.0f)), Mat(3, 1, CV_32FC1, Scalar::all(0.0f)), this->K, Mat(), map_);
        map_ = map_.reshape(2, this->rows);
        convertMaps(map_, Mat(), xy_, fxy_, CV_16SC2);

        //map for converting from Spherical coordinate space to Euclidean space
        euclideanMap_.create(this->rows, this->cols);
        Matx<T, 3, 3> km(this->K);
        float invFx = (float)(1.0f / km(0, 0)), cx = (float)(km(0, 2));
        double invFy = 1.0f / (km(1, 1)), cy = km(1, 2);
        for (int i = 0; i < this->rows; i++)
        {
            float y = (float)((i - cy) * invFy);
            for (int j = 0; j < this->cols; j++)
            {
                float x = (j - cx) * invFx;
                float theta = std::atan(x);
                float phi = std::asin(y / std::sqrt(x * x + y * y + 1.0f));

                euclideanMap_(i, j) = Vec2f((theta - min_theta) / theta_step_, (phi - min_phi) / phi_step_);
            }
        }
        //convert map to 2 maps in short format for increasing speed in remap function
        convertMaps(euclideanMap_, Mat(), invxy_, invfxy_, CV_16SC2);

        // Update the kernels: the steps are due to the fact that derivatives will be computed on a grid where
        // the step is not 1. Only need to do it on one dimension as it computes derivatives in only one direction
        kx_dx_ /= theta_step_;
        ky_dy_ /= phi_step_;

        this->cacheIsDirty = false;
    }

    /** Compute the normals
     * @param r
     * @return
     */
    virtual void compute(const Mat& in, Mat& normals_out) const CV_OVERRIDE
    {
        const Mat_<T>& r_non_interp = in;

        // Interpolate the radial image to make derivatives meaningful
        Mat_<T> r;
        // higher quality remapping does not help here
        remap(r_non_interp, r, xy_, fxy_, INTER_LINEAR);

        // Compute the derivatives with respect to theta and phi
        // TODO add bilateral filtering (as done in kinfu)
        Mat_<T> r_theta, r_phi;
        sepFilter2D(r, r_theta, r.depth(), kx_dx_, ky_dx_);
        //current OpenCV version sometimes corrupts r matrix after second call of sepFilter2D
        //it depends on resolution, be careful
        sepFilter2D(r, r_phi, r.depth(), kx_dy_, ky_dy_);

        // Fill the result matrix
        Mat_<Vec4T> normals(this->rows, this->cols);

        const T* r_theta_ptr = r_theta[0], * r_theta_ptr_end = r_theta_ptr + this->rows * this->cols;
        const T* r_phi_ptr = r_phi[0];
        const Mat33T* R = reinterpret_cast<const Mat33T*>(R_hat_[0]);
        const T* r_ptr = r[0];
        Vec4T* normal = normals[0];
        for (; r_theta_ptr != r_theta_ptr_end; ++r_theta_ptr, ++r_phi_ptr, ++R, ++r_ptr, ++normal)
        {
            if (cvIsNaN(*r_ptr))
            {
                (*normal)[0] = *r_ptr;
                (*normal)[1] = *r_ptr;
                (*normal)[2] = *r_ptr;
                (*normal)[3] = 0;
            }
            else
            {
                T r_theta_over_r = (*r_theta_ptr) / (*r_ptr);
                T r_phi_over_r = (*r_phi_ptr) / (*r_ptr);
                // R(1,1) is 0
                signNormal((*R)(0, 0) + (*R)(0, 1) * r_theta_over_r + (*R)(0, 2) * r_phi_over_r,
                           (*R)(1, 0) + (*R)(1, 2) * r_phi_over_r,
                           (*R)(2, 0) + (*R)(2, 1) * r_theta_over_r + (*R)(2, 2) * r_phi_over_r, *normal);
            }
        }

        remap(normals, normals_out, invxy_, invfxy_, INTER_LINEAR);
        normal = normals_out.ptr<Vec4T>(0);
        Vec4T* normal_end = normal + this->rows * this->cols;
        for (; normal != normal_end; ++normal)
            signNormal((*normal)[0], (*normal)[1], (*normal)[2], *normal);
    }

    // Cached data
    /** Stores R */
    mutable Mat_<Vec9T> R_hat_;
    mutable float phi_step_, theta_step_;

    /** Derivative kernels */
    mutable Mat kx_dx_, ky_dx_, kx_dy_, ky_dy_;
    /** mapping function to get an SRI image */
    mutable Mat_<Vec2f> map_;
    mutable Mat xy_, fxy_;

    mutable Mat_<Vec2f> euclideanMap_;
    mutable Mat invxy_, invfxy_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* Uses the simpliest possible method for normals calculation: calculates cross product between two vectors
(pointAt(x+1, y) - pointAt(x, y)) and (pointAt(x, y+1) - pointAt(x, y)) */

template<typename DataType>
class CrossProduct : public RgbdNormalsImpl<DataType>
{
public:
    typedef Vec<DataType, 3> Vec3T;
    typedef Vec<DataType, 4> Vec4T;
    typedef Point3_<DataType> Point3T;

    CrossProduct(int _rows, int _cols, int _windowSize, const Mat& _K) :
        RgbdNormalsImpl<DataType>(_rows, _cols, _windowSize, _K, RgbdNormals::RGBD_NORMALS_METHOD_CROSS_PRODUCT)
    { }

    /** Compute cached data
     */
    virtual void cache() const CV_OVERRIDE
    {
        this->cacheIsDirty = false;
    }

    static inline Point3T fromVec(Vec4T v)
    {
        return {v[0], v[1], v[2]};
    }

    static inline Vec4T toVec4(Point3T p)
    {
        return {p.x, p.y, p.z, 0};
    }

    static inline bool haveNaNs(Point3T p)
    {
        return cvIsNaN(p.x) || cvIsNaN(p.y) || cvIsNaN(p.z);
    }

    /** Compute the normals
     * @param points reprojected depth points
     * @param normals generated normals
     * @return
     */
    virtual void compute(const Mat& points, Mat& normals) const CV_OVERRIDE
    {
        for(int y = 0; y < this->rows; y++)
        {
            const Vec4T* ptsRow0 = points.ptr<Vec4T>(y);
            const Vec4T* ptsRow1 = (y < this->rows - 1) ? points.ptr<Vec4T>(y + 1) : nullptr;
            Vec4T *normRow = normals.ptr<Vec4T>(y);

            for (int x = 0; x < this->cols; x++)
            {
                Point3T v00 = fromVec(ptsRow0[x]);
                const float qnan = std::numeric_limits<float>::quiet_NaN();
                Point3T n(qnan, qnan, qnan);

                if ((x < this->cols - 1) && (y < this->rows - 1) && !haveNaNs(v00))
                {
                    Point3T v01 = fromVec(ptsRow0[x + 1]);
                    Point3T v10 = fromVec(ptsRow1[x]);

                    if (!haveNaNs(v01) && !haveNaNs(v10))
                    {
                        Vec3T vec = (v10 - v00).cross(v01 - v00);
                        n = normalize(vec);
                    }
                }

                normRow[x] = toVec4(n);
            }
        }
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Ptr<RgbdNormals> RgbdNormals::create(int rows, int cols, int depth, InputArray K, int windowSize, float diffThreshold, RgbdNormalsMethod method)
{
    CV_Assert(rows > 0 && cols > 0 && (depth == CV_32F || depth == CV_64F));
    CV_Assert(windowSize == 1 || windowSize == 3 || windowSize == 5 || windowSize == 7);
    CV_Assert(K.cols() == 3 && K.rows() == 3 && (K.depth() == CV_32F || K.depth() == CV_64F));

    Mat mK = K.getMat();
    Ptr<RgbdNormals> ptr;
    switch (method)
    {
    case (RGBD_NORMALS_METHOD_FALS):
    {
        if (depth == CV_32F)
            ptr = makePtr<FALS<float> >(rows, cols, windowSize, mK);
        else
            ptr = makePtr<FALS<double>>(rows, cols, windowSize, mK);
        break;
    }
    case (RGBD_NORMALS_METHOD_LINEMOD):
    {
        CV_Assert(diffThreshold > 0);
        if (depth == CV_32F)
            ptr = makePtr<LINEMOD<float> >(rows, cols, windowSize, mK, diffThreshold);
        else
            ptr = makePtr<LINEMOD<double>>(rows, cols, windowSize, mK, diffThreshold);
        break;
    }
    case RGBD_NORMALS_METHOD_SRI:
    {
        if (depth == CV_32F)
            ptr = makePtr<SRI<float> >(rows, cols, windowSize, mK);
        else
            ptr = makePtr<SRI<double>>(rows, cols, windowSize, mK);
        break;
    }
    case RGBD_NORMALS_METHOD_CROSS_PRODUCT:
    {
        if (depth == CV_32F)
            ptr = makePtr<CrossProduct<float> >(rows, cols, windowSize, mK);
        else
            ptr = makePtr<CrossProduct<double>>(rows, cols, windowSize, mK);
        break;
    }
    default:
        CV_Error(Error::StsBadArg, "Unknown normals compute algorithm");
    }

    return ptr;
}

} // namespace cv
