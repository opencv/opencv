#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv
{

class CV_EXPORTS_W UMat
{
public:
    //! default constructor
    CV_WRAP UMat(UMatUsageFlags usageFlags = USAGE_DEFAULT);
    //! constructs 2D matrix of the specified size and type
    // (_type is CV_8UC1, CV_64FC3, CV_32SC(12) etc.)
    CV_WRAP UMat(int rows, int cols, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    CV_WRAP UMat(Size size, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    //! constucts 2D matrix and fills it with the specified value _s.
    CV_WRAP UMat(int rows, int cols, int type, const Scalar& s, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    CV_WRAP UMat(Size size, int type, const Scalar& s, UMatUsageFlags usageFlags = USAGE_DEFAULT);

    //! Mat is mappable to UMat
    CV_WRAP_MAPPABLE(Ptr<Mat>);

    //! returns the OpenCL queue used by OpenCV UMat
    CV_WRAP_PHANTOM(static void* queue());

    //! returns the OpenCL context used by OpenCV UMat
    CV_WRAP_PHANTOM(static void* context());

    //! copy constructor
    CV_WRAP UMat(const UMat& m);

    //! creates a matrix header for a part of the bigger matrix
    CV_WRAP UMat(const UMat& m, const Range& rowRange, const Range& colRange = Range::all());
    CV_WRAP UMat(const UMat& m, const Rect& roi);
    CV_WRAP UMat(const UMat& m, const std::vector<Range>& ranges);

    //CV_WRAP_AS(get) Mat getMat(int flags CV_WRAP_DEFAULT(ACCESS_RW)) const;
    //! returns a numpy matrix
    CV_WRAP_PHANTOM(Mat get() const);

    //! returns true iff the matrix data is continuous
    // (i.e. when there are no gaps between successive rows).
    // similar to CV_IS_MAT_CONT(cvmat->type)
    CV_WRAP bool isContinuous() const;

    //! returns true if the matrix is a submatrix of another matrix
    CV_WRAP bool isSubmatrix() const;

    /*! Returns the OpenCL buffer handle on which UMat operates on.
    The UMat instance should be kept alive during the use of the handle to prevent the buffer to be
    returned to the OpenCV buffer pool.
    */
    CV_WRAP void* handle(AccessFlag accessFlags) const;

    // offset of the submatrix (or 0)
    CV_PROP_RW size_t offset;
};

} // namespace cv
