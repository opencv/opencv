#include "test_precomp.hpp"
#include <time.h>

#define CALIB3D_HOMOGRAPHY_ERROR_MATRIX_SIZE 1
#define CALIB3D_HOMOGRAPHY_ERROR_MATRIX_DIFF 2
#define CALIB3D_HOMOGRAPHY_ERROR_REPROJ_DIFF 3
#define CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK 4
#define CALIB3D_HOMOGRAPHY_ERROR_RANSAC_DIFF 5

#define MESSAGE_MATRIX_SIZE "Homography matrix must have 3*3 sizes."
#define MESSAGE_MATRIX_DIFF "Accuracy of homography transformation matrix less than required."
#define MESSAGE_REPROJ_DIFF_1 "Reprojection error for current pair of points more than required."
#define MESSAGE_REPROJ_DIFF_2 "Reprojection error is not optimal."
#define MESSAGE_RANSAC_MASK_1 "Sizes of inliers/outliers mask are incorrect."
#define MESSAGE_RANSAC_MASK_2 "Mask mustn't have any outliers."
#define MESSAGE_RANSAC_MASK_3 "All values of mask must be 1 (true) or 0 (false)."
#define MESSAGE_RANSAC_MASK_4 "Mask of inliers/outliers is incorrect."
#define MESSAGE_RANSAC_MASK_5 "Inlier in original mask shouldn't be outlier in found mask."
#define MESSAGE_RANSAC_DIFF "Reprojection error for current pair of points more than required."

#define MAX_COUNT_OF_POINTS 303
#define COUNT_NORM_TYPES 3
#define METHODS_COUNT 3

size_t NORM_TYPE[COUNT_NORM_TYPES] = {cv::NORM_L1, cv::NORM_L2, cv::NORM_INF};
size_t METHOD[METHODS_COUNT] = {0, CV_RANSAC, CV_LMEDS};

using namespace cv;
using namespace std;

class CV_HomographyTest: public cvtest::ArrayTest
{
 public:
	     
	    CV_HomographyTest();
		~CV_HomographyTest();

		int read_params( CvFileStorage* fs );
		void fill_array( int test_case_idx, int i, int j, Mat& arr );
		int prepare_test_case( int test_case_idx );
		void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
        void run (int);

		bool check_matrix (const Mat& H);
		bool check_transform (const Mat& src, const Mat& dst, const Mat& H);
	    

		void prepare_to_validation( int test_case_idx );

 protected:

	    int method;
		int image_size;
		int square_size;
		double reproj_threshold;
        double sigma;
        bool test_cpp;
		 
		double get_success_error_level( int test_case_idx, int i, int j );
		void test_projectPoints(Mat& src_2d, Mat& dst_2d, const Mat& H, RNG* rng, double sigma);  // checking for quality of perpective transformation
		
 private:
	float max_diff, max_2diff;
	bool check_matrix_size(const cv::Mat& H);
	bool check_matrix_diff(const cv::Mat& original, const cv::Mat& found, const int norm_type, double &diff);
	// bool check_reproj_error(const cv::Mat& src_3d, const cv::Mat& dst_3d, const int norm_type = NORM_L2);
	int check_ransac_mask_1(const Mat& src, const Mat& mask);
	int check_ransac_mask_2(const Mat& original_mask, const Mat& found_mask);

	void print_information_1(int j, int N, int method, const Mat& H);
	void print_information_2(int j, int N, int method, const Mat& H, const Mat& H_res, int k, double diff);
	void print_information_3(int j, int N, const Mat& mask);
	void print_information_4(int method, int j, int N, int k, int l, double diff);
	void print_information_5(int method, int j, int N, int l, double diff);
	void print_information_6(int j, int N, int k, double diff, bool value);
	void print_information_7(int j, int N, int k, double diff, bool original_value, bool found_value);
	void print_information_8(int j, int N, int k, int l, double diff);

	void check_transform_quality(cv::InputArray src_points, cv::InputArray dst_poits, const cv::Mat& H, const int norm_type = NORM_L2);
	void check_transform_quality(const cv::InputArray src_points, const vector <cv::Point2f> dst_points, const cv::Mat& H, const int norm_type = NORM_L2);
	void check_transform_quality(const vector <cv::Point2f> src_points, const cv::InputArray dst_points, const cv::Mat& H, const int norm_type = NORM_L2); 
	void check_transform_quality(const vector <cv::Point2f> src_points, const vector <cv::Point2f> dst_points, const cv::Mat& H, const int norm_type = NORM_L2);
};

CV_HomographyTest::CV_HomographyTest() : max_diff(1e-2), max_2diff(2e-2)
{
 test_array[INPUT].push_back(NULL);
 test_array[INPUT].push_back(NULL);
 test_array[INPUT].push_back(NULL);
 test_array[INPUT].push_back(NULL);
 test_array[INPUT].push_back(NULL);
 test_array[INPUT].push_back(NULL);
 test_array[TEMP].push_back(NULL);
 test_array[TEMP].push_back(NULL);
 test_array[OUTPUT].push_back(NULL);
 test_array[OUTPUT].push_back(NULL);
 test_array[REF_OUTPUT].push_back(NULL);
 test_array[REF_OUTPUT].push_back(NULL);

 element_wise_relative_error = false;

 method = 0;
 image_size = 1e+2;
 reproj_threshold = 3.0;
 sigma = 0.01;

 test_cpp = false;
}

CV_HomographyTest::~CV_HomographyTest() {}

void CV_HomographyTest::get_test_array_types_and_sizes( int /*test_case_idx*/, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int pt_depth = CV_32F;
    double pt_count_exp = cvtest::randReal(rng)*6 + 1;
    int pt_count = cvRound(exp(pt_count_exp));

    /* dims = cvtest::randInt(rng) % 2 + 2;
    method = 1 << (cvtest::randInt(rng) % 4);

    if( method == CV_FM_7POINT )
        pt_count = 7;
    else
    {
        pt_count = MAX( pt_count, 8 + (method == CV_FM_8POINT) );
        if( pt_count >= 8 && cvtest::randInt(rng) % 2 )
            method |= CV_FM_8POINT;
    } */

	types[INPUT][0] = CV_MAKETYPE(pt_depth, 2);
	
	types[INPUT][1] = types[INPUT][0];

	types[OUTPUT][0] = CV_MAKETYPE(pt_depth, 1);
	
    /* if( cvtest::randInt(rng) % 2 )
        sizes[INPUT][0] = cvSize(pt_count, dims);
    else
    {
        sizes[INPUT][0] = cvSize(dims, pt_count);
        if( cvtest::randInt(rng) % 2 )
        {
            types[INPUT][0] = CV_MAKETYPE(pt_depth, dims);
            if( cvtest::randInt(rng) % 2 )
                sizes[INPUT][0] = cvSize(pt_count, 1);
            else
                sizes[INPUT][0] = cvSize(1, pt_count);
        }
    }

    sizes[INPUT][1] = sizes[INPUT][0];
    types[INPUT][1] = types[INPUT][0];

    sizes[INPUT][2] = cvSize(pt_count, 1 );
    types[INPUT][2] = CV_64FC3;

    sizes[INPUT][3] = cvSize(4,3);
    types[INPUT][3] = CV_64FC1;

    sizes[INPUT][4] = sizes[INPUT][5] = cvSize(3,3);
    types[INPUT][4] = types[INPUT][5] = CV_MAKETYPE(CV_64F, 1);

    sizes[TEMP][0] = cvSize(3,3);
    types[TEMP][0] = CV_64FC1;
    sizes[TEMP][1] = cvSize(pt_count,1);
    types[TEMP][1] = CV_8UC1;

    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(3,1);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_64FC1;
    sizes[OUTPUT][1] = sizes[REF_OUTPUT][1] = cvSize(pt_count,1);
    types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_8UC1;
    
    test_cpp = (cvtest::randInt(rng) & 256) == 0;
	*/
}

int CV_HomographyTest::read_params(CvFileStorage *fs)
{
 int code = cvtest::ArrayTest::read_params(fs);
 return code;
}

double CV_HomographyTest::get_success_error_level(int test_case_idx, int i, int j) 
{
 return max_diff;
}

void CV_HomographyTest::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
 double t[9]={0};
 RNG& rng = ts->get_rng();

 if ( i != INPUT )
 {
  cvtest::ArrayTest::fill_array( test_case_idx, i, j, arr );
  return;
 }

 switch( j )
    {
    case 0:
    case 1:
        return; // fill them later in prepare_test_case
    case 2:
        {
         double* p = arr.ptr<double>();
         for( i = 0; i < arr.cols*3; i += 3 )
         {
          /* p[i] = cvtest::randReal(rng)*square_size;
          p[i+1] = cvtest::randReal(rng)*square_size;
          p[i+2] = cvtest::randReal(rng)*square_size + square_size; */
         }
        }
        break;
    case 3:
        {
         double r[3];
         Mat rot_vec( 3, 1, CV_64F, r );
         Mat rot_mat( 3, 3, CV_64F, t, 4*sizeof(t[0]) );
         r[0] = cvtest::randReal(rng)*CV_PI*2;
         r[1] = cvtest::randReal(rng)*CV_PI*2;
         r[2] = cvtest::randReal(rng)*CV_PI*2;

         cvtest::Rodrigues( rot_vec, rot_mat );
         /* t[3] = cvtest::randReal(rng)*square_size;
         t[7] = cvtest::randReal(rng)*square_size;
         t[11] = cvtest::randReal(rng)*square_size; */
         Mat( 3, 4, CV_64F, t ).convertTo(arr, arr.type());
        }
        break;
    case 4:
    case 5:
		{
         /* t[0] = t[4] = cvtest::randReal(rng)*(max_f - min_f) + min_f;
            t[2] = (img_size*0.5 + cvtest::randReal(rng)*4. - 2.)*t[0];
            t[5] = (img_size*0.5 + cvtest::randReal(rng)*4. - 2.)*t[4];
            t[8] = 1.0f;
            Mat( 3, 3, CV_64F, t ).convertTo( arr, arr.type() ); */
         break;
		}
    }
}

int CV_HomographyTest::prepare_test_case(int test_case_idx)
{
 int code = cvtest::ArrayTest::prepare_test_case(test_case_idx);

 if (code > 0)  
 {
  Mat& src = test_mat[INPUT][0];
  RNG& rng = ts->get_rng();

  float Hdata[] = { sqrt(2.0f)/2, -sqrt(2.0f)/2, 0.0f, 
	                sqrt(2.0f)/2,  sqrt(2.0f)/2, 0.0f, 
					0.0f,		   0.0f,		 1.0f };
  
  Mat H( 3, 3, CV_32F, Hdata );

  cv::Mat dst(1, src.cols, CV_32FC2);
  
  int k;

  for( k = 0; k < 2; k++ )
  {
   const Mat& H = test_mat[OUTPUT][0];
   Mat& dst = test_mat[INPUT][k == 0 ? 1 : 2];

   for (int i = 0; i < src.cols; ++i)
   {
	float *s = src.ptr<float>()+2*i;
    float *d = dst.ptr<float>()+2*i;

    d[0] = Hdata[0]*s[0] + Hdata[1]*s[1] + Hdata[2];
	d[1] = Hdata[3]*s[0] + Hdata[4]*s[1] + Hdata[5];
   }

   test_projectPoints( src, dst, H, &rng, sigma );
  }
 }

 return code;
}

static void test_convertHomogeneous( const Mat& _src, Mat& _dst )
{
 Mat src = _src, dst = _dst;
 
 int i, count, sdims, ddims;
 int sstep1, sstep2, dstep1, dstep2;

 if( src.depth() != CV_64F ) _src.convertTo(src, CV_64F);
    
 if( dst.depth() != CV_64F ) dst.create(dst.size(), CV_MAKETYPE(CV_64F, _dst.channels()));

 if( src.rows > src.cols )
 {
  count = src.rows;
  sdims = src.channels()*src.cols;
  sstep1 = (int)(src.step/sizeof(double));
  sstep2 = 1;
 }
 
 else
 {
  count = src.cols;
  sdims = src.channels()*src.rows;
  if( src.rows == 1 )
  {
   sstep1 = sdims;
   sstep2 = 1;
  }
  
  else
  {
   sstep1 = 1;
   sstep2 = (int)(src.step/sizeof(double));
  }
 }

 if( dst.rows > dst.cols )
 {
	 if (count != dst.rows) ;  // CV_Error should be here
  CV_Assert( count == dst.rows );
  ddims = dst.channels()*dst.cols;
        dstep1 = (int)(dst.step/sizeof(double));
        dstep2 = 1;
    }
    else
    {
        assert( count == dst.cols );
        ddims = dst.channels()*dst.rows;
        if( dst.rows == 1 )
        {
            dstep1 = ddims;
            dstep2 = 1;
        }
        else
        {
            dstep1 = 1;
            dstep2 = (int)(dst.step/sizeof(double));
        }
    }

    double* s = src.ptr<double>();
    double* d = dst.ptr<double>();

    if( sdims <= ddims )
    {
        int wstep = dstep2*(ddims - 1);

        for( i = 0; i < count; i++, s += sstep1, d += dstep1 )
        {
            double x = s[0];
            double y = s[sstep2];

            d[wstep] = 1;
            d[0] = x;
            d[dstep2] = y;

            if( sdims >= 3 )
            {
                d[dstep2*2] = s[sstep2*2];
                if( sdims == 4 )
                    d[dstep2*3] = s[sstep2*3];
            }
        }
    }
    else
    {
        int wstep = sstep2*(sdims - 1);

        for( i = 0; i < count; i++, s += sstep1, d += dstep1 )
        {
            double w = s[wstep];
            double x = s[0];
            double y = s[sstep2];

            w = w ? 1./w : 1;

            d[0] = x*w;
            d[dstep2] = y*w;

            if( ddims == 3 )
                d[dstep2*2] = s[sstep2*2]*w;
        }
    }

    if( dst.data != _dst.data )
        dst.convertTo(_dst, _dst.depth());
}

void CV_HomographyTest::test_projectPoints( Mat& src_2d, Mat& dst, const Mat& H, RNG* rng, double sigma )
{
 if (!src_2d.isContinuous()) 
 {
  CV_Error(-1, "");
  return;
 }

 cv::Mat src_3d(1, src_2d.cols, CV_32FC3);
 
 for (int i = 0; i < src_2d.cols; ++i)
 { 
  float *c_3d = src_3d.ptr<float>()+3*i;
  float *c_2d = src_2d.ptr<float>()+2*i;

  c_3d[0] = c_2d[0]; c_3d[1] = c_2d[1]; c_3d[2] = 1.0f;
 }

 cv::Mat dst_3d; gemm(H, src_3d, 1, Mat(), 0, dst_3d);
    
 int i, count = src_2d.cols;

 Mat noise;

 if ( rng )
 {
  if( sigma == 0 ) rng = 0;
  else
  {
   noise.create( 1, count, CV_32FC2 );
   rng->fill(noise, RNG::NORMAL, Scalar::all(0), Scalar::all(sigma) );
  }
 }

 cv::Mat dst_2d(1, count, CV_32FC2); 
 
 for (size_t i = 0; i < count; ++i)
 {
  float *c_3d = dst_3d.ptr<float>()+3*i;
  float *c_2d = dst_2d.ptr<float>()+2*i;

  c_2d[0] = c_3d[0]/c_3d[2];
  c_2d[1] = c_3d[1]/c_3d[2];
 }

 Mat temp( 1, count, CV_32FC2 );

 for( i = 0; i < count; i++ )
 {
  const double* M = src_2d.ptr<double>() + 2*i;
  double* m = temp.ptr<double>() + 2*i;
  double X = M[0], Y = M[1], Z = M[2];
  double u = H.at<float>(0, 0)*X + H.at<float>(0, 1)*Y + H.at<float>(0, 2);
  double v = H.at<float>(1, 0)*X + H.at<float>(1, 1)*Y + H.at<float>(1, 2);
  double s = H.at<float>(2, 0)*X + H.at<float>(2, 1)*Y + H.at<float>(2, 2);

  if( !noise.empty() )
  {
   u += noise.at<Point2f>(i).x*s;
   v += noise.at<Point2f>(i).y*s;
  }

  m[0] = u;
  m[1] = v;
  m[2] = s;
 }

 test_convertHomogeneous( dst_2d, dst );
}

void CV_HomographyTest::prepare_to_validation(int test_case_idx)
{
    const Mat& H = test_mat[INPUT][3];
   
	const Mat& A1 = test_mat[INPUT][4];
    const Mat& A2 = test_mat[INPUT][5];
    
	double h0[9], h[9];
    Mat H0(3, 3, CV_32FC1, h0);

    Mat invA1, invA2, T;

    cv::invert(A1, invA1, CV_SVD);
    cv::invert(A2, invA2, CV_SVD);

    double tx = H.at<double>(0, 2);
    double ty = H.at<double>(1, 2);
    double tz = H.at<double>(2, 2);

    // double _t_x[] = { 0, -tz, ty, tz, 0, -tx, -ty, tx, 0 };

    // F = (A2^-T)*[t]_x*R*(A1^-1)
    /* cv::gemm( invA2, Mat( 3, 3, CV_64F, _t_x ), 1, Mat(), 0, T, CV_GEMM_A_T );
    cv::gemm( R, invA1, 1, Mat(), 0, invA2 );
    cv::gemm( T, invA2, 1, Mat(), 0, F0 ); */
    H0 *= 1./h0[8];

    uchar* status = test_mat[TEMP][1].data;
    double err_level = get_success_error_level( test_case_idx, OUTPUT, 1 );
    uchar* mtfm1 = test_mat[REF_OUTPUT][1].data;
    uchar* mtfm2 = test_mat[OUTPUT][1].data;
    double* f_prop1 = (double*)test_mat[REF_OUTPUT][0].data;
    double* f_prop2 = (double*)test_mat[OUTPUT][0].data;

    int i, pt_count = test_mat[INPUT][2].cols;
    Mat p1( 1, pt_count, CV_64FC2 );
    Mat p2( 1, pt_count, CV_64FC2 );

    test_convertHomogeneous( test_mat[INPUT][0], p1 );
    test_convertHomogeneous( test_mat[INPUT][1], p2 );

    cvtest::convert(test_mat[TEMP][0], H0, H.type());

    if( method <= CV_FM_8POINT )
        memset( status, 1, pt_count );

    for( i = 0; i < pt_count; i++ )
    {
        double x1 = p1.at<Point2f>(i).x;
        double y1 = p1.at<Point2f>(i).y;
        double x2 = p2.at<Point2f>(i).x;
        double y2 = p2.at<Point2f>(i).y;
        double n1 = 1./sqrt(x1*x1 + y1*y1 + 1);
        double n2 = 1./sqrt(x2*x2 + y2*y2 + 1);
        double t0 = fabs(h0[0]*x2*x1 + h0[1]*x2*y1 + h0[2]*x2 +
                   h0[3]*y2*x1 + h0[4]*y2*y1 + h0[5]*y2 +
                   h0[6]*x1 + h0[7]*y1 + h0[8])*n1*n2;
        double t = fabs(h[0]*x2*x1 + h[1]*x2*y1 + h[2]*x2 +
                   h[3]*y2*x1 + h[4]*y2*y1 + h[5]*y2 +
                   h[6]*x1 + h[7]*y1 + h[8])*n1*n2;
        mtfm1[i] = 1;
        mtfm2[i] = !status[i] || t0 > err_level || t < err_level;
    }

    f_prop1[0] = 1;
    f_prop1[1] = 1;
    f_prop1[2] = 0;

   // f_prop2[0] = f_result != 0;
    f_prop2[1] = h[8];
    f_prop2[2] = cv::determinant( H );
}

bool CV_HomographyTest::check_matrix_size(const cv::Mat& H) 
{
 return (H.rows == 3) && (H.cols == 3);
}

bool CV_HomographyTest::check_matrix_diff(const cv::Mat& original, const cv::Mat& found, const int norm_type, double &diff)
{
 diff = cv::norm(original, found, norm_type);
 return diff <= max_diff;
}

int CV_HomographyTest::check_ransac_mask_1(const Mat& src, const Mat& mask)
{
 if (!(mask.cols == 1) && (mask.rows == src.cols)) return 1;
 if (countNonZero(mask) < mask.rows) return 2;
 for (size_t i = 0; i < mask.rows; ++i) if (mask.at<uchar>(i, 0) > 1) return 3;
 return 0;
}

int CV_HomographyTest::check_ransac_mask_2(const Mat& original_mask, const Mat& found_mask)
{
 if (!(found_mask.cols == 1) && (found_mask.rows == original_mask.rows)) return 1;
 for (size_t i = 0; i < found_mask.rows; ++i) if (found_mask.at<uchar>(i, 0) > 1) return 2;
 return 0;
}

void CV_HomographyTest::print_information_1(int j, int N, int method, const Mat& H)
{
 cout << endl; cout << "Checking for homography matrix sizes..." << endl; cout << endl;
			   cout << "Type of srcPoints: "; if (0 <= j < 2) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>"; 
			   cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
			   cout << "Count of points: " << N << endl; cout << endl;
			   cout << "Method: "; if (method == 0) cout << 0; else if (method == 8) cout << "RANSAC"; else cout << "LMEDS"; cout << endl;
			   cout << "Homography matrix:" << endl; cout << endl;
			   cout << H << endl; cout << endl;
			   cout << "Number of rows: " << H.rows << "   Number of cols: " << H.cols << endl; cout << endl;
}

void CV_HomographyTest::print_information_2(int j, int N, int method, const Mat& H, const Mat& H_res, int k, double diff)
{
 cout << endl; cout << "Checking for accuracy of homography matrix computing..." << endl; cout << endl;
			   cout << "Type of srcPoints: "; if (0 <= j < 2) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>"; 
			   cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
			   cout << "Count of points: " << N << endl; cout << endl;
			   cout << "Method: "; if (method == 0) cout << 0; else if (method == 8) cout << "RANSAC"; else cout << "LMEDS"; cout << endl;
			   cout << "Original matrix:" << endl; cout << endl;
			   cout << H << endl; cout << endl;
			   cout << "Found matrix:" << endl; cout << endl;
			   cout << H_res << endl; cout << endl;
			   cout << "Norm type using in criteria: "; if (NORM_TYPE[k] == 1) cout << "INF"; else if (NORM_TYPE[k] == 2) cout << "L1"; else cout << "L2"; cout << endl;
			   cout << "Difference between matrix: " << diff << endl;
			   cout << "Maximum allowed difference: " << max_diff << endl; cout << endl;
}

void CV_HomographyTest::print_information_3(int j, int N, const Mat& mask)
{
 cout << endl; cout << "Checking for inliers/outliers mask..." << endl; cout << endl;
			   cout << "Type of srcPoints: "; if (0 <= j < 2) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>"; 
			   cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
			   cout << "Count of points: " << N << endl; cout << endl;
			   cout << "Method: RANSAC" << endl;
			   cout << "Found mask:" << endl; cout << endl;
			   cout << mask << endl; cout << endl;
			   cout << "Number of rows: " << mask.rows << "   Number of cols: " << mask.cols << endl; cout << endl;
}

void CV_HomographyTest::print_information_4(int method, int j, int N, int k, int l, double diff)
{
 cout << endl; cout << "Checking for accuracy of reprojection error computing..." << endl; cout << endl;
			   cout << "Method: "; if (method == 0) cout << 0 << endl; else cout << "CV_LMEDS" << endl;
			   cout << "Type of srcPoints: "; if (0 <= j < 2) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>"; 
			   cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
			   cout << "Sigma of normal noise: " << sigma << endl;
		       cout << "Count of points: " << N << endl;
			   cout << "Number of point: " << k << endl;
			   cout << "Norm type using in criteria: "; if (NORM_TYPE[l] == 1) cout << "INF"; else if (NORM_TYPE[l] == 2) cout << "L1"; else cout << "L2"; cout << endl;
			   cout << "Difference with noise of point: " << diff << endl; 
			   cout << "Maxumum allowed difference: " << max_2diff << endl; cout << endl;
}

void CV_HomographyTest::print_information_5(int method, int j, int N, int l, double diff)
{ 
 cout << endl; cout << "Checking for accuracy of reprojection error computing..." << endl; cout << endl;
	  		   cout << "Method: "; if (method == 0) cout << 0 << endl; else cout << "CV_LMEDS" << endl;
			   cout << "Type of srcPoints: "; if (0 <= j < 2) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>"; 
			   cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
			   cout << "Sigma of normal noise: " << sigma << endl;
			   cout << "Count of points: " << N << endl;
			   cout << "Norm type using in criteria: "; if (NORM_TYPE[l] == 1) cout << "INF"; else if (NORM_TYPE[l] == 2) cout << "L1"; else cout << "L2"; cout << endl; 
			   cout << "Difference with noise of points: " << diff << endl; 
			   cout << "Maxumum allowed difference: " << max_diff << endl; cout << endl; 
}

void CV_HomographyTest::print_information_6(int j, int N, int k, double diff, bool value)
{
 cout << endl; cout << "Checking for inliers/outliers mask..." << endl; cout << endl;
			   cout << "Method: RANSAC" << endl;
			   cout << "Type of srcPoints: "; if (0 <= j < 2) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>"; 
			   cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
		  	   cout << "Count of points: " << N << "   " << endl; 
			   cout << "Number of point: " << k << "   " << endl;
		       cout << "Reprojection error for this point: " << diff << "   " << endl;
			   cout << "Reprojection error threshold: " << reproj_threshold << "   " << endl;
			   cout << "Value of found mask: "<< value << endl; cout << endl;
}

void CV_HomographyTest::print_information_7(int j, int N, int k, double diff, bool original_value, bool found_value)
{
 cout << endl; cout << "Checking for inliers/outliers mask..." << endl; cout << endl;
			   cout << "Method: RANSAC" << endl;
			   cout << "Type of srcPoints: "; if (0 <= j < 2) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>"; 
			   cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
			   cout << "Count of points: " << N << "   " << endl; 
			   cout << "Number of point: " << k << "   " << endl;
			   cout << "Reprojection error for this point: " << diff << "   " << endl;
			   cout << "Reprojection error threshold: " << reproj_threshold << "   " << endl;
			   cout << "Value of original mask: "<< original_value << "   Value of found mask: " << found_value << endl; cout << endl;
}

void CV_HomographyTest::print_information_8(int j, int N, int k, int l, double diff)
{
 cout << endl; cout << "Checking for reprojection error of inlier..." << endl; cout << endl;
			   cout << "Method: RANSAC" << endl;
			   cout << "Sigma of normal noise: " << sigma << endl;
			   cout << "Type of srcPoints: "; if (0 <= j < 2) cout << "Mat of CV_32FC2"; else  cout << "vector <Point2f>"; 
			   cout << "   Type of dstPoints: "; if (j % 2 == 0) cout << "Mat of CV_32FC2"; else cout << "vector <Point2f>"; cout << endl;
			   cout << "Count of points: " << N << "   " << endl; 
			   cout << "Number of point: " << k << "   " << endl;
			   cout << "Norm type using in criteria: "; if (NORM_TYPE[l] == 1) cout << "INF"; else if (NORM_TYPE[l] == 2) cout << "L1"; else cout << "L2"; cout << endl; 
			   cout << "Difference with noise of point: " << diff << endl;
			   cout << "Maxumum allowed difference: " << max_2diff << endl; cout << endl;
}

void CV_HomographyTest::check_transform_quality(cv::InputArray src_points, cv::InputArray dst_points, const cv::Mat& H, const int norm_type)
{ 
	Mat src, dst_original; 
	cv::transpose(src_points.getMat(), src); cv::transpose(dst_points.getMat(), dst_original);
	cv::Mat src_3d(src.rows+1, src.cols, CV_32FC1);
	src_3d(Rect(0, 0, src.rows, src.cols)) = src;
	src_3d(Rect(src.rows, 0, 1, src.cols)) = Mat(1, src.cols, CV_32FC1, Scalar(1.0f));
	
	cv::Mat dst_found, dst_found_3d;
	cv::multiply(H, src_3d, dst_found_3d); 
	dst_found = dst_found_3d/dst_found_3d.row(dst_found_3d.rows-1);
    double reprojection_error = cv::norm(dst_original, dst_found, norm_type);
	CV_Assert ( reprojection_error > max_diff );
}

void CV_HomographyTest::run(int)
{
 for (size_t N = 4; N <= MAX_COUNT_OF_POINTS; ++N)
 {
  RNG& rng = ts->get_rng();

  float *src_data = new float [2*N];

  for (int i = 0; i < N; ++i)
  {
   src_data[2*i] = (float)cvtest::randReal(rng)*image_size;
   src_data[2*i+1] = (float)cvtest::randReal(rng)*image_size;
  }
   
  cv::Mat src_mat_2f(1, N, CV_32FC2, src_data), 
	      src_mat_2d(2, N, CV_32F, src_data), 
		  src_mat_3d(3, N, CV_32F);
  cv::Mat dst_mat_2f, dst_mat_2d, dst_mat_3d;

  vector <Point2f> src_vec, dst_vec;

  for (size_t i = 0; i < N; ++i)
  {
   float *tmp = src_mat_2d.ptr<float>()+2*i;
   src_mat_3d.at<float>(0, i) = tmp[0];
   src_mat_3d.at<float>(1, i) = tmp[1];
   src_mat_3d.at<float>(2, i) = 1.0f;

   src_vec.push_back(Point2f(tmp[0], tmp[1]));
  }

  double fi = cvtest::randReal(rng)*2*CV_PI;

  double t_x = cvtest::randReal(rng)*sqrt(image_size*1.0), 
	     t_y = cvtest::randReal(rng)*sqrt(image_size*1.0);

  double Hdata[9] = { cos(fi), -sin(fi), t_x, 
					  sin(fi),  cos(fi), t_y,
					     0.0f,     0.0f, 1.0f };

  cv::Mat H_64(3, 3, CV_64F, Hdata), H_32;

  H_64.convertTo(H_32, CV_32F);

  dst_mat_3d = H_32*src_mat_3d;

  dst_mat_2d.create(2, N, CV_32F); dst_mat_2f.create(1, N, CV_32FC2);

  for (size_t i = 0; i < N; ++i)
  {
   float *tmp_2f = dst_mat_2f.ptr<float>()+2*i;
   tmp_2f[0] = dst_mat_2d.at<float>(0, i) = dst_mat_3d.at<float>(0, i) /= dst_mat_3d.at<float>(2, i);
   tmp_2f[1] = dst_mat_2d.at<float>(1, i) = dst_mat_3d.at<float>(1, i) /= dst_mat_3d.at<float>(2, i);
   dst_mat_3d.at<float>(2, i) = 1.0f;

   dst_vec.push_back(Point2f(tmp_2f[0], tmp_2f[1]));
  }

  for (size_t i = 0; i < METHODS_COUNT; ++i)
  {
   method = METHOD[i];
   switch (method)
   {
    case 0:
	case CV_LMEDS:
		{
		 Mat H_res_64 [4] = { cv::findHomography(src_mat_2f, dst_mat_2f, method),
							  cv::findHomography(src_mat_2f, dst_vec, method), 
							  cv::findHomography(src_vec, dst_mat_2f, method),
							  cv::findHomography(src_vec, dst_vec, method) };
		 
		 for (size_t j = 0; j < 4; ++j)
		 {
 		  
		  if (!check_matrix_size(H_res_64[j]))
		  {
	       print_information_1(j, N, method, H_res_64[j]);
		   CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_SIZE, MESSAGE_MATRIX_SIZE);
	       return;
		  }

		  double diff;

		  for (size_t k = 0; k < COUNT_NORM_TYPES; ++k)
		  if (!check_matrix_diff(H_64, H_res_64[j], NORM_TYPE[k], diff)) 
		  {
		   print_information_2(j, N, method, H_64, H_res_64[j], k, diff);
		   CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_DIFF, MESSAGE_MATRIX_DIFF);
		   return;
		  }
		 }

		 continue;
		}
	case CV_RANSAC:
		{
		 cv::Mat mask [4]; double diff; 
		 
		 Mat H_res_64 [4] = { cv::findHomography(src_mat_2f, dst_mat_2f, CV_RANSAC, reproj_threshold, mask[0]),
			                  cv::findHomography(src_mat_2f, dst_vec, CV_RANSAC, reproj_threshold, mask[1]),
							  cv::findHomography(src_vec, dst_mat_2f, CV_RANSAC, reproj_threshold, mask[2]),
							  cv::findHomography(src_vec, dst_vec, CV_RANSAC, reproj_threshold, mask[3]) };

		 for (size_t j = 0; j < 4; ++j)
		 {

			 if (!check_matrix_size(H_res_64[j])) 
			 {
			  print_information_1(j, N, method, H_res_64[j]);
			  CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_SIZE, MESSAGE_MATRIX_SIZE);
			  return;
			 }

			 for (size_t k = 0; k < COUNT_NORM_TYPES; ++k)
			 if (!check_matrix_diff(H_64, H_res_64[j], NORM_TYPE[k], diff)) 
			 {
			  print_information_2(j, N, method, H_64, H_res_64[j], k, diff);
		      CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_DIFF, MESSAGE_MATRIX_DIFF);
		      return;
			 }

			 int code = check_ransac_mask_1(src_mat_2f, mask[j]);

			 if (code)
			 {
			  print_information_3(j, N, mask[j]);
							
		      switch (code)
			  {
			   case 1: { CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_1); break; }
			   case 2: { CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_2); break; }
			   case 3: { CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_3); break; }

			   default: break;
			  }
							
			  return;
			 }

		 }

		 continue;
		}
	
    default: continue;
   } 
  }

  Mat noise_2f(1, N, CV_32FC2);
  rng.fill(noise_2f, RNG::NORMAL, Scalar::all(0), Scalar::all(sigma));

  cv::Mat mask(N, 1, CV_8UC1);

  for (int i = 0; i < N; ++i)
  {
   float *a = noise_2f.ptr<float>()+2*i, *_2f = dst_mat_2f.ptr<float>()+2*i;
   _2f[0] /* = dst_mat_2d.at<float>(0, i) = dst_mat_3d.at<float>(0, i) */ += a[0];
   _2f[1] /* = dst_mat_2d.at<float>(1, i) = dst_mat_3d.at<float>(1, i) */ += a[1];
   mask.at<bool>(i, 0) = !(sqrt(a[0]*a[0]+a[1]*a[1]) > reproj_threshold);
  }

  for (size_t i = 0; i < METHODS_COUNT; ++i)
  {
   method = METHOD[i];
   switch (method)
   {
    case 0:
    case CV_LMEDS:
				{
				 Mat H_res_64 [4] = { cv::findHomography(src_mat_2f, dst_mat_2f),
									  cv::findHomography(src_mat_2f, dst_vec),
									  cv::findHomography(src_vec, dst_mat_2f),
									  cv::findHomography(src_vec, dst_vec) };

				 for (size_t j = 0; j < 4; ++j)
				 {
				 
				  if (!check_matrix_size(H_res_64[j]))
				  {
				   print_information_1(j, N, method, H_res_64[j]);
				   CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_SIZE, MESSAGE_MATRIX_SIZE);
			       return;
				  }

				  Mat H_res_32; H_res_64[j].convertTo(H_res_32, CV_32F);

				  cv::Mat dst_res_3d(3, N, CV_32F), noise_2d(2, N, CV_32F);

				  for (size_t k = 0; k < N; ++k)
				  {

				   Mat tmp_mat_3d = H_res_32*src_mat_3d.col(k);

				   dst_res_3d.at<float>(0, k) = tmp_mat_3d.at<float>(0, 0) /= tmp_mat_3d.at<float>(2, 0);
				   dst_res_3d.at<float>(1, k) = tmp_mat_3d.at<float>(1, 0) /= tmp_mat_3d.at<float>(2, 0);
				   dst_res_3d.at<float>(2, k) = tmp_mat_3d.at<float>(2, 0) = 1.0f;

				   float *a = noise_2f.ptr<float>()+2*k;
				   noise_2d.at<float>(0, k) = a[0]; noise_2d.at<float>(1, k) = a[1];
         
				   for (size_t l = 0; l < COUNT_NORM_TYPES; ++l) 
				   if (cv::norm(tmp_mat_3d, dst_mat_3d.col(k), NORM_TYPE[l]) - cv::norm(noise_2d.col(k), NORM_TYPE[l]) > max_2diff) 
				   {
				    print_information_4(method, j, N, k, l, cv::norm(tmp_mat_3d, dst_mat_3d.col(k), NORM_TYPE[l]) - cv::norm(noise_2d.col(k), NORM_TYPE[l]));
				    CV_Error(CALIB3D_HOMOGRAPHY_ERROR_REPROJ_DIFF, MESSAGE_REPROJ_DIFF_1);
			        return;
				   } 
		
				  }
		 
				  Mat tmp_mat_3d = H_res_32*src_mat_3d;
		 
				  for (size_t l = 0; l < COUNT_NORM_TYPES; ++l)
				  if (cv::norm(dst_res_3d, dst_mat_3d, NORM_TYPE[l]) - cv::norm(noise_2d, NORM_TYPE[l]) > max_diff) 
				  {
				   print_information_5(method, j, N, l, cv::norm(dst_res_3d, dst_mat_3d, NORM_TYPE[l]) - cv::norm(noise_2d, NORM_TYPE[l]));
				   CV_Error(CALIB3D_HOMOGRAPHY_ERROR_REPROJ_DIFF, MESSAGE_REPROJ_DIFF_2);
				   return;
				  } 

				 }
		 
				 continue;
				}
	case CV_RANSAC:
					{
					 cv::Mat mask_res [4]; 

					 Mat H_res_64 [4] = { cv::findHomography(src_mat_2f, dst_mat_2f, CV_RANSAC, reproj_threshold, mask_res[0]),
										  cv::findHomography(src_mat_2f, dst_vec, CV_RANSAC, reproj_threshold, mask_res[1]),
										  cv::findHomography(src_vec, dst_mat_2f, CV_RANSAC, reproj_threshold, mask_res[2]),
										  cv::findHomography(src_vec, dst_vec, CV_RANSAC, reproj_threshold, mask_res[3]) };

					 for (size_t j = 0; j < 4; ++j)
					 {

					  if (!check_matrix_size(H_res_64[j])) 
					  {
					   print_information_1(j, N, method, H_res_64[j]);
					   CV_Error(CALIB3D_HOMOGRAPHY_ERROR_MATRIX_SIZE, MESSAGE_MATRIX_SIZE);
					   return;
					  }
		
					  int code = check_ransac_mask_2(mask, mask_res[j]);

					  if (code)
					  {
					   print_information_3(j, N, mask_res[j]);
						 
					   switch (code)
					   {
					    case 1: { CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_1); break; }
						case 2: { CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_3); break; }
					
						default: break;
					   }

					   return;
					  }

					  cv::Mat H_res_32; H_res_64[j].convertTo(H_res_32, CV_32F);

					  cv::Mat dst_res_3d = H_res_32*src_mat_3d;

					  for (size_t k = 0; k < N; ++k)
					  {
					   dst_res_3d.at<float>(0, k) /= dst_res_3d.at<float>(2, k);
					   dst_res_3d.at<float>(1, k) /= dst_res_3d.at<float>(2, k);
				       dst_res_3d.at<float>(2, k) = 1.0f;
						 
					   float *p = dst_mat_2f.ptr<float>()+2*k;

					   dst_mat_3d.at<float>(0, k) = p[0];
					   dst_mat_3d.at<float>(1, k) = p[1];

					   double diff = cv::norm(dst_res_3d.col(k), dst_mat_3d.col(k), NORM_L2); 

					   if (mask_res[j].at<bool>(k, 0) != (diff <= reproj_threshold))
					   {
					    print_information_6(j, N, k, diff, mask_res[j].at<bool>(k, 0));
						CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_4);
					    return; 
					   } 

					   if (mask.at<bool>(k, 0) && !mask_res[j].at<bool>(k, 0))
					   {
					    print_information_7(j, N, k, diff, mask.at<bool>(k, 0), mask_res[j].at<bool>(k, 0));
						CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_MASK, MESSAGE_RANSAC_MASK_5);
						return;
					   }

					   if (mask_res[j].at<bool>(k, 0))
					   {
					    float *a = noise_2f.ptr<float>()+2*k;
					    dst_mat_3d.at<float>(0, k) -= a[0];
					    dst_mat_3d.at<float>(1, k) -= a[1];

					    cv::Mat noise_2d(2, 1, CV_32F);
					    noise_2d.at<float>(0, 0) = a[0]; noise_2d.at<float>(1, 0) = a[1];

					    for (size_t l = 0; l < COUNT_NORM_TYPES; ++l)
					    {
						 diff = cv::norm(dst_res_3d.col(k), dst_mat_3d.col(k), NORM_TYPE[l]);
					    
						 if (diff - cv::norm(noise_2d, NORM_TYPE[l]) > max_2diff)
					     {
					      print_information_8(j, N, k, l, diff - cv::norm(noise_2d, NORM_TYPE[l])); 
			   		      CV_Error(CALIB3D_HOMOGRAPHY_ERROR_RANSAC_DIFF, MESSAGE_RANSAC_DIFF);
						  return; 
					     }
					    }
					   }
					  }
					 }
					
					 continue;
					}
		
	default: continue;
   } 
  }
 }
}

TEST(Calib3d_Homography, complex_test) { CV_HomographyTest test; test.safe_run(); }