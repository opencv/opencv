#ifdef WIN32
#include "cv.h"
#include "highgui.h"
#else
#include "opencv/cv.h"
#include "opencv/highgui.h"
#endif
#include <stdio.h>
#include <string.h>
#include <time.h>

// example command line (for copy-n-paste):
// calibration -w 6 -h 8 -s 2 -n 10 -o camera.yml -op -oe [<list_of_views.txt>]

/* The list of views may look as following (discard the starting and ending ------ separators):
-------------------
view000.png
view001.png
#view002.png
view003.png
view010.png
one_extra_view.jpg
-------------------
that is, the file will contain 6 lines, view002.png will not be used for calibration,
other ones will be (those, in which the chessboard pattern will be found)
*/

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

double compute_reprojection_error( const CvMat* object_points,
        const CvMat* rot_vects, const CvMat* trans_vects,
        const CvMat* camera_matrix, const CvMat* dist_coeffs,
        const CvMat* image_points, const CvMat* point_counts,
        CvMat* per_view_errors );


int run_calibration( CvSeq* image_points_seq, CvSize img_size, CvSize board_size,
                     float square_size, float aspect_ratio, int flags,
                     CvMat* camera_matrix, CvMat* dist_coeffs, CvMat** extr_params,
                     CvMat** reproj_errs, double* avg_reproj_err );


void save_camera_params( const char* out_filename, int image_count, CvSize img_size,
                         CvSize board_size, float square_size,
                         float aspect_ratio, int flags,
                         const CvMat* camera_matrix, CvMat* dist_coeffs,
                         const CvMat* extr_params, const CvSeq* image_points_seq,
                         const CvMat* reproj_errs, double avg_reproj_err );

int calibrate( int argc, char** argv );
