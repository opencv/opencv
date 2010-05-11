#define CV_AUTOSTEP  0x7fffffff

#define cvGetSubArr cvGetSubRect

#define CV_MAX_ARR 10

#define CV_NO_DEPTH_CHECK     1

#define CV_NO_CN_CHECK        2

#define CV_NO_SIZE_CHECK      4

#define cvZero  cvSetZero

#define cvCvtScale cvConvertScale

#define cvScale  cvConvertScale

#define cvCvtScaleAbs  cvConvertScaleAbs

#define CV_CMP_EQ   0

#define CV_CMP_GT   1

#define CV_CMP_GE   2

#define CV_CMP_LT   3

#define CV_CMP_LE   4

#define CV_CMP_NE   5

#define  CV_CHECK_RANGE    1

#define  CV_CHECK_QUIET    2

#define cvCheckArray cvCheckArr

#define CV_RAND_UNI      0

#define CV_RAND_NORMAL   1

#define CV_SORT_EVERY_ROW 0

#define CV_SORT_EVERY_COLUMN 1

#define CV_SORT_ASCENDING 0

#define CV_SORT_DESCENDING 16

#define CV_GEMM_A_T 1

#define CV_GEMM_B_T 2

#define CV_GEMM_C_T 4

#define cvMatMulAddEx cvGEMM

#define cvMatMulAddS cvTransform

#define cvT cvTranspose

#define cvMirror cvFlip

#define CV_SVD_MODIFY_A   1

#define CV_SVD_U_T        2

#define CV_SVD_V_T        4

#define CV_LU  0

#define CV_SVD 1

#define CV_SVD_SYM 2

#define CV_CHOLESKY 3

#define CV_QR  4

#define CV_NORMAL 16

#define cvInv cvInvert

#define CV_COVAR_SCRAMBLED 0

#define CV_COVAR_NORMAL    1

#define CV_COVAR_USE_AVG   2

#define CV_COVAR_SCALE     4

#define CV_COVAR_ROWS      8

#define CV_COVAR_COLS     16

#define CV_PCA_DATA_AS_ROW 0

#define CV_PCA_DATA_AS_COL 1

#define CV_PCA_USE_AVG 2

#define cvMahalonobis  cvMahalanobis

#define CV_C            1

#define CV_L1           2

#define CV_L2           4

#define CV_NORM_MASK    7

#define CV_RELATIVE     8

#define CV_DIFF         16

#define CV_MINMAX       32

#define CV_DIFF_C       (CV_DIFF | CV_C)

#define CV_DIFF_L1      (CV_DIFF | CV_L1)

#define CV_DIFF_L2      (CV_DIFF | CV_L2)

#define CV_RELATIVE_C   (CV_RELATIVE | CV_C)

#define CV_RELATIVE_L1  (CV_RELATIVE | CV_L1)

#define CV_RELATIVE_L2  (CV_RELATIVE | CV_L2)

#define CV_REDUCE_SUM 0

#define CV_REDUCE_AVG 1

#define CV_REDUCE_MAX 2

#define CV_REDUCE_MIN 3

#define CV_DXT_FORWARD  0

#define CV_DXT_INVERSE  1

#define CV_DXT_SCALE    2 

#define CV_DXT_INV_SCALE (CV_DXT_INVERSE + CV_DXT_SCALE)

#define CV_DXT_INVERSE_SCALE CV_DXT_INV_SCALE

#define CV_DXT_ROWS     4 

#define CV_DXT_MUL_CONJ 8 

#define cvFFT cvDFT

#define CV_FRONT 1

#define CV_BACK 0

#define cvGraphFindEdge cvFindGraphEdge

#define cvGraphFindEdgeByPtr cvFindGraphEdgeByPtr

#define  CV_GRAPH_VERTEX        1

#define  CV_GRAPH_TREE_EDGE     2

#define  CV_GRAPH_BACK_EDGE     4

#define  CV_GRAPH_FORWARD_EDGE  8

#define  CV_GRAPH_CROSS_EDGE    16

#define  CV_GRAPH_ANY_EDGE      30

#define  CV_GRAPH_NEW_TREE      32

#define  CV_GRAPH_BACKTRACKING  64

#define  CV_GRAPH_OVER          -1

#define  CV_GRAPH_ALL_ITEMS    -1

#define  CV_GRAPH_ITEM_VISITED_FLAG  (1 << 30)

#define  CV_GRAPH_SEARCH_TREE_NODE_FLAG   (1 << 29)

#define  CV_GRAPH_FORWARD_EDGE_FLAG       (1 << 28)

#define CV_FILLED -1

#define CV_AA 16

#define cvDrawRect cvRectangle

#define cvDrawLine cvLine

#define cvDrawCircle cvCircle

#define cvDrawEllipse cvEllipse

#define cvDrawPolyLine cvPolyLine

#define CV_FONT_HERSHEY_SIMPLEX         0

#define CV_FONT_HERSHEY_PLAIN           1

#define CV_FONT_HERSHEY_DUPLEX          2

#define CV_FONT_HERSHEY_COMPLEX         3

#define CV_FONT_HERSHEY_TRIPLEX         4

#define CV_FONT_HERSHEY_COMPLEX_SMALL   5

#define CV_FONT_HERSHEY_SCRIPT_SIMPLEX  6

#define CV_FONT_HERSHEY_SCRIPT_COMPLEX  7

#define CV_FONT_ITALIC                 16

#define CV_FONT_VECTOR0    CV_FONT_HERSHEY_SIMPLEX

#define CV_KMEANS_USE_INITIAL_LABELS    1

#define CV_ErrModeLeaf     0   

#define CV_ErrModeParent   1   

#define CV_ErrModeSilent   2   

#define CV_MAJOR_VERSION    2

#define CV_MINOR_VERSION    0

#define CV_SUBMINOR_VERSION 0

#define CV_VERSION          CVAUX_STR(CV_MAJOR_VERSION) "." CVAUX_STR(CV_MINOR_VERSION) "." CVAUX_STR(CV_SUBMINOR_VERSION)

#define CV_PI   3.1415926535897932384626433832795

#define CV_LOG2 0.69314718055994530941723212145818

#define IPL_DEPTH_SIGN 0x80000000

#define IPL_DEPTH_1U     1

#define IPL_DEPTH_8U     8

#define IPL_DEPTH_16U   16

#define IPL_DEPTH_32F   32

#define IPL_DEPTH_8S  (IPL_DEPTH_SIGN| 8)

#define IPL_DEPTH_16S (IPL_DEPTH_SIGN|16)

#define IPL_DEPTH_32S (IPL_DEPTH_SIGN|32)

#define IPL_DATA_ORDER_PIXEL  0

#define IPL_DATA_ORDER_PLANE  1

#define IPL_ORIGIN_TL 0

#define IPL_ORIGIN_BL 1

#define IPL_ALIGN_4BYTES   4

#define IPL_ALIGN_8BYTES   8

#define IPL_ALIGN_16BYTES 16

#define IPL_ALIGN_32BYTES 32

#define IPL_ALIGN_DWORD   IPL_ALIGN_4BYTES

#define IPL_ALIGN_QWORD   IPL_ALIGN_8BYTES

#define IPL_BORDER_CONSTANT   0

#define IPL_BORDER_REPLICATE  1

#define IPL_BORDER_REFLECT    2

#define IPL_BORDER_WRAP       3

#define IPL_IMAGE_HEADER 1

#define IPL_IMAGE_DATA   2

#define IPL_IMAGE_ROI    4

#define IPL_BORDER_REFLECT_101    4

#define IPL_IMAGE_MAGIC_VAL  ((int)sizeof(IplImage))

#define CV_TYPE_NAME_IMAGE "opencv-image"

#define IPL_DEPTH_64F  64

#define CV_CN_MAX     64

#define CV_CN_SHIFT   3

#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)

#define CV_8U   0

#define CV_8S   1

#define CV_16U  2

#define CV_16S  3

#define CV_32S  4

#define CV_32F  5

#define CV_64F  6

#define CV_USRTYPE1 7

#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)

#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))

#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U,1)

#define CV_8UC2 CV_MAKETYPE(CV_8U,2)

#define CV_8UC3 CV_MAKETYPE(CV_8U,3)

#define CV_8UC4 CV_MAKETYPE(CV_8U,4)

#define CV_8SC1 CV_MAKETYPE(CV_8S,1)

#define CV_8SC2 CV_MAKETYPE(CV_8S,2)

#define CV_8SC3 CV_MAKETYPE(CV_8S,3)

#define CV_8SC4 CV_MAKETYPE(CV_8S,4)

#define CV_16UC1 CV_MAKETYPE(CV_16U,1)

#define CV_16UC2 CV_MAKETYPE(CV_16U,2)

#define CV_16UC3 CV_MAKETYPE(CV_16U,3)

#define CV_16UC4 CV_MAKETYPE(CV_16U,4)

#define CV_16SC1 CV_MAKETYPE(CV_16S,1)

#define CV_16SC2 CV_MAKETYPE(CV_16S,2)

#define CV_16SC3 CV_MAKETYPE(CV_16S,3)

#define CV_16SC4 CV_MAKETYPE(CV_16S,4)

#define CV_32SC1 CV_MAKETYPE(CV_32S,1)

#define CV_32SC2 CV_MAKETYPE(CV_32S,2)

#define CV_32SC3 CV_MAKETYPE(CV_32S,3)

#define CV_32SC4 CV_MAKETYPE(CV_32S,4)

#define CV_32FC1 CV_MAKETYPE(CV_32F,1)

#define CV_32FC2 CV_MAKETYPE(CV_32F,2)

#define CV_32FC3 CV_MAKETYPE(CV_32F,3)

#define CV_32FC4 CV_MAKETYPE(CV_32F,4)

#define CV_64FC1 CV_MAKETYPE(CV_64F,1)

#define CV_64FC2 CV_MAKETYPE(CV_64F,2)

#define CV_64FC3 CV_MAKETYPE(CV_64F,3)

#define CV_64FC4 CV_MAKETYPE(CV_64F,4)

#define CV_AUTO_STEP  0x7fffffff

#define CV_WHOLE_ARR  cvSlice( 0, 0x3fffffff )

#define CV_MAT_CN_MASK          ((CV_CN_MAX - 1) << CV_CN_SHIFT)

#define CV_MAT_TYPE_MASK        (CV_DEPTH_MAX*CV_CN_MAX - 1)

#define CV_MAT_TYPE(flags)      ((flags) & CV_MAT_TYPE_MASK)

#define CV_MAT_CONT_FLAG_SHIFT  14

#define CV_MAT_CONT_FLAG        (1 << CV_MAT_CONT_FLAG_SHIFT)

#define CV_IS_CONT_MAT          CV_IS_MAT_CONT

#define CV_MAT_TEMP_FLAG_SHIFT  15

#define CV_MAT_TEMP_FLAG        (1 << CV_MAT_TEMP_FLAG_SHIFT)

#define CV_MAGIC_MASK       0xFFFF0000

#define CV_MAT_MAGIC_VAL    0x42420000

#define CV_TYPE_NAME_MAT    "opencv-matrix"

#define CV_MATND_MAGIC_VAL    0x42430000

#define CV_TYPE_NAME_MATND    "opencv-nd-matrix"

#define CV_MAX_DIM            32

#define CV_MAX_DIM_HEAP       (1 << 16)

#define CV_SPARSE_MAT_MAGIC_VAL    0x42440000

#define CV_TYPE_NAME_SPARSE_MAT    "opencv-sparse-matrix"

#define CV_HIST_MAGIC_VAL     0x42450000

#define CV_HIST_UNIFORM_FLAG  (1 << 10)

#define CV_HIST_RANGES_FLAG   (1 << 11)

#define CV_HIST_ARRAY         0

#define CV_HIST_SPARSE        1

#define CV_HIST_TREE          CV_HIST_SPARSE

#define CV_HIST_UNIFORM       1

#define CV_TERMCRIT_ITER    1

#define CV_TERMCRIT_NUMBER  CV_TERMCRIT_ITER

#define CV_TERMCRIT_EPS     2

#define CV_WHOLE_SEQ_END_INDEX 0x3fffffff

#define CV_WHOLE_SEQ  cvSlice(0, CV_WHOLE_SEQ_END_INDEX)

#define CV_STORAGE_MAGIC_VAL    0x42890000

#define CV_TYPE_NAME_SEQ             "opencv-sequence"

#define CV_TYPE_NAME_SEQ_TREE        "opencv-sequence-tree"

#define CV_SET_ELEM_IDX_MASK   ((1 << 26) - 1)

#define CV_SET_ELEM_FREE_FLAG  (1 << (sizeof(int)*8-1))

#define CV_TYPE_NAME_GRAPH "opencv-graph"

#define CV_SEQ_MAGIC_VAL             0x42990000

#define CV_SET_MAGIC_VAL             0x42980000

#define CV_SEQ_ELTYPE_BITS           9

#define CV_SEQ_ELTYPE_MASK           ((1 << CV_SEQ_ELTYPE_BITS) - 1)

#define CV_SEQ_ELTYPE_POINT          CV_32SC2  

#define CV_SEQ_ELTYPE_CODE           CV_8UC1   

#define CV_SEQ_ELTYPE_GENERIC        0

#define CV_SEQ_ELTYPE_PTR            CV_USRTYPE1

#define CV_SEQ_ELTYPE_PPOINT         CV_SEQ_ELTYPE_PTR  

#define CV_SEQ_ELTYPE_INDEX          CV_32SC1  

#define CV_SEQ_ELTYPE_GRAPH_EDGE     0  

#define CV_SEQ_ELTYPE_GRAPH_VERTEX   0  

#define CV_SEQ_ELTYPE_TRIAN_ATR      0  

#define CV_SEQ_ELTYPE_CONNECTED_COMP 0  

#define CV_SEQ_ELTYPE_POINT3D        CV_32FC3  

#define CV_SEQ_KIND_BITS        3

#define CV_SEQ_KIND_MASK        (((1 << CV_SEQ_KIND_BITS) - 1)<<CV_SEQ_ELTYPE_BITS)

#define CV_SEQ_KIND_GENERIC     (0 << CV_SEQ_ELTYPE_BITS)

#define CV_SEQ_KIND_CURVE       (1 << CV_SEQ_ELTYPE_BITS)

#define CV_SEQ_KIND_BIN_TREE    (2 << CV_SEQ_ELTYPE_BITS)

#define CV_SEQ_KIND_GRAPH       (3 << CV_SEQ_ELTYPE_BITS)

#define CV_SEQ_KIND_SUBDIV2D    (4 << CV_SEQ_ELTYPE_BITS)

#define CV_SEQ_FLAG_SHIFT       (CV_SEQ_KIND_BITS + CV_SEQ_ELTYPE_BITS)

#define CV_SEQ_FLAG_CLOSED     (1 << CV_SEQ_FLAG_SHIFT)

#define CV_SEQ_FLAG_SIMPLE     (2 << CV_SEQ_FLAG_SHIFT)

#define CV_SEQ_FLAG_CONVEX     (4 << CV_SEQ_FLAG_SHIFT)

#define CV_SEQ_FLAG_HOLE       (8 << CV_SEQ_FLAG_SHIFT)

#define CV_GRAPH_FLAG_ORIENTED (1 << CV_SEQ_FLAG_SHIFT)

#define CV_GRAPH               CV_SEQ_KIND_GRAPH

#define CV_ORIENTED_GRAPH      (CV_SEQ_KIND_GRAPH|CV_GRAPH_FLAG_ORIENTED)

#define CV_SEQ_POINT_SET       (CV_SEQ_KIND_GENERIC| CV_SEQ_ELTYPE_POINT)

#define CV_SEQ_POINT3D_SET     (CV_SEQ_KIND_GENERIC| CV_SEQ_ELTYPE_POINT3D)

#define CV_SEQ_POLYLINE        (CV_SEQ_KIND_CURVE  | CV_SEQ_ELTYPE_POINT)

#define CV_SEQ_POLYGON         (CV_SEQ_FLAG_CLOSED | CV_SEQ_POLYLINE )

#define CV_SEQ_CONTOUR         CV_SEQ_POLYGON

#define CV_SEQ_SIMPLE_POLYGON  (CV_SEQ_FLAG_SIMPLE | CV_SEQ_POLYGON  )

#define CV_SEQ_CHAIN           (CV_SEQ_KIND_CURVE  | CV_SEQ_ELTYPE_CODE)

#define CV_SEQ_CHAIN_CONTOUR   (CV_SEQ_FLAG_CLOSED | CV_SEQ_CHAIN)

#define CV_SEQ_POLYGON_TREE    (CV_SEQ_KIND_BIN_TREE  | CV_SEQ_ELTYPE_TRIAN_ATR)

#define CV_SEQ_CONNECTED_COMP  (CV_SEQ_KIND_GENERIC  | CV_SEQ_ELTYPE_CONNECTED_COMP)

#define CV_SEQ_INDEX           (CV_SEQ_KIND_GENERIC  | CV_SEQ_ELTYPE_INDEX)

#define CV_STORAGE_READ          0

#define CV_STORAGE_WRITE         1

#define CV_STORAGE_WRITE_TEXT    CV_STORAGE_WRITE

#define CV_STORAGE_WRITE_BINARY  CV_STORAGE_WRITE

#define CV_STORAGE_APPEND        2

#define CV_NODE_NONE        0

#define CV_NODE_INT         1

#define CV_NODE_INTEGER     CV_NODE_INT

#define CV_NODE_REAL        2

#define CV_NODE_FLOAT       CV_NODE_REAL

#define CV_NODE_STR         3

#define CV_NODE_STRING      CV_NODE_STR

#define CV_NODE_REF         4 

#define CV_NODE_SEQ         5

#define CV_NODE_MAP         6

#define CV_NODE_TYPE_MASK   7

#define CV_NODE_FLOW        8 

#define CV_NODE_USER        16

#define CV_NODE_EMPTY       32

#define CV_NODE_NAMED       64

#define CV_NODE_SEQ_SIMPLE 256

#define CV_StsOk                    0  

#define CV_StsBackTrace            -1  

#define CV_StsError                -2  

#define CV_StsInternal             -3  

#define CV_StsNoMem                -4  

#define CV_StsBadArg               -5  

#define CV_StsBadFunc              -6  

#define CV_StsNoConv               -7  

#define CV_StsAutoTrace            -8  

#define CV_HeaderIsNull            -9  

#define CV_BadImageSize            -10 

#define CV_BadOffset               -11 

#define CV_BadDataPtr              -12 

#define CV_BadStep                 -13 

#define CV_BadModelOrChSeq         -14 

#define CV_BadNumChannels          -15 

#define CV_BadNumChannel1U         -16 

#define CV_BadDepth                -17 

#define CV_BadAlphaChannel         -18 

#define CV_BadOrder                -19 

#define CV_BadOrigin               -20 

#define CV_BadAlign                -21 

#define CV_BadCallBack             -22 

#define CV_BadTileSize             -23 

#define CV_BadCOI                  -24 

#define CV_BadROISize              -25 

#define CV_MaskIsTiled             -26 

#define CV_StsNullPtr                -27 

#define CV_StsVecLengthErr           -28 

#define CV_StsFilterStructContentErr -29 

#define CV_StsKernelStructContentErr -30 

#define CV_StsFilterOffsetErr        -31 

#define CV_StsBadSize                -201 

#define CV_StsDivByZero              -202 

#define CV_StsInplaceNotSupported    -203 

#define CV_StsObjectNotFound         -204 

#define CV_StsUnmatchedFormats       -205 

#define CV_StsBadFlag                -206 

#define CV_StsBadPoint               -207 

#define CV_StsBadMask                -208 

#define CV_StsUnmatchedSizes         -209 

#define CV_StsUnsupportedFormat      -210 

#define CV_StsOutOfRange             -211 

#define CV_StsParseError             -212 

#define CV_StsNotImplemented         -213 

#define CV_StsBadMemBlock            -214 

#define CV_StsAssert                 -215 

#define CV_BLUR_NO_SCALE 0

#define CV_BLUR  1

#define CV_GAUSSIAN  2

#define CV_MEDIAN 3

#define CV_BILATERAL 4

#define CV_INPAINT_NS      0

#define CV_INPAINT_TELEA   1

#define CV_SCHARR -1

#define CV_MAX_SOBEL_KSIZE 7

#define  CV_BGR2BGRA    0

#define  CV_RGB2RGBA    CV_BGR2BGRA

#define  CV_BGRA2BGR    1

#define  CV_RGBA2RGB    CV_BGRA2BGR

#define  CV_BGR2RGBA    2

#define  CV_RGB2BGRA    CV_BGR2RGBA

#define  CV_RGBA2BGR    3

#define  CV_BGRA2RGB    CV_RGBA2BGR

#define  CV_BGR2RGB     4

#define  CV_RGB2BGR     CV_BGR2RGB

#define  CV_BGRA2RGBA   5

#define  CV_RGBA2BGRA   CV_BGRA2RGBA

#define  CV_BGR2GRAY    6

#define  CV_RGB2GRAY    7

#define  CV_GRAY2BGR    8

#define  CV_GRAY2RGB    CV_GRAY2BGR

#define  CV_GRAY2BGRA   9

#define  CV_GRAY2RGBA   CV_GRAY2BGRA

#define  CV_BGRA2GRAY   10

#define  CV_RGBA2GRAY   11

#define  CV_BGR2BGR565  12

#define  CV_RGB2BGR565  13

#define  CV_BGR5652BGR  14

#define  CV_BGR5652RGB  15

#define  CV_BGRA2BGR565 16

#define  CV_RGBA2BGR565 17

#define  CV_BGR5652BGRA 18

#define  CV_BGR5652RGBA 19

#define  CV_GRAY2BGR565 20

#define  CV_BGR5652GRAY 21

#define  CV_BGR2BGR555  22

#define  CV_RGB2BGR555  23

#define  CV_BGR5552BGR  24

#define  CV_BGR5552RGB  25

#define  CV_BGRA2BGR555 26

#define  CV_RGBA2BGR555 27

#define  CV_BGR5552BGRA 28

#define  CV_BGR5552RGBA 29

#define  CV_GRAY2BGR555 30

#define  CV_BGR5552GRAY 31

#define  CV_BGR2XYZ     32

#define  CV_RGB2XYZ     33

#define  CV_XYZ2BGR     34

#define  CV_XYZ2RGB     35

#define  CV_BGR2YCrCb   36

#define  CV_RGB2YCrCb   37

#define  CV_YCrCb2BGR   38

#define  CV_YCrCb2RGB   39

#define  CV_BGR2HSV     40

#define  CV_RGB2HSV     41

#define  CV_BGR2Lab     44

#define  CV_RGB2Lab     45

#define  CV_BayerBG2BGR 46

#define  CV_BayerGB2BGR 47

#define  CV_BayerRG2BGR 48

#define  CV_BayerGR2BGR 49

#define  CV_BayerBG2RGB CV_BayerRG2BGR

#define  CV_BayerGB2RGB CV_BayerGR2BGR

#define  CV_BayerRG2RGB CV_BayerBG2BGR

#define  CV_BayerGR2RGB CV_BayerGB2BGR

#define  CV_BGR2Luv     50

#define  CV_RGB2Luv     51

#define  CV_BGR2HLS     52

#define  CV_RGB2HLS     53

#define  CV_HSV2BGR     54

#define  CV_HSV2RGB     55

#define  CV_Lab2BGR     56

#define  CV_Lab2RGB     57

#define  CV_Luv2BGR     58

#define  CV_Luv2RGB     59

#define  CV_HLS2BGR     60

#define  CV_HLS2RGB     61

#define  CV_COLORCVT_MAX  100

#define  CV_INTER_NN        0

#define  CV_INTER_LINEAR    1

#define  CV_INTER_CUBIC     2

#define  CV_INTER_AREA      3

#define  CV_WARP_FILL_OUTLIERS 8

#define  CV_WARP_INVERSE_MAP  16

#define  CV_SHAPE_RECT      0

#define  CV_SHAPE_CROSS     1

#define  CV_SHAPE_ELLIPSE   2

#define  CV_SHAPE_CUSTOM    100

#define CV_MOP_OPEN         2

#define CV_MOP_CLOSE        3

#define CV_MOP_GRADIENT     4

#define CV_MOP_TOPHAT       5

#define CV_MOP_BLACKHAT     6

#define  CV_TM_SQDIFF        0

#define  CV_TM_SQDIFF_NORMED 1

#define  CV_TM_CCORR         2

#define  CV_TM_CCORR_NORMED  3

#define  CV_TM_CCOEFF        4

#define  CV_TM_CCOEFF_NORMED 5

#define  CV_LKFLOW_PYR_A_READY       1

#define  CV_LKFLOW_PYR_B_READY       2

#define  CV_LKFLOW_INITIAL_GUESSES   4

#define  CV_LKFLOW_GET_MIN_EIGENVALS 8

#define CV_POLY_APPROX_DP 0

#define CV_DOMINANT_IPAN 1

#define CV_CONTOURS_MATCH_I1  1

#define CV_CONTOURS_MATCH_I2  2

#define CV_CONTOURS_MATCH_I3  3

#define  CV_CONTOUR_TREES_MATCH_I1  1

#define CV_CLOCKWISE         1

#define CV_COUNTER_CLOCKWISE 2

#define CV_COMP_CORREL        0

#define CV_COMP_CHISQR        1

#define CV_COMP_INTERSECT     2

#define CV_COMP_BHATTACHARYYA 3

#define  CV_VALUE  1

#define  CV_ARRAY  2

#define CV_DIST_MASK_3   3

#define CV_DIST_MASK_5   5

#define CV_DIST_MASK_PRECISE 0

#define CV_THRESH_BINARY      0  

#define CV_THRESH_BINARY_INV  1  

#define CV_THRESH_TRUNC       2  

#define CV_THRESH_TOZERO      3  

#define CV_THRESH_TOZERO_INV  4  

#define CV_THRESH_MASK        7

#define CV_THRESH_OTSU        8  

#define CV_ADAPTIVE_THRESH_MEAN_C  0

#define CV_ADAPTIVE_THRESH_GAUSSIAN_C  1

#define CV_FLOODFILL_FIXED_RANGE (1 << 16)

#define CV_FLOODFILL_MASK_ONLY   (1 << 17)

#define CV_CANNY_L2_GRADIENT  (1 << 31)

#define CV_HOUGH_STANDARD 0

#define CV_HOUGH_PROBABILISTIC 1

#define CV_HOUGH_MULTI_SCALE 2

#define CV_HOUGH_GRADIENT 3

#define CV_HAAR_DO_CANNY_PRUNING    1

#define CV_HAAR_SCALE_IMAGE         2

#define CV_HAAR_FIND_BIGGEST_OBJECT 4

#define CV_HAAR_DO_ROUGH_SEARCH     8

#define CV_LMEDS 4

#define CV_RANSAC 8

#define CV_CALIB_CB_ADAPTIVE_THRESH  1

#define CV_CALIB_CB_NORMALIZE_IMAGE  2

#define CV_CALIB_CB_FILTER_QUADS     4

#define CV_CALIB_USE_INTRINSIC_GUESS  1

#define CV_CALIB_FIX_ASPECT_RATIO     2

#define CV_CALIB_FIX_PRINCIPAL_POINT  4

#define CV_CALIB_ZERO_TANGENT_DIST    8

#define CV_CALIB_FIX_FOCAL_LENGTH 16

#define CV_CALIB_FIX_K1  32

#define CV_CALIB_FIX_K2  64

#define CV_CALIB_FIX_K3  128

#define CV_CALIB_FIX_INTRINSIC  256

#define CV_CALIB_SAME_FOCAL_LENGTH 512

#define CV_CALIB_ZERO_DISPARITY 1024

#define CV_FM_7POINT 1

#define CV_FM_8POINT 2

#define CV_FM_LMEDS_ONLY  CV_LMEDS

#define CV_FM_RANSAC_ONLY CV_RANSAC

#define CV_FM_LMEDS CV_LMEDS

#define CV_FM_RANSAC CV_RANSAC

#define CV_STEREO_BM_NORMALIZED_RESPONSE  0

#define CV_STEREO_BM_BASIC 0

#define CV_STEREO_BM_FISH_EYE 1

#define CV_STEREO_BM_NARROW 2

#define CV_STEREO_GC_OCCLUDED  SHRT_MAX

#define CV_RETR_EXTERNAL 0

#define CV_RETR_LIST     1

#define CV_RETR_CCOMP    2

#define CV_RETR_TREE     3

#define CV_CHAIN_CODE               0

#define CV_CHAIN_APPROX_NONE        1

#define CV_CHAIN_APPROX_SIMPLE      2

#define CV_CHAIN_APPROX_TC89_L1     3

#define CV_CHAIN_APPROX_TC89_KCOS   4

#define CV_LINK_RUNS                5

#define CV_SUBDIV2D_VIRTUAL_POINT_FLAG (1 << 30)

#define CV_DIST_USER    -1  

#define CV_DIST_L1      1   

#define CV_DIST_L2      2   

#define CV_DIST_C       3   

#define CV_DIST_L12     4   

#define CV_DIST_FAIR    5   

#define CV_DIST_WELSCH  6   

#define CV_DIST_HUBER   7   

#define CV_HAAR_MAGIC_VAL    0x42500000

#define CV_TYPE_NAME_HAAR    "opencv-haar-classifier"

#define CV_HAAR_FEATURE_MAX  3

