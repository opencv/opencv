#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

//extra color conversions supported implicitly
enum
{
	CX_BGRA2HLS      = CV_COLORCVT_MAX + CV_BGR2HLS,
	CX_BGRA2HLS_FULL = CV_COLORCVT_MAX + CV_BGR2HLS_FULL,
	CX_BGRA2HSV      = CV_COLORCVT_MAX + CV_BGR2HSV,
	CX_BGRA2HSV_FULL = CV_COLORCVT_MAX + CV_BGR2HSV_FULL,
	CX_BGRA2Lab      = CV_COLORCVT_MAX + CV_BGR2Lab,
	CX_BGRA2Luv      = CV_COLORCVT_MAX + CV_BGR2Luv,
	CX_BGRA2XYZ      = CV_COLORCVT_MAX + CV_BGR2XYZ,
	CX_BGRA2YCrCb    = CV_COLORCVT_MAX + CV_BGR2YCrCb,
	CX_BGRA2YUV      = CV_COLORCVT_MAX + CV_BGR2YUV,
	CX_HLS2BGRA      = CV_COLORCVT_MAX + CV_HLS2BGR,
	CX_HLS2BGRA_FULL = CV_COLORCVT_MAX + CV_HLS2BGR_FULL,
	CX_HLS2RGBA      = CV_COLORCVT_MAX + CV_HLS2RGB,
	CX_HLS2RGBA_FULL = CV_COLORCVT_MAX + CV_HLS2RGB_FULL,
	CX_HSV2BGRA      = CV_COLORCVT_MAX + CV_HSV2BGR,
	CX_HSV2BGRA_FULL = CV_COLORCVT_MAX + CV_HSV2BGR_FULL,
	CX_HSV2RGBA      = CV_COLORCVT_MAX + CV_HSV2RGB,
	CX_HSV2RGBA_FULL = CV_COLORCVT_MAX + CV_HSV2RGB_FULL,
	CX_Lab2BGRA      = CV_COLORCVT_MAX + CV_Lab2BGR,
	CX_Lab2LBGRA     = CV_COLORCVT_MAX + CV_Lab2LBGR,
	CX_Lab2LRGBA     = CV_COLORCVT_MAX + CV_Lab2LRGB,
	CX_Lab2RGBA      = CV_COLORCVT_MAX + CV_Lab2RGB,
	CX_LBGRA2Lab     = CV_COLORCVT_MAX + CV_LBGR2Lab,
	CX_LBGRA2Luv     = CV_COLORCVT_MAX + CV_LBGR2Luv,
	CX_LRGBA2Lab     = CV_COLORCVT_MAX + CV_LRGB2Lab,
	CX_LRGBA2Luv     = CV_COLORCVT_MAX + CV_LRGB2Luv,
	CX_Luv2BGRA      = CV_COLORCVT_MAX + CV_Luv2BGR,
	CX_Luv2LBGRA     = CV_COLORCVT_MAX + CV_Luv2LBGR,
	CX_Luv2LRGBA     = CV_COLORCVT_MAX + CV_Luv2LRGB,
	CX_Luv2RGBA      = CV_COLORCVT_MAX + CV_Luv2RGB,
	CX_RGBA2HLS      = CV_COLORCVT_MAX + CV_RGB2HLS,
	CX_RGBA2HLS_FULL = CV_COLORCVT_MAX + CV_RGB2HLS_FULL,
	CX_RGBA2HSV      = CV_COLORCVT_MAX + CV_RGB2HSV,
	CX_RGBA2HSV_FULL = CV_COLORCVT_MAX + CV_RGB2HSV_FULL,
	CX_RGBA2Lab      = CV_COLORCVT_MAX + CV_RGB2Lab,
	CX_RGBA2Luv      = CV_COLORCVT_MAX + CV_RGB2Luv,
	CX_RGBA2XYZ      = CV_COLORCVT_MAX + CV_RGB2XYZ,
	CX_RGBA2YCrCb    = CV_COLORCVT_MAX + CV_RGB2YCrCb,
	CX_RGBA2YUV      = CV_COLORCVT_MAX + CV_RGB2YUV,
	CX_XYZ2BGRA      = CV_COLORCVT_MAX + CV_XYZ2BGR,
	CX_XYZ2RGBA      = CV_COLORCVT_MAX + CV_XYZ2RGB,
	CX_YCrCb2BGRA    = CV_COLORCVT_MAX + CV_YCrCb2BGR,
	CX_YCrCb2RGBA    = CV_COLORCVT_MAX + CV_YCrCb2RGB,
	CX_YUV2BGRA      = CV_COLORCVT_MAX + CV_YUV2BGR,
	CX_YUV2RGBA      = CV_COLORCVT_MAX + CV_YUV2RGB
};

CV_ENUM(CvtMode, 
	CV_BayerBG2BGR, CV_BayerBG2BGR_VNG, CV_BayerBG2GRAY,
	CV_BayerGB2BGR, CV_BayerGB2BGR_VNG, CV_BayerGB2GRAY,
	CV_BayerGR2BGR, CV_BayerGR2BGR_VNG, CV_BayerGR2GRAY,
	CV_BayerRG2BGR, CV_BayerRG2BGR_VNG, CV_BayerRG2GRAY,
	
	CV_BGR2BGR555, CV_BGR2BGR565, CV_BGR2BGRA, CV_BGR2GRAY, 
	CV_BGR2HLS, CV_BGR2HLS_FULL, CV_BGR2HSV, CV_BGR2HSV_FULL,
	CV_BGR2Lab, CV_BGR2Luv, CV_BGR2RGB, CV_BGR2RGBA, CV_BGR2XYZ,
	CV_BGR2YCrCb, CV_BGR2YUV, CV_BGR5552BGR, CV_BGR5552BGRA,
	
	CV_BGR5552GRAY, CV_BGR5552RGB, CV_BGR5552RGBA, CV_BGR5652BGR,
	CV_BGR5652BGRA, CV_BGR5652GRAY, CV_BGR5652RGB, CV_BGR5652RGBA,
	
	CV_BGRA2BGR, CV_BGRA2BGR555, CV_BGRA2BGR565, CV_BGRA2GRAY, CV_BGRA2RGBA,
	CX_BGRA2HLS, CX_BGRA2HLS_FULL, CX_BGRA2HSV, CX_BGRA2HSV_FULL,
	CX_BGRA2Lab, CX_BGRA2Luv, CX_BGRA2XYZ,
	CX_BGRA2YCrCb, CX_BGRA2YUV,
	
	CV_GRAY2BGR, CV_GRAY2BGR555, CV_GRAY2BGR565, CV_GRAY2BGRA,

	CV_HLS2BGR, CV_HLS2BGR_FULL, CV_HLS2RGB, CV_HLS2RGB_FULL,
	CX_HLS2BGRA, CX_HLS2BGRA_FULL, CX_HLS2RGBA, CX_HLS2RGBA_FULL,	
	
	CV_HSV2BGR, CV_HSV2BGR_FULL, CV_HSV2RGB, CV_HSV2RGB_FULL,
	CX_HSV2BGRA, CX_HSV2BGRA_FULL, CX_HSV2RGBA,	CX_HSV2RGBA_FULL,
	
	CV_Lab2BGR, CV_Lab2LBGR, CV_Lab2LRGB, CV_Lab2RGB,
	CX_Lab2BGRA, CX_Lab2LBGRA, CX_Lab2LRGBA, CX_Lab2RGBA,
	
	CV_LBGR2Lab, CV_LBGR2Luv, CV_LRGB2Lab, CV_LRGB2Luv,
	CX_LBGRA2Lab, CX_LBGRA2Luv, CX_LRGBA2Lab, CX_LRGBA2Luv,
	
	CV_Luv2BGR, CV_Luv2LBGR, CV_Luv2LRGB, CV_Luv2RGB,
	CX_Luv2BGRA, CX_Luv2LBGRA, CX_Luv2LRGBA, CX_Luv2RGBA,
	
	CV_RGB2BGR555, CV_RGB2BGR565, CV_RGB2GRAY,
	CV_RGB2HLS, CV_RGB2HLS_FULL, CV_RGB2HSV, CV_RGB2HSV_FULL,
	CV_RGB2Lab, CV_RGB2Luv, CV_RGB2XYZ, CV_RGB2YCrCb, CV_RGB2YUV,
	
	CV_RGBA2BGR, CV_RGBA2BGR555, CV_RGBA2BGR565, CV_RGBA2GRAY,
	CX_RGBA2HLS, CX_RGBA2HLS_FULL, CX_RGBA2HSV, CX_RGBA2HSV_FULL,
	CX_RGBA2Lab, CX_RGBA2Luv, CX_RGBA2XYZ,
	CX_RGBA2YCrCb, CX_RGBA2YUV,
	
	CV_XYZ2BGR, CV_XYZ2RGB, CX_XYZ2BGRA, CX_XYZ2RGBA,
	
	CV_YCrCb2BGR, CV_YCrCb2RGB, CX_YCrCb2BGRA, CX_YCrCb2RGBA,
	CV_YUV2BGR, CV_YUV2RGB, CX_YUV2BGRA, CX_YUV2RGBA
	)

CV_ENUM(CvtMode2, CV_YUV420i2BGR, CV_YUV420i2BGRA, CV_YUV420i2RGB, CV_YUV420i2RGBA, CV_YUV420sp2BGR, CV_YUV420sp2BGRA, CV_YUV420sp2RGB, CV_YUV420sp2RGBA)
	
struct ChPair
{
	ChPair(int _scn, int _dcn): scn(_scn), dcn(_dcn) {}
	int scn, dcn;
};

ChPair getConversionInfo(int cvtMode)
{
	switch(cvtMode)
	{
	case CV_BayerBG2GRAY: case CV_BayerGB2GRAY:
	case CV_BayerGR2GRAY: case CV_BayerRG2GRAY:
		return ChPair(1,1);
	case CV_GRAY2BGR555: case CV_GRAY2BGR565:
		return ChPair(1,2);
	case CV_BayerBG2BGR: case CV_BayerBG2BGR_VNG:
	case CV_BayerGB2BGR: case CV_BayerGB2BGR_VNG:
	case CV_BayerGR2BGR: case CV_BayerGR2BGR_VNG:
	case CV_BayerRG2BGR: case CV_BayerRG2BGR_VNG:
	case CV_GRAY2BGR: case CV_YUV420i2BGR:
	case CV_YUV420i2RGB: case CV_YUV420sp2BGR:
	case CV_YUV420sp2RGB:
		return ChPair(1,3);
	case CV_GRAY2BGRA: case CV_YUV420i2BGRA:
	case CV_YUV420i2RGBA: case CV_YUV420sp2BGRA:
	case CV_YUV420sp2RGBA:
		return ChPair(1,4);
	case CV_BGR5552GRAY: case CV_BGR5652GRAY:
		return ChPair(2,1);
	case CV_BGR5552BGR: case CV_BGR5552RGB:
	case CV_BGR5652BGR: case CV_BGR5652RGB:
		return ChPair(2,3);
	case CV_BGR5552BGRA: case CV_BGR5552RGBA:
	case CV_BGR5652BGRA: case CV_BGR5652RGBA:
		return ChPair(2,4);
	case CV_BGR2GRAY: case CV_RGB2GRAY:
		return ChPair(3,1);
	case CV_BGR2BGR555: case CV_BGR2BGR565:
	case CV_RGB2BGR555: case CV_RGB2BGR565:
		return ChPair(3,2);
	case CV_BGR2HLS: case CV_BGR2HLS_FULL:
	case CV_BGR2HSV: case CV_BGR2HSV_FULL:
	case CV_BGR2Lab: case CV_BGR2Luv:
	case CV_BGR2RGB: case CV_BGR2XYZ:
	case CV_BGR2YCrCb: case CV_BGR2YUV:
	case CV_HLS2BGR: case CV_HLS2BGR_FULL:
	case CV_HLS2RGB: case CV_HLS2RGB_FULL:
	case CV_HSV2BGR: case CV_HSV2BGR_FULL:
	case CV_HSV2RGB: case CV_HSV2RGB_FULL:
	case CV_Lab2BGR: case CV_Lab2LBGR:
	case CV_Lab2LRGB: case CV_Lab2RGB:
	case CV_LBGR2Lab: case CV_LBGR2Luv:
	case CV_LRGB2Lab: case CV_LRGB2Luv:
	case CV_Luv2BGR: case CV_Luv2LBGR:
	case CV_Luv2LRGB: case CV_Luv2RGB:
	case CV_RGB2HLS: case CV_RGB2HLS_FULL:
	case CV_RGB2HSV: case CV_RGB2HSV_FULL:
	case CV_RGB2Lab: case CV_RGB2Luv:
	case CV_RGB2XYZ: case CV_RGB2YCrCb:
	case CV_RGB2YUV: case CV_XYZ2BGR:
	case CV_XYZ2RGB: case CV_YCrCb2BGR:
	case CV_YCrCb2RGB: case CV_YUV2BGR:
	case CV_YUV2RGB:
		return ChPair(3,3);
	case CV_BGR2BGRA: case CV_BGR2RGBA:
	case CX_HLS2BGRA: case CX_HLS2BGRA_FULL:
	case CX_HLS2RGBA: case CX_HLS2RGBA_FULL:
	case CX_HSV2BGRA: case CX_HSV2BGRA_FULL:
	case CX_HSV2RGBA: case CX_HSV2RGBA_FULL:
	case CX_Lab2BGRA: case CX_Lab2LBGRA:
	case CX_Lab2LRGBA: case CX_Lab2RGBA:
	case CX_Luv2BGRA: case CX_Luv2LBGRA:
	case CX_Luv2LRGBA: case CX_Luv2RGBA:
	case CX_XYZ2BGRA: case CX_XYZ2RGBA:
	case CX_YCrCb2BGRA: case CX_YCrCb2RGBA:
	case CX_YUV2BGRA: case CX_YUV2RGBA:
		return ChPair(3,4);
	case CV_BGRA2GRAY: case CV_RGBA2GRAY:
		return ChPair(4,1);
	case CV_BGRA2BGR555: case CV_BGRA2BGR565:
	case CV_RGBA2BGR555: case CV_RGBA2BGR565:
		return ChPair(4,2);
	case CV_BGRA2BGR: case CX_BGRA2HLS:
	case CX_BGRA2HLS_FULL: case CX_BGRA2HSV:
	case CX_BGRA2HSV_FULL: case CX_BGRA2Lab:
	case CX_BGRA2Luv: case CX_BGRA2XYZ:
	case CX_BGRA2YCrCb: case CX_BGRA2YUV:
	case CX_LBGRA2Lab: case CX_LBGRA2Luv:
	case CX_LRGBA2Lab: case CX_LRGBA2Luv:
	case CV_RGBA2BGR: case CX_RGBA2HLS:
	case CX_RGBA2HLS_FULL: case CX_RGBA2HSV:
	case CX_RGBA2HSV_FULL: case CX_RGBA2Lab:
	case CX_RGBA2Luv: case CX_RGBA2XYZ:
	case CX_RGBA2YCrCb: case CX_RGBA2YUV:
		return ChPair(4,3);
	case CV_BGRA2RGBA:
		return ChPair(4,4);
	default:
		ADD_FAILURE() << "Unknown conversion type";
		break;
	};
	return ChPair(0,0);
}

typedef std::tr1::tuple<Size, CvtMode> Size_CvtMode_t;
typedef perf::TestBaseWithParam<Size_CvtMode_t> Size_CvtMode;

PERF_TEST_P(Size_CvtMode, cvtColor8u,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::ValuesIn(CvtMode::all())
                )
            )
{
	Size sz = get<0>(GetParam());
	int mode = get<1>(GetParam());
	ChPair ch = getConversionInfo(mode);
	mode %= CV_COLORCVT_MAX;
    
	Mat src(sz, CV_8UC(ch.scn));
	Mat dst(sz, CV_8UC(ch.dcn));
	
	declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE() cvtColor(src, dst, mode, ch.dcn);
    
    SANITY_CHECK(dst, 1);
}

typedef std::tr1::tuple<Size, CvtMode2> Size_CvtMode2_t;
typedef perf::TestBaseWithParam<Size_CvtMode2_t> Size_CvtMode2;

PERF_TEST_P(Size_CvtMode2, cvtColorYUV420,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p, Size(130, 60)),
                testing::ValuesIn(CvtMode2::all())
                )
            )
{
    Size sz = get<0>(GetParam());
	int mode = get<1>(GetParam());
	ChPair ch = getConversionInfo(mode);

    Mat src(sz.height + sz.height / 2, sz.width, CV_8UC(ch.scn));
    Mat dst(sz, CV_8UC(ch.dcn));

    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE() cvtColor(src, dst, mode, ch.dcn);
    
    SANITY_CHECK(dst, 1);
}