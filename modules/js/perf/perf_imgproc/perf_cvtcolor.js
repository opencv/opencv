const isNodeJs = (typeof window) === 'undefined'? true : false;

if (isNodeJs) {
  var Benchmark = require('benchmark');
  var cv = require('../../opencv');
  var HelpFunc = require('../perf_helpfunc');
  var Base = require('../base');
} else {
  var paramsElement = document.getElementById('params');
  var runButton = document.getElementById('runButton');
  var logElement = document.getElementById('log');
}

cv.onRuntimeInitialized = () => {
  console.log('opencv.js loaded');
  if (isNodeJs) {
    global.cv = cv;
    global.combine = HelpFunc.combine;
    global.cvtStr2cvSize = HelpFunc.cvtStr2cvSize;
    global.cvSize = Base.cvSize;
  } else {
    runButton.removeAttribute('disabled');
    runButton.setAttribute('class', 'btn btn-primary');
    runButton.innerHTML = 'Run';
  }
  let totalCaseNum, currentCaseId;

  //extra color conversions supported implicitly
  {
    cv.CX_BGRA2HLS      = cv.COLOR_COLORCVT_MAX + cv.COLOR_BGR2HLS,
    cv.CX_BGRA2HLS_FULL = cv.COLOR_COLORCVT_MAX + cv.COLOR_BGR2HLS_FULL,
    cv.CX_BGRA2HSV      = cv.COLOR_COLORCVT_MAX + cv.COLOR_BGR2HSV,
    cv.CX_BGRA2HSV_FULL = cv.COLOR_COLORCVT_MAX + cv.COLOR_BGR2HSV_FULL,
    cv.CX_BGRA2Lab      = cv.COLOR_COLORCVT_MAX + cv.COLOR_BGR2Lab,
    cv.CX_BGRA2Luv      = cv.COLOR_COLORCVT_MAX + cv.COLOR_BGR2Luv,
    cv.CX_BGRA2XYZ      = cv.COLOR_COLORCVT_MAX + cv.COLOR_BGR2XYZ,
    cv.CX_BGRA2YCrCb    = cv.COLOR_COLORCVT_MAX + cv.COLOR_BGR2YCrCb,
    cv.CX_BGRA2YUV      = cv.COLOR_COLORCVT_MAX + cv.COLOR_BGR2YUV,
    cv.CX_HLS2BGRA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_HLS2BGR,
    cv.CX_HLS2BGRA_FULL = cv.COLOR_COLORCVT_MAX + cv.COLOR_HLS2BGR_FULL,
    cv.CX_HLS2RGBA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_HLS2RGB,
    cv.CX_HLS2RGBA_FULL = cv.COLOR_COLORCVT_MAX + cv.COLOR_HLS2RGB_FULL,
    cv.CX_HSV2BGRA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_HSV2BGR,
    cv.CX_HSV2BGRA_FULL = cv.COLOR_COLORCVT_MAX + cv.COLOR_HSV2BGR_FULL,
    cv.CX_HSV2RGBA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_HSV2RGB,
    cv.CX_HSV2RGBA_FULL = cv.COLOR_COLORCVT_MAX + cv.COLOR_HSV2RGB_FULL,
    cv.CX_Lab2BGRA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_Lab2BGR,
    cv.CX_Lab2LBGRA     = cv.COLOR_COLORCVT_MAX + cv.COLOR_Lab2LBGR,
    cv.CX_Lab2LRGBA     = cv.COLOR_COLORCVT_MAX + cv.COLOR_Lab2LRGB,
    cv.CX_Lab2RGBA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_Lab2RGB,
    cv.CX_LBGRA2Lab     = cv.COLOR_COLORCVT_MAX + cv.COLOR_LBGR2Lab,
    cv.CX_LBGRA2Luv     = cv.COLOR_COLORCVT_MAX + cv.COLOR_LBGR2Luv,
    cv.CX_LRGBA2Lab     = cv.COLOR_COLORCVT_MAX + cv.COLOR_LRGB2Lab,
    cv.CX_LRGBA2Luv     = cv.COLOR_COLORCVT_MAX + cv.COLOR_LRGB2Luv,
    cv.CX_Luv2BGRA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_Luv2BGR,
    cv.CX_Luv2LBGRA     = cv.COLOR_COLORCVT_MAX + cv.COLOR_Luv2LBGR,
    cv.CX_Luv2LRGBA     = cv.COLOR_COLORCVT_MAX + cv.COLOR_Luv2LRGB,
    cv.CX_Luv2RGBA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_Luv2RGB,
    cv.CX_RGBA2HLS      = cv.COLOR_COLORCVT_MAX + cv.COLOR_RGB2HLS,
    cv.CX_RGBA2HLS_FULL = cv.COLOR_COLORCVT_MAX + cv.COLOR_RGB2HLS_FULL,
    cv.CX_RGBA2HSV      = cv.COLOR_COLORCVT_MAX + cv.COLOR_RGB2HSV,
    cv.CX_RGBA2HSV_FULL = cv.COLOR_COLORCVT_MAX + cv.COLOR_RGB2HSV_FULL,
    cv.CX_RGBA2Lab      = cv.COLOR_COLORCVT_MAX + cv.COLOR_RGB2Lab,
    cv.CX_RGBA2Luv      = cv.COLOR_COLORCVT_MAX + cv.COLOR_RGB2Luv,
    cv.CX_RGBA2XYZ      = cv.COLOR_COLORCVT_MAX + cv.COLOR_RGB2XYZ,
    cv.CX_RGBA2YCrCb    = cv.COLOR_COLORCVT_MAX + cv.COLOR_RGB2YCrCb,
    cv.CX_RGBA2YUV      = cv.COLOR_COLORCVT_MAX + cv.COLOR_RGB2YUV,
    cv.CX_XYZ2BGRA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_XYZ2BGR,
    cv.CX_XYZ2RGBA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_XYZ2RGB,
    cv.CX_YCrCb2BGRA    = cv.COLOR_COLORCVT_MAX + cv.COLOR_YCrCb2BGR,
    cv.CX_YCrCb2RGBA    = cv.COLOR_COLORCVT_MAX + cv.COLOR_YCrCb2RGB,
    cv.CX_YUV2BGRA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_YUV2BGR,
    cv.CX_YUV2RGBA      = cv.COLOR_COLORCVT_MAX + cv.COLOR_YUV2RGB
  };

  const CvtMode = [
    "COLOR_BGR2BGR555", "COLOR_BGR2BGR565", "COLOR_BGR2BGRA", "COLOR_BGR2GRAY",
    "COLOR_BGR2HLS", "COLOR_BGR2HLS_FULL", "COLOR_BGR2HSV", "COLOR_BGR2HSV_FULL",
    "COLOR_BGR2Lab", "COLOR_BGR2Luv", "COLOR_BGR2RGB", "COLOR_BGR2RGBA", "COLOR_BGR2XYZ",
    "COLOR_BGR2YCrCb", "COLOR_BGR2YUV", "COLOR_BGR5552BGR", "COLOR_BGR5552BGRA",

    "COLOR_BGR5552GRAY", "COLOR_BGR5552RGB", "COLOR_BGR5552RGBA", "COLOR_BGR5652BGR",
    "COLOR_BGR5652BGRA", "COLOR_BGR5652GRAY", "COLOR_BGR5652RGB", "COLOR_BGR5652RGBA",

    "COLOR_BGRA2BGR", "COLOR_BGRA2BGR555", "COLOR_BGRA2BGR565", "COLOR_BGRA2GRAY", "COLOR_BGRA2RGBA",
    "CX_BGRA2HLS", "CX_BGRA2HLS_FULL", "CX_BGRA2HSV", "CX_BGRA2HSV_FULL",
    "CX_BGRA2Lab", "CX_BGRA2Luv", "CX_BGRA2XYZ",
    "CX_BGRA2YCrCb", "CX_BGRA2YUV",

    "COLOR_GRAY2BGR", "COLOR_GRAY2BGR555", "COLOR_GRAY2BGR565", "COLOR_GRAY2BGRA",

    "COLOR_HLS2BGR", "COLOR_HLS2BGR_FULL", "COLOR_HLS2RGB", "COLOR_HLS2RGB_FULL",
    "CX_HLS2BGRA", "CX_HLS2BGRA_FULL", "CX_HLS2RGBA", "CX_HLS2RGBA_FULL",

    "COLOR_HSV2BGR", "COLOR_HSV2BGR_FULL", "COLOR_HSV2RGB", "COLOR_HSV2RGB_FULL",
    "CX_HSV2BGRA", "CX_HSV2BGRA_FULL", "CX_HSV2RGBA", "CX_HSV2RGBA_FULL",

    "COLOR_Lab2BGR", "COLOR_Lab2LBGR", "COLOR_Lab2LRGB", "COLOR_Lab2RGB",
    "CX_Lab2BGRA", "CX_Lab2LBGRA", "CX_Lab2LRGBA", "CX_Lab2RGBA",

    "COLOR_LBGR2Lab", "COLOR_LBGR2Luv", "COLOR_LRGB2Lab", "COLOR_LRGB2Luv",
    "CX_LBGRA2Lab", "CX_LBGRA2Luv", "CX_LRGBA2Lab", "CX_LRGBA2Luv",

    "COLOR_Luv2BGR", "COLOR_Luv2LBGR", "COLOR_Luv2LRGB", "COLOR_Luv2RGB",
    "CX_Luv2BGRA", "CX_Luv2LBGRA", "CX_Luv2LRGBA", "CX_Luv2RGBA",

    "COLOR_RGB2BGR555", "COLOR_RGB2BGR565", "COLOR_RGB2GRAY",
    "COLOR_RGB2HLS", "COLOR_RGB2HLS_FULL", "COLOR_RGB2HSV", "COLOR_RGB2HSV_FULL",
    "COLOR_RGB2Lab", "COLOR_RGB2Luv", "COLOR_RGB2XYZ", "COLOR_RGB2YCrCb", "COLOR_RGB2YUV",

    "COLOR_RGBA2BGR", "COLOR_RGBA2BGR555", "COLOR_RGBA2BGR565", "COLOR_RGBA2GRAY",
    "CX_RGBA2HLS", "CX_RGBA2HLS_FULL", "CX_RGBA2HSV", "CX_RGBA2HSV_FULL",
    "CX_RGBA2Lab", "CX_RGBA2Luv", "CX_RGBA2XYZ",
    "CX_RGBA2YCrCb", "CX_RGBA2YUV",

    "COLOR_XYZ2BGR", "COLOR_XYZ2RGB", "CX_XYZ2BGRA", "CX_XYZ2RGBA",

    "COLOR_YCrCb2BGR", "COLOR_YCrCb2RGB", "CX_YCrCb2BGRA", "CX_YCrCb2RGBA",
    "COLOR_YUV2BGR", "COLOR_YUV2RGB", "CX_YUV2BGRA", "CX_YUV2RGBA"
  ];
  const CvtModeSize = [cvSize.szODD, cvSize.szVGA, cvSize.sz1080p];
  const combiCvtMode = combine(CvtModeSize, CvtMode);

  // didn't support 16u and 32f perf tests according to
  // https://github.com/opencv/opencv/commit/4e679e1cc5b075ec006b29a58b4fe117523fba1d
  const CvtMode16U = [
    "COLOR_BGR2BGRA", "COLOR_BGR2GRAY",
    "COLOR_BGR2RGB", "COLOR_BGR2RGBA", "COLOR_BGR2XYZ",
    "COLOR_BGR2YCrCb", "COLOR_BGR2YUV",

    "COLOR_BGRA2BGR", "COLOR_BGRA2GRAY", "COLOR_BGRA2RGBA",
    "CX_BGRA2XYZ",
    "CX_BGRA2YCrCb", "CX_BGRA2YUV",

    "COLOR_GRAY2BGR", "COLOR_GRAY2BGRA",

    "COLOR_RGB2GRAY",
    "COLOR_RGB2XYZ", "COLOR_RGB2YCrCb", "COLOR_RGB2YUV",

    "COLOR_RGBA2BGR", "COLOR_RGBA2GRAY",
    "CX_RGBA2XYZ",
    "CX_RGBA2YCrCb", "CX_RGBA2YUV",

    "COLOR_XYZ2BGR", "COLOR_XYZ2RGB", "CX_XYZ2BGRA", "CX_XYZ2RGBA",

    "COLOR_YCrCb2BGR", "COLOR_YCrCb2RGB", "CX_YCrCb2BGRA", "CX_YCrCb2RGBA",
    "COLOR_YUV2BGR", "COLOR_YUV2RGB", "CX_YUV2BGRA", "CX_YUV2RGBA"
  ];
  const CvtMode16USize = [cvSize.szODD, cvSize.szVGA, cvSize.sz1080p];
  const combiCvtMode16U = combine(CvtMode16USize, CvtMode16U);

  const CvtMode32F = [
    "COLOR_BGR2BGRA", "COLOR_BGR2GRAY",
    "COLOR_BGR2HLS", "COLOR_BGR2HLS_FULL", "COLOR_BGR2HSV", "COLOR_BGR2HSV_FULL",
    "COLOR_BGR2Lab", "COLOR_BGR2Luv", "COLOR_BGR2RGB", "COLOR_BGR2RGBA", "COLOR_BGR2XYZ",
    "COLOR_BGR2YCrCb", "COLOR_BGR2YUV",

    "COLOR_BGRA2BGR", "COLOR_BGRA2GRAY", "COLOR_BGRA2RGBA",
    "CX_BGRA2HLS", "CX_BGRA2HLS_FULL", "CX_BGRA2HSV", "CX_BGRA2HSV_FULL",
    "CX_BGRA2Lab", "CX_BGRA2Luv", "CX_BGRA2XYZ",
    "CX_BGRA2YCrCb", "CX_BGRA2YUV",

    "COLOR_GRAY2BGR", "COLOR_GRAY2BGRA",

    "COLOR_HLS2BGR", "COLOR_HLS2BGR_FULL", "COLOR_HLS2RGB", "COLOR_HLS2RGB_FULL",
    "CX_HLS2BGRA", "CX_HLS2BGRA_FULL", "CX_HLS2RGBA", "CX_HLS2RGBA_FULL",

    "COLOR_HSV2BGR", "COLOR_HSV2BGR_FULL", "COLOR_HSV2RGB", "COLOR_HSV2RGB_FULL",
    "CX_HSV2BGRA", "CX_HSV2BGRA_FULL", "CX_HSV2RGBA", "CX_HSV2RGBA_FULL",

    "COLOR_Lab2BGR", "COLOR_Lab2LBGR", "COLOR_Lab2LRGB", "COLOR_Lab2RGB",
    "CX_Lab2BGRA", "CX_Lab2LBGRA", "CX_Lab2LRGBA", "CX_Lab2RGBA",

    "COLOR_LBGR2Lab", "COLOR_LBGR2Luv", "COLOR_LRGB2Lab", "COLOR_LRGB2Luv",
    "CX_LBGRA2Lab", "CX_LBGRA2Luv", "CX_LRGBA2Lab", "CX_LRGBA2Luv",

    "COLOR_Luv2BGR", "COLOR_Luv2LBGR", "COLOR_Luv2LRGB", "COLOR_Luv2RGB",
    "CX_Luv2BGRA", "CX_Luv2LBGRA", "CX_Luv2LRGBA", "CX_Luv2RGBA",

    "COLOR_RGB2GRAY",
    "COLOR_RGB2HLS", "COLOR_RGB2HLS_FULL", "COLOR_RGB2HSV", "COLOR_RGB2HSV_FULL",
    "COLOR_RGB2Lab", "COLOR_RGB2Luv", "COLOR_RGB2XYZ", "COLOR_RGB2YCrCb", "COLOR_RGB2YUV",

    "COLOR_RGBA2BGR", "COLOR_RGBA2GRAY",
    "CX_RGBA2HLS", "CX_RGBA2HLS_FULL", "CX_RGBA2HSV", "CX_RGBA2HSV_FULL",
    "CX_RGBA2Lab", "CX_RGBA2Luv", "CX_RGBA2XYZ",
    "CX_RGBA2YCrCb", "CX_RGBA2YUV",

    "COLOR_XYZ2BGR", "COLOR_XYZ2RGB", "CX_XYZ2BGRA", "CX_XYZ2RGBA",

    "COLOR_YCrCb2BGR", "COLOR_YCrCb2RGB", "CX_YCrCb2BGRA", "CX_YCrCb2RGBA",
    "COLOR_YUV2BGR", "COLOR_YUV2RGB", "CX_YUV2BGRA", "CX_YUV2RGBA"
  ];
  const CvtMode32FSize = [cvSize.szODD, cvSize.szVGA, cvSize.sz1080p];
  const combiCvtMode32F = combine(CvtMode32FSize, CvtMode32F);

  const CvtModeBayer = [
    "COLOR_BayerBG2BGR", "COLOR_BayerBG2BGRA", "COLOR_BayerBG2BGR_VNG", "COLOR_BayerBG2GRAY",
    "COLOR_BayerGB2BGR", "COLOR_BayerGB2BGRA", "COLOR_BayerGB2BGR_VNG", "COLOR_BayerGB2GRAY",
    "COLOR_BayerGR2BGR", "COLOR_BayerGR2BGRA", "COLOR_BayerGR2BGR_VNG", "COLOR_BayerGR2GRAY",
    "COLOR_BayerRG2BGR", "COLOR_BayerRG2BGRA", "COLOR_BayerRG2BGR_VNG", "COLOR_BayerRG2GRAY"
  ];
  const CvtModeBayerSize = [cvSize.szODD, cvSize.szVGA];
  const combiCvtModeBayer = combine(CvtModeBayerSize, CvtModeBayer);


  const CvtMode2 = [
    "COLOR_YUV2BGR_NV12", "COLOR_YUV2BGRA_NV12", "COLOR_YUV2RGB_NV12", "COLOR_YUV2RGBA_NV12", "COLOR_YUV2BGR_NV21", "COLOR_YUV2BGRA_NV21", "COLOR_YUV2RGB_NV21", "COLOR_YUV2RGBA_NV21",
    "COLOR_YUV2BGR_YV12", "COLOR_YUV2BGRA_YV12", "COLOR_YUV2RGB_YV12", "COLOR_YUV2RGBA_YV12", "COLOR_YUV2BGR_IYUV", "COLOR_YUV2BGRA_IYUV", "COLOR_YUV2RGB_IYUV", "COLOR_YUV2RGBA_IYUV",
    "COLOR_YUV2GRAY_420", "COLOR_YUV2RGB_UYVY", "COLOR_YUV2BGR_UYVY", "COLOR_YUV2RGBA_UYVY", "COLOR_YUV2BGRA_UYVY", "COLOR_YUV2RGB_YUY2", "COLOR_YUV2BGR_YUY2", "COLOR_YUV2RGB_YVYU",
    "COLOR_YUV2BGR_YVYU", "COLOR_YUV2RGBA_YUY2", "COLOR_YUV2BGRA_YUY2", "COLOR_YUV2RGBA_YVYU", "COLOR_YUV2BGRA_YVYU"
  ];
  const CvtMode2Size = [cvSize.szVGA, cvSize.sz1080p, cvSize.sz130x60];
  const combiCvtMode2 = combine(CvtMode2Size, CvtMode2);

  const CvtMode3 = [
    "COLOR_RGB2YUV_IYUV", "COLOR_BGR2YUV_IYUV", "COLOR_RGBA2YUV_IYUV", "COLOR_BGRA2YUV_IYUV",
    "COLOR_RGB2YUV_YV12", "COLOR_BGR2YUV_YV12", "COLOR_RGBA2YUV_YV12", "COLOR_BGRA2YUV_YV12"
  ];
  const CvtMode3Size = [cvSize.szVGA, cvSize.sz720p, cvSize.sz1080p, cvSize.sz130x60];
  const combiCvtMode3 = combine(CvtMode3Size, CvtMode3);

  const EdgeAwareBayerMode = [
    "COLOR_BayerBG2BGR_EA", "COLOR_BayerGB2BGR_EA", "COLOR_BayerRG2BGR_EA", "COLOR_BayerGR2BGR_EA"
  ];
  const EdgeAwareBayerModeSize = [cvSize.szVGA, cvSize.sz720p, cvSize.sz1080p, cvSize.sz130x60];
  const combiEdgeAwareBayer = combine(EdgeAwareBayerModeSize, EdgeAwareBayerMode);

  // This function returns an array. The 1st element is the channel number of
  // source mat and 2nd element is the channel number of destination mat.
  function getConversionInfo(cvtMode) {
    switch(cvtMode) {
      case "COLOR_BayerBG2GRAY": case "COLOR_BayerGB2GRAY":
      case "COLOR_BayerGR2GRAY": case "COLOR_BayerRG2GRAY":
      case "COLOR_YUV2GRAY_420":
        return [1, 1];
      case "COLOR_GRAY2BGR555": case "COLOR_GRAY2BGR565":
        return [1, 2];
      case "COLOR_BayerBG2BGR": case "COLOR_BayerBG2BGR_VNG":
      case "COLOR_BayerGB2BGR": case "COLOR_BayerGB2BGR_VNG":
      case "COLOR_BayerGR2BGR": case "COLOR_BayerGR2BGR_VNG":
      case "COLOR_BayerRG2BGR": case "COLOR_BayerRG2BGR_VNG":
      case "COLOR_GRAY2BGR":
      case "COLOR_YUV2BGR_NV12": case "COLOR_YUV2RGB_NV12":
      case "COLOR_YUV2BGR_NV21": case "COLOR_YUV2RGB_NV21":
      case "COLOR_YUV2BGR_YV12": case "COLOR_YUV2RGB_YV12":
      case "COLOR_YUV2BGR_IYUV": case "COLOR_YUV2RGB_IYUV":
        return [1, 3];
      case "COLOR_GRAY2BGRA":
      case "COLOR_YUV2BGRA_NV12": case "COLOR_YUV2RGBA_NV12":
      case "COLOR_YUV2BGRA_NV21": case "COLOR_YUV2RGBA_NV21":
      case "COLOR_YUV2BGRA_YV12": case "COLOR_YUV2RGBA_YV12":
      case "COLOR_YUV2BGRA_IYUV": case "COLOR_YUV2RGBA_IYUV":
      case "COLOR_BayerBG2BGRA": case "COLOR_BayerGB2BGRA":
      case "COLOR_BayerGR2BGRA": case "COLOR_BayerRG2BGRA":
        return [1, 4];
      case "COLOR_BGR5552GRAY": case "COLOR_BGR5652GRAY":
        return [2, 1];
      case "COLOR_BGR5552BGR": case "COLOR_BGR5552RGB":
      case "COLOR_BGR5652BGR": case "COLOR_BGR5652RGB":
      case "COLOR_YUV2RGB_UYVY": case "COLOR_YUV2BGR_UYVY":
      case "COLOR_YUV2RGB_YUY2": case "COLOR_YUV2BGR_YUY2":
      case "COLOR_YUV2RGB_YVYU": case "COLOR_YUV2BGR_YVYU":
        return [2, 3];
      case "COLOR_BGR5552BGRA": case "COLOR_BGR5552RGBA":
      case "COLOR_BGR5652BGRA": case "COLOR_BGR5652RGBA":
      case "COLOR_YUV2RGBA_UYVY": case "COLOR_YUV2BGRA_UYVY":
      case "COLOR_YUV2RGBA_YUY2": case "COLOR_YUV2BGRA_YUY2":
      case "COLOR_YUV2RGBA_YVYU": case "COLOR_YUV2BGRA_YVYU":
        return [2, 4];
      case "COLOR_BGR2GRAY": case "COLOR_RGB2GRAY":
      case "COLOR_RGB2YUV_IYUV": case "COLOR_RGB2YUV_YV12":
      case "COLOR_BGR2YUV_IYUV": case "COLOR_BGR2YUV_YV12":
        return [3, 1];
      case "COLOR_BGR2BGR555": case "COLOR_BGR2BGR565":
      case "COLOR_RGB2BGR555": case "COLOR_RGB2BGR565":
        return [3, 2];
      case "COLOR_BGR2HLS": case "COLOR_BGR2HLS_FULL":
      case "COLOR_BGR2HSV": case "COLOR_BGR2HSV_FULL":
      case "COLOR_BGR2Lab": case "COLOR_BGR2Luv":
      case "COLOR_BGR2RGB": case "COLOR_BGR2XYZ":
      case "COLOR_BGR2YCrCb": case "COLOR_BGR2YUV":
      case "COLOR_HLS2BGR": case "COLOR_HLS2BGR_FULL":
      case "COLOR_HLS2RGB": case "COLOR_HLS2RGB_FULL":
      case "COLOR_HSV2BGR": case "COLOR_HSV2BGR_FULL":
      case "COLOR_HSV2RGB": case "COLOR_HSV2RGB_FULL":
      case "COLOR_Lab2BGR": case "COLOR_Lab2LBGR":
      case "COLOR_Lab2LRGB": case "COLOR_Lab2RGB":
      case "COLOR_LBGR2Lab": case "COLOR_LBGR2Luv":
      case "COLOR_LRGB2Lab": case "COLOR_LRGB2Luv":
      case "COLOR_Luv2BGR": case "COLOR_Luv2LBGR":
      case "COLOR_Luv2LRGB": case "COLOR_Luv2RGB":
      case "COLOR_RGB2HLS": case "COLOR_RGB2HLS_FULL":
      case "COLOR_RGB2HSV": case "COLOR_RGB2HSV_FULL":
      case "COLOR_RGB2Lab": case "COLOR_RGB2Luv":
      case "COLOR_RGB2XYZ": case "COLOR_RGB2YCrCb":
      case "COLOR_RGB2YUV": case "COLOR_XYZ2BGR":
      case "COLOR_XYZ2RGB": case "COLOR_YCrCb2BGR":
      case "COLOR_YCrCb2RGB": case "COLOR_YUV2BGR":
      case "COLOR_YUV2RGB":
        return [3, 3];
      case "COLOR_BGR2BGRA": case "COLOR_BGR2RGBA":
      case "CX_HLS2BGRA": case "CX_HLS2BGRA_FULL":
      case "CX_HLS2RGBA": case "CX_HLS2RGBA_FULL":
      case "CX_HSV2BGRA": case "CX_HSV2BGRA_FULL":
      case "CX_HSV2RGBA": case "CX_HSV2RGBA_FULL":
      case "CX_Lab2BGRA": case "CX_Lab2LBGRA":
      case "CX_Lab2LRGBA": case "CX_Lab2RGBA":
      case "CX_Luv2BGRA": case "CX_Luv2LBGRA":
      case "CX_Luv2LRGBA": case "CX_Luv2RGBA":
      case "CX_XYZ2BGRA": case "CX_XYZ2RGBA":
      case "CX_YCrCb2BGRA": case "CX_YCrCb2RGBA":
      case "CX_YUV2BGRA": case "CX_YUV2RGBA":
        return [3, 4];
      case "COLOR_BGRA2GRAY": case "COLOR_RGBA2GRAY":
      case "COLOR_RGBA2YUV_IYUV": case "COLOR_RGBA2YUV_YV12":
      case "COLOR_BGRA2YUV_IYUV": case "COLOR_BGRA2YUV_YV12":
        return [4, 1];
      case "COLOR_BGRA2BGR555": case "COLOR_BGRA2BGR565":
      case "COLOR_RGBA2BGR555": case "COLOR_RGBA2BGR565":
        return [4, 2];
      case "COLOR_BGRA2BGR": case "CX_BGRA2HLS":
      case "CX_BGRA2HLS_FULL": case "CX_BGRA2HSV":
      case "CX_BGRA2HSV_FULL": case "CX_BGRA2Lab":
      case "CX_BGRA2Luv": case "CX_BGRA2XYZ":
      case "CX_BGRA2YCrCb": case "CX_BGRA2YUV":
      case "CX_LBGRA2Lab": case "CX_LBGRA2Luv":
      case "CX_LRGBA2Lab": case "CX_LRGBA2Luv":
      case "COLOR_RGBA2BGR": case "CX_RGBA2HLS":
      case "CX_RGBA2HLS_FULL": case "CX_RGBA2HSV":
      case "CX_RGBA2HSV_FULL": case "CX_RGBA2Lab":
      case "CX_RGBA2Luv": case "CX_RGBA2XYZ":
      case "CX_RGBA2YCrCb": case "CX_RGBA2YUV":
        return [4, 3];
      case "COLOR_BGRA2RGBA":
        return [4, 4];
      default:
        console.error("Unknown conversion type");
        break;
      };
      return [0, 0];
  }

  function getMatType(chPair) {
    let dataType = "8U";  // now just support "8U" data type, we can set it as a param to extend the data type later.
    let mat1Type, mat2Type;
    if (chPair[0] === 0) {
      mat1Type = `CV_${dataType}C`;
    } else {
      mat1Type = `CV_${dataType}C${chPair[0].toString()}`;
    }
    if (chPair[1] === 0) {
      mat2Type = `CV_${dataType}C`;
    } else {
      mat2Type = `CV_${dataType}C${chPair[1].toString()}`;
    }
    return [mat1Type, mat2Type];
  }

  function addCvtColorCase(suite) {
    suite.add('cvtColor', function() {
      cv.cvtColor(mat1, mat2, mode, 0);
      }, {
        'setup': function() {
          let size = this.params.size;
          let matType = this.params.matType;
          let mode = cv[this.params.mode]%cv.COLOR_COLORCVT_MAX;
          let mat1 = new cv.Mat(size[1], size[0], cv[matType[0]]);
          let mat2 = new cv.Mat(size[1], size[0], cv[matType[1]]);
            },
        'teardown': function() {
          mat1.delete();
          mat2.delete();
        }
    });
  }

  function addCvtModeCase(suite, combination) {
    totalCaseNum += combination.length;
    for(let i = 0; i < combination.length; ++i) {
      let size = combination[i][0];
      let mode = combination[i][1];
      let chPair = getConversionInfo(mode);
      let matType = getMatType(chPair);
      let sizeArray = [size.width, size.height];

      addCvtColorCase(suite);
      // set init params
      let index = suite.length - 1;
      suite[index].params = {
        size: sizeArray,
        matType: matType,
        mode: mode
      };
    };
  }

  function addCvtModeBayerCase(suite, combination) {
    totalCaseNum += combination.length;
    for(let i = 0; i < combination.length; ++i) {
      let size = combination[i][0];
      let mode = combination[i][1];
      let chPair = getConversionInfo(mode);
      let matType = getMatType(chPair);
      let sizeArray = [size.width, size.height];

      addCvtColorCase(suite);
      // set init params
      let index = suite.length - 1;
      suite[index].params = {
        size: sizeArray,
        matType: matType,
        mode: mode
      };
    };
  }

  function addCvtMode2Case(suite, combination) {
    totalCaseNum += combination.length;
    for(let i = 0; i < combination.length; ++i) {
      let size = combination[i][0];
      let mode = combination[i][1];
      let chPair = getConversionInfo(mode);
      let matType = getMatType(chPair);
      let sizeArray = [size.width, size.height+size.height/2];

      addCvtColorCase(suite);
      // set init params
      let index = suite.length - 1;
      suite[index].params = {
        size: sizeArray,
        matType: matType,
        mode: mode
      };
    };
  }

  function addCvtMode3Case(suite, combination) {
    totalCaseNum += combination.length;
    for(let i = 0; i < combination.length; ++i) {
      let size = combination[i][0];
      let mode = combination[i][1];
      let chPair = getConversionInfo(mode);
      let matType = getMatType(chPair);
      let sizeArray = [size.width, size.height+size.height/2];

      addCvtColorCase(suite);
      // set init params
      let index = suite.length - 1;
      suite[index].params = {
        size: sizeArray,
        matType: matType,
        mode: mode
      };
    };
  }

  function addEdgeAwareBayerModeCase(suite, combination) {
    totalCaseNum += combination.length;
    for(let i = 0; i < combination.length; ++i) {
      let size = combination[i][0];
      let mode = combination[i][1];
      let chPair = getConversionInfo(mode);
      let matType = getMatType(chPair);
      let sizeArray = [size.width, size.height];

      addCvtColorCase(suite);
      // set init params
      let index = suite.length - 1;
      suite[index].params = {
        size: sizeArray,
        matType: matType,
        mode: mode
      };
    };
  }

  function decodeParams2Case(suite, params) {
    let sizeStr = (params.match(/[0-9]+/g) || []).slice(0, 2).toString();
    let mode = (params.match(/CX\_[A-z]+2[A-z]+/) || params.match(/COLOR\_[A-z]+2[A-z]+/) || []).toString();
    let size = cvtStr2cvSize(sizeStr);

    // check if the params match and add case
    for (let i = 0; i < combinations.length; ++i) {
      let combination = combinations[i];
      for (let j = 0; j < combination.length; ++j) {
        if (size === combination[j][0] && mode === combination[j][1]) {
          cvtFunc[i](suite, [combination[j]]);
        }
      }
    }
  }

  function log(message) {
    console.log(message);
    if (!isNodeJs) {
      logElement.innerHTML += `\n${'\t' + message}`;
    }
  }

  function setBenchmarkSuite(suite) {
    suite
    // add listeners
    .on('cycle', function(event) {
      ++currentCaseId;
      let params = event.target.params;
      let mode = params.mode;
      let size = params.size;
      log(`=== ${event.target.name} ${currentCaseId} ===`);
      log(`params: (${parseInt(size[0])}x${parseInt(size[1])}, ${mode})`);
      log('elapsed time:' +String(event.target.times.elapsed*1000)+' ms');
      log('mean time:' +String(event.target.stats.mean*1000)+' ms');
      log('stddev time:' +String(event.target.stats.deviation*1000)+' ms');
      log(String(event.target));
    })
    .on('error', function(event) { log(`test case ${event.target.name} failed`); })
    .on('complete', function(event) {
      log(`\n ###################################`)
      log(`Finished testing ${event.currentTarget.length} cases \n`);
      if (!isNodeJs) {
        runButton.removeAttribute('disabled');
        runButton.setAttribute('class', 'btn btn-primary');
        runButton.innerHTML = 'Run';
      }
    });
  }

  function genBenchmarkCase(paramsContent) {
    let suite = new Benchmark.Suite;
    totalCaseNum = 0;
    currentCaseId = 0;
    if (/\([0-9]+x[0-9]+,[\ ]*\w+\)/g.test(paramsContent.toString())) {
      let params = paramsContent.toString().match(/\([0-9]+x[0-9]+,[\ ]*\w+\)/g)[0];
      decodeParams2Case(suite, params);
    } else {
      log("no filter or getting invalid params, run all the cases");
      addCvtModeCase(suite, combiCvtMode);
      addCvtModeBayerCase(suite, combiCvtModeBayer);
      addCvtMode2Case(suite, combiCvtMode2);
      addCvtMode3Case(suite, combiCvtMode3);
    }
    setBenchmarkSuite(suite);
    log(`Running ${totalCaseNum} tests from CvtColor`);
    suite.run({ 'async': true }); // run the benchmark
  }



  // init
  let cvtFunc = [addCvtModeCase, addCvtModeBayerCase, addCvtMode2Case, addCvtMode3Case];//, addEdgeAwareBayerModeCase];
  let combinations = [combiCvtMode, combiCvtModeBayer, combiCvtMode2, combiCvtMode3];//, combiEdgeAwareBayer];

  // set test filter params
  if (isNodeJs) {
    const args = process.argv.slice(2);
    let paramsContent = '';
    if (/--test_param_filter=\([0-9]+x[0-9]+,[\ ]*\w+\)/g.test(args.toString())) {
      paramsContent = args.toString().match(/\([0-9]+x[0-9]+,[\ ]*\w+\)/g)[0];
    }
    genBenchmarkCase(paramsContent);
  } else {
    runButton.onclick = function()　{
      let paramsContent = paramsElement.value;
      genBenchmarkCase(paramsContent);
      if (totalCaseNum !== 0) {
        runButton.setAttribute("disabled", "disabled");
        runButton.setAttribute('class', 'btn btn-primary disabled');
        runButton.innerHTML = "Running";
      }
    }
  }
};