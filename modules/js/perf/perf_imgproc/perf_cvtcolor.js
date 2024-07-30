var isNodeJs = (typeof window) === 'undefined'? true : false;

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

function perf() {

  console.log('opencv.js loaded');
  if (isNodeJs) {
    global.cv = cv;
    global.combine = HelpFunc.combine;
    global.constructMode = HelpFunc.constructMode;
    global.log = HelpFunc.log;
    global.decodeParams2Case = HelpFunc.decodeParams2Case;
    global.setBenchmarkSuite = HelpFunc.setBenchmarkSuite;
    global.addKernelCase = HelpFunc.addKernelCase;
    global.cvSize = Base.getCvSize();
  } else {
    enableButton();
    cvSize = getCvSize();
  }
  let totalCaseNum, currentCaseId;

  // extra color conversions supported implicitly
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

  // didn't support 16u and 32f perf tests according to
  // https://github.com/opencv/opencv/commit/4e679e1cc5b075ec006b29a58b4fe117523fba1d
  function constructCvtMode16U() {
    let cvtMode16U = [];
    cvtMode16U = cvtMode16U.concat(constructMode("COLOR_", "BGR", ["BGRA", "GRAY", "RGB", "RGBA", "XYZ", "YCrCb", "YUV"]));
    cvtMode16U = cvtMode16U.concat(constructMode("COLOR_", "BGRA", ["BGR", "GRAY", "RGBA"]));
    cvtMode16U = cvtMode16U.concat(constructMode("CX_", "BGRA", ["XYZ", "YCrCb", "YUV"]));
    cvtMode16U = cvtMode16U.concat(constructMode("COLOR_", "GRAY", ["BGR", "BGRA"]));
    cvtMode16U = cvtMode16U.concat(constructMode("COLOR_", "RGB", ["GRAY", "XYZ", "YCrCb", "YUV"]));
    cvtMode16U = cvtMode16U.concat(constructMode("COLOR_", "RGBA", ["BGR", "GRAY"]));
    cvtMode16U = cvtMode16U.concat(constructMode("CX_", "RGBA", ["XYZ", "YCrCb", "YUV"]));
    cvtMode16U = cvtMode16U.concat(constructMode("COLOR_", "XYZ", ["BGR", "RGB"]));
    cvtMode16U = cvtMode16U.concat(constructMode("CX_", "XYZ", ["BGRA", "RGBA"]));
    cvtMode16U = cvtMode16U.concat(constructMode("COLOR_", "YCrCb", ["BGR", "RGB"]));
    cvtMode16U = cvtMode16U.concat(constructMode("CX_", "YCrCb", ["BGRA", "RGBA"]));
    cvtMode16U = cvtMode16U.concat(constructMode("COLOR_", "YUV", ["BGR", "RGB"]));
    cvtMode16U = cvtMode16U.concat(constructMode("CX_", "YUV", ["BGRA", "RGBA"]));

    return cvtMode16U;
  }

  const CvtMode16U = constructCvtMode16U();

  const CvtMode16USize = [cvSize.szODD, cvSize.szVGA, cvSize.sz1080p];
  const combiCvtMode16U = combine(CvtMode16USize, CvtMode16U);

  function constructCvtMode32F(source) {
    let cvtMode32F = source;
    cvtMode32F = cvtMode32F.concat(constructMode("COLOR_", "BGR", ["HLS", "HLS_FULL", "HSV", "HSV_FULL", "Lab", "Luv"]));
    cvtMode32F = cvtMode32F.concat(constructMode("CX_", "BGRA", ["HLS", "HLS_FULL", "HSV", "HSV_FULL", "Lab", "Luv"]));
    cvtMode32F = cvtMode32F.concat(constructMode("COLOR_", "HLS", ["BGR", "BGR_FULL", "RGB", "RGB_FULL"]));
    cvtMode32F = cvtMode32F.concat(constructMode("CX_", "HLS", ["BGRA", "BGRA_FULL", "RGBA", "RGBA_FULL"]));
    cvtMode32F = cvtMode32F.concat(constructMode("COLOR_", "HSV", ["BGR", "BGR_FULL", "RGB", "RGB_FULL"]));
    cvtMode32F = cvtMode32F.concat(constructMode("CX_", "HSV", ["BGRA", "BGRA_FULL", "RGBA", "RGBA_FULL"]));
    cvtMode32F = cvtMode32F.concat(constructMode("COLOR_", "Lab", ["BGR", "LBGR", "RGB", "LRGB"]));
    cvtMode32F = cvtMode32F.concat(constructMode("CX_", "Lab", ["BGRA", "LBGRA", "RGBA", "LRGBA"]));
    cvtMode32F = cvtMode32F.concat(constructMode("COLOR_", "Luv", ["BGR", "LBGR", "RGB", "LRGB"]));
    cvtMode32F = cvtMode32F.concat(constructMode("CX_", "Luv", ["BGRA", "LBGRA", "RGBA", "LRGBA"]));
    cvtMode32F = cvtMode32F.concat(constructMode("COLOR_", "LBGR", ["Lab", "Luv"]));
    cvtMode32F = cvtMode32F.concat(constructMode("CX_", "LBGRA", ["Lab", "Luv"]));
    cvtMode32F = cvtMode32F.concat(constructMode("COLOR_", "LRGB", ["Lab", "Luv"]));
    cvtMode32F = cvtMode32F.concat(constructMode("CX_", "LRGBA", ["Lab", "Luv"]));
    cvtMode32F = cvtMode32F.concat(constructMode("COLOR_", "RGB", ["HLS", "HLS_FULL", "HSV", "HSV_FULL", "Lab", "Luv"]));
    cvtMode32F = cvtMode32F.concat(constructMode("CX_", "RGBA", ["HLS", "HLS_FULL", "HSV", "HSV_FULL", "Lab", "Luv"]));

    return cvtMode32F;
  }

  const CvtMode32F = constructCvtMode32F(CvtMode16U);

  const CvtMode32FSize = [cvSize.szODD, cvSize.szVGA, cvSize.sz1080p];
  const combiCvtMode32F = combine(CvtMode32FSize, CvtMode32F);

  function constructeCvtMode(source) {
    let cvtMode = source
    cvtMode = cvtMode.concat(constructMode("COLOR_", "BGR", ["BGR555", "BGR565"]));
    cvtMode = cvtMode.concat(constructMode("COLOR_", "BGR555", ["BGR", "BGRA", "GRAY", "RGB", "RGBA"]));
    cvtMode = cvtMode.concat(constructMode("COLOR_", "BGR565", ["BGR", "BGRA", "GRAY", "RGB", "RGBA"]));
    cvtMode = cvtMode.concat(constructMode("COLOR_", "BGRA", ["BGR555", "BGR565"]));
    cvtMode = cvtMode.concat(constructMode("COLOR_", "GRAY", ["BGR555", "BGR565"]));
    cvtMode = cvtMode.concat(constructMode("COLOR_", "RGB", ["BGR555", "BGR565"]));
    cvtMode = cvtMode.concat(constructMode("COLOR_", "RGBA", ["BGR555", "BGR565"]));

    return cvtMode;
  }

  const CvtMode = constructeCvtMode(CvtMode32F);

  const CvtModeSize = [cvSize.szODD, cvSize.szVGA, cvSize.sz1080p];
  // combiCvtMode permute size and mode
  const combiCvtMode = combine(CvtModeSize, CvtMode);

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

  function addCvtColorCase(suite, type) {
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

  function addCvtModeCase(suite, combination, type) {
    totalCaseNum += combination.length;
    for(let i = 0; i < combination.length; ++i) {
      let size = combination[i][0];
      let mode = combination[i][1];
      let chPair = getConversionInfo(mode);
      let matType = getMatType(chPair);
      let sizeArray;
      if (type == 0) {
        sizeArray = [size.width, size.height];
      } else {
        sizeArray = [size.width, size.height+size.height/2];
      }
      let params = {size:sizeArray, matType: matType, mode: mode};
      addKernelCase(suite, params, type, addCvtColorCase);
    };
  }

  function genBenchmarkCase(paramsContent) {
    let suite = new Benchmark.Suite;
    totalCaseNum = 0;
    currentCaseId = 0;
    if (/\([0-9]+x[0-9]+,[\ ]*\w+\)/g.test(paramsContent.toString())) {
      let params = paramsContent.toString().match(/\([0-9]+x[0-9]+,[\ ]*\w+\)/g)[0];
      let paramObjs = [];
      paramObjs.push({name:"mode", value:"", reg:["/CX\_[A-z]+2[A-z]+/", "/COLOR\_[A-z]+2[A-z]+/"], index:1});
      paramObjs.push({name:"size", value:"", reg:[""], index:0});

      let locationList = decodeParams2Case(params, paramObjs,combinations);
      for (let i = 0; i < locationList.length; i++){
        let first = locationList[i][0];
        let second = locationList[i][1];
        if (first < 2) {
          addCvtModeCase(suite, [combinations[first][second]], 0);
        } else {
          addCvtModeCase(suite, [combinations[first][second]], 1);
        }
      }
    } else {
      log("no filter or getting invalid params, run all the cases");
      addCvtModeCase(suite, combiCvtMode, 0);
      addCvtModeCase(suite, combiCvtModeBayer, 0);
      addCvtModeCase(suite, combiCvtMode2, 1);
      addCvtModeCase(suite, combiCvtMode3, 1);
    }
    setBenchmarkSuite(suite, "cvtcolor", currentCaseId);
    log(`Running ${totalCaseNum} tests from CvtColor`);
    suite.run({ 'async': true }); // run the benchmark
  }

  // init
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
        disableButton();
      }
    }
  }
};

async function main() {
  if (cv instanceof Promise) {
    cv = await cv;
    perf();
  } else {
    cv.onRuntimeInitialized = perf;
  }
}

main();