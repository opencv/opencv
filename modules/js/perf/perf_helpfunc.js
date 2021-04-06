const isNodeJs = (typeof window) === 'undefined'? true : false;

if(isNodeJs) {
  var Base = require("./base");
  global.getCvSize = Base.getCvSize;
}

var fillGradient = function(cv, img, delta=5) {
  let ch = img.channels();
  console.assert(!img.empty() && img.depth() == cv.CV_8U && ch <= 4);

  let n = 255 / delta;
  for(let r = 0; r < img.rows; ++r) {
    let kR = r % (2*n);
    let valR = (kR<=n) ? delta*kR : delta*(2*n-kR);
    for(let c = 0; c < img.cols; ++c) {
        let kC = c % (2*n);
        let valC = (kC<=n) ? delta*kC : delta*(2*n-kC);
        let vals = [valR, valC, 200*r/img.rows, 255];
        let p = img.ptr(r, c);
        for(let i = 0; i < ch; ++i) p[i] = vals[i];
    }
  }
}

var smoothBorder = function(cv, img, color, delta=5) {
  let ch = img.channels();
  console.assert(!img.empty() && img.depth() == cv.CV_8U && ch <= 4);

  let n = 100/delta;
  let nR = Math.min(n, (img.rows+1)/2);
  let nC = Math.min(n, (img.cols+1)/2);
  let s = new cv.Scalar();

  for (let r = 0; r < nR; r++) {
    let k1 = r*delta/100.0, k2 = 1-k1;
    for(let c = 0; c < img.cols; c++) {
      let view = img.ptr(r, c);
      for(let i = 0; i < ch; i++) s[i] = view[i];
      for(let i = 0; i < ch; i++) view[i] = s[i]*k1 + color[i] * k2;
    }
    for(let c=0; c < img.cols; c++) {
      let view = img.ptr(img.rows-r-1, c);
      for(let i = 0; i < ch; i++) s[i] = view[i];
      for(let i = 0; i < ch; i++) view[i] = s[i]*k1 + color[i] * k2;
    }
  }
  for (let r = 0; r < img.rows; r++) {
    for(let c = 0; c < nC; c++) {
      let k1 = c*delta/100.0, k2 = 1-k1;
      let view = img.ptr(r, c);
      for(let i = 0; i < ch; i++) s[i] = view[i];
      for(let i = 0; i < ch; i++) view[i] = s[i]*k1 + color[i] * k2;
    }
    for(let c = 0; c < n; c++) {
      let k1 = c*delta/100.0, k2 = 1-k1;
      let view = img.ptr(r, img.cols-c-1);
      for(let i = 0; i < ch; i++) s[i] = view[i];
      for(let i = 0; i < ch; i++) view[i] = s[i]*k1 + color[i] * k2;
    }
  }
}

var cvtStr2cvSize = function(strSize) {
  let size;
  let cvSize = getCvSize();

  switch(strSize) {
    case "127,61": size = cvSize.szODD;break;
    case '320,240': size = cvSize.szQVGA;break;
    case '640,480': size = cvSize.szVGA;break;
    case '800,600': size = cvSize.szSVGA;break;
    case '960,540': size = cvSize.szqHD;break;
    case '1024,768': size = cvSize.szXGA;break;
    case '1280,720': size = cvSize.sz720p;break;
    case '1280,1024': size = cvSize.szSXGA;break;
    case '1920,1080': size = cvSize.sz1080p;break;
    case "130,60": size = cvSize.sz130x60;break;
    case '213,120': size = cvSize.sz213x120;break;
    default: console.error("unsupported size for this case");
  }
  return size;
}

var combine = function() {
  let result = [[]];
  for (let i = 0; i < arguments.length; ++i) {
    result = permute(result, arguments[i]);
  }
  return result;
}

function permute (source, target) {
  let result = [];
  for (let i = 0; i < source.length; ++i) {
    for (let j = 0; j < target.length; ++j) {
      let tmp = source[i].slice();
      tmp.push(target[j]);
      result.push(tmp);
    }
  }
  return result;
}

var constructMode = function (startStr, sChannel, dChannel) {
  let modeList = []
  for (let j in dChannel) {
    modeList.push(startStr+sChannel+"2"+dChannel[j])
  }
  return modeList;
}

var enableButton = function () {
  runButton.removeAttribute('disabled');
  runButton.setAttribute('class', 'btn btn-primary');
  runButton.innerHTML = 'Run';
}

var disableButton = function () {
  runButton.setAttribute("disabled", "disabled");
  runButton.setAttribute('class', 'btn btn-primary disabled');
  runButton.innerHTML = "Running";
}

var log = function (message) {
  console.log(message);
  if (!isNodeJs) {
    logElement.innerHTML += `\n${'\t' + message}`;
  }
}

var addKernelCase = function (suite, params, type, kernelFunc) {
  kernelFunc(suite, type);
  let index = suite.length - 1;
  suite[index].params = params;
}

function constructParamLog(params, kernel) {
  let paramLog = '';
  if (kernel == "cvtcolor") {
    let mode = params.mode;
    let size = params.size;
    paramLog = `params: (${parseInt(size[0])}x${parseInt(size[1])}, ${mode})`;
  } else if (kernel == "resize") {
    let matType = params.matType;
    let size1 = params.from;
    let size2 = params.to;
    paramLog = `params: (${matType},${parseInt(size1.width)}x${parseInt(size1.height)},`+
    `${parseInt(size2.width)}x${parseInt(size2.height)})`;
  } else if (kernel == "threshold") {
    let matSize = params.matSize;
    let matType = params.matType;
    let threshType = params.threshType;
    paramLog = `params: (${parseInt(matSize.width)}x${parseInt(matSize.height)},`+
    `${matType},${threshType})`;
  } else if (kernel == "sobel") {
    let size = params.size;
    let ddepth = params.ddepth;
    let dxdy = params.dxdy;
    let ksize = params.ksize;
    let borderType = params.borderType;
    paramLog = `params: (${parseInt(size[0])}x${parseInt(size[1])},`+
    `${ddepth},${dxdy},${borderType}, ksize:${ksize})`;
  } else if (kernel == "filter2d") {
    let size = params.size;
    let ksize = params.ksize;
    let borderMode = params.borderMode;
    paramLog = `params: (${parseInt(size.width)}x${parseInt(size.height)},`+
    `${ksize},${borderMode})`;
  } else if (kernel == "scharr") {
    let size = params.size;
    let ddepth = params.ddepth;
    let dxdy = params.dxdy;
    let borderType = params.borderType;
    paramLog = `params: (${parseInt(size[0])}x${parseInt(size[1])},`+
    `${ddepth},${dxdy},${borderType})`;
  } else if (kernel == "gaussianBlur" || kernel == "blur") {
    let size = params.size;
    let matType = params.matType;
    let borderType = params.borderType;
    let ksize = params.ksize;
    paramLog = `params: (${parseInt(size.width)}x${parseInt(size.height)},`+
    `${matType},${borderType}, ksize: (${ksize}x${ksize}))`;
  } else if (kernel == "medianBlur") {
    let size = params.size;
    let matType = params.matType;
    let ksize = params.ksize;
    paramLog = `params: (${parseInt(size.width)}x${parseInt(size.height)},`+
    `${matType}, ksize: ${ksize})`;
  } else if (kernel == "erode" || kernel == "dilate" || kernel == "pyrDown") {
    let size = params.size;
    let matType = params.matType;
    paramLog = `params: (${parseInt(size.width)}x${parseInt(size.height)},`+
    `${matType})`;
  } else if (kernel == "remap") {
    let size = params.size;
    let matType = params.matType;
    let mapType = params.mapType;
    let interType = params.interType;
    paramLog = `params: (${parseInt(size.width)}x${parseInt(size.height)},`+
    `${matType}, ${mapType}, ${interType})`;
  } else if (kernel == "warpAffine" || kernel == "warpPerspective") {
    let size = params.size;
    let interType = params.interType;
    let borderMode = params.borderMode;
    paramLog = `params: (${parseInt(size.width)}x${parseInt(size.height)},`+
    `${interType}, ${borderMode})`;
  }
  return paramLog;
}

var setBenchmarkSuite =  function (suite, kernel, currentCaseId) {
  suite
  // add listeners
  .on('cycle', function(event) {
    ++currentCaseId;
    let params = event.target.params;
    paramLog = constructParamLog(params, kernel);

    log(`=== ${event.target.name} ${currentCaseId} ===`);
    log(paramLog);
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

var decodeParams2Case = function(paramContent, paramsList, combinations) {
  let sizeString = (paramContent.match(/[0-9]+x[0-9]+/g) || []).toString();
  let sizes = (sizeString.match(/[0-9]+/g) || []);
  let paramSize = paramsList.length;
  let paramObjs = []
  let sizeCount = 0;
  for (let i = 0; i < paramSize; i++) {
      let param = paramsList[i];
      let paramName = param.name;
      let paramValue = param.value;
      let paramReg = param.reg;
      let paramIndex = param.index;

      if(paramValue != "") {
        paramObjs.push({name: paramName, value: paramValue, index: paramIndex});
      } else if (paramName.startsWith('size')) {
        let sizeStr = sizes.slice(sizeCount, sizeCount+2).toString();
        paramValue = cvtStr2cvSize(sizeStr);
        sizeCount += 2;
        paramObjs.push({name: paramName, value: paramValue, index: paramIndex});
      } else {
        for (let index in paramReg) {
          let reg = eval(paramReg[index]);
          if ('loc' in param) {
            paramValue = (paramContent.match(reg) || [])[param.loc].toString();
          } else {
            paramValue = (paramContent.match(reg) || []).toString();
          }

          if (paramValue != "") {
            paramObjs.push({name: paramName, value: paramValue, index: paramIndex});
            break;
          }
        }
      }
  }

  let location = [];
  for (let i = 0; i < combinations.length; ++i) {
    let combination = combinations[i];
    for (let j = 0; j < combination.length; ++j) {
      if (judgeCombin(combination[j], paramObjs)) {
        location.push([i,j]);
      }
    }
  }
  return location;
}

function judgeCombin(combination, paramObjs) {
  for (let i =0; i < paramObjs.length; i++) {
    if (paramObjs[i].value != combination[paramObjs[i].index]){
      return false;
    }
  }
  return true;
}


if (typeof window === 'undefined') {
  exports.enableButton = enableButton;
  exports.disableButton = disableButton;
  exports.fillGradient = fillGradient;
  exports.smoothBorder = smoothBorder;
  exports.cvtStr2cvSize = cvtStr2cvSize;
  exports.combine = combine;
  exports.constructMode = constructMode;
  exports.log = log;
  exports.decodeParams2Case = decodeParams2Case;
  exports.setBenchmarkSuite = setBenchmarkSuite;
  exports.addKernelCase = addKernelCase;
}