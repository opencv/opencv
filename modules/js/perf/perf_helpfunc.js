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

var cvtStr2cvSize = function(strSize) {
  let size;
  switch(strSize) {
    case "127,61": size = cvSize.szODD;break;
    case '320,240': size = cvSize.szQVGA;break;
    case '640,480': size = cvSize.szVGA;break;
    case '960,540': size = cvSize.szqHD;break;
    case '1280,720': size = cvSize.sz720p;break;
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

if (typeof window === 'undefined') {
  exports.fillGradient = fillGradient;
  exports.cvtStr2cvSize = cvtStr2cvSize;
  exports.combine = combine;
}