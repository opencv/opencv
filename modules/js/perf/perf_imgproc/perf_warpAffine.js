var isNodeJs = (typeof window) === 'undefined'? true : false;

if　(isNodeJs)　{
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
      global.fillGradient = HelpFunc.fillGradient;
      global.smoothBorder = HelpFunc.smoothBorder;
      global.combine = HelpFunc.combine;
      global.log = HelpFunc.log;
      global.decodeParams2Case = HelpFunc.decodeParams2Case;
      global.setBenchmarkSuite = HelpFunc.setBenchmarkSuite;
      global.addKernelCase = HelpFunc.addKernelCase
      global.cvSize = Base.getCvSize();
    } else {
      enableButton();
      cvSize = getCvSize();
    }
    let totalCaseNum, currentCaseId;

    const WarpAffineSize = [cvSize.szVGA, cvSize.sz720p, cvSize.sz1080p];
    const InterType = ["INTER_NEAREST", "INTER_LINEAR"];
    const BorderMode = ["BORDER_CONSTANT", "BORDER_REPLICATE"]
    const combiWarpAffine = combine(WarpAffineSize, InterType, BorderMode);

    function addWarpAffineCase(suite, type) {
        suite.add('warpAffine', function() {
            cv.warpAffine(src, dst, warpMat, sz, interType, borderMode, borderColor);
          }, {
              'setup': function() {
                let sz = this.params.size;
                let interType = cv[this.params.interType];
                let borderMode = cv[this.params.borderMode];
                let srcSize = new cv.Size(512, 512);

                let borderColor = new cv.Scalar.all(150);
                let src = new cv.Mat(srcSize, cv.CV_8UC4);
                let dst = new cv.Mat(sz, cv.CV_8UC4);
                fillGradient(cv, src);
                if (borderMode == cv.BORDER_CONSTANT) {
                  smoothBorder(cv, src, borderMode, 1);
                }

                let point = new cv.Point(src.cols/2.0, src.rows/2.0);
                let warpMat = cv.getRotationMatrix2D(point, 30.0, 2.2);
                },
              'teardown': function() {
                src.delete();
                dst.delete();
                warpMat.delete();
              }
          });
    }

    function addWarpAffineModeCase(suite, combination, type) {
      totalCaseNum += combination.length;
      for (let i = 0; i < combination.length; ++i) {
        let size =  combination[i][0];
        let interType = combination[i][1];
        let borderMode = combination[i][2];

        let params = {size: size, interType:interType, borderMode:borderMode};
        addKernelCase(suite, params, type, addWarpAffineCase);
      }
    }

    function genBenchmarkCase(paramsContent) {
      let suite = new Benchmark.Suite;
      totalCaseNum = 0;
      currentCaseId = 0;

      if (/\([0-9]+x[0-9]+,[\ ]*INTER\_\w+,[\ ]*BORDER\_\w+\)/g.test(paramsContent.toString())) {
          let params = paramsContent.toString().match(/\([0-9]+x[0-9]+,[\ ]*INTER\_\w+,[\ ]*BORDER\_\w+\)/g)[0];
          let paramObjs = [];
          paramObjs.push({name:"size", value:"", reg:[""], index:0});
          paramObjs.push({name:"interType", value: "", reg:["/INTER\_\\w+/"], index:1});
          paramObjs.push({name:"borderMode", value: "", reg:["/BORDER\_\\w+/"], index:2});
          let locationList = decodeParams2Case(params, paramObjs, warpAffineCombinations);

          for (let i = 0; i < locationList.length; i++){
              let first = locationList[i][0];
              let second = locationList[i][1];
              addWarpAffineModeCase(suite, [warpAffineCombinations[first][second]], first);
            }
      } else {
        log("no filter or getting invalid params, run all the cases");
        addWarpAffineModeCase(suite, combiWarpAffine, 0);
      }
      setBenchmarkSuite(suite, "warpAffine", currentCaseId);
      log(`Running ${totalCaseNum} tests from warpAffine`);
      suite.run({ 'async': true }); // run the benchmark
  }

    let warpAffineCombinations = [combiWarpAffine];

    if (isNodeJs) {
      const args = process.argv.slice(2);
      let paramsContent = '';
      if (/--test_param_filter=\([0-9]+x[0-9]+,[\ ]*INTER\_\w+,[\ ]*BORDER\_\w+\)/g.test(args.toString())) {
        paramsContent = args.toString().match(/\([0-9]+x[0-9]+,[\ ]*INTER\_\w+,[\ ]*BORDER\_\w+\)/g)[0];
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