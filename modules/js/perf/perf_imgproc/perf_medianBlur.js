const isNodeJs = (typeof window) === 'undefined'? true : false;

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
      global.combine = HelpFunc.combine;
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

    const MedianBlurSize = [cvSize.szODD, cvSize.szQVGA, cvSize.szVGA, cvSize.sz720p];
    const MedianBlurType = ["CV_8UC1", "CV_8UC4", "CV_16UC1", "CV_16SC1", "CV_32FC1"];
    const combiMedianBlur = combine(MedianBlurSize, MedianBlurType, [3,5]);

    function addMedianBlurCase(suite, type) {
        suite.add('medianBlur', function() {
            cv.medianBlur(src, dst, ksize);
          }, {
              'setup': function() {
                let size = this.params.size;
                let matType = cv[this.params.matType];
                let ksize = this.params.ksize;
                let src = new cv.Mat(size, matType);
                let dst = new cv.Mat(size, matType);
                },
              'teardown': function() {
                src.delete();
                dst.delete();
              }
          });
    }

    function addMedianBlurModeCase(suite, combination, type) {
      totalCaseNum += combination.length;
      for (let i = 0; i < combination.length; ++i) {
        let size =  combination[i][0];
        let matType = combination[i][1];
        let ksize = combination[i][2];

        let params = {size: size, matType:matType, ksize: ksize};
        addKernelCase(suite, params, type, addMedianBlurCase);
      }
    }

    function genBenchmarkCase(paramsContent) {
      let suite = new Benchmark.Suite;
      totalCaseNum = 0;
      currentCaseId = 0;

      if (/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*(3|5)\)/g.test(paramsContent.toString())) {
          let params = paramsContent.toString().match(/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*(3|5)\)/g)[0];
          let paramObjs = [];
          paramObjs.push({name:"size", value:"", reg:[""], index:0});
          paramObjs.push({name:"matType", value:"", reg:["/CV\_[0-9]+[FSUfsu]C[0-9]/"], index:1});
          paramObjs.push({name:"ksize", value: "", reg:["/\\b[0-9]\\b/"], index:2});
          let locationList = decodeParams2Case(params, paramObjs, medianBlurCombinations);

          for (let i = 0; i < locationList.length; i++){
              let first = locationList[i][0];
              let second = locationList[i][1];
              addMedianBlurModeCase(suite, [medianBlurCombinations[first][second]], first);
            }
      } else {
        log("no filter or getting invalid params, run all the cases");
        addMedianBlurModeCase(suite, combiMedianBlur, 0);
      }
      setBenchmarkSuite(suite, "medianBlur", currentCaseId);
      log(`Running ${totalCaseNum} tests from medianBlur`);
      suite.run({ 'async': true }); // run the benchmark
  }

    let medianBlurCombinations = [combiMedianBlur];

    if (isNodeJs) {
      const args = process.argv.slice(2);
      let paramsContent = '';
      if (/--test_param_filter=\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*(3|5)\)/g.test(args.toString())) {
        paramsContent = args.toString().match(/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*(3|5)\)/g)[0];
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