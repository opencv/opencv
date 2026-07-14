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

    const BlurSize = [cvSize.szODD, cvSize.szQVGA, cvSize.szVGA, cvSize.sz720p];
    const Blur5x16Size = [cvSize.szVGA, cvSize.sz720p];
    const BlurType = ["CV_8UC1", "CV_8UC4", "CV_16UC1", "CV_16SC1", "CV_32FC1"];
    const BlurType5x5 = ["CV_8UC1", "CV_8UC4", "CV_16UC1", "CV_16SC1", "CV_32FC1", "CV_32FC3"];
    const BorderType3x3 = ["BORDER_REPLICATE", "BORDER_CONSTANT"];
    const BorderTypeAll = ["BORDER_REPLICATE", "BORDER_CONSTANT", "BORDER_REFLECT", "BORDER_REFLECT101"];

    const combiBlur3x3 = combine(BlurSize, BlurType, BorderType3x3);
    const combiBlur16x16 = combine(Blur5x16Size, BlurType, BorderTypeAll);
    const combiBlur5x5 = combine(Blur5x16Size, BlurType5x5, BorderTypeAll);

    function addBlurCase(suite, type) {
        suite.add('blur', function() {
            cv.blur(src, dst, ksize, new cv.Point(-1,-1), borderType);
          }, {
              'setup': function() {
                let size = this.params.size;
                let matType = cv[this.params.matType];
                let borderType = cv[this.params.borderType];
                let ksizeNum = this.params.ksize;
                let ksize = new cv.Size(ksizeNum, ksizeNum);
                let src = new cv.Mat(size, matType);
                let dst = new cv.Mat(size, matType);
                },
              'teardown': function() {
                src.delete();
                dst.delete();
              }
          });
    }

    function addBlurModeCase(suite, combination, type) {
      totalCaseNum += combination.length;
      for (let i = 0; i < combination.length; ++i) {
        let size =  combination[i][0];
        let matType = combination[i][1];
        let borderType = combination[i][2];
        let ksizeArray = [3, 16, 5];

        let params = {size: size, matType:matType, ksize: ksizeArray[type], borderType:borderType};
        addKernelCase(suite, params, type, addBlurCase);
      }
    }

    function genBenchmarkCase(paramsContent) {
      let suite = new Benchmark.Suite;
      totalCaseNum = 0;
      currentCaseId = 0;

      if (/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*BORDER\_\w+\)/g.test(paramsContent.toString())) {
          let params = paramsContent.toString().match(/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*BORDER\_\w+\)/g)[0];
          let paramObjs = [];
          paramObjs.push({name:"size", value:"", reg:[""], index:0});
          paramObjs.push({name:"matType", value:"", reg:["/CV\_[0-9]+[FSUfsu]C[0-9]/"], index:1});
          paramObjs.push({name:"borderMode", value: "", reg:["/BORDER\_\\w+/"], index:2});
          let locationList = decodeParams2Case(params, paramObjs,blurCombinations);

          for (let i = 0; i < locationList.length; i++){
              let first = locationList[i][0];
              let second = locationList[i][1];
              addBlurModeCase(suite, [blurCombinations[first][second]], first);
            }
      } else {
        log("no filter or getting invalid params, run all the cases");
        addBlurModeCase(suite, combiBlur3x3, 0);
        addBlurModeCase(suite, combiBlur16x16, 1);
        addBlurModeCase(suite, combiBlur5x5, 2);
      }
      setBenchmarkSuite(suite, "blur", currentCaseId);
      log(`Running ${totalCaseNum} tests from blur`);
      suite.run({ 'async': true }); // run the benchmark
  }

    let blurCombinations = [combiBlur3x3, combiBlur16x16, combiBlur5x5];

    if (isNodeJs) {
      const args = process.argv.slice(2);
      let paramsContent = '';
      if (/--test_param_filter=\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*BORDER\_\w+\)/g.test(args.toString())) {
        paramsContent = args.toString().match(/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*BORDER\_\w+\)/g)[0];
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