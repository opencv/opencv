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

    const Filter2dSize = [cvSize.szQVGA, cvSize.sz1080p];
    const Filter2dKsize = ["3", "5"];
    const Filter2dBorderMode = ["BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT_101"];
    const DISABLED_Filter2dBorderMode = ["BORDER_CONSTANT", "BORDER_REPLICATE"];
    const combiFilter2dCase = combine(Filter2dSize, Filter2dKsize, Filter2dBorderMode);
    const combiDISABLEDFilter2dCase = combine(Filter2dSize, Filter2dKsize, DISABLED_Filter2dBorderMode);

    function addFilter2dCase(suite, type) {
        suite.add('filter2d', function() {
            cv.filter2D(src, dst, cv.CV_8UC4, kernel, new cv.Point(1, 1), 0.0, borderMode);
          }, {
              'setup': function() {
                let size = this.params.size;
                let ksize = parseInt(this.params.ksize);
                let borderMode = cv[this.params.borderMode];

                let src = new cv.Mat(size, cv.CV_8UC4);
                let dst = new cv.Mat(size, cv.CV_8UC4);
                let kernelElement = [];
                for (let i = 0; i < ksize*ksize; i++) {
                    let randNum = Math.random();
                    kernelElement.push(-3.0+randNum*13.0);
                }
                let kernel = cv.matFromArray(ksize, ksize, cv.CV_32FC1, kernelElement);
                },
              'teardown': function() {
                src.delete();
                dst.delete();
              }
          });
    }

    function addFilter2dModeCase(suite, combination, type) {
      totalCaseNum += combination.length;
      for (let i = 0; i < combination.length; ++i) {
        let size =  combination[i][0];
        let ksize = combination[i][1];
        let borderMode = combination[i][2];
        let params = {size: size, ksize: ksize, borderMode:borderMode};
        addKernelCase(suite, params, type, addFilter2dCase);
      }
    }

    function genBenchmarkCase(paramsContent) {
        let suite = new Benchmark.Suite;
        totalCaseNum = 0;
        currentCaseId = 0;

        if (/\([0-9]+x[0-9]+,[\ ]*[0-9],[\ ]*BORDER\_\w+\)/g.test(paramsContent.toString())) {
            let params = paramsContent.toString().match(/\([0-9]+x[0-9]+,[\ ]*[0-9],[\ ]*BORDER\_\w+\)/g)[0];
            let paramObjs = [];
            paramObjs.push({name:"size", value:"", reg:[""], index:0});
            paramObjs.push({name:"ksize", value:"", reg:["/\\b[0-9]\\b/"], index:1});
            paramObjs.push({name:"borderMode", value: "", reg:["/BORDER\_\\w+/"], index:2});
            let locationList = decodeParams2Case(params, paramObjs,filter2dCombinations);

            for (let i = 0; i < locationList.length; i++){
                let first = locationList[i][0];
                let second = locationList[i][1];
                addFilter2dModeCase(suite, [filter2dCombinations[first][second]], 0);
              }
        } else {
          log("no filter or getting invalid params, run all the cases");
          addFilter2dModeCase(suite, combiFilter2dCase, 0);
        }
        setBenchmarkSuite(suite, "filter2d", currentCaseId);
        log(`Running ${totalCaseNum} tests from Filter2d`);
        suite.run({ 'async': true }); // run the benchmark
    }

    let filter2dCombinations = [combiFilter2dCase];//,combiDISABLEDFilter2dCase];

    if (isNodeJs) {
        const args = process.argv.slice(2);
        let paramsContent = '';
        if (/--test_param_filter=\([0-9]+x[0-9]+,[\ ]*[0-9],[\ ]*BORDER\_\w+\)/g.test(args.toString())) {
          paramsContent = args.toString().match(/\([0-9]+x[0-9]+,[\ ]*[0-9],[\ ]*BORDER\_\w+\)/g)[0];
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