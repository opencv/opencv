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

    const PyrDownSize = [cvSize.sz1080p, cvSize.sz720p, cvSize.szVGA, cvSize.szQVGA, cvSize.szODD];
    const PyrDownType = ["CV_8UC1", "CV_8UC3", "CV_8UC4", "CV_16SC1", "CV_16SC3", "CV_16SC4", "CV_32FC1", "CV_32FC3", "CV_32FC4"];

    const combiPyrDown = combine(PyrDownSize, PyrDownType);

    function addPryDownCase(suite, type) {
        suite.add('pyrDown', function() {
            cv.pyrDown(src, dst);
          }, {
              'setup': function() {
                let size = this.params.size;
                let matType = cv[this.params.matType];
                let src = new cv.Mat(size, matType);
                let dst = new cv.Mat((size.height + 1)/2, (size.height + 1)/2, matType)
                },
              'teardown': function() {
                src.delete();
                dst.delete();
              }
          });
    }

    function addPyrDownModeCase(suite, combination, type) {
      totalCaseNum += combination.length;
      for (let i = 0; i < combination.length; ++i) {
        let size =  combination[i][0];
        let matType = combination[i][1];

        let params = {size: size, matType:matType};
        addKernelCase(suite, params, type, addPryDownCase);
      }
    }

    function genBenchmarkCase(paramsContent) {
      let suite = new Benchmark.Suite;
      totalCaseNum = 0;
      currentCaseId = 0;

      if (/\([0-9]+x[0-9]+,[\ ]*CV\_\w+\)/g.test(paramsContent.toString())) {
          let params = paramsContent.toString().match(/\([0-9]+x[0-9]+,[\ ]*CV\_\w+\)/g)[0];
          let paramObjs = [];
          paramObjs.push({name:"size", value:"", reg:[""], index:0});
          paramObjs.push({name:"matType", value:"", reg:["/CV\_[0-9]+[FSUfsu]C[0-9]/"], index:1});
          let locationList = decodeParams2Case(params, paramObjs, pyrDownCombinations);

          for (let i = 0; i < locationList.length; i++){
              let first = locationList[i][0];
              let second = locationList[i][1];
              addPyrDownModeCase(suite, [pyrDownCombinations[first][second]], first);
            }
      } else {
        log("no filter or getting invalid params, run all the cases");
        addPyrDownModeCase(suite, combiPyrDown, 0);
      }
      setBenchmarkSuite(suite, "pyrDown", currentCaseId);
      log(`Running ${totalCaseNum} tests from pyrDown`);
      suite.run({ 'async': true }); // run the benchmark
  }

    let pyrDownCombinations = [combiPyrDown];

    if (isNodeJs) {
      const args = process.argv.slice(2);
      let paramsContent = '';
      if (/--test_param_filter=\([0-9]+x[0-9]+,[\ ]*CV\_\w+\)/g.test(args.toString())) {
        paramsContent = args.toString().match(/\([0-9]+x[0-9]+,[\ ]*CV\_\w+\)/g)[0];
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