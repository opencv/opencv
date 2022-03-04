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

  const typicalMatSizes = [cvSize.szVGA, cvSize.sz720p, cvSize.sz1080p, cvSize.szODD];
  const matTypes = ['CV_8UC1', 'CV_16SC1', 'CV_32FC1', 'CV_64FC1'];
  const threshTypes = ['THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV'];

  const combiSizeMatTypeThreshType = combine(typicalMatSizes, matTypes, threshTypes);
  const combiSizeOnly = combine(typicalMatSizes, ['CV_8UC1'], ['THRESH_BINARY|THRESH_OTSU']);


  function addThresholdCase(suite, type) {
    suite.add('threshold', function() {
      if (type == "sizeonly") {
        cv.threshold(src, dst, threshold, thresholdMax, cv.THRESH_BINARY|cv.THRESH_OTSU);
      } else {
        cv.threshold(src, dst, threshold, thresholdMax, threshType);
      }
      }, {
        'setup': function() {
          let matSize = this.params.matSize;
          let type =  this.params.modeType;
          let src, dst, matType, threshType;
          if (type == "sizeonly") {
            src = new cv.Mat(matSize, cv.CV_8UC1);
            dst = new cv.Mat(matSize, cv.CV_8UC1);
          } else {
            matType = cv[this.params.matType];
            threshType = cv[this.params.threshType];
            src = new cv.Mat(matSize, matType);
            dst = new cv.Mat(matSize, matType);
          }
          let threshold = 127.0;
          let thresholdMax = 210.0;
          let srcView = src.data;
          srcView[0] = 0;
          srcView[1] = 100;
          srcView[2] = 200;
            },
        'teardown': function() {
          src.delete();
          dst.delete();
        }
    });
  }

  function addThresholdModecase(suite, combination, type) {
    totalCaseNum += combination.length;
    for (let i = 0; i < combination.length; ++i) {
      let matSize = combination[i][0];
      let matType = 'CV_8UC1';
      let threshType = 'THRESH_BINARY|THRESH_OTSU';
      if (type != "sizeonly") {
        matType = combination[i][1];
        threshType = combination[i][2];
      }
      let params = {matSize: matSize, matType: matType, threshType: threshType, modeType: type};
      addKernelCase(suite, params, type, addThresholdCase);
    }
  }

  function genBenchmarkCase(paramsContent) {
    let suite = new Benchmark.Suite;
    totalCaseNum = 0;
    currentCaseId = 0;
    let params = "";
    let paramObjs = [];
    paramObjs.push({name:"size", value:"", reg:[""], index:0});

    if (/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*THRESH\_\w+\)/g.test(paramsContent.toString())) {
      params = paramsContent.toString().match(/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*THRESH\_\w+\)/g)[0];
      paramObjs.push({name:"matType", value:"", reg:["/CV\_[0-9]+[A-z][A-z][0-9]/"], index:1});
      paramObjs.push({name:"threshType", value:"", reg:["/THRESH\_[A-z]+\_?[A-z]*/"], index:2});
    } else if (/[\ ]*[0-9]+x[0-9]+[\ ]*/g.test(paramsContent.toString())) {
      params = paramsContent.toString().match(/[\ ]*[0-9]+x[0-9]+[\ ]*/g)[0];
      paramObjs.push({name:"matType", value:"CV_8UC1", reg:[""], index:1});
      paramObjs.push({name:"threshType", value:"THRESH_BINARY|THRESH_OTSU", reg:[""], index:2});
    }

    if(params != ""){
      let locationList = decodeParams2Case(params, paramObjs,combinations);
      for (let i = 0; i < locationList.length; i++){
        let first = locationList[i][0];
        let second = locationList[i][1];
        if (first == 0) {
          addThresholdModecase(suite, [combinations[first][second]], "normal");
        } else {
          addThresholdModecase(suite, [combinations[first][second]], "sizeonly");
        }
      }
    } else {
      log("no filter or getting invalid params, run all the cases");
      addThresholdModecase(suite, combiSizeMatTypeThreshType, "normal");
      addThresholdModecase(suite, combiSizeOnly, "sizeonly");
    }
    setBenchmarkSuite(suite, "threshold", currentCaseId);
    log(`Running ${totalCaseNum} tests from Threshold`);
    suite.run({ 'async': true }); // run the benchmark
  }

  // init
  let combinations = [combiSizeMatTypeThreshType, combiSizeOnly];

  // set test filter params
  if (isNodeJs) {
    const args = process.argv.slice(2);
    let paramsContent = '';
    if (/--test_param_filter=\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*THRESH\_\w+\)/g.test(args.toString())) {
      paramsContent = args.toString().match(/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*THRESH\_\w+\)/g)[0];
    } else if (/--test_param_filter=[\ ]*[0-9]+x[0-9]+[\ ]*/g.test(args.toString())) {
      paramsContent = args.toString().match(/[\ ]*[0-9]+x[0-9]+[\ ]*/g)[0];
    }
    genBenchmarkCase(paramsContent);
  } else {
    runButton.onclick = function() {
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