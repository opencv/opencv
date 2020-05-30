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

  const typicalMatSizes = [cvSize.szVGA, cvSize.sz720p, cvSize.sz1080p, cvSize.szODD];
  const matTypes = ['CV_8UC1', 'CV_16SC1', 'CV_32FC1', 'CV_64FC1'];
  const threshTypes = ['THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV'];

  const combiSizeMatTypeThreshType = combine(typicalMatSizes, matTypes, threshTypes);
  const combiSizeOnly = combine(typicalMatSizes, ['CV_8UC1'], ['THRESH_BINARY|THRESH_OTSU']);

  function addSizeMatTypeThreshTypeCase(suite, combination) {
    totalCaseNum += combination.length;
    for (let i = 0; i < combination.length; ++i) {
      let matSize = combination[i][0];
      let matType = combination[i][1];
      let threshType = combination[i][2];

      suite.add('threshold', function() {
        cv.threshold(src, dst, threshold, thresholdMax, threshType);
        }, {
          'setup': function() {
            let matSize = this.params.matSize;
            let matType = cv[this.params.matType];
            let threshType = cv[this.params.threshType];
            let threshold = 127.0;
            let thresholdMax = 210.0;
            let src = new cv.Mat(matSize, matType);
            let dst = new cv.Mat(matSize, matType);
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

      // set init params
      let index = suite.length - 1;
      suite[index].params = {
        matSize: matSize,
        matType: matType,
        threshType: threshType
      };
    }
  }

  function addSizeOnlyCase(suite, combination) {
    totalCaseNum += combination.length;
    for (let i = 0; i < combination.length; ++i) {
      let matSize = combination[i][0];

      suite.add('threshold', function() {
        cv.threshold(src, dst, threshold, thresholdMax, cv.THRESH_BINARY|cv.THRESH_OTSU);
        }, {
          'setup': function() {
            let matSize = this.params.matSize;
            let threshold = 127.0;
            let thresholdMax = 210.0;
            let src = new cv.Mat(matSize, cv.CV_8UC1);
            let dst = new cv.Mat(matSize, cv.CV_8UC1);
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

      // set init params
      let index = suite.length - 1;
      suite[index].params = {
        matSize: matSize,
        matType: 'CV_8UC1',
        threshType: 'THRESH_BINARY|THRESH_OTSU'
      };
    }
  }

  function decodeParams2Case(suite, params, isSizeOnly) {
    let sizeString = params.match(/[0-9]+x[0-9]+/g).toString();
    let sizes = sizeString.match(/[0-9]+/g);
    let size1Str = sizes.slice(0, 2).toString();
    let matSize = cvtStr2cvSize(size1Str);
    let matType, threshType;
    if (isSizeOnly) {
      matType = 'CV_8UC1';
      threshType = 'THRESH_BINARY|THRESH_OTSU';
    } else {
      matType = (params.match(/CV\_[0-9]+[A-z][A-z][0-9]/) || []).toString();
      threshType = (params.match(/THRESH\_[A-z]+\_?[A-z]*/) || []).toString();
    }
    // check if the params match and add case
    for (let i = 0; i < combinations.length; ++i) {
      let combination = combinations[i];
      for (let j = 0; j < combination.length; ++j) {
        if (matSize === combination[j][0] && matType === combination[j][1] && threshType === combination[j][2]) {
          thresholdFunc[i](suite, [combination[j]]);
        }
      }
    }
  }

  function log(message) {
    console.log(message);1
    if (!isNodeJs) {
      logElement.innerHTML += `\n${'\t'.repeat(1) + message}`;
    }
  }

  function setBenchmarkSuite(suite) {
    suite
    // add listeners
    .on('cycle', function(event) {
      ++currentCaseId;
      let params = event.target.params;
      let matSize = params.matSize;
      let matType = params.matType;
      let threshType = params.threshType;
      log(`=== ${event.target.name} ${currentCaseId} ===`);
      log(`params: (${parseInt(matSize.width)}x${parseInt(matSize.height)},`+
          `${matType},${threshType})`);
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
    if (/\([0-9]+x[0-9]+,[\ ]*\w+,[\ ]*\w+\)/g.test(paramsContent.toString())) {
      let params = paramsContent.toString().match(/\([0-9]+x[0-9]+,[\ ]*\w+,[\ ]*\w+\)/g)[0];
      let isSizeOnly = 0;
      decodeParams2Case(suite, params, isSizeOnly);
    } else if (/[\ ]*[0-9]+x[0-9]+[\ ]*/g.test(paramsContent.toString())) {
      let params = paramsContent.toString().match(/[\ ]*[0-9]+x[0-9]+[\ ]*/g)[0];
      let isSizeOnly = 1;
      decodeParams2Case(suite, params, isSizeOnly);
    }
    else {
      log("no filter or getting invalid params, run all the cases");
      addSizeMatTypeThreshTypeCase(suite, combiSizeMatTypeThreshType);
      addSizeOnlyCase(suite, combiSizeOnly);
    }
    setBenchmarkSuite(suite);
    log(`Running ${totalCaseNum} tests from Threshold`);
    suite.run({ 'async': true }); // run the benchmark
  }

  // init
  let thresholdFunc = [addSizeMatTypeThreshTypeCase, addSizeOnlyCase];
  let combinations = [combiSizeMatTypeThreshType, combiSizeOnly];

  // set test filter params
  if (isNodeJs) {
    const args = process.argv.slice(2);
    let paramsContent = '';
    if (/--test_param_filter=\([0-9]+x[0-9]+,[\ ]*\w+,[\ ]*\w+\)/g.test(args.toString())) {
      paramsContent = args.toString().match(/\([0-9]+x[0-9]+,[\ ]*\w+,[\ ]*\w+\)/g)[0];
    } else if (/--test_param_filter=[\ ]*[0-9]+x[0-9]+[\ ]*/g.test(args.toString())) {
      paramsContent = args.toString().match(/[\ ]*[0-9]+x[0-9]+[\ ]*/g)[0];
    }
    genBenchmarkCase(paramsContent);
  } else {
    runButton.onclick = function() {
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