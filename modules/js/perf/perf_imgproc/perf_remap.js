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

    const RemapSize = [cvSize.szVGA, cvSize.sz1080p];
    const RemapSrcType = ["CV_16UC1", "CV_16SC1", "CV_32FC1"];
    const RemapType = ["CV_16SC2", "CV_32FC1", "CV_32FC2"];
    const InterType = ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_LANCZOS4"];
    const combiRemap = combine(RemapSize, RemapSrcType, RemapType, InterType);

    function addRemapCase(suite, type) {
        suite.add('remap', function() {
            cv.remap(src, dst, map1, map2, interType);
          }, {
              'setup': function() {
                let size = this.params.size;
                let matType = cv[this.params.matType];
                let mapType = cv[this.params.mapType];
                let interType = cv[this.params.interType];


                let src = new cv.Mat(size, matType);
                let dst = new cv.Mat(size, matType);
                let map1 = new cv.Mat(size, mapType);
                let map2;
                if (mapType == cv.CV_32FC1) {
                  map2 = new cv.Mat(size, mapType);
                } else if (interType != cv.INTER_NEAREST && mapType == cv.CV_16SC2) {
                  map2 = new cv.Mat.zeros(size, cv.CV_16UC1);
                } else {
                  map2 = new cv.Mat();
                }

                for (let j = 0; j < map1.rows; j++) {
                  for (let i = 0; i <  map1.cols; i++) {
                    let randNum = Math.random();
                    let view, view1;
                    switch(matType) {
                      case cv.CV_16UC1:
                        view = src.ushortPtr(j,i);
                        view[0] = Math.floor(randNum*256);
                        break;
                      case cv.CV_16SC1:
                        view = src.shortPtr(j,i);
                        view[0] = Math.floor(randNum*256);
                        break;
                      case cv.CV_32FC1:
                        view = src.floatPtr(j,i);
                        view[0] = randNum*256;
                        break;
                      default:
                        console.error("Unknown conversion type 1");
                        break;
                    }

                    switch(mapType) {
                      case cv.CV_32FC1:
                        view1 = map1.floatPtr(j,i);
                        let view2 = map2.floatPtr(j,i);
                        view1[0] = src.cols - i - 1;
                        view2[0] = j;
                        break;
                      case cv.CV_32FC2:
                        view1 = map1.floatPtr(j,i);
                        view1[0] = src.cols - i - 1;
                        view1[1] = j;
                        break;
                      case cv.CV_16SC2:
                        view1 = map1.shortPtr(j,i);
                        view1[0] = src.cols - i - 1;
                        view1[1] = j;
                        break;
                      default:
                        console.error("Unknown conversion type 2");
                        break;
                    }
                  }
                }
                },
              'teardown': function() {
                src.delete();
                dst.delete();
                map1.delete();
                map2.delete();
              }
          });
    }

    function addRemapModeCase(suite, combination, type) {
      totalCaseNum += combination.length;
      for (let i = 0; i < combination.length; ++i) {
        let size =  combination[i][0];
        let matType = combination[i][1];
        let mapType = combination[i][2];
        let interType = combination[i][3];

        let params = {size: size, matType:matType, mapType:mapType, interType:interType};
        addKernelCase(suite, params, type, addRemapCase);
      }
    }

    function genBenchmarkCase(paramsContent) {
      let suite = new Benchmark.Suite;
      totalCaseNum = 0;
      currentCaseId = 0;

      if (/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*CV\_\w+,[\ ]*INTER\_\w+\)/g.test(paramsContent.toString())) {
          let params = paramsContent.toString().match(/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*CV\_\w+,[\ ]*INTER\_\w+\)/g)[0];
          let paramObjs = [];
          paramObjs.push({name:"size", value:"", reg:[""], index:0});
          paramObjs.push({name:"matType", value:"", reg:["/CV\_[0-9]+[FSUfsu]C[0-9]/"], index:1});
          paramObjs.push({name:"mapType", value:"", reg:["/CV\_[0-9]+[FSUfsu]C[0-9]/g"], index:2, loc:1});
          paramObjs.push({name:"interType", value: "", reg:["/INTER\_\\w+/"], index:3});
          let locationList = decodeParams2Case(params, paramObjs, remapCombinations);

          for (let i = 0; i < locationList.length; i++){
              let first = locationList[i][0];
              let second = locationList[i][1];
              addRemapModeCase(suite, [remapCombinations[first][second]], first);
            }
      } else {
        log("no filter or getting invalid params, run all the cases");
        addRemapModeCase(suite, combiRemap, 0);
      }
      setBenchmarkSuite(suite, "remap", currentCaseId);
      log(`Running ${totalCaseNum} tests from remap`);
      suite.run({ 'async': true }); // run the benchmark
  }

    let remapCombinations = [combiRemap];

    if (isNodeJs) {
      const args = process.argv.slice(2);
      let paramsContent = '';
      if (/--test_param_filter=\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*CV\_\w+,[\ ]*INTER\_\w+\)/g.test(args.toString())) {
        paramsContent = args.toString().match(/\([0-9]+x[0-9]+,[\ ]*CV\_\w+,[\ ]*CV\_\w+,[\ ]*INTER\_\w+\)/g)[0];
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