var testrunner = require('qunit');
testrunner.options.maxBlockDuration = 20000; // cause opencv_js.js need time to load


testrunner.run({
    code: 'opencv.js',
    tests: ['test_mat.js', 'test_utils.js', 'test_imgproc.js', 'test_objdetect.js', 'test_video.js']}, function(err, report) {
        console.log(report.failed + ' failed, ' + report.passed + ' passed');
});
