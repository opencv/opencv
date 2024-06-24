// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

QUnit.test("init_cv", (assert) => {
    if (cv instanceof Promise) {
        const done = assert.async();
        cv.then((ready_cv) => {
            cv = ready_cv;
            done();
        });
    } else if (cv.getBuildInformation === undefined) {
        const done = assert.async();
        cv['onRuntimeInitialized'] = () => {
            done();
        }
    }
    assert.ok(true);
});
