// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

if (cv.getBuildInformation === undefined) {
    // WASM
    QUnit.test("init_cv", (assert) => {
        const done = assert.async();
        assert.ok(true);
        if (cv instanceof Promise) {
            cv.then((ready_cv) => {
                cv = ready_cv;
                done();
            });
        } else {
            cv['onRuntimeInitialized'] = () => {
                done();
            }
        }
    });
}
