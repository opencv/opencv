// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

if (cv instanceof Promise) {
    QUnit.test("init_cv", (assert) => {
        const done = assert.async();
        assert.ok(true);
        cv.then((ready_cv) => {
            cv = ready_cv;
            done();
        });
    });
}
