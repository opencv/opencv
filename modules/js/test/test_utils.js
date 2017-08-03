/*/////////////////////////////////////////////////////////////////////////////
AUTHOR: Sajjad Taheri sajjadt[at]uci[at]edu

                             LICENSE AGREEMENT
Copyright (c) 2015, University of california, Irvine

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement:
   This product includes software developed by the UC Irvine.
4. Neither the name of the UC Irvine nor the
   names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UC IRVINE ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL UC IRVINE OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/////////////////////////////////////////////////////////////////////////////*/
if (typeof module !== 'undefined' && module.exports) {
    // The envrionment is Node.js
    var cv = require('./opencv.js');
}
QUnit.module ("Utils", {});
QUnit.test("Test vectors", function(assert) {
	{
		let pointVector = new cv.PointVector();
		for (var i=0 ; i<100; ++i) {
			pointVector.push_back({x: i, y: 2*i});
		}

		assert.equal(pointVector.size(), 100);

		let index = 10;
		let item = pointVector.get(index);
		assert.equal(item.x, index);
		assert.equal(item.y, 2*index);

		index = 0;
		item = pointVector.get(index);
		assert.equal(item.x, index);
		assert.equal(item.y, 2*index);

		index = 99;
		item = pointVector.get(index);
		assert.equal(item.x, index);
		assert.equal(item.y, 2*index);

		pointVector.delete();
	}

	{
		let pointVector = new cv.PointVector();
		for (var i=0 ; i<100; ++i) {
			pointVector.push_back(new cv.Point(i, 2*i));
		}

		pointVector.push_back(new cv.Point());

		assert.equal(pointVector.size(), 101);

		let index = 10;
		let item = pointVector.get(index);
		assert.equal(item.x, index);
		assert.equal(item.y, 2*index);

		index = 0;
		item = pointVector.get(index);
		assert.equal(item.x, index);
		assert.equal(item.y, 2*index);

		index = 99;
		item = pointVector.get(index);
		assert.equal(item.x, index);
		assert.equal(item.y, 2*index);

		index = 100;
		item = pointVector.get(index);
		assert.equal(item.x, 0);
		assert.equal(item.y, 0);

		pointVector.delete();
	}
});
QUnit.test("Test Rect", function(assert) {
	let rectVector = new cv.RectVector();
	let rect = {x: 1, y: 2, width: 3, height: 4};
	rectVector.push_back(rect);
	rectVector.push_back(new cv.Rect());
	rectVector.push_back(new cv.Rect(rect));
	rectVector.push_back(new cv.Rect({x: 5, y: 6}, {width: 7, height: 8}));
	rectVector.push_back(new cv.Rect(9, 10, 11, 12));

	assert.equal(rectVector.size(), 5);

	let item = rectVector.get(0);
	assert.equal(item.x, 1);
	assert.equal(item.y, 2);
	assert.equal(item.width, 3);
	assert.equal(item.height, 4);

	item = rectVector.get(1);
	assert.equal(item.x, 0);
	assert.equal(item.y, 0);
	assert.equal(item.width, 0);
	assert.equal(item.height, 0);

	item = rectVector.get(2);
	assert.equal(item.x, 1);
	assert.equal(item.y, 2);
	assert.equal(item.width, 3);
	assert.equal(item.height, 4);

	item = rectVector.get(3);
	assert.equal(item.x, 5);
	assert.equal(item.y, 6);
	assert.equal(item.width, 7);
	assert.equal(item.height, 8);

	item = rectVector.get(4);
	assert.equal(item.x, 9);
	assert.equal(item.y, 10);
	assert.equal(item.width, 11);
	assert.equal(item.height, 12);

	rectVector.delete();
});
QUnit.test("Test Size", function(assert) {
	 {
        let mat = new cv.Mat();
        mat.create({width: 5, height: 10}, cv.CV_8UC4);
        let size = mat.size();

        assert.ok(mat.type() === cv.CV_8UC4);
        assert.ok(size[0] === 10);
        assert.ok(size[1] === 5);
        assert.ok(mat.channels() === 4);

        mat.delete();
    }

     {
        let mat = new cv.Mat();
        mat.create(new cv.Size(5, 10), cv.CV_8UC4);
        let size = mat.size();

        assert.ok(mat.type() === cv.CV_8UC4);
        assert.ok(size[0] === 10);
        assert.ok(size[1] === 5);
        assert.ok(mat.channels() === 4);

        mat.delete();
    }
});


QUnit.test("test_rotated_rect", function(assert) {
  {
    let rect = {center: {x: 100, y: 100}, size: {height: 100, width: 50}, angle: 30};

    assert.equal(rect.center.x, 100);
    assert.equal(rect.center.y, 100);
    assert.equal(rect.angle, 30);
    assert.equal(rect.size.height, 100);
    assert.equal(rect.size.width, 50);
  }

  {
    let rect = new cv.RotatedRect();

    assert.equal(rect.center.x, 0);
    assert.equal(rect.center.y, 0);
    assert.equal(rect.angle, 0);
    assert.equal(rect.size.height, 0);
    assert.equal(rect.size.width, 0);

    let points = rect.points();

    assert.equal(points[0].x, 0);
    assert.equal(points[0].y, 0);
    assert.equal(points[1].x, 0);
    assert.equal(points[1].y, 0);
    assert.equal(points[2].x, 0);
    assert.equal(points[2].y, 0);
    assert.equal(points[3].x, 0);
    assert.equal(points[3].y, 0);
  }

  {
    let rect = new cv.RotatedRect({x: 100, y: 100}, {height: 100, width: 50}, 30);

    assert.equal(rect.center.x, 100);
    assert.equal(rect.center.y, 100);
    assert.equal(rect.angle, 30);
    assert.equal(rect.size.height, 100);
    assert.equal(rect.size.width, 50);

    let points = rect.points();

    assert.equal(points[0].x, rect.boundingRect2f().x);
    assert.equal(points[1].y, rect.boundingRect2f().y);
  }
});
