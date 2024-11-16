#ifndef __POINT_H__
#define __POINT_H__

/*
 *  Point.hpp
 *  zxing
 *
 *  Copyright 2010 ZXing authors All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

namespace zxing {
class PointI {
public:
  int x;
  int y;
};

class Point {
public:
  Point() : x(0.0f), y(0.0f) {};
  Point(float x_, float y_) : x(x_), y(y_) {};

  float x;
  float y;
};

class Line {
public:
  Line(Point start_, Point end_) : start(start_), end(end_) {};

  Point start;
  Point end;
};
}
#endif  // POINT_H_
