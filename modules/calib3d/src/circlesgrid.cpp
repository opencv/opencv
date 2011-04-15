/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "circlesgrid.hpp"
//#define DEBUG_CIRCLES

#ifdef DEBUG_CIRCLES
#include <opencv2/highgui/highgui.hpp>
#endif

using namespace cv;
using namespace std;

void CirclesGridClusterFinder::hierarchicalClustering(const vector<Point2f> points, const Size &patternSize, vector<Point2f> &patternPoints)
{
  Mat dists(points.size(), points.size(), CV_32FC1, Scalar(0));
  Mat distsMask(dists.size(), CV_8UC1, Scalar(0));
  for(size_t i=0; i<points.size(); i++)
  {
    for(size_t j=i+1; j<points.size(); j++)
    {
      dists.at<float>(i, j) = norm(points[i] - points[j]);
      distsMask.at<uchar>(i, j) = 255;
      //TODO: use symmetry
      distsMask.at<uchar>(j, i) = distsMask.at<uchar>(i, j);
      dists.at<float>(j, i) = dists.at<float>(i, j);
    }
  }

  vector<std::list<size_t> > clusters(points.size());
  for(size_t i=0; i<points.size(); i++)
  {
    clusters[i].push_back(i);
  }

  int patternClusterIdx = 0;
  while(clusters[patternClusterIdx].size() < patternSize.area() && countNonZero(distsMask == 255) > 0)
  {
    Point minLoc;
    minMaxLoc(dists, 0, 0, &minLoc, 0, distsMask);
    int minIdx = std::min(minLoc.x, minLoc.y);
    int maxIdx = std::max(minLoc.x, minLoc.y);

    distsMask.row(maxIdx).setTo(0);
    distsMask.col(maxIdx).setTo(0);
    Mat newDists = cv::min(dists.row(minLoc.x), dists.row(minLoc.y));
    Mat tmpLine = dists.row(minIdx);
    newDists.copyTo(tmpLine);
    tmpLine = dists.col(minIdx);
    newDists.copyTo(tmpLine);

    clusters[minIdx].splice(clusters[minIdx].end(), clusters[maxIdx]);
    patternClusterIdx = minIdx;
  }

  patternPoints.clear();
  if(clusters[patternClusterIdx].size() != patternSize.area())
  {
    return;
  }
  patternPoints.reserve(clusters[patternClusterIdx].size());
  for(std::list<size_t>::iterator it = clusters[patternClusterIdx].begin(); it != clusters[patternClusterIdx].end(); it++)
  {
    patternPoints.push_back(points[*it]);
  }
}

void CirclesGridClusterFinder::findGrid(const std::vector<cv::Point2f> points, cv::Size patternSize, vector<Point2f>& centers)
{
  centers.clear();
  if(points.empty())
  {
    return;
  }

  vector<Point2f> patternPoints;
  hierarchicalClustering(points, patternSize, patternPoints);
  if(patternPoints.empty())
  {
    return;
  }

  vector<Point2f> hull2f;
  convexHull(Mat(patternPoints), hull2f, false);
  const size_t cornersCount = 6;
  if(hull2f.size() < cornersCount)
    return;

  vector<Point2f> corners;
  findCorners(hull2f, corners);
  if(corners.size() != cornersCount)
    return;

  vector<Point2f> outsideCorners;
  findOutsideCorners(corners, outsideCorners);
  const size_t outsideCornersCount = 2;
  if(outsideCorners.size() != outsideCornersCount)
    return;

  vector<Point2f> sortedCorners;
  getSortedCorners(hull2f, corners, outsideCorners, sortedCorners);
  if(sortedCorners.size() != cornersCount)
    return;

  vector<Point2f> rectifiedPatternPoints;
  rectifyPatternPoints(patternSize, patternPoints, sortedCorners, rectifiedPatternPoints);
  if(patternPoints.size() != rectifiedPatternPoints.size())
    return;

  parsePatternPoints(patternSize, patternPoints, rectifiedPatternPoints, centers);
}

void CirclesGridClusterFinder::findCorners(const std::vector<cv::Point2f> &hull2f, std::vector<cv::Point2f> &corners)
{
  //find angles (cosines) of vertices in convex hull
  vector<float> angles;
  for(size_t i=0; i<hull2f.size(); i++)
  {
    Point2f vec1 = hull2f[(i+1) % hull2f.size()] - hull2f[i % hull2f.size()];
    Point2f vec2 = hull2f[(i-1 + static_cast<int>(hull2f.size())) % hull2f.size()] - hull2f[i % hull2f.size()];
    float angle = vec1.ddot(vec2) / (norm(vec1) * norm(vec2));
    angles.push_back(angle);
  }

  //sort angles by cosine
  //corners are the most sharp angles (6)
  Mat anglesMat = Mat(angles);
  Mat sortedIndices;
  sortIdx(anglesMat, sortedIndices, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
  CV_Assert(sortedIndices.type() == CV_32SC1);
  const int cornersCount = 6;
  corners.clear();
  for(int i=0; i<cornersCount; i++)
  {
    corners.push_back(hull2f[sortedIndices.at<int>(i, 0)]);
  }
}

void CirclesGridClusterFinder::findOutsideCorners(const std::vector<cv::Point2f> &corners, std::vector<cv::Point2f> &outsideCorners)
{
  //find two pairs of the most nearest corners
  double min1 = std::numeric_limits<double>::max();
  double min2 = std::numeric_limits<double>::max();
  Point minLoc1, minLoc2;

  for(size_t i=0; i<corners.size(); i++)
  {
    for(size_t j=i+1; j<corners.size(); j++)
    {
      double dist = norm(corners[i] - corners[j]);
      Point loc(j, i);
      if(dist < min1)
      {
        min2 = min1;
        minLoc2 = minLoc1;
        min1 = dist;
        minLoc1 = loc;
      }
      else
      {
        if(dist < min2)
        {
          min2 = dist;
          minLoc2 = loc;
        }
      }
    }
  }
  std::set<int> outsideCornersIndices;
  for(size_t i=0; i<corners.size(); i++)
  {
    outsideCornersIndices.insert(i);
  }
  outsideCornersIndices.erase(minLoc1.x);
  outsideCornersIndices.erase(minLoc1.y);
  outsideCornersIndices.erase(minLoc2.x);
  outsideCornersIndices.erase(minLoc2.y);

  outsideCorners.clear();
  for(std::set<int>::iterator it = outsideCornersIndices.begin(); it != outsideCornersIndices.end(); it++)
  {
    outsideCorners.push_back(corners[*it]);
  }
}

void CirclesGridClusterFinder::getSortedCorners(const std::vector<cv::Point2f> &hull2f, const std::vector<cv::Point2f> &corners, const std::vector<cv::Point2f> &outsideCorners, std::vector<cv::Point2f> &sortedCorners)
{
  Point2f center = std::accumulate(corners.begin(), corners.end(), Point2f(0.0f, 0.0f));
  center *= 1.0 / corners.size();

  vector<Point2f> centerToCorners;
  for(size_t i=0; i<outsideCorners.size(); i++)
  {
    centerToCorners.push_back(outsideCorners[i] - center);
  }

  //TODO: use CirclesGridFinder::getDirection
  float crossProduct = centerToCorners[0].x * centerToCorners[1].y - centerToCorners[0].y * centerToCorners[1].x;
  //y axis is inverted in computer vision so we check > 0
  bool isClockwise = crossProduct > 0;
  Point2f firstCorner  = isClockwise ? outsideCorners[1] : outsideCorners[0];

  std::vector<Point2f>::const_iterator firstCornerIterator = std::find(hull2f.begin(), hull2f.end(), firstCorner);
  sortedCorners.clear();
  for(vector<Point2f>::const_iterator it = firstCornerIterator; it != hull2f.end(); it++)
  {
    vector<Point2f>::const_iterator itCorners = std::find(corners.begin(), corners.end(), *it);
    if(itCorners != corners.end())
    {
      sortedCorners.push_back(*it);
    }
  }
  for(vector<Point2f>::const_iterator it = hull2f.begin(); it != firstCornerIterator; it++)
  {
    vector<Point2f>::const_iterator itCorners = std::find(corners.begin(), corners.end(), *it);
    if(itCorners != corners.end())
    {
      sortedCorners.push_back(*it);
    }
  }
}

void CirclesGridClusterFinder::rectifyPatternPoints(const cv::Size &patternSize, const std::vector<cv::Point2f> &patternPoints, const std::vector<cv::Point2f> &sortedCorners, std::vector<cv::Point2f> &rectifiedPatternPoints)
{
  //indices of corner points in pattern
  vector<Point> trueIndices;
  trueIndices.push_back(Point(0, 0));
  trueIndices.push_back(Point(patternSize.width - 1, 0));
  trueIndices.push_back(Point(patternSize.width - 1, 1));
  trueIndices.push_back(Point(patternSize.width - 1, patternSize.height - 2));
  trueIndices.push_back(Point(patternSize.width - 1, patternSize.height - 1));
  trueIndices.push_back(Point(0, patternSize.height - 1));

  vector<Point2f> idealPoints;
  for(size_t idx=0; idx<trueIndices.size(); idx++)
  {
    int i = trueIndices[idx].y;
    int j = trueIndices[idx].x;
    idealPoints.push_back(Point2f((2*j + i % 2)*squareSize, i*squareSize));
  }

  Mat homography = findHomography(Mat(sortedCorners), Mat(idealPoints), 0);
  Mat rectifiedPointsMat;
  transform(Mat(patternPoints), rectifiedPointsMat, homography);
  rectifiedPatternPoints.clear();
  convertPointsHomogeneous(rectifiedPointsMat, rectifiedPatternPoints);
}

void CirclesGridClusterFinder::parsePatternPoints(const cv::Size &patternSize, const std::vector<cv::Point2f> &patternPoints, const std::vector<cv::Point2f> &rectifiedPatternPoints, std::vector<cv::Point2f> &centers)
{
  flann::LinearIndexParams flannIndexParams;
  flann::Index flannIndex(Mat(rectifiedPatternPoints).reshape(1), flannIndexParams);

  centers.clear();
  for( int i = 0; i < patternSize.height; i++ )
  {
    for( int j = 0; j < patternSize.width; j++ )
    {
      Point2f idealPt((2*j + i % 2)*squareSize, i*squareSize);
      vector<float> query = Mat(idealPt);
      int knn = 1;
      vector<int> indices(knn);
      vector<float> dists(knn);
      flannIndex.knnSearch(query, indices, dists, knn, flann::SearchParams());
      centers.push_back(patternPoints.at(indices[0]));

      if(dists[0] > maxRectifiedDistance)
      {
        centers.clear();
        return;
      }
    }
  }
}

Graph::Graph(size_t n)
{
  for (size_t i = 0; i < n; i++)
  {
    addVertex(i);
  }
}

bool Graph::doesVertexExist(size_t id) const
{
  return (vertices.find(id) != vertices.end());
}

void Graph::addVertex(size_t id)
{
  assert( !doesVertexExist( id ) );

  vertices.insert(pair<size_t, Vertex> (id, Vertex()));
}

void Graph::addEdge(size_t id1, size_t id2)
{
  assert( doesVertexExist( id1 ) );
  assert( doesVertexExist( id2 ) );

  vertices[id1].neighbors.insert(id2);
  vertices[id2].neighbors.insert(id1);
}

void Graph::removeEdge(size_t id1, size_t id2)
{
  assert( doesVertexExist( id1 ) );
  assert( doesVertexExist( id2 ) );

  vertices[id1].neighbors.erase(id2);
  vertices[id2].neighbors.erase(id1);
}

bool Graph::areVerticesAdjacent(size_t id1, size_t id2) const
{
  assert( doesVertexExist( id1 ) );
  assert( doesVertexExist( id2 ) );

  Vertices::const_iterator it = vertices.find(id1);
  return it->second.neighbors.find(id2) != it->second.neighbors.end();
}

size_t Graph::getVerticesCount() const
{
  return vertices.size();
}

size_t Graph::getDegree(size_t id) const
{
  assert( doesVertexExist(id) );

  Vertices::const_iterator it = vertices.find(id);
  return it->second.neighbors.size();
}

void Graph::floydWarshall(cv::Mat &distanceMatrix, int infinity) const
{
  const int edgeWeight = 1;

  const size_t n = getVerticesCount();
  distanceMatrix.create(n, n, CV_32SC1);
  distanceMatrix.setTo(infinity);
  for (Vertices::const_iterator it1 = vertices.begin(); it1 != vertices.end(); it1++)
  {
    distanceMatrix.at<int> (it1->first, it1->first) = 0;
    for (Neighbors::const_iterator it2 = it1->second.neighbors.begin(); it2 != it1->second.neighbors.end(); it2++)
    {
      assert( it1->first != *it2 );
      distanceMatrix.at<int> (it1->first, *it2) = edgeWeight;
    }
  }

  for (Vertices::const_iterator it1 = vertices.begin(); it1 != vertices.end(); it1++)
  {
    for (Vertices::const_iterator it2 = vertices.begin(); it2 != vertices.end(); it2++)
    {
      for (Vertices::const_iterator it3 = vertices.begin(); it3 != vertices.end(); it3++)
      {
        int val1 = distanceMatrix.at<int> (it2->first, it3->first);
        int val2;
        if (distanceMatrix.at<int> (it2->first, it1->first) == infinity || distanceMatrix.at<int> (it1->first,
                                                                                                   it3->first)
            == infinity)
          val2 = val1;
        else
        {
          val2 = distanceMatrix.at<int> (it2->first, it1->first) + distanceMatrix.at<int> (it1->first, it3->first);
        }
        distanceMatrix.at<int> (it2->first, it3->first) = (val1 == infinity) ? val2 : std::min(val1, val2);
      }
    }
  }
}

const Graph::Neighbors& Graph::getNeighbors(size_t id) const
{
  assert( doesVertexExist(id) );

  Vertices::const_iterator it = vertices.find(id);
  return it->second.neighbors;
}

CirclesGridFinder::Segment::Segment(cv::Point2f _s, cv::Point2f _e) :
  s(_s), e(_e)
{
}

void computeShortestPath(Mat &predecessorMatrix, int v1, int v2, vector<int> &path);
void computePredecessorMatrix(const Mat &dm, int verticesCount, Mat &predecessorMatrix);

CirclesGridFinderParameters::CirclesGridFinderParameters()
{
  minDensity = 10;
  densityNeighborhoodSize = Size2f(16, 16);
  minDistanceToAddKeypoint = 20;
  kmeansAttempts = 100;
  convexHullFactor = 1.1f;
  keypointScale = 1;

  minGraphConfidence = 9;
  vertexGain = 2;
  vertexPenalty = -5;
  edgeGain = 1;
  edgePenalty = -5;
  existingVertexGain = 0;

  minRNGEdgeSwitchDist = 5.f;
  gridType = SYMMETRIC_GRID;
}

CirclesGridFinder::CirclesGridFinder(Size _patternSize, const vector<Point2f> &testKeypoints,
                                     const CirclesGridFinderParameters &_parameters) :
  patternSize(static_cast<size_t> (_patternSize.width), static_cast<size_t> (_patternSize.height))
{
  CV_Assert(_patternSize.height >= 0 && _patternSize.width >= 0);

  keypoints = testKeypoints;
  parameters = _parameters;
  largeHoles = 0;
  smallHoles = 0;
}

bool CirclesGridFinder::findHoles()
{
  switch (parameters.gridType)
  {
    case CirclesGridFinderParameters::SYMMETRIC_GRID:
    {
      vector<Point2f> vectors, filteredVectors, basis;
      Graph rng(0);
      computeRNG(rng, vectors);
      filterOutliersByDensity(vectors, filteredVectors);
      vector<Graph> basisGraphs;
      findBasis(filteredVectors, basis, basisGraphs);
      findMCS(basis, basisGraphs);
      break;
    }

    case CirclesGridFinderParameters::ASYMMETRIC_GRID:
    {
      vector<Point2f> vectors, tmpVectors, filteredVectors, basis;
      Graph rng(0);
      computeRNG(rng, tmpVectors);
      rng2gridGraph(rng, vectors);
      filterOutliersByDensity(vectors, filteredVectors);
      vector<Graph> basisGraphs;
      findBasis(filteredVectors, basis, basisGraphs);
      findMCS(basis, basisGraphs);
      eraseUsedGraph(basisGraphs);
      holes2 = holes;
      holes.clear();
      findMCS(basis, basisGraphs);
      break;
    }

    default:
      CV_Error(CV_StsBadArg, "Unkown pattern type");
  }
  return (isDetectionCorrect());
  //CV_Error( 0, "Detection is not correct" );
}

void CirclesGridFinder::rng2gridGraph(Graph &rng, std::vector<cv::Point2f> &vectors) const
{
  for (size_t i = 0; i < rng.getVerticesCount(); i++)
  {
    Graph::Neighbors neighbors1 = rng.getNeighbors(i);
    for (Graph::Neighbors::iterator it1 = neighbors1.begin(); it1 != neighbors1.end(); it1++)
    {
      Graph::Neighbors neighbors2 = rng.getNeighbors(*it1);
      for (Graph::Neighbors::iterator it2 = neighbors2.begin(); it2 != neighbors2.end(); it2++)
      {
        if (i < *it2)
        {
          Point2f vec1 = keypoints[i] - keypoints[*it1];
          Point2f vec2 = keypoints[*it1] - keypoints[*it2];
          if (norm(vec1 - vec2) < parameters.minRNGEdgeSwitchDist || norm(vec1 + vec2)
              < parameters.minRNGEdgeSwitchDist)
            continue;

          vectors.push_back(keypoints[i] - keypoints[*it2]);
          vectors.push_back(keypoints[*it2] - keypoints[i]);
        }
      }
    }
  }
}

void CirclesGridFinder::eraseUsedGraph(vector<Graph> &basisGraphs) const
{
  for (size_t i = 0; i < holes.size(); i++)
  {
    for (size_t j = 0; j < holes[i].size(); j++)
    {
      for (size_t k = 0; k < basisGraphs.size(); k++)
      {
        if (i != holes.size() - 1 && basisGraphs[k].areVerticesAdjacent(holes[i][j], holes[i + 1][j]))
        {
          basisGraphs[k].removeEdge(holes[i][j], holes[i + 1][j]);
        }

        if (j != holes[i].size() - 1 && basisGraphs[k].areVerticesAdjacent(holes[i][j], holes[i][j + 1]))
        {
          basisGraphs[k].removeEdge(holes[i][j], holes[i][j + 1]);
        }
      }
    }
  }
}

bool CirclesGridFinder::isDetectionCorrect()
{
  switch (parameters.gridType)
  {
    case CirclesGridFinderParameters::SYMMETRIC_GRID:
    {
      if (holes.size() != patternSize.height)
        return false;

      set<size_t> vertices;
      for (size_t i = 0; i < holes.size(); i++)
      {
        if (holes[i].size() != patternSize.width)
          return false;

        for (size_t j = 0; j < holes[i].size(); j++)
        {
          vertices.insert(holes[i][j]);
        }
      }

      return vertices.size() == patternSize.area();
    }

    case CirclesGridFinderParameters::ASYMMETRIC_GRID:
    {
      if (holes.size() < holes2.size() || holes[0].size() < holes2[0].size())
      {
        largeHoles = &holes2;
        smallHoles = &holes;
      }
      else
      {
        largeHoles = &holes;
        smallHoles = &holes2;
      }

      size_t largeWidth = patternSize.width;
      size_t largeHeight = ceil(patternSize.height / 2.);
      size_t smallWidth = patternSize.width;
      size_t smallHeight = floor(patternSize.height / 2.);

      size_t sw = smallWidth, sh = smallHeight, lw = largeWidth, lh = largeHeight;
      if (largeHoles->size() != largeHeight)
      {
        std::swap(lh, lw);
      }
      if (smallHoles->size() != smallHeight)
      {
        std::swap(sh, sw);
      }

      if (largeHoles->size() != lh || smallHoles->size() != sh)
      {
        return false;
      }

      set<size_t> vertices;
      for (size_t i = 0; i < largeHoles->size(); i++)
      {
        if (largeHoles->at(i).size() != lw)
        {
          return false;
        }

        for (size_t j = 0; j < largeHoles->at(i).size(); j++)
        {
          vertices.insert(largeHoles->at(i)[j]);
        }

        if (i < smallHoles->size())
        {
          if (smallHoles->at(i).size() != sw)
          {
            return false;
          }

          for (size_t j = 0; j < smallHoles->at(i).size(); j++)
          {
            vertices.insert(smallHoles->at(i)[j]);
          }
        }
      }
      return (vertices.size() == largeHeight * largeWidth + smallHeight * smallWidth);
    }

    default:
      CV_Error(0, "Unknown pattern type");
  }

  return false;
}

void CirclesGridFinder::findMCS(const vector<Point2f> &basis, vector<Graph> &basisGraphs)
{
  holes.clear();
  Path longestPath;
  size_t bestGraphIdx = findLongestPath(basisGraphs, longestPath);
  vector<size_t> holesRow = longestPath.vertices;

  while (holesRow.size() > std::max(patternSize.width, patternSize.height))
  {
    holesRow.pop_back();
    holesRow.erase(holesRow.begin());
  }

  if (bestGraphIdx == 0)
  {
    holes.push_back(holesRow);
    size_t w = holes[0].size();
    size_t h = holes.size();

    //parameters.minGraphConfidence = holes[0].size() * parameters.vertexGain + (holes[0].size() - 1) * parameters.edgeGain;
    //parameters.minGraphConfidence = holes[0].size() * parameters.vertexGain + (holes[0].size() / 2) * parameters.edgeGain;
    //parameters.minGraphConfidence = holes[0].size() * parameters.existingVertexGain + (holes[0].size() / 2) * parameters.edgeGain;
    parameters.minGraphConfidence = holes[0].size() * parameters.existingVertexGain;
    for (size_t i = h; i < patternSize.height; i++)
    {
      addHolesByGraph(basisGraphs, true, basis[1]);
    }

    //parameters.minGraphConfidence = holes.size() * parameters.existingVertexGain + (holes.size() / 2) * parameters.edgeGain;
    parameters.minGraphConfidence = holes.size() * parameters.existingVertexGain;

    for (size_t i = w; i < patternSize.width; i++)
    {
      addHolesByGraph(basisGraphs, false, basis[0]);
    }
  }
  else
  {
    holes.resize(holesRow.size());
    for (size_t i = 0; i < holesRow.size(); i++)
      holes[i].push_back(holesRow[i]);

    size_t w = holes[0].size();
    size_t h = holes.size();

    parameters.minGraphConfidence = holes.size() * parameters.existingVertexGain;
    for (size_t i = w; i < patternSize.width; i++)
    {
      addHolesByGraph(basisGraphs, false, basis[0]);
    }

    parameters.minGraphConfidence = holes[0].size() * parameters.existingVertexGain;
    for (size_t i = h; i < patternSize.height; i++)
    {
      addHolesByGraph(basisGraphs, true, basis[1]);
    }
  }
}

Mat CirclesGridFinder::rectifyGrid(Size detectedGridSize, const vector<Point2f>& centers,
                                   const vector<Point2f> &keypoints, vector<Point2f> &warpedKeypoints)
{
  assert( !centers.empty() );
  const float edgeLength = 30;
  const Point2f offset(150, 150);

  vector<Point2f> dstPoints;
  for (int i = 0; i < detectedGridSize.height; i++)
  {
    for (int j = 0; j < detectedGridSize.width; j++)
    {
      dstPoints.push_back(offset + Point2f(edgeLength * j, edgeLength * i));
    }
  }

  Mat H = findHomography(Mat(centers), Mat(dstPoints), CV_RANSAC);
  //Mat H = findHomography( Mat( corners ), Mat( dstPoints ) );

  vector<Point2f> srcKeypoints;
  for (size_t i = 0; i < keypoints.size(); i++)
  {
    srcKeypoints.push_back(keypoints[i]);
  }

  Mat dstKeypointsMat;
  transform(Mat(srcKeypoints), dstKeypointsMat, H);
  vector<Point2f> dstKeypoints;
  convertPointsHomogeneous(dstKeypointsMat, dstKeypoints);

  warpedKeypoints.clear();
  for (size_t i = 0; i < dstKeypoints.size(); i++)
  {
    Point2f pt = dstKeypoints[i];
    warpedKeypoints.push_back(pt);
  }

  return H;
}

size_t CirclesGridFinder::findNearestKeypoint(Point2f pt) const
{
  size_t bestIdx = -1;
  double minDist = std::numeric_limits<double>::max();
  for (size_t i = 0; i < keypoints.size(); i++)
  {
    double dist = norm(pt - keypoints[i]);
    if (dist < minDist)
    {
      minDist = dist;
      bestIdx = i;
    }
  }
  return bestIdx;
}

void CirclesGridFinder::addPoint(Point2f pt, vector<size_t> &points)
{
  size_t ptIdx = findNearestKeypoint(pt);
  if (norm(keypoints[ptIdx] - pt) > parameters.minDistanceToAddKeypoint)
  {
    Point2f kpt = Point2f(pt);
    keypoints.push_back(kpt);
    points.push_back(keypoints.size() - 1);
  }
  else
  {
    points.push_back(ptIdx);
  }
}

void CirclesGridFinder::findCandidateLine(vector<size_t> &line, size_t seedLineIdx, bool addRow, Point2f basisVec,
                                          vector<size_t> &seeds)
{
  line.clear();
  seeds.clear();

  if (addRow)
  {
    for (size_t i = 0; i < holes[seedLineIdx].size(); i++)
    {
      Point2f pt = keypoints[holes[seedLineIdx][i]] + basisVec;
      addPoint(pt, line);
      seeds.push_back(holes[seedLineIdx][i]);
    }
  }
  else
  {
    for (size_t i = 0; i < holes.size(); i++)
    {
      Point2f pt = keypoints[holes[i][seedLineIdx]] + basisVec;
      addPoint(pt, line);
      seeds.push_back(holes[i][seedLineIdx]);
    }
  }

  assert( line.size() == seeds.size() );
}

void CirclesGridFinder::findCandidateHoles(vector<size_t> &above, vector<size_t> &below, bool addRow, Point2f basisVec,
                                           vector<size_t> &aboveSeeds, vector<size_t> &belowSeeds)
{
  above.clear();
  below.clear();
  aboveSeeds.clear();
  belowSeeds.clear();

  findCandidateLine(above, 0, addRow, -basisVec, aboveSeeds);
  size_t lastIdx = addRow ? holes.size() - 1 : holes[0].size() - 1;
  findCandidateLine(below, lastIdx, addRow, basisVec, belowSeeds);

  assert( below.size() == above.size() );
  assert( belowSeeds.size() == aboveSeeds.size() );
  assert( below.size() == belowSeeds.size() );
}

bool CirclesGridFinder::areCentersNew(const vector<size_t> &newCenters, const vector<vector<size_t> > &holes)
{
  for (size_t i = 0; i < newCenters.size(); i++)
  {
    for (size_t j = 0; j < holes.size(); j++)
    {
      if (holes[j].end() != std::find(holes[j].begin(), holes[j].end(), newCenters[i]))
      {
        return false;
      }
    }
  }

  return true;
}

void CirclesGridFinder::insertWinner(float aboveConfidence, float belowConfidence, float minConfidence, bool addRow,
                                     const vector<size_t> &above, const vector<size_t> &below,
                                     vector<vector<size_t> > &holes)
{
  if (aboveConfidence < minConfidence && belowConfidence < minConfidence)
    return;

  if (addRow)
  {
    if (aboveConfidence >= belowConfidence)
    {
      if (!areCentersNew(above, holes))
        CV_Error( 0, "Centers are not new" );

      holes.insert(holes.begin(), above);
    }
    else
    {
      if (!areCentersNew(below, holes))
        CV_Error( 0, "Centers are not new" );

      holes.insert(holes.end(), below);
    }
  }
  else
  {
    if (aboveConfidence >= belowConfidence)
    {
      if (!areCentersNew(above, holes))
        CV_Error( 0, "Centers are not new" );

      for (size_t i = 0; i < holes.size(); i++)
      {
        holes[i].insert(holes[i].begin(), above[i]);
      }
    }
    else
    {
      if (!areCentersNew(below, holes))
        CV_Error( 0, "Centers are not new" );

      for (size_t i = 0; i < holes.size(); i++)
      {
        holes[i].insert(holes[i].end(), below[i]);
      }
    }
  }
}

float CirclesGridFinder::computeGraphConfidence(const vector<Graph> &basisGraphs, bool addRow,
                                                const vector<size_t> &points, const vector<size_t> &seeds)
{
  assert( points.size() == seeds.size() );
  float confidence = 0;
  const size_t vCount = basisGraphs[0].getVerticesCount();
  assert( basisGraphs[0].getVerticesCount() == basisGraphs[1].getVerticesCount() );

  for (size_t i = 0; i < seeds.size(); i++)
  {
    if (seeds[i] < vCount && points[i] < vCount)
    {
      if (!basisGraphs[addRow].areVerticesAdjacent(seeds[i], points[i]))
      {
        confidence += parameters.vertexPenalty;
      }
      else
      {
        confidence += parameters.vertexGain;
      }
    }

    if (points[i] < vCount)
    {
      confidence += parameters.existingVertexGain;
    }
  }

  for (size_t i = 1; i < points.size(); i++)
  {
    if (points[i - 1] < vCount && points[i] < vCount)
    {
      if (!basisGraphs[!addRow].areVerticesAdjacent(points[i - 1], points[i]))
      {
        confidence += parameters.edgePenalty;
      }
      else
      {
        confidence += parameters.edgeGain;
      }
    }
  }
  return confidence;

}

void CirclesGridFinder::addHolesByGraph(const vector<Graph> &basisGraphs, bool addRow, Point2f basisVec)
{
  vector<size_t> above, below, aboveSeeds, belowSeeds;
  findCandidateHoles(above, below, addRow, basisVec, aboveSeeds, belowSeeds);
  float aboveConfidence = computeGraphConfidence(basisGraphs, addRow, above, aboveSeeds);
  float belowConfidence = computeGraphConfidence(basisGraphs, addRow, below, belowSeeds);

  insertWinner(aboveConfidence, belowConfidence, parameters.minGraphConfidence, addRow, above, below, holes);
}

void CirclesGridFinder::filterOutliersByDensity(const vector<Point2f> &samples, vector<Point2f> &filteredSamples)
{
  if (samples.empty())
    CV_Error( 0, "samples is empty" );

  filteredSamples.clear();

  for (size_t i = 0; i < samples.size(); i++)
  {
    Rect_<float> rect(samples[i] - Point2f(parameters.densityNeighborhoodSize) * 0.5,
                      parameters.densityNeighborhoodSize);
    int neighborsCount = 0;
    for (size_t j = 0; j < samples.size(); j++)
    {
      if (rect.contains(samples[j]))
        neighborsCount++;
    }
    if (neighborsCount >= parameters.minDensity)
      filteredSamples.push_back(samples[i]);
  }

  if (filteredSamples.empty())
    CV_Error( 0, "filteredSamples is empty" );
}

void CirclesGridFinder::findBasis(const vector<Point2f> &samples, vector<Point2f> &basis, vector<Graph> &basisGraphs)
{
  basis.clear();
  Mat bestLabels;
  TermCriteria termCriteria;
  Mat centers;
  const int clustersCount = 4;
  kmeans(Mat(samples).reshape(1, 0), clustersCount, bestLabels, termCriteria, parameters.kmeansAttempts,
         KMEANS_RANDOM_CENTERS, &centers);
  assert( centers.type() == CV_32FC1 );

  vector<int> basisIndices;
  //TODO: only remove duplicate
  for (int i = 0; i < clustersCount; i++)
  {
    int maxIdx = (fabs(centers.at<float> (i, 0)) < fabs(centers.at<float> (i, 1)));
    if (centers.at<float> (i, maxIdx) > 0)
    {
      Point2f vec(centers.at<float> (i, 0), centers.at<float> (i, 1));
      basis.push_back(vec);
      basisIndices.push_back(i);
    }
  }
  if (basis.size() != 2)
    CV_Error(0, "Basis size is not 2");

  if (basis[1].x > basis[0].x)
  {
    std::swap(basis[0], basis[1]);
    std::swap(basisIndices[0], basisIndices[1]);
  }

  const float minBasisDif = 2;
  if (norm(basis[0] - basis[1]) < minBasisDif)
    CV_Error(0, "degenerate basis" );

  vector<vector<Point2f> > clusters(2), hulls(2);
  for (size_t k = 0; k < samples.size(); k++)
  {
    int label = bestLabels.at<int> (k, 0);
    int idx = -1;
    if (label == basisIndices[0])
      idx = 0;
    if (label == basisIndices[1])
      idx = 1;
    if (idx >= 0)
    {
      clusters[idx].push_back(basis[idx] + parameters.convexHullFactor * (samples[k] - basis[idx]));
    }
  }
  for (size_t i = 0; i < basis.size(); i++)
  {
    convexHull(Mat(clusters[i]), hulls[i]);
  }

  basisGraphs.resize(basis.size(), Graph(keypoints.size()));
  for (size_t i = 0; i < keypoints.size(); i++)
  {
    for (size_t j = 0; j < keypoints.size(); j++)
    {
      if (i == j)
        continue;

      Point2f vec = keypoints[i] - keypoints[j];

      for (size_t k = 0; k < hulls.size(); k++)
      {
        if (pointPolygonTest(Mat(hulls[k]), vec, false) >= 0)
        {
          basisGraphs[k].addEdge(i, j);
        }
      }
    }
  }
  if (basisGraphs.size() != 2)
    CV_Error(0, "Number of basis graphs is not 2");
}

void CirclesGridFinder::computeRNG(Graph &rng, std::vector<cv::Point2f> &vectors, Mat *drawImage) const
{
  rng = Graph(keypoints.size());
  vectors.clear();

  //TODO: use more fast algorithm instead of naive N^3
  for (size_t i = 0; i < keypoints.size(); i++)
  {
    for (size_t j = 0; j < keypoints.size(); j++)
    {
      if (i == j)
        continue;

      Point2f vec = keypoints[i] - keypoints[j];
      double dist = norm(vec);

      bool isNeighbors = true;
      for (size_t k = 0; k < keypoints.size(); k++)
      {
        if (k == i || k == j)
          continue;

        double dist1 = norm(keypoints[i] - keypoints[k]);
        double dist2 = norm(keypoints[j] - keypoints[k]);
        if (dist1 < dist && dist2 < dist)
        {
          isNeighbors = false;
          break;
        }
      }

      if (isNeighbors)
      {
        rng.addEdge(i, j);
        vectors.push_back(keypoints[i] - keypoints[j]);
        if (drawImage != 0)
        {
          line(*drawImage, keypoints[i], keypoints[j], Scalar(255, 0, 0), 2);
          circle(*drawImage, keypoints[i], 3, Scalar(0, 0, 255), -1);
          circle(*drawImage, keypoints[j], 3, Scalar(0, 0, 255), -1);
        }
      }
    }
  }
}

void computePredecessorMatrix(const Mat &dm, int verticesCount, Mat &predecessorMatrix)
{
  assert( dm.type() == CV_32SC1 );
  predecessorMatrix.create(verticesCount, verticesCount, CV_32SC1);
  predecessorMatrix = -1;
  for (int i = 0; i < predecessorMatrix.rows; i++)
  {
    for (int j = 0; j < predecessorMatrix.cols; j++)
    {
      int dist = dm.at<int> (i, j);
      for (int k = 0; k < verticesCount; k++)
      {
        if (dm.at<int> (i, k) == dist - 1 && dm.at<int> (k, j) == 1)
        {
          predecessorMatrix.at<int> (i, j) = k;
          break;
        }
      }
    }
  }
}

void computeShortestPath(Mat &predecessorMatrix, size_t v1, size_t v2, vector<size_t> &path)
{
  if (predecessorMatrix.at<int> (v1, v2) < 0)
  {
    path.push_back(v1);
    return;
  }

  computeShortestPath(predecessorMatrix, v1, predecessorMatrix.at<int> (v1, v2), path);
  path.push_back(v2);
}

size_t CirclesGridFinder::findLongestPath(vector<Graph> &basisGraphs, Path &bestPath)
{
  vector<Path> longestPaths(1);
  vector<int> confidences;

  size_t bestGraphIdx = 0;
  const int infinity = -1;
  for (size_t graphIdx = 0; graphIdx < basisGraphs.size(); graphIdx++)
  {
    const Graph &g = basisGraphs[graphIdx];
    Mat distanceMatrix;
    g.floydWarshall(distanceMatrix, infinity);
    Mat predecessorMatrix;
    computePredecessorMatrix(distanceMatrix, g.getVerticesCount(), predecessorMatrix);

    double maxVal;
    Point maxLoc;
    assert (infinity < 0);
    minMaxLoc(distanceMatrix, 0, &maxVal, 0, &maxLoc);

    if (maxVal > longestPaths[0].length)
    {
      longestPaths.clear();
      confidences.clear();
      bestGraphIdx = graphIdx;
    }
    if (longestPaths.empty() || (maxVal == longestPaths[0].length && graphIdx == bestGraphIdx))
    {
      Path path = Path(maxLoc.x, maxLoc.y, cvRound(maxVal));
      CV_Assert(maxLoc.x >= 0 && maxLoc.y >= 0)
        ;
      size_t id1 = static_cast<size_t> (maxLoc.x);
      size_t id2 = static_cast<size_t> (maxLoc.y);
      computeShortestPath(predecessorMatrix, id1, id2, path.vertices);
      longestPaths.push_back(path);

      int conf = 0;
      for (size_t v2 = 0; v2 < path.vertices.size(); v2++)
      {
        conf += basisGraphs[1 - (int)graphIdx].getDegree(v2);
      }
      confidences.push_back(conf);
    }
  }
  //if( bestGraphIdx != 0 )
  //CV_Error( 0, "" );

  int maxConf = -1;
  int bestPathIdx = -1;
  for (size_t i = 0; i < confidences.size(); i++)
  {
    if (confidences[i] > maxConf)
    {
      maxConf = confidences[i];
      bestPathIdx = i;
    }
  }

  //int bestPathIdx = rand() % longestPaths.size();
  bestPath = longestPaths.at(bestPathIdx);
  bool needReverse = (bestGraphIdx == 0 && keypoints[bestPath.lastVertex].x < keypoints[bestPath.firstVertex].x)
      || (bestGraphIdx == 1 && keypoints[bestPath.lastVertex].y < keypoints[bestPath.firstVertex].y);
  if (needReverse)
  {
    std::swap(bestPath.lastVertex, bestPath.firstVertex);
    std::reverse(bestPath.vertices.begin(), bestPath.vertices.end());
  }
  return bestGraphIdx;
}

void CirclesGridFinder::drawBasis(const vector<Point2f> &basis, Point2f origin, Mat &drawImg) const
{
  for (size_t i = 0; i < basis.size(); i++)
  {
    Point2f pt(basis[i]);
    line(drawImg, origin, origin + pt, Scalar(0, i * 255, 0), 2);
  }
}

void CirclesGridFinder::drawBasisGraphs(const vector<Graph> &basisGraphs, Mat &drawImage, bool drawEdges,
                                        bool drawVertices) const
{
  //const int vertexRadius = 1;
  const int vertexRadius = 3;
  const Scalar vertexColor = Scalar(0, 0, 255);
  const int vertexThickness = -1;

  const Scalar edgeColor = Scalar(255, 0, 0);
  //const int edgeThickness = 1;
  const int edgeThickness = 2;

  if (drawEdges)
  {
    for (size_t i = 0; i < basisGraphs.size(); i++)
    {
      for (size_t v1 = 0; v1 < basisGraphs[i].getVerticesCount(); v1++)
      {
        for (size_t v2 = 0; v2 < basisGraphs[i].getVerticesCount(); v2++)
        {
          if (basisGraphs[i].areVerticesAdjacent(v1, v2))
          {
            line(drawImage, keypoints[v1], keypoints[v2], edgeColor, edgeThickness);
          }
        }
      }
    }
  }
  if (drawVertices)
  {
    for (size_t v = 0; v < basisGraphs[0].getVerticesCount(); v++)
    {
      circle(drawImage, keypoints[v], vertexRadius, vertexColor, vertexThickness);
    }
  }
}

void CirclesGridFinder::drawHoles(const Mat &srcImage, Mat &drawImage) const
{
  //const int holeRadius = 4;
  //const int holeRadius = 2;
  //const int holeThickness = 1;
  const int holeRadius = 3;
  const int holeThickness = -1;

  //const Scalar holeColor = Scalar(0, 0, 255);
  const Scalar holeColor = Scalar(0, 255, 0);

  if (srcImage.channels() == 1)
    cvtColor(srcImage, drawImage, CV_GRAY2RGB);
  else
    srcImage.copyTo(drawImage);

  for (size_t i = 0; i < holes.size(); i++)
  {
    for (size_t j = 0; j < holes[i].size(); j++)
    {
      if (j != holes[i].size() - 1)
        line(drawImage, keypoints[holes[i][j]], keypoints[holes[i][j + 1]], Scalar(255, 0, 0), 2);
      if (i != holes.size() - 1)
        line(drawImage, keypoints[holes[i][j]], keypoints[holes[i + 1][j]], Scalar(255, 0, 0), 2);

      //circle(drawImage, keypoints[holes[i][j]], holeRadius, holeColor, holeThickness);
      circle(drawImage, keypoints[holes[i][j]], holeRadius, holeColor, holeThickness);
    }
  }
}

Size CirclesGridFinder::getDetectedGridSize() const
{
  if (holes.size() == 0)
    return Size(0, 0);

  return Size(holes[0].size(), holes.size());
}

void CirclesGridFinder::getHoles(vector<Point2f> &outHoles) const
{
  outHoles.clear();

  for (size_t i = 0; i < holes.size(); i++)
  {
    for (size_t j = 0; j < holes[i].size(); j++)
    {
      outHoles.push_back(keypoints[holes[i][j]]);
    }
  }
}

bool areIndicesCorrect(Point pos, vector<vector<size_t> > *points)
{
  if (pos.y < 0 || pos.x < 0)
    return false;
  return (static_cast<size_t> (pos.y) < points->size() && static_cast<size_t> (pos.x) < points->at(pos.y).size());
}

void CirclesGridFinder::getAsymmetricHoles(std::vector<cv::Point2f> &outHoles) const
{
  outHoles.clear();

  vector<Point> largeCornerIndices, smallCornerIndices;
  vector<Point> firstSteps, secondSteps;
  size_t cornerIdx = getFirstCorner(largeCornerIndices, smallCornerIndices, firstSteps, secondSteps);
  CV_Assert(largeHoles != 0 && smallHoles != 0)
    ;

  Point srcLargePos = largeCornerIndices[cornerIdx];
  Point srcSmallPos = smallCornerIndices[cornerIdx];

  while (areIndicesCorrect(srcLargePos, largeHoles) || areIndicesCorrect(srcSmallPos, smallHoles))
  {
    Point largePos = srcLargePos;
    while (areIndicesCorrect(largePos, largeHoles))
    {
      outHoles.push_back(keypoints[largeHoles->at(largePos.y)[largePos.x]]);
      largePos += firstSteps[cornerIdx];
    }
    srcLargePos += secondSteps[cornerIdx];

    Point smallPos = srcSmallPos;
    while (areIndicesCorrect(smallPos, smallHoles))
    {
      outHoles.push_back(keypoints[smallHoles->at(smallPos.y)[smallPos.x]]);
      smallPos += firstSteps[cornerIdx];
    }
    srcSmallPos += secondSteps[cornerIdx];
  }
}

double CirclesGridFinder::getDirection(Point2f p1, Point2f p2, Point2f p3)
{
  Point2f a = p3 - p1;
  Point2f b = p2 - p1;
  return a.x * b.y - a.y * b.x;
}

bool CirclesGridFinder::areSegmentsIntersecting(Segment seg1, Segment seg2)
{
  bool doesStraddle1 = (getDirection(seg2.s, seg2.e, seg1.s) * getDirection(seg2.s, seg2.e, seg1.e)) < 0;
  bool doesStraddle2 = (getDirection(seg1.s, seg1.e, seg2.s) * getDirection(seg1.s, seg1.e, seg2.e)) < 0;
  return doesStraddle1 && doesStraddle2;

  /*
   Point2f t1 = e1-s1;
   Point2f n1(t1.y, -t1.x);
   double c1 = -n1.ddot(s1);

   Point2f t2 = e2-s2;
   Point2f n2(t2.y, -t2.x);
   double c2 = -n2.ddot(s2);

   bool seg1 = ((n1.ddot(s2) + c1) * (n1.ddot(e2) + c1)) <= 0;
   bool seg1 = ((n2.ddot(s1) + c2) * (n2.ddot(e1) + c2)) <= 0;

   return seg1 && seg2;
   */
}

void CirclesGridFinder::getCornerSegments(const vector<vector<size_t> > &points, vector<vector<Segment> > &segments,
                                          vector<Point> &cornerIndices, vector<Point> &firstSteps,
                                          vector<Point> &secondSteps) const
{
  segments.clear();
  cornerIndices.clear();
  firstSteps.clear();
  secondSteps.clear();
  size_t h = points.size();
  size_t w = points[0].size();
  CV_Assert(h >= 2 && w >= 2)
    ;

  //all 8 segments with one end in a corner
  vector<Segment> corner;
  corner.push_back(Segment(keypoints[points[1][0]], keypoints[points[0][0]]));
  corner.push_back(Segment(keypoints[points[0][0]], keypoints[points[0][1]]));
  segments.push_back(corner);
  cornerIndices.push_back(Point(0, 0));
  firstSteps.push_back(Point(1, 0));
  secondSteps.push_back(Point(0, 1));
  corner.clear();

  corner.push_back(Segment(keypoints[points[0][w - 2]], keypoints[points[0][w - 1]]));
  corner.push_back(Segment(keypoints[points[0][w - 1]], keypoints[points[1][w - 1]]));
  segments.push_back(corner);
  cornerIndices.push_back(Point(w - 1, 0));
  firstSteps.push_back(Point(0, 1));
  secondSteps.push_back(Point(-1, 0));
  corner.clear();

  corner.push_back(Segment(keypoints[points[h - 2][w - 1]], keypoints[points[h - 1][w - 1]]));
  corner.push_back(Segment(keypoints[points[h - 1][w - 1]], keypoints[points[h - 1][w - 2]]));
  segments.push_back(corner);
  cornerIndices.push_back(Point(w - 1, h - 1));
  firstSteps.push_back(Point(-1, 0));
  secondSteps.push_back(Point(0, -1));
  corner.clear();

  corner.push_back(Segment(keypoints[points[h - 1][1]], keypoints[points[h - 1][0]]));
  corner.push_back(Segment(keypoints[points[h - 1][0]], keypoints[points[h - 2][0]]));
  cornerIndices.push_back(Point(0, h - 1));
  firstSteps.push_back(Point(0, -1));
  secondSteps.push_back(Point(1, 0));
  segments.push_back(corner);
  corner.clear();

  //y axis is inverted in computer vision so we check < 0
  bool isClockwise =
      getDirection(keypoints[points[0][0]], keypoints[points[0][w - 1]], keypoints[points[h - 1][w - 1]]) < 0;
  if (!isClockwise)
  {
#ifdef DEBUG_CIRCLES
    cout << "Corners are counterclockwise" << endl;
#endif
    std::reverse(segments.begin(), segments.end());
  }
}

bool CirclesGridFinder::doesIntersectionExist(const vector<Segment> &corner, const vector<vector<Segment> > &segments)
{
  for (size_t i = 0; i < corner.size(); i++)
  {
    for (size_t j = 0; j < segments.size(); j++)
    {
      for (size_t k = 0; k < segments[j].size(); k++)
      {
        if (areSegmentsIntersecting(corner[i], segments[j][k]))
          return true;
      }
    }
  }

  return false;
}

size_t CirclesGridFinder::getFirstCorner(vector<Point> &largeCornerIndices, vector<Point> &smallCornerIndices, vector<
    Point> &firstSteps, vector<Point> &secondSteps) const
{
  vector<vector<Segment> > largeSegments;
  vector<vector<Segment> > smallSegments;

  getCornerSegments(*largeHoles, largeSegments, largeCornerIndices, firstSteps, secondSteps);
  getCornerSegments(*smallHoles, smallSegments, smallCornerIndices, firstSteps, secondSteps);

  const size_t cornersCount = 4;
  CV_Assert(largeSegments.size() == cornersCount)
    ;

  bool isInsider[cornersCount];

  for (size_t i = 0; i < cornersCount; i++)
  {
    isInsider[i] = doesIntersectionExist(largeSegments[i], smallSegments);
  }

  int cornerIdx = 0;
  bool waitOutsider = true;

  while (true)
  {
    if (waitOutsider)
    {
      if (!isInsider[(cornerIdx + 1) % cornersCount])
        waitOutsider = false;
    }
    else
    {
      if (isInsider[(cornerIdx + 1) % cornersCount])
        break;
    }

    cornerIdx = (cornerIdx + 1) % cornersCount;
  }

  return cornerIdx;
}
