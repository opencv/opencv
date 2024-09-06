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

#include "precomp.hpp"
#include "circlesgrid.hpp"
#include <limits>

 // Requires CMake flag: DEBUG_opencv_calib=ON
//#define DEBUG_CIRCLES

#ifdef DEBUG_CIRCLES
#  include <iostream>
#  include "opencv2/opencv_modules.hpp"
#  ifdef HAVE_OPENCV_HIGHGUI
#    include "opencv2/highgui.hpp"
#  else
#    undef DEBUG_CIRCLES
#  endif
#endif

namespace cv {

#ifdef DEBUG_CIRCLES
void drawPoints(const std::vector<Point2f> &points, Mat &outImage, int radius = 2,  Scalar color = Scalar::all(255), int thickness = -1)
{
  for(size_t i=0; i<points.size(); i++)
  {
    circle(outImage, points[i], radius, color, thickness);
  }
}
#endif

void CirclesGridClusterFinder::hierarchicalClustering(const std::vector<Point2f> &points, const Size &patternSz, std::vector<Point2f> &patternPoints)
{
    int j, n = (int)points.size();
    size_t pn = static_cast<size_t>(patternSz.area());

    patternPoints.clear();
    if (pn >= points.size())
    {
        if (pn == points.size())
            patternPoints = points;
        return;
    }

    Mat dists(n, n, CV_32FC1, Scalar(0));
    Mat distsMask(dists.size(), CV_8UC1, Scalar(0));
    for(int i = 0; i < n; i++)
    {
        for(j = i+1; j < n; j++)
        {
            dists.at<float>(i, j) = (float)norm(points[i] - points[j]);
            distsMask.at<uchar>(i, j) = 255;
            //TODO: use symmetry
            distsMask.at<uchar>(j, i) = 255;//distsMask.at<uchar>(i, j);
            dists.at<float>(j, i) = dists.at<float>(i, j);
        }
    }

    std::vector<std::list<size_t> > clusters(points.size());
    for(size_t i=0; i<points.size(); i++)
    {
        clusters[i].push_back(i);
    }

    int patternClusterIdx = 0;
    while(clusters[patternClusterIdx].size() < pn)
    {
        Point minLoc;
        minMaxLoc(dists, 0, 0, &minLoc, 0, distsMask);
        int minIdx = std::min(minLoc.x, minLoc.y);
        int maxIdx = std::max(minLoc.x, minLoc.y);

        distsMask.row(maxIdx).setTo(0);
        distsMask.col(maxIdx).setTo(0);
        Mat tmpRow = dists.row(minIdx);
        Mat tmpCol = dists.col(minIdx);
        cv::min(dists.row(minLoc.x), dists.row(minLoc.y), tmpRow);
        tmpRow = tmpRow.t();
        tmpRow.copyTo(tmpCol);

        clusters[minIdx].splice(clusters[minIdx].end(), clusters[maxIdx]);
        patternClusterIdx = minIdx;
    }

    //the largest cluster can have more than pn points -- we need to filter out such situations
    if(clusters[patternClusterIdx].size() != static_cast<size_t>(patternSz.area()))
    {
      return;
    }

    patternPoints.reserve(clusters[patternClusterIdx].size());
    for(std::list<size_t>::iterator it = clusters[patternClusterIdx].begin(); it != clusters[patternClusterIdx].end();++it)
    {
        patternPoints.push_back(points[*it]);
    }
}

void CirclesGridClusterFinder::findGrid(const std::vector<cv::Point2f> &points, cv::Size _patternSize, std::vector<Point2f>& centers)
{
  patternSize = _patternSize;
  centers.clear();
  if(points.empty())
  {
    return;
  }

  std::vector<Point2f> patternPoints;
  hierarchicalClustering(points, patternSize, patternPoints);
  if(patternPoints.empty())
  {
    return;
  }

#ifdef DEBUG_CIRCLES
  Mat patternPointsImage(1024, 1248, CV_8UC1, Scalar(0));
  drawPoints(patternPoints, patternPointsImage);
  imshow("pattern points", patternPointsImage);
#endif

  std::vector<Point2f> hull2f;
  convexHull(patternPoints, hull2f, false);
  const size_t cornersCount = isAsymmetricGrid ? 6 : 4;
  if(hull2f.size() < cornersCount)
    return;

  std::vector<Point2f> corners;
  findCorners(hull2f, corners);
  if(corners.size() != cornersCount)
    return;

  std::vector<Point2f> outsideCorners, sortedCorners;
  if(isAsymmetricGrid)
  {
    findOutsideCorners(corners, outsideCorners);
    const size_t outsideCornersCount = 2;
    if(outsideCorners.size() != outsideCornersCount)
      return;
  }
  getSortedCorners(hull2f, patternPoints, corners, outsideCorners, sortedCorners);
  if(sortedCorners.size() != cornersCount)
    return;

  std::vector<Point2f> rectifiedPatternPoints;
  rectifyPatternPoints(patternPoints, sortedCorners, rectifiedPatternPoints);
  if(patternPoints.size() != rectifiedPatternPoints.size())
    return;

  parsePatternPoints(patternPoints, rectifiedPatternPoints, centers);
}

void CirclesGridClusterFinder::findCorners(const std::vector<cv::Point2f> &hull2f, std::vector<cv::Point2f> &corners)
{
  //find angles (cosines) of vertices in convex hull
  std::vector<float> angles;
  for(size_t i=0; i<hull2f.size(); i++)
  {
    Point2f vec1 = hull2f[(i+1) % hull2f.size()] - hull2f[i % hull2f.size()];
    Point2f vec2 = hull2f[(i-1 + static_cast<int>(hull2f.size())) % hull2f.size()] - hull2f[i % hull2f.size()];
    float angle = (float)(vec1.ddot(vec2) / (norm(vec1) * norm(vec2)));
    angles.push_back(angle);
  }

  //sort angles by cosine
  //corners are the most sharp angles (6)
  Mat anglesMat = Mat(angles);
  Mat sortedIndices;
  sortIdx(anglesMat, sortedIndices, SORT_EVERY_ROW + SORT_DESCENDING);
  CV_Assert(sortedIndices.type() == CV_32SC1);
  CV_Assert(sortedIndices.rows == 1);
  const int cornersCount = isAsymmetricGrid ? 6 : 4;
  Mat cornersIndices;
  cv::sort(sortedIndices.colRange(0, cornersCount), cornersIndices, SORT_EVERY_ROW + SORT_ASCENDING);
  corners.clear();
  for(int i=0; i<cornersCount; i++)
  {
    corners.push_back(hull2f[cornersIndices.at<int>(i)]);
  }
}

void CirclesGridClusterFinder::findOutsideCorners(const std::vector<cv::Point2f> &corners, std::vector<cv::Point2f> &outsideCorners)
{
  CV_Assert(!corners.empty());
  outsideCorners.clear();
  //find two pairs of the most nearest corners
  const size_t n = corners.size();

#ifdef DEBUG_CIRCLES
  Mat cornersImage(1024, 1248, CV_8UC1, Scalar(0));
  drawPoints(corners, cornersImage);
  imshow("corners", cornersImage);
#endif

  std::vector<Point2f> tangentVectors(n);
  for(size_t k=0; k < n; k++)
  {
    Point2f diff = corners[(k + 1) % n] - corners[k];
    tangentVectors[k] = diff * (1.0f / norm(diff));
  }

  //compute angles between all sides
  Mat cosAngles((int)n, (int)n, CV_32FC1, 0.0f);
  for(size_t i = 0; i < n; i++)
  {
    for(size_t j = i + 1; j < n; j++)
    {
      float val = fabs(tangentVectors[i].dot(tangentVectors[j]));
      cosAngles.at<float>((int)i, (int)j) = val;
      cosAngles.at<float>((int)j, (int)i) = val;
    }
  }

  //find two parallel sides to which outside corners belong
  Point maxLoc;
  minMaxLoc(cosAngles, 0, 0, 0, &maxLoc);
  const int diffBetweenFalseLines = 3;
  if(abs(maxLoc.x - maxLoc.y) == diffBetweenFalseLines)
  {
    cosAngles.row(maxLoc.x).setTo(0.0f);
    cosAngles.col(maxLoc.x).setTo(0.0f);
    cosAngles.row(maxLoc.y).setTo(0.0f);
    cosAngles.col(maxLoc.y).setTo(0.0f);
    minMaxLoc(cosAngles, 0, 0, 0, &maxLoc);
  }

#ifdef DEBUG_CIRCLES
  Mat linesImage(1024, 1248, CV_8UC1, Scalar(0));
  line(linesImage, corners[maxLoc.y], corners[(maxLoc.y + 1) % n], Scalar(255));
  line(linesImage, corners[maxLoc.x], corners[(maxLoc.x + 1) % n], Scalar(255));
  imshow("lines", linesImage);
#endif

  int maxIdx = std::max(maxLoc.x, maxLoc.y);
  int minIdx = std::min(maxLoc.x, maxLoc.y);
  const int bigDiff = 4;
  if(maxIdx - minIdx == bigDiff)
  {
    minIdx += (int)n;
    std::swap(maxIdx, minIdx);
  }
  if(maxIdx - minIdx != (int)n - bigDiff)
  {
    return;
  }

  int outsidersSegmentIdx = (minIdx + maxIdx) / 2;

  outsideCorners.push_back(corners[outsidersSegmentIdx % n]);
  outsideCorners.push_back(corners[(outsidersSegmentIdx + 1) % n]);

#ifdef DEBUG_CIRCLES
  drawPoints(outsideCorners, cornersImage, 2, Scalar(128));
  imshow("corners", cornersImage);
#endif
}

namespace {
double pointLineDistance(const cv::Point2f &p, const cv::Vec4f &line)
{
  Vec3f pa( line[0], line[1], 1 );
  Vec3f pb( line[2], line[3], 1 );
  Vec3f l = pa.cross(pb);
  return std::abs((p.x * l[0] + p.y * l[1] + l[2])) * 1.0 /
         std::sqrt(double(l[0] * l[0] + l[1] * l[1]));
}
}

void CirclesGridClusterFinder::getSortedCorners(const std::vector<cv::Point2f> &hull2f, const std::vector<cv::Point2f> &patternPoints, const std::vector<cv::Point2f> &corners, const std::vector<cv::Point2f> &outsideCorners, std::vector<cv::Point2f> &sortedCorners)
{
  Point2f firstCorner;
  if(isAsymmetricGrid)
  {
    Point2f center = std::accumulate(corners.begin(), corners.end(), Point2f(0.0f, 0.0f));
    center *= 1.0 / corners.size();

    std::vector<Point2f> centerToCorners;
    for(size_t i=0; i<outsideCorners.size(); i++)
    {
      centerToCorners.push_back(outsideCorners[i] - center);
    }

    //TODO: use CirclesGridFinder::getDirection
    float crossProduct = centerToCorners[0].x * centerToCorners[1].y - centerToCorners[0].y * centerToCorners[1].x;
    //y axis is inverted in computer vision so we check > 0
    bool isClockwise = crossProduct > 0;
    firstCorner  = isClockwise ? outsideCorners[1] : outsideCorners[0];
  }
  else
  {
    firstCorner = corners[0];
  }

  std::vector<Point2f>::const_iterator firstCornerIterator = std::find(hull2f.begin(), hull2f.end(), firstCorner);
  sortedCorners.clear();
  for(std::vector<Point2f>::const_iterator it = firstCornerIterator; it != hull2f.end();++it)
  {
    std::vector<Point2f>::const_iterator itCorners = std::find(corners.begin(), corners.end(), *it);
    if(itCorners != corners.end())
    {
      sortedCorners.push_back(*it);
    }
  }
  for(std::vector<Point2f>::const_iterator it = hull2f.begin(); it != firstCornerIterator;++it)
  {
    std::vector<Point2f>::const_iterator itCorners = std::find(corners.begin(), corners.end(), *it);
    if(itCorners != corners.end())
    {
      sortedCorners.push_back(*it);
    }
  }

  if(!isAsymmetricGrid)
  {
    double dist01 = norm(sortedCorners[0] - sortedCorners[1]);
    double dist12 = norm(sortedCorners[1] - sortedCorners[2]);
    // Use half the average distance between circles on the shorter side as threshold for determining whether a point lies on an edge.
    double thresh = min(dist01, dist12) / min(patternSize.width, patternSize.height) / 2;

    size_t circleCount01 = 0;
    size_t circleCount12 = 0;
    Vec4f line01( sortedCorners[0].x, sortedCorners[0].y, sortedCorners[1].x, sortedCorners[1].y );
    Vec4f line12( sortedCorners[1].x, sortedCorners[1].y, sortedCorners[2].x, sortedCorners[2].y );
    // Count the circles along both edges.
    for (size_t i = 0; i < patternPoints.size(); i++)
    {
      if (pointLineDistance(patternPoints[i], line01) < thresh)
        circleCount01++;
      if (pointLineDistance(patternPoints[i], line12) < thresh)
        circleCount12++;
    }

    // Ensure that the edge from sortedCorners[0] to sortedCorners[1] is the one with more circles (i.e. it is interpreted as the pattern's width).
    if ((circleCount01 > circleCount12 && patternSize.height > patternSize.width) || (circleCount01 < circleCount12 && patternSize.height < patternSize.width))
    {
      for(size_t i=0; i<sortedCorners.size()-1; i++)
      {
        sortedCorners[i] = sortedCorners[i+1];
      }
      sortedCorners[sortedCorners.size() - 1] = firstCorner;
    }
  }
}

void CirclesGridClusterFinder::rectifyPatternPoints(const std::vector<cv::Point2f> &patternPoints, const std::vector<cv::Point2f> &sortedCorners, std::vector<cv::Point2f> &rectifiedPatternPoints)
{
  //indices of corner points in pattern
  std::vector<Point> trueIndices;
  trueIndices.emplace_back(0, 0);
  trueIndices.emplace_back(patternSize.width - 1, 0);
  if(isAsymmetricGrid)
  {
    trueIndices.emplace_back(patternSize.width - 1, 1);
    trueIndices.emplace_back(patternSize.width - 1, patternSize.height - 2);
  }
  trueIndices.emplace_back(patternSize.width - 1, patternSize.height - 1);
  trueIndices.emplace_back(0, patternSize.height - 1);

  std::vector<Point2f> idealPoints;
  for(size_t idx=0; idx<trueIndices.size(); idx++)
  {
    int i = trueIndices[idx].y;
    int j = trueIndices[idx].x;
    if(isAsymmetricGrid)
    {
      idealPoints.emplace_back((2*j + i % 2)*squareSize, i*squareSize);
    }
    else
    {
      idealPoints.emplace_back(j*squareSize, i*squareSize);
    }
  }

  Mat homography = findHomography(sortedCorners, idealPoints, 0);
  Mat rectifiedPointsMat;
  transform(patternPoints, rectifiedPointsMat, homography);
  rectifiedPatternPoints.clear();
  convertPointsFromHomogeneous(rectifiedPointsMat, rectifiedPatternPoints);
}

void CirclesGridClusterFinder::parsePatternPoints(const std::vector<cv::Point2f> &patternPoints,
                                                  const std::vector<cv::Point2f> &rectifiedPatternPoints,
                                                  std::vector<cv::Point2f> &centers)
{
#ifndef HAVE_OPENCV_FLANN
  CV_UNUSED(patternPoints);
  CV_UNUSED(rectifiedPatternPoints);
  CV_UNUSED(centers);
  CV_Error(Error::StsNotImplemented, "The desired functionality requires flann module, which was disabled.");
#else
  flann::LinearIndexParams flannIndexParams;
  flann::Index flannIndex(Mat(rectifiedPatternPoints).reshape(1,
                (int)rectifiedPatternPoints.size()), flannIndexParams);

  centers.clear();
  for( int i = 0; i < patternSize.height; i++ )
  {
    for( int j = 0; j < patternSize.width; j++ )
    {
      Point2f idealPt;
      if(isAsymmetricGrid)
        idealPt = Point2f((2*j + i % 2)*squareSize, i*squareSize);
      else
        idealPt = Point2f(j*squareSize, i*squareSize);

      Mat query(1, 2, CV_32F, &idealPt);
      const int knn = 1;
      int indicesbuf[knn] = {0};
      float distsbuf[knn] = {0.f};
      Mat indices(1, knn, CV_32S, &indicesbuf);
      Mat dists(1, knn, CV_32F, &distsbuf);
      flannIndex.knnSearch(query, indices, dists, knn, flann::SearchParams());
      centers.push_back(patternPoints.at(indicesbuf[0]));

      if(distsbuf[0] > maxRectifiedDistance)
      {
#ifdef DEBUG_CIRCLES
        std::cout << "Pattern not detected: too large rectified distance" << std::endl;
#endif
        centers.clear();
        return;
      }
    }
  }
#endif
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
  CV_Assert( !doesVertexExist( id ) );

  vertices.emplace(id, Vertex());
}

void Graph::addEdge(size_t id1, size_t id2)
{
  CV_Assert( doesVertexExist( id1 ) );
  CV_Assert( doesVertexExist( id2 ) );

  vertices[id1].neighbors.insert(id2);
  vertices[id2].neighbors.insert(id1);
}

void Graph::removeEdge(size_t id1, size_t id2)
{
  CV_Assert( doesVertexExist( id1 ) );
  CV_Assert( doesVertexExist( id2 ) );

  vertices[id1].neighbors.erase(id2);
  vertices[id2].neighbors.erase(id1);
}

bool Graph::areVerticesAdjacent(size_t id1, size_t id2) const
{
  Vertices::const_iterator it = vertices.find(id1);
  CV_Assert(it != vertices.end());
  const Neighbors & neighbors = it->second.neighbors;
  return neighbors.find(id2) != neighbors.end();
}

size_t Graph::getVerticesCount() const
{
  return vertices.size();
}

size_t Graph::getDegree(size_t id) const
{
  Vertices::const_iterator it = vertices.find(id);
  CV_Assert( it != vertices.end() );
  return it->second.neighbors.size();
}

void Graph::floydWarshall(cv::Mat &distanceMatrix, int infinity) const
{
  const int edgeWeight = 1;

  const int n = (int)getVerticesCount();
  distanceMatrix.create(n, n, CV_32SC1);
  distanceMatrix.setTo(infinity);
  for (Vertices::const_iterator it1 = vertices.begin(); it1 != vertices.end();++it1)
  {
    distanceMatrix.at<int> ((int)it1->first, (int)it1->first) = 0;
    for (Neighbors::const_iterator it2 = it1->second.neighbors.begin(); it2 != it1->second.neighbors.end();++it2)
    {
      CV_Assert( it1->first != *it2 );
      distanceMatrix.at<int> ((int)it1->first, (int)*it2) = edgeWeight;
    }
  }

  for (Vertices::const_iterator it1 = vertices.begin(); it1 != vertices.end();++it1)
  {
    for (Vertices::const_iterator it2 = vertices.begin(); it2 != vertices.end();++it2)
    {
      for (Vertices::const_iterator it3 = vertices.begin(); it3 != vertices.end();++it3)
      {
      int i1 = (int)it1->first, i2 = (int)it2->first, i3 = (int)it3->first;
        int val1 = distanceMatrix.at<int> (i2, i3);
        int val2;
        if (distanceMatrix.at<int> (i2, i1) == infinity ||
      distanceMatrix.at<int> (i1, i3) == infinity)
          val2 = val1;
        else
        {
          val2 = distanceMatrix.at<int> (i2, i1) + distanceMatrix.at<int> (i1, i3);
        }
        distanceMatrix.at<int> (i2, i3) = (val1 == infinity) ? val2 : std::min(val1, val2);
      }
    }
  }
}

const Graph::Neighbors& Graph::getNeighbors(size_t id) const
{
  Vertices::const_iterator it = vertices.find(id);
  CV_Assert( it != vertices.end() );
  return it->second.neighbors;
}

CirclesGridFinder::Segment::Segment(cv::Point2f _s, cv::Point2f _e) :
  s(_s), e(_e)
{
}

void computeShortestPath(Mat &predecessorMatrix, int v1, int v2, std::vector<int> &path);
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
  vertexGain = 1;
  vertexPenalty = -0.6f;
  edgeGain = 1;
  edgePenalty = -0.6f;
  existingVertexGain = 10000;

  minRNGEdgeSwitchDist = 5.f;
  gridType = SYMMETRIC_GRID;

  squareSize = 1.0f;
  maxRectifiedDistance = squareSize/2.0f;
}

CirclesGridFinder::CirclesGridFinder(Size _patternSize, const std::vector<Point2f> &testKeypoints,
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
      std::vector<Point2f> vectors, filteredVectors, basis;
      Graph rng(0);
      computeRNG(rng, vectors);
      filterOutliersByDensity(vectors, filteredVectors);
      std::vector<Graph> basisGraphs;
      findBasis(filteredVectors, basis, basisGraphs);
      findMCS(basis, basisGraphs);
      break;
    }

    case CirclesGridFinderParameters::ASYMMETRIC_GRID:
    {
      std::vector<Point2f> vectors, tmpVectors, filteredVectors, basis;
      Graph rng(0);
      computeRNG(rng, tmpVectors);
      rng2gridGraph(rng, vectors);
      filterOutliersByDensity(vectors, filteredVectors);
      std::vector<Graph> basisGraphs;
      findBasis(filteredVectors, basis, basisGraphs);
      findMCS(basis, basisGraphs);
      eraseUsedGraph(basisGraphs);
      holes2 = holes;
      holes.clear();
      findMCS(basis, basisGraphs);
      break;
    }

    default:
      CV_Error(Error::StsBadArg, "Unknown pattern type");
  }
  return (isDetectionCorrect());
}

void CirclesGridFinder::rng2gridGraph(Graph &rng, std::vector<cv::Point2f> &vectors) const
{
  for (size_t i = 0; i < rng.getVerticesCount(); i++)
  {
    Graph::Neighbors neighbors1 = rng.getNeighbors(i);
    for (Graph::Neighbors::iterator it1 = neighbors1.begin(); it1 != neighbors1.end(); ++it1)
    {
      Graph::Neighbors neighbors2 = rng.getNeighbors(*it1);
      for (Graph::Neighbors::iterator it2 = neighbors2.begin(); it2 != neighbors2.end(); ++it2)
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

void CirclesGridFinder::eraseUsedGraph(std::vector<Graph> &basisGraphs) const
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
      rotatedGrid = holes.size() != patternSize.height && holes.size() == patternSize.width;
      if (holes.size() != patternSize.height && holes.size() != patternSize.width)
        return false;

      size_t num_vertices = 0ull;
      for (size_t i = 0; i < holes.size(); i++)
      {
        if (holes[i].size() != patternSize.width && rotatedGrid == false)
        {
          return false;
        }
        else if (holes[i].size() != patternSize.height && rotatedGrid == true)
        {
          rotatedGrid = false;
          return false;
        }

        num_vertices += holes[i].size();
      }
      return num_vertices == patternSize.area();
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
      size_t largeHeight = (size_t)ceil(patternSize.height / 2.);
      size_t smallWidth = patternSize.width;
      size_t smallHeight = (size_t)floor(patternSize.height / 2.);

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

      std::set<size_t> vertices;
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
  }
  CV_Error(Error::StsBadArg, "Unknown pattern type");
}

void CirclesGridFinder::findMCS(const std::vector<Point2f> &basis, std::vector<Graph> &basisGraphs)
{
  holes.clear();
  Path longestPath;
  size_t bestGraphIdx = findLongestPath(basisGraphs, longestPath);
  std::vector<size_t> holesRow = longestPath.vertices;

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

Mat CirclesGridFinder::rectifyGrid(Size detectedGridSize, const std::vector<Point2f>& centers,
                                   const std::vector<Point2f> &keypoints, std::vector<Point2f> &warpedKeypoints)
{
  CV_Assert( !centers.empty() );
  const float edgeLength = 30;
  const Point2f offset(150, 150);

  std::vector<Point2f> dstPoints;
  bool isClockwiseBefore =
      getDirection(centers[0], centers[detectedGridSize.width - 1], centers[centers.size() - 1]) < 0;

  int iStart = isClockwiseBefore ? 0 : detectedGridSize.height - 1;
  int iEnd = isClockwiseBefore ? detectedGridSize.height : -1;
  int iStep = isClockwiseBefore ? 1 : -1;
  for (int i = iStart; i != iEnd; i += iStep)
  {
    for (int j = 0; j < detectedGridSize.width; j++)
    {
      dstPoints.push_back(offset + Point2f(edgeLength * j, edgeLength * i));
    }
  }

  Mat H = findHomography(centers, dstPoints, RANSAC);
  //Mat H = findHomography(corners, dstPoints);

  if (H.empty())
  {
      H = Mat::zeros(3, 3, CV_64FC1);
      warpedKeypoints.clear();
      return H;
  }

  std::vector<Point2f> srcKeypoints;
  for (size_t i = 0; i < keypoints.size(); i++)
  {
    srcKeypoints.push_back(keypoints[i]);
  }

  Mat dstKeypointsMat;
  transform(srcKeypoints, dstKeypointsMat, H);
  std::vector<Point2f> dstKeypoints;
  convertPointsFromHomogeneous(dstKeypointsMat, dstKeypoints);

  warpedKeypoints.clear();
  for (auto &pt:dstKeypoints)
  {
    warpedKeypoints.emplace_back(std::move(pt));
  }

  return H;
}

size_t CirclesGridFinder::findNearestKeypoint(Point2f pt) const
{
  size_t bestIdx = 0;
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

void CirclesGridFinder::addPoint(Point2f pt, std::vector<size_t> &points)
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

void CirclesGridFinder::findCandidateLine(std::vector<size_t> &line, size_t seedLineIdx, bool addRow, Point2f basisVec,
                                          std::vector<size_t> &seeds)
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

  CV_Assert( line.size() == seeds.size() );
}

void CirclesGridFinder::findCandidateHoles(std::vector<size_t> &above, std::vector<size_t> &below, bool addRow, Point2f basisVec,
                                           std::vector<size_t> &aboveSeeds, std::vector<size_t> &belowSeeds)
{
  above.clear();
  below.clear();
  aboveSeeds.clear();
  belowSeeds.clear();

  findCandidateLine(above, 0, addRow, -basisVec, aboveSeeds);
  size_t lastIdx = addRow ? holes.size() - 1 : holes[0].size() - 1;
  findCandidateLine(below, lastIdx, addRow, basisVec, belowSeeds);

  CV_Assert( below.size() == above.size() );
  CV_Assert( belowSeeds.size() == aboveSeeds.size() );
  CV_Assert( below.size() == belowSeeds.size() );
}

bool CirclesGridFinder::areCentersNew(const std::vector<size_t> &newCenters, const std::vector<std::vector<size_t> > &holes)
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
                                     const std::vector<size_t> &above, const std::vector<size_t> &below,
                                     std::vector<std::vector<size_t> > &holes)
{
  if (aboveConfidence < minConfidence && belowConfidence < minConfidence)
    return;

  if (addRow)
  {
    if (aboveConfidence >= belowConfidence)
    {
      if (!areCentersNew(above, holes))
        CV_Error( cv::Error::StsError, "Centers are not new" );

      holes.insert(holes.begin(), above);
    }
    else
    {
      if (!areCentersNew(below, holes))
        CV_Error( cv::Error::StsError, "Centers are not new" );

      holes.insert(holes.end(), below);
    }
  }
  else
  {
    if (aboveConfidence >= belowConfidence)
    {
      if (!areCentersNew(above, holes))
        CV_Error( cv::Error::StsError, "Centers are not new" );

      for (size_t i = 0; i < holes.size(); i++)
      {
        holes[i].insert(holes[i].begin(), above[i]);
      }
    }
    else
    {
      if (!areCentersNew(below, holes))
        CV_Error( cv::Error::StsError, "Centers are not new" );

      for (size_t i = 0; i < holes.size(); i++)
      {
        holes[i].insert(holes[i].end(), below[i]);
      }
    }
  }
}

float CirclesGridFinder::computeGraphConfidence(const std::vector<Graph> &basisGraphs, bool addRow,
                                                const std::vector<size_t> &points, const std::vector<size_t> &seeds)
{
  CV_Assert( points.size() == seeds.size() );
  float confidence = 0;
  const size_t vCount = basisGraphs[0].getVerticesCount();
  CV_Assert( basisGraphs[0].getVerticesCount() == basisGraphs[1].getVerticesCount() );

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

void CirclesGridFinder::addHolesByGraph(const std::vector<Graph> &basisGraphs, bool addRow, Point2f basisVec)
{
  std::vector<size_t> above, below, aboveSeeds, belowSeeds;
  findCandidateHoles(above, below, addRow, basisVec, aboveSeeds, belowSeeds);
  float aboveConfidence = computeGraphConfidence(basisGraphs, addRow, above, aboveSeeds);
  float belowConfidence = computeGraphConfidence(basisGraphs, addRow, below, belowSeeds);

  insertWinner(aboveConfidence, belowConfidence, parameters.minGraphConfidence, addRow, above, below, holes);
}

void CirclesGridFinder::filterOutliersByDensity(const std::vector<Point2f> &samples, std::vector<Point2f> &filteredSamples)
{
  if (samples.empty())
    CV_Error( cv::Error::StsError, "samples is empty" );

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
    CV_Error( cv::Error::StsError, "filteredSamples is empty" );
}

void CirclesGridFinder::findBasis(const std::vector<Point2f> &samples, std::vector<Point2f> &basis, std::vector<Graph> &basisGraphs)
{
  basis.clear();
  Mat bestLabels;
  TermCriteria termCriteria;
  Mat centers;
  const int clustersCount = 4;
  int nsamples = (int)samples.size();
  kmeans(Mat(samples).reshape(1, nsamples), clustersCount, bestLabels, termCriteria, parameters.kmeansAttempts,
         KMEANS_RANDOM_CENTERS, centers);
  CV_Assert( centers.type() == CV_32FC1 );

  std::vector<int> basisIndices;
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
    CV_Error(cv::Error::StsError, "Basis size is not 2");

  if (basis[1].x > basis[0].x)
  {
    std::swap(basis[0], basis[1]);
    std::swap(basisIndices[0], basisIndices[1]);
  }

  const float minBasisDif = 2;
  if (norm(basis[0] - basis[1]) < minBasisDif)
    CV_Error(cv::Error::StsError, "degenerate basis" );

  std::vector<std::vector<Point2f> > clusters(2), hulls(2);
  for (int k = 0; k < (int)samples.size(); k++)
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
    convexHull(clusters[i], hulls[i]);
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
        if (pointPolygonTest(hulls[k], vec, false) >= 0)
        {
          basisGraphs[k].addEdge(i, j);
        }
      }
    }
  }
  if (basisGraphs.size() != 2)
    CV_Error(cv::Error::StsError, "Number of basis graphs is not 2");
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
  CV_Assert( dm.type() == CV_32SC1 );
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

static void computeShortestPath(Mat &predecessorMatrix, size_t v1, size_t v2, std::vector<size_t> &path)
{
  if (predecessorMatrix.at<int> ((int)v1, (int)v2) < 0)
  {
    path.push_back(v1);
    return;
  }

  computeShortestPath(predecessorMatrix, v1, predecessorMatrix.at<int> ((int)v1, (int)v2), path);
  path.push_back(v2);
}

size_t CirclesGridFinder::findLongestPath(std::vector<Graph> &basisGraphs, Path &bestPath)
{
  std::vector<Path> longestPaths(1);
  std::vector<int> confidences;

  size_t bestGraphIdx = 0;
  const int infinity = -1;
  for (size_t graphIdx = 0; graphIdx < basisGraphs.size(); graphIdx++)
  {
    const Graph &g = basisGraphs[graphIdx];
    Mat distanceMatrix;
    g.floydWarshall(distanceMatrix, infinity);
    Mat predecessorMatrix;
    computePredecessorMatrix(distanceMatrix, (int)g.getVerticesCount(), predecessorMatrix);

    double maxVal;
    Point maxLoc;
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
      for (int v2 = 0; v2 < (int)path.vertices.size(); v2++)
      {
        conf += (int)basisGraphs[1 - (int)graphIdx].getDegree(v2);
      }
      confidences.push_back(conf);
    }
  }

  int maxConf = -1;
  int bestPathIdx = -1;
  for (int i = 0; i < (int)confidences.size(); i++)
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

void CirclesGridFinder::drawBasis(const std::vector<Point2f> &basis, Point2f origin, Mat &drawImg) const
{
  for (size_t i = 0; i < basis.size(); i++)
  {
    Point2f pt(basis[i]);
    line(drawImg, origin, origin + pt, Scalar(0, (double)(i * 255), 0), 2);
  }
}

void CirclesGridFinder::drawBasisGraphs(const std::vector<Graph> &basisGraphs, Mat &drawImage, bool drawEdges,
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
    cvtColor(srcImage, drawImage, COLOR_GRAY2RGB);
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

      circle(drawImage, keypoints[holes[i][j]], holeRadius, holeColor, holeThickness);
    }
  }
}

Size CirclesGridFinder::getDetectedGridSize() const
{
  if (holes.size() == 0)
    return Size(0, 0);

  return Size((int)holes[0].size(), (int)holes.size());
}

void CirclesGridFinder::getHoles(std::vector<Point2f> &outHoles) const
{
  outHoles.clear();
  if (rotatedGrid == false)
  {
    for (size_t i = 0ull; i < holes.size(); i++)
    {
      for (size_t j = 0ull; j < holes[i].size(); j++)
      {
        outHoles.push_back(keypoints[holes[i][j]]);
      }
    }
  }
  else
  {
    bool visit_all = false;
    size_t j = 0ull;
    while (visit_all != true)
    {
      visit_all = true;
      for (size_t i = 0ull; i < holes.size(); i++)
      {
        if (j < holes[i].size())
        {
          outHoles.push_back(keypoints[holes[i][j]]);
          visit_all = false;
        }
      }
      j++;
    }
  }
}

static bool areIndicesCorrect(Point pos, std::vector<std::vector<size_t> > *points)
{
  if (pos.y < 0 || pos.x < 0)
    return false;
  return (static_cast<size_t> (pos.y) < points->size() && static_cast<size_t> (pos.x) < points->at(pos.y).size());
}

void CirclesGridFinder::getAsymmetricHoles(std::vector<cv::Point2f> &outHoles) const
{
  outHoles.clear();

  std::vector<Point> largeCornerIndices, smallCornerIndices;
  std::vector<Point> firstSteps, secondSteps;
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

void CirclesGridFinder::getCornerSegments(const std::vector<std::vector<size_t> > &points, std::vector<std::vector<Segment> > &segments,
                                          std::vector<Point> &cornerIndices, std::vector<Point> &firstSteps,
                                          std::vector<Point> &secondSteps) const
{
  segments.clear();
  cornerIndices.clear();
  firstSteps.clear();
  secondSteps.clear();
  int h = (int)points.size();
  int w = (int)points[0].size();
  CV_Assert(h >= 2 && w >= 2)
    ;

  //all 8 segments with one end in a corner
  std::vector<Segment> corner;
  corner.emplace_back(keypoints[points[1][0]], keypoints[points[0][0]]);
  corner.emplace_back(keypoints[points[0][0]], keypoints[points[0][1]]);
  segments.push_back(corner);
  cornerIndices.emplace_back(0, 0);
  firstSteps.emplace_back(1, 0);
  secondSteps.emplace_back(0, 1);
  corner.clear();

  corner.emplace_back(keypoints[points[0][w - 2]], keypoints[points[0][w - 1]]);
  corner.emplace_back(keypoints[points[0][w - 1]], keypoints[points[1][w - 1]]);
  segments.push_back(corner);
  cornerIndices.emplace_back(w - 1, 0);
  firstSteps.emplace_back(0, 1);
  secondSteps.emplace_back(-1, 0);
  corner.clear();

  corner.emplace_back(keypoints[points[h - 2][w - 1]], keypoints[points[h - 1][w - 1]]);
  corner.emplace_back(keypoints[points[h - 1][w - 1]], keypoints[points[h - 1][w - 2]]);
  segments.push_back(corner);
  cornerIndices.emplace_back(w - 1, h - 1);
  firstSteps.emplace_back(-1, 0);
  secondSteps.emplace_back(0, -1);
  corner.clear();

  corner.emplace_back(keypoints[points[h - 1][1]], keypoints[points[h - 1][0]]);
  corner.emplace_back(keypoints[points[h - 1][0]], keypoints[points[h - 2][0]]);
  cornerIndices.emplace_back(0, h - 1);
  firstSteps.emplace_back(0, -1);
  secondSteps.emplace_back(1, 0);
  segments.push_back(corner);
  corner.clear();

  //y axis is inverted in computer vision so we check < 0
  bool isClockwise =
      getDirection(keypoints[points[0][0]], keypoints[points[0][w - 1]], keypoints[points[h - 1][w - 1]]) < 0;
  if (!isClockwise)
  {
#ifdef DEBUG_CIRCLES
    std::cout << "Corners are counterclockwise" << std::endl;
#endif
    std::reverse(segments.begin(), segments.end());
    std::reverse(cornerIndices.begin(), cornerIndices.end());
    std::reverse(firstSteps.begin(), firstSteps.end());
    std::reverse(secondSteps.begin(), secondSteps.end());
    std::swap(firstSteps, secondSteps);
  }
}

bool CirclesGridFinder::doesIntersectionExist(const std::vector<Segment> &corner, const std::vector<std::vector<Segment> > &segments)
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

size_t CirclesGridFinder::getFirstCorner(std::vector<Point> &largeCornerIndices, std::vector<Point> &smallCornerIndices, std::vector<
    Point> &firstSteps, std::vector<Point> &secondSteps) const
{
  std::vector<std::vector<Segment> > largeSegments;
  std::vector<std::vector<Segment> > smallSegments;

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

  for (size_t i = 0; i < cornersCount * 2; ++i)
  {
    if (waitOutsider)
    {
      if (!isInsider[(cornerIdx + 1) % cornersCount])
        waitOutsider = false;
    }
    else
    {
      if (isInsider[(cornerIdx + 1) % cornersCount])
        return cornerIdx;
    }

    cornerIdx = (cornerIdx + 1) % cornersCount;
  }

  CV_Error(Error::StsNoConv, "isInsider array has the same values");
}

}
