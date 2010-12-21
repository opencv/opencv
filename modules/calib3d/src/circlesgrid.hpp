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

#ifndef CIRCLESGRID_HPP_
#define CIRCLESGRID_HPP_

#include <fstream>
#include <iostream>
#include <string>
#include <set>

#include "precomp.hpp"

class Graph
{
public:
  typedef std::set<int> Neighbors;
  struct Vertex
  {
    Neighbors neighbors;
  };
  typedef std::map<int, Vertex> Vertices;

  Graph( int n);
  bool doesVertexExist( int id ) const;
  void addVertex( int id );
  void addEdge( int id1, int id2 );
  bool areVerticesAdjacent( int id1, int id2 ) const;
  size_t getVerticesCount() const;
  size_t getDegree( int id ) const;
  void floydWarshall(cv::Mat &distanceMatrix, int infinity = -1) const;

private:
  Vertices vertices;
};

struct Path
{
  int firstVertex;
  int lastVertex;
  int length;

  std::vector<int> vertices;

  Path(int first = -1, int last = -1, int len = -1)
  {
    firstVertex = first;
    lastVertex = last;
    length = len;
  }
};

struct CirclesGridFinderParameters
{
  CirclesGridFinderParameters();
  cv::Size2f densityNeighborhoodSize;
  float minDensity;
  int kmeansAttempts;
  int minDistanceToAddKeypoint;
  int keypointScale;
  int minGraphConfidence;
  float vertexGain;
  float vertexPenalty;
  int existingVertexGain;
  float edgeGain;
  float edgePenalty;
  float convexHullFactor;
};

class CirclesGridFinder
{
public:
  CirclesGridFinder(cv::Size patternSize, const std::vector<cv::Point2f> &testKeypoints,
                    const CirclesGridFinderParameters &parameters = CirclesGridFinderParameters());
  bool findHoles();
  static cv::Mat rectifyGrid(cv::Size detectedGridSize, const std::vector<cv::Point2f>& centers,
                          const std::vector<cv::Point2f> &keypoint, std::vector<cv::Point2f> &warpedKeypoints);

  void getHoles(std::vector<cv::Point2f> &holes) const;
  cv::Size getDetectedGridSize() const;

  void drawBasis(const std::vector<cv::Point2f> &basis, cv::Point2f origin, cv::Mat &drawImg) const;
  void drawBasisGraphs(const std::vector<Graph> &basisGraphs, cv::Mat &drawImg, bool drawEdges = true, bool drawVertices =
      true) const;
  void drawHoles(const cv::Mat &srcImage, cv::Mat &drawImage) const;
private:
  void computeEdgeVectorsOfRNG(std::vector<cv::Point2f> &vectors, cv::Mat *drawImage = 0) const;
  void filterOutliersByDensity(const std::vector<cv::Point2f> &samples, std::vector<cv::Point2f> &filteredSamples);
  void findBasis(const std::vector<cv::Point2f> &samples, std::vector<cv::Point2f> &basis, std::vector<Graph> &basisGraphs);
  void findMCS(const std::vector<cv::Point2f> &basis, std::vector<Graph> &basisGraphs);
  size_t findLongestPath(std::vector<Graph> &basisGraphs, Path &bestPath);
  float computeGraphConfidence(const std::vector<Graph> &basisGraphs, bool addRow, const std::vector<int> &points, const std::vector<
      int> &seeds);
  void addHolesByGraph(const std::vector<Graph> &basisGraphs, bool addRow, cv::Point2f basisVec);

  int findNearestKeypoint(cv::Point2f pt) const;
  void addPoint(cv::Point2f pt, std::vector<int> &points);
  void findCandidateLine(std::vector<int> &line, int seedLineIdx, bool addRow, cv::Point2f basisVec, std::vector<int> &seeds);
  void findCandidateHoles(std::vector<int> &above, std::vector<int> &below, bool addRow, cv::Point2f basisVec,
                          std::vector<int> &aboveSeeds, std::vector<int> &belowSeeds);
  static bool areCentersNew( const std::vector<int> &newCenters, const std::vector<std::vector<int> > &holes );
  bool isDetectionCorrect();

  static void insertWinner(float aboveConfidence, float belowConfidence, float minConfidence,
                           bool addRow,
                           const std::vector<int> &above, const std::vector<int> &below, std::vector<std::vector<int> > &holes);
  static bool areVerticesAdjacent(const Graph &graph, int vertex1, int vertex2);

  std::vector<cv::Point2f> keypoints;

  std::vector<std::vector<int> > holes;
  const cv::Size patternSize;
  CirclesGridFinderParameters parameters;
};

#endif /* CIRCLESGRID_HPP_ */
