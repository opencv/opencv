/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install, copy or use the software.
//
// Copyright (C) 2009, Farhad Dadgostar
// Intel Corporation and third party copyrights are property of their respective owners.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

CvFuzzyPoint::CvFuzzyPoint(double _x, double _y)
{
    x = _x;
    y = _y;
}

bool CvFuzzyCurve::between(double x, double x1, double x2)
{
    if ((x >= x1) && (x <= x2))
        return true;
    else if ((x >= x2) && (x <= x1))
        return true;

    return false;
}

CvFuzzyCurve::CvFuzzyCurve()
{
    value = 0;
}

CvFuzzyCurve::~CvFuzzyCurve()
{
    // nothing to do
}

void CvFuzzyCurve::setCentre(double _centre)
{
    centre = _centre;
}

double CvFuzzyCurve::getCentre()
{
    return centre;
}

void CvFuzzyCurve::clear()
{
    points.clear();
}

void CvFuzzyCurve::addPoint(double x, double y)
{
    points.push_back(CvFuzzyPoint(x, y));
}

double CvFuzzyCurve::calcValue(double param)
{
    int size = (int)points.size();
    double x1, y1, x2, y2, m, y;
    for (int i = 1; i < size; i++)
    {
        x1 = points[i-1].x;
        x2 = points[i].x;
        if (between(param, x1, x2)) {
            y1 = points[i-1].y;
            y2 = points[i].y;
            if (x2 == x1)
                return y2;
            m = (y2-y1)/(x2-x1);
            y = m*(param-x1)+y1;
            return y;
        }
    }
    return 0;
}

double CvFuzzyCurve::getValue()
{
    return value;
}

void CvFuzzyCurve::setValue(double _value)
{
    value = _value;
}


CvFuzzyFunction::CvFuzzyFunction()
{
    // nothing to do
}

CvFuzzyFunction::~CvFuzzyFunction()
{
    curves.clear();
}

void CvFuzzyFunction::addCurve(CvFuzzyCurve *curve, double value)
{
    curves.push_back(*curve);
    curve->setValue(value);
}

void CvFuzzyFunction::resetValues()
{
    int numCurves = (int)curves.size();
    for (int i = 0; i < numCurves; i++)
        curves[i].setValue(0);
}

double CvFuzzyFunction::calcValue()
{
    double s1 = 0, s2 = 0, v;
    int numCurves = (int)curves.size();
    for (int i = 0; i < numCurves; i++)
    {
        v = curves[i].getValue();
        s1 += curves[i].getCentre() * v;
        s2 += v;
    }

    if (s2 != 0)
        return s1/s2;
    else
        return 0;
}

CvFuzzyCurve *CvFuzzyFunction::newCurve()
{
    CvFuzzyCurve *c;
    c = new CvFuzzyCurve();
    addCurve(c);
    return c;
}

CvFuzzyRule::CvFuzzyRule()
{
    fuzzyInput1 = NULL;
    fuzzyInput2 = NULL;
    fuzzyOutput = NULL;
}

CvFuzzyRule::~CvFuzzyRule()
{
    if (fuzzyInput1 != NULL)
        delete fuzzyInput1;

    if (fuzzyInput2 != NULL)
        delete fuzzyInput2;

    if (fuzzyOutput != NULL)
        delete fuzzyOutput;
}

void CvFuzzyRule::setRule(CvFuzzyCurve *c1, CvFuzzyCurve *c2, CvFuzzyCurve *o1)
{
    fuzzyInput1 = c1;
    fuzzyInput2 = c2;
    fuzzyOutput = o1;
}

double CvFuzzyRule::calcValue(double param1, double param2)
{
    double v1, v2;
    v1 = fuzzyInput1->calcValue(param1);
    if (fuzzyInput2 != NULL)
    {
        v2 = fuzzyInput2->calcValue(param2);
        if (v1 < v2)
            return v1;
        else
            return v2;
    }
    else
        return v1;
}

CvFuzzyCurve *CvFuzzyRule::getOutputCurve()
{
    return fuzzyOutput;
}

CvFuzzyController::CvFuzzyController()
{
    // nothing to do
}

CvFuzzyController::~CvFuzzyController()
{
    int size = (int)rules.size();
    for(int i = 0; i < size; i++)
        delete rules[i];
}

void CvFuzzyController::addRule(CvFuzzyCurve *c1, CvFuzzyCurve *c2, CvFuzzyCurve *o1)
{
    CvFuzzyRule *f = new CvFuzzyRule();
    rules.push_back(f);
    f->setRule(c1, c2, o1);
}

double CvFuzzyController::calcOutput(double param1, double param2)
{
    double v;
    CvFuzzyFunction list;
    int size = (int)rules.size();

    for(int i = 0; i < size; i++)
    {
        v = rules[i]->calcValue(param1, param2);
        if (v != 0)
            list.addCurve(rules[i]->getOutputCurve(), v);
    }
    v = list.calcValue();
    return v;
}

CvFuzzyMeanShiftTracker::FuzzyResizer::FuzzyResizer()
{
    CvFuzzyCurve *i1L, *i1M, *i1H;
    CvFuzzyCurve *oS, *oZE, *oE;
    CvFuzzyCurve *c;

    double MedStart = 0.1, MedWidth = 0.15;

    c = iInput.newCurve();
    c->addPoint(0, 1);
    c->addPoint(0.1, 0);
    c->setCentre(0);
    i1L = c;

    c = iInput.newCurve();
    c->addPoint(0.05, 0);
    c->addPoint(MedStart, 1);
    c->addPoint(MedStart+MedWidth, 1);
    c->addPoint(MedStart+MedWidth+0.05, 0);
    c->setCentre(MedStart+(MedWidth/2));
    i1M = c;

    c = iInput.newCurve();
    c->addPoint(MedStart+MedWidth, 0);
    c->addPoint(1, 1);
    c->addPoint(1000, 1);
    c->setCentre(1);
    i1H = c;

    c = iOutput.newCurve();
    c->addPoint(-10000, 1);
    c->addPoint(-5, 1);
    c->addPoint(-0.5, 0);
    c->setCentre(-5);
    oS = c;

    c = iOutput.newCurve();
    c->addPoint(-1, 0);
    c->addPoint(-0.05, 1);
    c->addPoint(0.05, 1);
    c->addPoint(1, 0);
    c->setCentre(0);
    oZE = c;

    c = iOutput.newCurve();
    c->addPoint(-0.5, 0);
    c->addPoint(5, 1);
    c->addPoint(1000, 1);
    c->setCentre(5);
    oE = c;

    fuzzyController.addRule(i1L, NULL, oS);
    fuzzyController.addRule(i1M, NULL, oZE);
    fuzzyController.addRule(i1H, NULL, oE);
}

int CvFuzzyMeanShiftTracker::FuzzyResizer::calcOutput(double edgeDensity, double density)
{
    return (int)fuzzyController.calcOutput(edgeDensity, density);
}

CvFuzzyMeanShiftTracker::SearchWindow::SearchWindow()
{
    x = 0;
    y = 0;
    width = 0;
    height = 0;
    maxWidth = 0;
    maxHeight = 0;
    xGc = 0;
    yGc = 0;
    m00 = 0;
    m01 = 0;
    m10 = 0;
    m11 = 0;
    m02 = 0;
    m20 = 0;
    ellipseHeight = 0;
    ellipseWidth = 0;
    ellipseAngle = 0;
    density = 0;
    depthLow = 0;
    depthHigh = 0;
    fuzzyResizer = NULL;
}

CvFuzzyMeanShiftTracker::SearchWindow::~SearchWindow()
{
    if (fuzzyResizer != NULL)
        delete fuzzyResizer;
}

void CvFuzzyMeanShiftTracker::SearchWindow::setSize(int _x, int _y, int _width, int _height)
{
    x = _x;
    y = _y;
    width = _width;
    height = _height;

    if (x < 0)
        x = 0;

    if (y < 0)
        y = 0;

    if (x + width > maxWidth)
        width = maxWidth - x;

    if (y + height > maxHeight)
        height = maxHeight - y;
}

void CvFuzzyMeanShiftTracker::SearchWindow::initDepthValues(IplImage *maskImage, IplImage *depthMap)
{
    unsigned int d=0, mind = 0xFFFF, maxd = 0, m0 = 0, m1 = 0, mc, dd;
    unsigned char *data = NULL;
    unsigned short *depthData = NULL;

    for (int j = 0; j < height; j++)
    {
        data = (unsigned char *)(maskImage->imageData + (maskImage->widthStep * (j + y)) + x);
        if (depthMap)
            depthData = (unsigned short *)(depthMap->imageData + (depthMap->widthStep * (j + y)) + x);

        for (int i = 0; i < width; i++)
        {
            if (*data)
            {
                m0 += 1;

                if (depthData)
                {
                    if (*depthData)
                    {
                        d = *depthData;
                        m1 += d;
                        if (d < mind)
                            mind = d;
                        if (d > maxd)
                            maxd = d;
                    }
                    depthData++;
                }
            }
            data++;
        }
    }

    if (m0 > 0)
    {
        mc = m1/m0;
        if ((mc - mind) > (maxd - mc))
            dd = maxd - mc;
        else
            dd = mc - mind;
        dd = dd - dd/10;
        depthHigh = mc + dd;
        depthLow = mc - dd;
    }
    else
    {
        depthHigh = 32000;
        depthLow = 0;
    }
}

bool CvFuzzyMeanShiftTracker::SearchWindow::shift()
{
    if ((xGc != (width/2)) || (yGc != (height/2)))
    {
        setSize(x + (xGc-(width/2)), y + (yGc-(height/2)), width, height);
        return true;
    }
    else
    {
        return false;
    }
}

void CvFuzzyMeanShiftTracker::SearchWindow::extractInfo(IplImage *maskImage, IplImage *depthMap, bool initDepth)
{
    m00 = 0;
    m10 = 0;
    m01 = 0;
    m11 = 0;
    density = 0;
    m02 = 0;
    m20 = 0;
    ellipseHeight = 0;
    ellipseWidth = 0;

    maxWidth = maskImage->width;
    maxHeight = maskImage->height;

    if (initDepth)
        initDepthValues(maskImage, depthMap);

    unsigned char *maskData = NULL;
    unsigned short *depthData = NULL, depth;
    bool isOk;
    unsigned long count;

    verticalEdgeLeft = 0;
    verticalEdgeRight = 0;
    horizontalEdgeTop = 0;
    horizontalEdgeBottom = 0;

    for (int j = 0; j < height; j++)
    {
        maskData = (unsigned char *)(maskImage->imageData + (maskImage->widthStep * (j + y)) + x);
        if (depthMap)
            depthData = (unsigned short *)(depthMap->imageData + (depthMap->widthStep * (j + y)) + x);

        count = 0;
        for (int i = 0; i < width; i++)
        {
            if (*maskData)
            {
                isOk = true;
                if (depthData)
                {
                    depth = (*depthData);
                    if ((depth > depthHigh) || (depth < depthLow))
                        isOk = false;

                    depthData++;
                }

                if (isOk)
                {
                    m00++;
                    m01 += j;
                    m10 += i;
                    m02 += (j * j);
                    m20 += (i * i);
                    m11 += (j * i);

                    if (i == 0)
                        verticalEdgeLeft++;
                    else if (i == width-1)
                        verticalEdgeRight++;
                    else if (j == 0)
                        horizontalEdgeTop++;
                    else if (j == height-1)
                        horizontalEdgeBottom++;

                    count++;
                }
            }
            maskData++;
        }
    }

    if (m00 > 0)
    {
        xGc = (int)(m10 / m00);
        yGc = (int)(m01 / m00);

        double a, b, c, e1, e2, e3;
        a = ((double)m20/(double)m00)-(xGc * xGc);
        b = 2*(((double)m11/(double)m00)-(xGc * yGc));
        c = ((double)m02/(double)m00)-(yGc * yGc);
        e1 = a+c;
        e3 = a-c;
        e2 = sqrt((b*b)+(e3*e3));
        ellipseHeight = int(sqrt(0.5*(e1+e2)));
        ellipseWidth = int(sqrt(0.5*(e1-e2)));
        if (e3 == 0)
            ellipseAngle = 0;
        else
            ellipseAngle = 0.5*atan(b/e3);

        density = (double)m00/(double)(width * height);
    }
    else
    {
        xGc = width / 2;
        yGc = height / 2;
        ellipseHeight = 0;
        ellipseWidth = 0;
        ellipseAngle = 0;
        density = 0;
    }
}

void CvFuzzyMeanShiftTracker::SearchWindow::getResizeAttribsEdgeDensityLinear(int &resizeDx, int &resizeDy, int &resizeDw, int &resizeDh) {
    int x1 = horizontalEdgeTop;
    int x2 = horizontalEdgeBottom;
    int y1 = verticalEdgeLeft;
    int y2 = verticalEdgeRight;
    int gx = (width*2)/5;
    int gy = (height*2)/5;
    int lx = width/10;
    int ly = height/10;

    resizeDy = 0;
    resizeDh = 0;
    resizeDx = 0;
    resizeDw = 0;

    if (x1 > gx) {
        resizeDy = -1;
    } else if (x1 < lx) {
        resizeDy = +1;
    }

    if (x2 > gx) {
        resizeDh = resizeDy + 1;
    } else if (x2 < lx) {
        resizeDh = - (resizeDy + 1);
    } else {
        resizeDh = - resizeDy;
    }

    if (y1 > gy) {
        resizeDx = -1;
    } else if (y1 < ly) {
        resizeDx = +1;
    }

    if (y2 > gy) {
        resizeDw = resizeDx + 1;
    } else if (y2 < ly) {
        resizeDw = - (resizeDx + 1);
    } else {
        resizeDw = - resizeDx;
    }
}

void CvFuzzyMeanShiftTracker::SearchWindow::getResizeAttribsInnerDensity(int &resizeDx, int &resizeDy, int &resizeDw, int &resizeDh)
{
    int newWidth, newHeight, dx, dy;
    double px, py;
    newWidth = int(sqrt(double(m00)*1.3));
    newHeight = int(newWidth*1.2);
    dx = (newWidth - width);
    dy = (newHeight - height);
    px = (double)xGc/(double)width;
    py = (double)yGc/(double)height;
    resizeDx = (int)(px*dx);
    resizeDy = (int)(py*dy);
    resizeDw = (int)((1-px)*dx);
    resizeDh = (int)((1-py)*dy);
}

void CvFuzzyMeanShiftTracker::SearchWindow::getResizeAttribsEdgeDensityFuzzy(int &resizeDx, int &resizeDy, int &resizeDw, int &resizeDh)
{
    double dx1=0, dx2, dy1, dy2;

    resizeDy = 0;
    resizeDh = 0;
    resizeDx = 0;
    resizeDw = 0;

    if (fuzzyResizer == NULL)
        fuzzyResizer = new FuzzyResizer();

    dx2 = fuzzyResizer->calcOutput(double(verticalEdgeRight)/double(height), density);
    if (dx1 == dx2)
    {
        resizeDx = int(-dx1);
        resizeDw = int(dx1+dx2);
    }

    dy1 = fuzzyResizer->calcOutput(double(horizontalEdgeTop)/double(width), density);
    dy2 = fuzzyResizer->calcOutput(double(horizontalEdgeBottom)/double(width), density);

    dx1 = fuzzyResizer->calcOutput(double(verticalEdgeLeft)/double(height), density);
    dx2 = fuzzyResizer->calcOutput(double(verticalEdgeRight)/double(height), density);
    //if (dx1 == dx2)
    {
        resizeDx = int(-dx1);
        resizeDw = int(dx1+dx2);
    }

    dy1 = fuzzyResizer->calcOutput(double(horizontalEdgeTop)/double(width), density);
    dy2 = fuzzyResizer->calcOutput(double(horizontalEdgeBottom)/double(width), density);
    //if (dy1 == dy2)
    {
        resizeDy = int(-dy1);
        resizeDh = int(dy1+dy2);
    }
}

bool CvFuzzyMeanShiftTracker::SearchWindow::meanShift(IplImage *maskImage, IplImage *depthMap, int maxIteration, bool initDepth)
{
    numShifts = 0;
    do
    {
        extractInfo(maskImage, depthMap, initDepth);
        if (! shift())
            return true;
    } while (++numShifts < maxIteration);

    return false;
}

void CvFuzzyMeanShiftTracker::findOptimumSearchWindow(SearchWindow &searchWindow, IplImage *maskImage, IplImage *depthMap, int maxIteration, int resizeMethod, bool initDepth)
{
    int resizeDx, resizeDy, resizeDw, resizeDh;
    resizeDx = 0;
    resizeDy = 0;
    resizeDw = 0;
    resizeDh = 0;
    searchWindow.numIters = 0;
    for (int i = 0; i < maxIteration; i++)
    {
        searchWindow.numIters++;
        searchWindow.meanShift(maskImage, depthMap, MaxMeanShiftIteration, initDepth);
        switch (resizeMethod)
        {
            case rmEdgeDensityLinear :
                searchWindow.getResizeAttribsEdgeDensityLinear(resizeDx, resizeDy, resizeDw, resizeDh);
                break;
            case rmEdgeDensityFuzzy :
                //searchWindow.getResizeAttribsEdgeDensityLinear(resizeDx, resizeDy, resizeDw, resizeDh);
                searchWindow.getResizeAttribsEdgeDensityFuzzy(resizeDx, resizeDy, resizeDw, resizeDh);
                break;
            case rmInnerDensity :
                searchWindow.getResizeAttribsInnerDensity(resizeDx, resizeDy, resizeDw, resizeDh);
                break;
            default:
                searchWindow.getResizeAttribsEdgeDensityLinear(resizeDx, resizeDy, resizeDw, resizeDh);
        }

        searchWindow.ldx = resizeDx;
        searchWindow.ldy = resizeDy;
        searchWindow.ldw = resizeDw;
        searchWindow.ldh = resizeDh;

        if ((resizeDx == 0) && (resizeDy == 0) && (resizeDw == 0) && (resizeDh == 0))
            break;

        searchWindow.setSize(searchWindow.x + resizeDx, searchWindow.y + resizeDy, searchWindow.width + resizeDw, searchWindow.height + resizeDh);
    }
}

CvFuzzyMeanShiftTracker::CvFuzzyMeanShiftTracker()
{
    searchMode = tsSetWindow;
}

CvFuzzyMeanShiftTracker::~CvFuzzyMeanShiftTracker()
{
    // nothing to do
}

void CvFuzzyMeanShiftTracker::track(IplImage *maskImage, IplImage *depthMap, int resizeMethod, bool resetSearch, int minKernelMass)
{
    bool initDepth = false;

    if (resetSearch)
        searchMode = tsSetWindow;

    switch (searchMode)
    {
        case tsDisabled:
            return;
        case tsSearching:
            return;
        case tsSetWindow:
            kernel.maxWidth = maskImage->width;
            kernel.maxHeight = maskImage->height;
            kernel.setSize(0, 0, maskImage->width, maskImage->height);
            initDepth = true;
        case tsTracking:
            searchMode = tsSearching;
            findOptimumSearchWindow(kernel, maskImage, depthMap, MaxSetSizeIteration, resizeMethod, initDepth);
            if ((kernel.density == 0) || (kernel.m00 < minKernelMass))
                searchMode = tsSetWindow;
            else
                searchMode = tsTracking;
    }
}
