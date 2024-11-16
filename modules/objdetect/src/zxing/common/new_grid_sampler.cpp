#include "new_grid_sampler.hpp"
#include "perspective_transform.hpp"
#include "../reader_exception.hpp"
#include "util/inireader.hpp"

#include <iostream>
#include <sstream>


namespace zxing {

NewGridSampler NewGridSampler::gridSampler;

NewGridSampler::NewGridSampler() {
}

// Samples an image for a rectangular matrix of bits of the given dimension.
Ref<ByteMatrix> NewGridSampler::sampleGrid(Ref<ByteMatrix> image, int dimension, Ref<PerspectiveTransform> transform, ErrorHandler & err_handler) {
    Ref<ByteMatrix> bits(new ByteMatrix(dimension));
    std::vector<float> points(dimension << 1, (const float)0.0f);
    
    int outlier = 0;
    int maxOutlier = dimension*dimension * 3 / 10 - 1;
    
    for (int y = 0; y < dimension; y++) {
        int max = points.size();
        float yValue = static_cast<float>(y) + 0.5f;
        for (int x = 0; x < max; x += 2) {
            points[x] = static_cast<float>(x >> 1) + 0.5f;
            points[x + 1] = yValue;
        }
        transform->transformPoints(points);
        // Quick check to see if points transformed to something inside the image;
        // sufficient to check the endpoings
        outlier += checkAndNudgePoints(image->getWidth(), image->getHeight(), points, err_handler);
        if (err_handler.ErrCode())   return Ref<ByteMatrix>();
        
        if (outlier >= maxOutlier)
        {
            std::ostringstream s;
            s << "Over 30% points out of bounds.";
            err_handler = ReaderErrorHandler(s.str().c_str());
            return Ref<ByteMatrix>();
        }
        
        for (int x = 0; x < max; x += 2) {
            bits->set(x >> 1, y, image->get(static_cast<int>(points[x]), static_cast<int>(points[x + 1])));
        }
    }
    return bits;
}

// Samples an image for a rectangular matrix of bits of the given dimension.
Ref<BitMatrix> NewGridSampler::sampleGrid(Ref<BitMatrix> image, int dimension, Ref<PerspectiveTransform> transform, float fInitialMS, ErrorHandler & err_handler) {
    Ref<BitMatrix> bits(new BitMatrix(dimension, err_handler));
    if (err_handler.ErrCode()) return Ref<BitMatrix>();
    
    std::vector<float> points(dimension << 1, (const float)0.0f);
    
    int outlier = 0;
    int maxOutlier = dimension*dimension * 3 / 10 - 1;
    int iHeight = image->getHeight();
    int iWidth = image->getWidth();
    
    {  // reject by pnt
        int iRatio = dimension / 2;
        
        std::vector<float> vcCornerPoints;
        float fDeltaAngle = 360.0 / zxing::wxcode::CODING_ROW_SUM * 2;
        float fCenterX = dimension / 2.0;
        float fCenterY = dimension / 2.0;
        
        for (int i = 0; i < zxing::wxcode::CODING_ROW_SUM / 2; ++i)
        {
            vcCornerPoints.push_back(fCenterX + iRatio * sin(fDeltaAngle * i));
            vcCornerPoints.push_back(fCenterY + iRatio * cos(fDeltaAngle * i));
        }
        
        transform->transformPoints(vcCornerPoints);
        
        for (size_t i = 0; i < vcCornerPoints.size(); i += 2)
        {
            if (vcCornerPoints[i] < 0 || vcCornerPoints[i] >= iWidth)
            {
                float outLen = vcCornerPoints[i] < 0 ? (-vcCornerPoints[i]) : (vcCornerPoints[i] - iWidth + 1);
                if (outLen / fInitialMS > GetIniParser()->GetReal("NEW_GRID_SAMPLER", "OUT_OF_BOUNDS", 9))
                {
                    err_handler = ReaderErrorHandler("width out of bounds.");
                    if (err_handler.ErrCode()) return Ref<BitMatrix>();
                }
            }
            if (vcCornerPoints[i+1] < 0 || vcCornerPoints[i+1] >= iHeight)
            {
                float outLen = vcCornerPoints[i+1] < 0 ? (-vcCornerPoints[i+1]) : (vcCornerPoints[i+1] - iHeight + 1);
                // printf("f\n", outLen / fInitialMS);
                if (outLen / fInitialMS > GetIniParser()->GetReal("NEW_GRID_SAMPLER", "OUT_OF_BOUNDS", 9))
                {
                    err_handler = ReaderErrorHandler("height out of bounds.");
                    if (err_handler.ErrCode()) return Ref<BitMatrix>();
                }
            }
        }
    }
    {  // rj by shape
        std::vector<float> vcCornerPoints;
        vcCornerPoints.push_back(0), vcCornerPoints.push_back(0);  // a 0 1
        vcCornerPoints.push_back(0), vcCornerPoints.push_back(dimension - 1);  // b 2 3
        vcCornerPoints.push_back(dimension - 1), vcCornerPoints.push_back(dimension - 1);  // c 4 5
        vcCornerPoints.push_back(dimension - 1), vcCornerPoints.push_back(0);  // d 6 7
        // a ----- D
        //|		  |
        //|		  |
        // b ----- C
        
        transform->transformPoints(vcCornerPoints);
        float fVecABx = vcCornerPoints[2] - vcCornerPoints[0], fVecABy = vcCornerPoints[3] - vcCornerPoints[1];
        float fVecADx = vcCornerPoints[6] - vcCornerPoints[0], fVecADy = vcCornerPoints[7] - vcCornerPoints[1];
        float fVecCBx = vcCornerPoints[4] - vcCornerPoints[2], fVecCBy = vcCornerPoints[5] - vcCornerPoints[3];
        float fVecCDx = vcCornerPoints[4] - vcCornerPoints[6], fVecCDy = vcCornerPoints[5] - vcCornerPoints[7];
        
        float fAreaABD = fabs(fVecABx * fVecADy - fVecADx * fVecABy) / 2;
        float fAreaBCD = fabs(fVecCBx * fVecCDy - fVecCDx * fVecCBy) / 2;
        
        float fAreaQua = fAreaABD + fAreaBCD;
        float fLenAB = sqrt(fVecABx * fVecABx + fVecABy * fVecABy);
        float fLenAD = sqrt(fVecADx * fVecADx + fVecADy * fVecADy);
        float fLenCB = sqrt(fVecCBx * fVecCBx + fVecCBy * fVecCBy);
        float fLenCD = sqrt(fVecCDx * fVecCDx + fVecCDy * fVecCDy);
        float fLenAvg = (fLenAB + fLenAD + fLenCB + fLenCD) / 4;
        float fAreaSqua = fLenAvg * fLenAvg;
        if (fAreaSqua > 1e-8
            && fAreaQua / fAreaSqua > GetIniParser()->GetReal("NEW_GRID_SAMPLER", "SHAPE_MIN_RATIO", 0.95)
            && fAreaQua / fAreaSqua < GetIniParser()->GetReal("NEW_GRID_SAMPLER", "SHAPE_MAX_RATIO", 1.05))
        {
        }
        else
        {
            err_handler = ReaderErrorHandler("shape not valid");
            if (err_handler.ErrCode()) return Ref<BitMatrix>();
        }
    }
    
    for (int y = 0; y < dimension; y++) {
        int max = points.size();
        float yValue = static_cast<float>(y) + 0.5f;
        for (int x = 0; x < max; x += 2) {
            points[x] = static_cast<float>(x >> 1) + 0.5f;
            points[x + 1] = yValue;
        }
        transform->transformPoints(points);
        // Quick check to see if points transformed to something inside the image;
        // sufficient to check the endpoings
        outlier += checkAndNudgePoints(image->getWidth(), image->getHeight(), points, err_handler);
        if (err_handler.ErrCode()) return Ref<BitMatrix>();
        
        if (outlier >= maxOutlier)
        {
            std::ostringstream s;
            s << "Over 30% points out of bounds.";
            err_handler = ReaderErrorHandler(s.str().c_str());
            if (err_handler.ErrCode()) return Ref<BitMatrix>();
        }
        
        for (int x = 0; x < max; x += 2) {
            if (image->get(static_cast<int>(points[x]), static_cast<int>(points[x + 1]))) {
                // Black (-ish) pixel
                bits->set(x >> 1, y);
            }
        }
    }
    return bits;
}

int NewGridSampler::checkAndNudgePoints(int width, int height, std::vector<float> &points, ErrorHandler & err_handler) {
    // Modified to support stlport : valiantliu
    float* pts = NULL;
    
    if (points.size() > 0)
    {
        pts = &points[0];
    }
    else
    {
        err_handler = ReaderErrorHandler("checkAndNudgePoints:: no points!");
        if (err_handler.ErrCode())   return -1;
    }
    
    // The Java code assumes that if the start and end points are in bounds, the rest will also be.
    // However, in some unusual cases points in the middle may also be out of bounds.
    // Since we can't rely on an ArrayIndexOutOfBoundsException like Java, we check every point.
    
    int outCount = 0;

    if (pts == NULL) {
        err_handler = ReaderErrorHandler("checkAndNudgePoints:: no points!");
        return -1;
    }

    for (size_t offset = 0; offset < points.size(); offset += 2) {
        int x = static_cast<int>(pts[offset]);
        int y = static_cast<int>(pts[offset + 1]);
        
        if (x < -1 || x > width || y < -1 || y > height)
        {
            outCount++;
        }

        if (x <= -1)
        {
            points[offset] = 0.0f;
        }
        else if (x >= width)
        {
            points[offset] = static_cast<float>(width - 1);
        }
        if (y <= -1)
        {
            points[offset + 1] = 0.0f;
        }
        else if (y >= height)
        {
            points[offset + 1] = static_cast<float>(height - 1);
        }
    }
    
    return outCount;
}

NewGridSampler &NewGridSampler::getInstance() {
    return gridSampler;
}
}  // namespace zxing
