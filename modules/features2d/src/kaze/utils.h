#ifndef __OPENCV_FEATURES_2D_KAZE_UTILS_H__
#define __OPENCV_FEATURES_2D_KAZE_UTILS_H__

/* ************************************************************************* */
/**
 * @brief This function computes the value of a 2D Gaussian function
 * @param x X Position
 * @param y Y Position
 * @param sig Standard Deviation
 */
inline float gaussian(float x, float y, float sigma) {
  return expf(-(x*x + y*y) / (2.0f*sigma*sigma));
}

/* ************************************************************************* */
/**
 * @brief This function checks descriptor limits
 * @param x X Position
 * @param y Y Position
 * @param width Image width
 * @param height Image height
 */
inline void checkDescriptorLimits(int &x, int &y, int width, int height) {

  if (x < 0) {
    x = 0;
  }

  if (y < 0) {
    y = 0;
  }

  if (x > width - 1) {
    x = width - 1;
  }

  if (y > height - 1) {
    y = height - 1;
  }
}

/* ************************************************************************* */
/**
 * @brief This funtion rounds float to nearest integer
 * @param flt Input float
 * @return dst Nearest integer
 */
inline int fRound(float flt) {
  return (int)(flt + 0.5f);
}

/* ************************************************************************* */
/**
 * @brief Exponentiation by squaring
 * @param flt Exponentiation base
 * @return dst Exponentiation value
 */
inline int fastpow(int base, int exp) {
    int res = 1;
    while(exp > 0) {
        if(exp & 1) {
            exp--;
            res *= base;
        } else {
            exp /= 2;
            base *= base;
        }
    }
    return res;
}

#endif
