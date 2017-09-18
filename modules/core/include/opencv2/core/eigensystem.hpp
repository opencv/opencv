#ifndef __EIGENSYSTEM_HPP__
#define __EIGENSYSTEM_HPP__

using namespace std;
using namespace cv;
/**
 *
 * This class is just a rip-off of the EigenvalueSolver in JAMA
 * (http://math.nist.gov/javanumerics/jama), which include public-domain
 * implementation of linear algebra algorithms. I give the credit to the
 * original authors (http://math.nist.gov/javanumerics/jama/#Authors).
 *
 * Copyright Notice in JAMA
 *
 * This software is a cooperative product of The MathWorks and
 * the National Institute of Standards and Technology (NIST) which has been
 * released to the public domain. Neither The MathWorks nor NIST assumes any
 * responsibility whatsoever for its use by other parties, and makes no
 * guarantees, expressed or implied, about its quality, reliability, or any
 * other characteristic.
 *
 */

class EigenvalueDecomposition {
private:
  int n;
  double cdivr, cdivi;

  double *d, *e, *ort;
  double **V, **H;

  template <typename _Tp> _Tp *alloc_1d(int m) { return new _Tp[m]; }

  template <typename _Tp> _Tp *alloc_1d(int m, _Tp val) {
    _Tp *arr = alloc_1d<_Tp>(m);
    for (int i = 0; i < m; i++)
      arr[i] = val;
    return arr;
  }

  template <typename _Tp> _Tp **alloc_2d(int m, int nn) {
    _Tp **arr = new _Tp *[m];
    for (int i = 0; i < m; i++)
      arr[i] = new _Tp[nn];
    return arr;
  }

  template <typename _Tp> _Tp **alloc_2d(int m, int nn, _Tp val) {
    _Tp **arr = alloc_2d<_Tp>(m, nn);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < nn; j++) {
        arr[i][j] = val;
      }
    }
    return arr;
  }

  void cdiv(double xr, double xi, double yr, double yi) {
    double r, dd;
    if (abs(yr) > abs(yi)) {
      r = yi / yr;
      dd = yr + r * yi;
      cdivr = (xr + r * xi) / dd;
      cdivi = (xi - r * xr) / dd;
    } else {
      r = yr / yi;
      dd = yi + r * yr;
      cdivr = (r * xr + xi) / dd;
      cdivi = (r * xi - xr) / dd;
    }
  }

  // Nonsymmetric reduction from Hessenberg to real Schur form.

  void hqr2() {

    //  This is derived from the Algol procedure hqr2,
    //  by Martin and Wilkinson, Handbook for Auto. Comp.,
    //  Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.

    // Initialize
    int nn = this->n;
    int n1 = nn - 1;
    int low = 0;
    int high = nn - 1;
    double eps = pow(2.0, -52.0);
    double exshift = 0.0;
    double p = 0, q = 0, r = 0, s = 0, z = 0, t, w, x, y;

    // Store roots isolated by balanc and compute matrix norm

    double norm = 0.0;
    for (int i = 0; i < nn; i++) {
      if ((i < low) | (i > high)) {
        d[i] = H[i][i];
        e[i] = 0.0;
      }
      for (int j = max(i - 1, 0); j < nn; j++) {
        norm = norm + abs(H[i][j]);
      }
    }

    // Outer loop over eigenvalue index
    int iter = 0;
    while (n1 >= low) {

      // Look for single small sub-diagonal element
      int l = n1;
      while (l > low) {
        s = abs(H[l - 1][l - 1]) + abs(H[l][l]);
        if (s == 0.0) {
          s = norm;
        }
        if (abs(H[l][l - 1]) < eps * s) {
          break;
        }
        l--;
      }

      // Check for convergence
      // One root found

      if (l == n1) {
        H[n1][n1] = H[n1][n1] + exshift;
        d[n1] = H[n1][n1];
        e[n1] = 0.0;
        n1--;
        iter = 0;

        // Two roots found

      } else if (l == n1 - 1) {
        w = H[n1][n1 - 1] * H[n1 - 1][n1];
        p = (H[n1 - 1][n1 - 1] - H[n1][n1]) / 2.0;
        q = p * p + w;
        z = sqrt(abs(q));
        H[n1][n1] = H[n1][n1] + exshift;
        H[n1 - 1][n1 - 1] = H[n1 - 1][n1 - 1] + exshift;
        x = H[n1][n1];

        // Real pair

        if (q >= 0) {
          if (p >= 0) {
            z = p + z;
          } else {
            z = p - z;
          }
          d[n1 - 1] = x + z;
          d[n1] = d[n1 - 1];
          if (z != 0.0) {
            d[n1] = x - w / z;
          }
          e[n1 - 1] = 0.0;
          e[n1] = 0.0;
          x = H[n1][n1 - 1];
          s = abs(x) + abs(z);
          p = x / s;
          q = z / s;
          r = sqrt(p * p + q * q);
          p = p / r;
          q = q / r;

          // Row modification

          for (int j = n1 - 1; j < nn; j++) {
            z = H[n1 - 1][j];
            H[n1 - 1][j] = q * z + p * H[n1][j];
            H[n1][j] = q * H[n1][j] - p * z;
          }

          // Column modification

          for (int i = 0; i <= n1; i++) {
            z = H[i][n1 - 1];
            H[i][n1 - 1] = q * z + p * H[i][n1];
            H[i][n1] = q * H[i][n1] - p * z;
          }

          // Accumulate transformations

          for (int i = low; i <= high; i++) {
            z = V[i][n1 - 1];
            V[i][n1 - 1] = q * z + p * V[i][n1];
            V[i][n1] = q * V[i][n1] - p * z;
          }

          // Complex pair

        } else {
          d[n1 - 1] = x + p;
          d[n1] = x + p;
          e[n1 - 1] = z;
          e[n1] = -z;
        }
        n1 = n1 - 2;
        iter = 0;

        // No convergence yet

      } else {

        // Form shift

        x = H[n1][n1];
        y = 0.0;
        w = 0.0;
        if (l < n1) {
          y = H[n1 - 1][n1 - 1];
          w = H[n1][n1 - 1] * H[n1 - 1][n1];
        }

        // Wilkinson's original ad hoc shift

        if (iter == 10) {
          exshift += x;
          for (int i = low; i <= n1; i++) {
            H[i][i] -= x;
          }
          s = abs(H[n1][n1 - 1]) + abs(H[n1 - 1][n1 - 2]);
          x = y = 0.75 * s;
          w = -0.4375 * s * s;
        }

        // MATLAB's new ad hoc shift

        if (iter == 30) {
          s = (y - x) / 2.0;
          s = s * s + w;
          if (s > 0) {
            s = sqrt(s);
            if (y < x) {
              s = -s;
            }
            s = x - w / ((y - x) / 2.0 + s);
            for (int i = low; i <= n1; i++) {
              H[i][i] -= s;
            }
            exshift += s;
            x = y = w = 0.964;
          }
        }

        iter = iter + 1; // (Could check iteration count here.)

        // Look for two consecutive small sub-diagonal elements
        int m = n1 - 2;
        while (m >= l) {
          z = H[m][m];
          r = x - z;
          s = y - z;
          p = (r * s - w) / H[m + 1][m] + H[m][m + 1];
          q = H[m + 1][m + 1] - z - r - s;
          r = H[m + 2][m + 1];
          s = abs(p) + abs(q) + abs(r);
          p = p / s;
          q = q / s;
          r = r / s;
          if (m == l) {
            break;
          }
          if (abs(H[m][m - 1]) * (abs(q) + abs(r)) <
              eps * (abs(p) *
                     (abs(H[m - 1][m - 1]) + abs(z) + abs(H[m + 1][m + 1])))) {
            break;
          }
          m--;
        }

        for (int i = m + 2; i <= n1; i++) {
          H[i][i - 2] = 0.0;
          if (i > m + 2) {
            H[i][i - 3] = 0.0;
          }
        }

        // Double QR step involving rows l:n1 and columns m:n1

        for (int k = m; k <= n1 - 1; k++) {
          bool notlast = (k != n1 - 1);
          if (k != m) {
            p = H[k][k - 1];
            q = H[k + 1][k - 1];
            r = (notlast ? H[k + 2][k - 1] : 0.0);
            x = abs(p) + abs(q) + abs(r);
            if (x != 0.0) {
              p = p / x;
              q = q / x;
              r = r / x;
            }
          }
          if (x == 0.0) {
            break;
          }
          s = sqrt(p * p + q * q + r * r);
          if (p < 0) {
            s = -s;
          }
          if (s != 0) {
            if (k != m) {
              H[k][k - 1] = -s * x;
            } else if (l != m) {
              H[k][k - 1] = -H[k][k - 1];
            }
            p = p + s;
            x = p / s;
            y = q / s;
            z = r / s;
            q = q / p;
            r = r / p;

            // Row modification

            for (int j = k; j < nn; j++) {
              p = H[k][j] + q * H[k + 1][j];
              if (notlast) {
                p = p + r * H[k + 2][j];
                H[k + 2][j] = H[k + 2][j] - p * z;
              }
              H[k][j] = H[k][j] - p * x;
              H[k + 1][j] = H[k + 1][j] - p * y;
            }

            // Column modification

            for (int i = 0; i <= min(n1, k + 3); i++) {
              p = x * H[i][k] + y * H[i][k + 1];
              if (notlast) {
                p = p + z * H[i][k + 2];
                H[i][k + 2] = H[i][k + 2] - p * r;
              }
              H[i][k] = H[i][k] - p;
              H[i][k + 1] = H[i][k + 1] - p * q;
            }

            // Accumulate transformations

            for (int i = low; i <= high; i++) {
              p = x * V[i][k] + y * V[i][k + 1];
              if (notlast) {
                p = p + z * V[i][k + 2];
                V[i][k + 2] = V[i][k + 2] - p * r;
              }
              V[i][k] = V[i][k] - p;
              V[i][k + 1] = V[i][k + 1] - p * q;
            }
          } // (s != 0)
        }   // k loop
      }     // check convergence
    }       // while (n1 >= low)

    // Backsubstitute to find vectors of upper triangular form

    if (norm == 0.0) {
      return;
    }

    for (n1 = nn - 1; n1 >= 0; n1--) {
      p = d[n1];
      q = e[n1];

      // Real vector

      if (q == 0) {
        int l = n1;
        H[n1][n1] = 1.0;
        for (int i = n1 - 1; i >= 0; i--) {
          w = H[i][i] - p;
          r = 0.0;
          for (int j = l; j <= n1; j++) {
            r = r + H[i][j] * H[j][n1];
          }
          if (e[i] < 0.0) {
            z = w;
            s = r;
          } else {
            l = i;
            if (e[i] == 0.0) {
              if (w != 0.0) {
                H[i][n1] = -r / w;
              } else {
                H[i][n1] = -r / (eps * norm);
              }

              // Solve real equations

            } else {
              x = H[i][i + 1];
              y = H[i + 1][i];
              q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
              t = (x * s - z * r) / q;
              H[i][n1] = t;
              if (abs(x) > abs(z)) {
                H[i + 1][n1] = (-r - w * t) / x;
              } else {
                H[i + 1][n1] = (-s - y * t) / z;
              }
            }

            // Overflow control

            t = abs(H[i][n1]);
            if ((eps * t) * t > 1) {
              for (int j = i; j <= n1; j++) {
                H[j][n1] = H[j][n1] / t;
              }
            }
          }
        }

        // Complex vector

      } else if (q < 0) {
        int l = n1 - 1;

        // Last vector component imaginary so matrix is triangular

        if (abs(H[n1][n1 - 1]) > abs(H[n1 - 1][n1])) {
          H[n1 - 1][n1 - 1] = q / H[n1][n1 - 1];
          H[n1 - 1][n1] = -(H[n1][n1] - p) / H[n1][n1 - 1];
        } else {
          cdiv(0.0, -H[n1 - 1][n1], H[n1 - 1][n1 - 1] - p, q);
          H[n1 - 1][n1 - 1] = cdivr;
          H[n1 - 1][n1] = cdivi;
        }
        H[n1][n1 - 1] = 0.0;
        H[n1][n1] = 1.0;
        for (int i = n1 - 2; i >= 0; i--) {
          double ra, sa, vr, vi;
          ra = 0.0;
          sa = 0.0;
          for (int j = l; j <= n1; j++) {
            ra = ra + H[i][j] * H[j][n1 - 1];
            sa = sa + H[i][j] * H[j][n1];
          }
          w = H[i][i] - p;

          if (e[i] < 0.0) {
            z = w;
            r = ra;
            s = sa;
          } else {
            l = i;
            if (e[i] == 0) {
              cdiv(-ra, -sa, w, q);
              H[i][n1 - 1] = cdivr;
              H[i][n1] = cdivi;
            } else {

              // Solve complex equations

              x = H[i][i + 1];
              y = H[i + 1][i];
              vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
              vi = (d[i] - p) * 2.0 * q;
              if ((vr == 0.0) & (vi == 0.0)) {
                vr = eps * norm * (abs(w) + abs(q) + abs(x) + abs(y) + abs(z));
              }
              cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi);
              H[i][n1 - 1] = cdivr;
              H[i][n1] = cdivi;
              if (abs(x) > (abs(z) + abs(q))) {
                H[i + 1][n1 - 1] = (-ra - w * H[i][n1 - 1] + q * H[i][n1]) / x;
                H[i + 1][n1] = (-sa - w * H[i][n1] - q * H[i][n1 - 1]) / x;
              } else {
                cdiv(-r - y * H[i][n1 - 1], -s - y * H[i][n1], z, q);
                H[i + 1][n1 - 1] = cdivr;
                H[i + 1][n1] = cdivi;
              }
            }

            // Overflow control

            t = max(abs(H[i][n1 - 1]), abs(H[i][n1]));
            if ((eps * t) * t > 1) {
              for (int j = i; j <= n1; j++) {
                H[j][n1 - 1] = H[j][n1 - 1] / t;
                H[j][n1] = H[j][n1] / t;
              }
            }
          }
        }
      }
    }

    // Vectors of isolated roots

    for (int i = 0; i < nn; i++) {
      if ((i < low) | (i > high)) {
        for (int j = i; j < nn; j++) {
          V[i][j] = H[i][j];
        }
      }
    }

    // Back transformation to get eigenvectors of original matrix

    for (int j = nn - 1; j >= low; j--) {
      for (int i = low; i <= high; i++) {
        z = 0.0;
        for (int k = low; k <= min(j, high); k++) {
          z = z + V[i][k] * H[k][j];
        }
        V[i][j] = z;
      }
    }
  }

  // Nonsymmetric reduction to Hessenberg form.
  void orthes() {

    //  This is derived from the Algol procedures orthes and ortran,
    //  by Martin and Wilkinson, Handbook for Auto. Comp.,
    //  Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutines in EISPACK.
    int low = 0;
    int high = n - 1;

    for (int m = low + 1; m <= high - 1; m++) {

      // Scale column.

      double scale = 0.0;
      for (int i = m; i <= high; i++) {
        scale = scale + abs(H[i][m - 1]);
      }
      if (scale != 0.0) {

        // Compute Householder transformation.

        double h = 0.0;
        for (int i = high; i >= m; i--) {
          ort[i] = H[i][m - 1] / scale;
          h += ort[i] * ort[i];
        }
        double g = sqrt(h);
        if (ort[m] > 0) {
          g = -g;
        }
        h = h - ort[m] * g;
        ort[m] = ort[m] - g;

        // Apply Householder similarity transformation
        // H = (I-u*u'/h)*H*(I-u*u')/h)

        for (int j = m; j < n; j++) {
          double f = 0.0;
          for (int i = high; i >= m; i--) {
            f += ort[i] * H[i][j];
          }
          f = f / h;
          for (int i = m; i <= high; i++) {
            H[i][j] -= f * ort[i];
          }
        }

        for (int i = 0; i <= high; i++) {
          double f = 0.0;
          for (int j = high; j >= m; j--) {
            f += ort[j] * H[i][j];
          }
          f = f / h;
          for (int j = m; j <= high; j++) {
            H[i][j] -= f * ort[j];
          }
        }
        ort[m] = scale * ort[m];
        H[m][m - 1] = scale * g;
      }
    }

    // Accumulate transformations (Algol's ortran).

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        V[i][j] = (i == j ? 1.0 : 0.0);
      }
    }

    for (int m = high - 1; m >= low + 1; m--) {
      if (H[m][m - 1] != 0.0) {
        for (int i = m + 1; i <= high; i++) {
          ort[i] = H[i][m - 1];
        }
        for (int j = m; j <= high; j++) {
          double g = 0.0;
          for (int i = m; i <= high; i++) {
            g += ort[i] * V[i][j];
          }
          // Double division avoids possible underflow
          g = (g / ort[m]) / H[m][m - 1];
          for (int i = m; i <= high; i++) {
            V[i][j] += g * ort[i];
          }
        }
      }
    }
  }

  void compute() {
    // allocate temporary matrices
    V = alloc_2d<double>(n, n, 0.0);
    d = alloc_1d<double>(n);
    e = alloc_1d<double>(n);
    ort = alloc_1d<double>(n);
    // Reduce to Hessenberg form.
    orthes();
    // Reduce Hessenberg to real Schur form.
    hqr2();
  }

public:
  EigenvalueDecomposition() : n(0) {}

  EigenvalueDecomposition(const Mat &src) : n(src.cols) { compute(src); }

  template <typename _Tp>
  EigenvalueDecomposition(const Mat_<_Tp> &src) : n(src.cols) {
    compute(src);
  }

  void compute(const Mat &src) { compute(Mat_<double>(src)); }

  template <typename _Tp> void compute(const Mat_<_Tp> &src) {
    // allocate the data to work on
    H = alloc_2d<double>(n, n);
    // now safely copy the data
    for (int i = 0; i < src.rows; i++) {
      for (int j = 0; j < src.cols; j++) {
        H[i][j] = src(i, j);
      }
    }
    // finally perform the eigenvalue decomposition of H
    compute();
  }

  ~EigenvalueDecomposition() {
    // free some memory
      delete[] d; delete[] e; delete[] ort;
    for (int i = 0; i < n; i++) {
      delete[] H[i];
      delete[] V[i];
    }
      delete[] H; delete[] V;
  }

  Mat eigenvalues() {
    Mat eigenval(1, n, CV_64FC1);
    for (int i = 0; i < n; i++) {
      eigenval.at<double>(0, i) = d[i];
    }
    return eigenval;
  }

  Mat eigenvectors() {
    Mat eigenvec(n, n, CV_64FC1);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        eigenvec.at<double>(i, j) = V[i][j];
    return eigenvec;
  }

  static void  eigensystem(InputArray src, OutputArray eigenvalues, OutputArray eigenvectors)
    {
        Mat srcIn;
        src.getMat().convertTo(srcIn, CV_64FC1);
        EigenvalueDecomposition es(srcIn);
        es.eigenvalues().copyTo(eigenvalues);
        es.eigenvectors().copyTo(eigenvectors);
    };
};

#endif /* eigensystem_h */
