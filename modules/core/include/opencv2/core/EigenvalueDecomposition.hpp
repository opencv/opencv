//
//  EigenvalueDecomposition.hpp
//  OpenCV
//
//  Created by Jasper Shemilt on 29/09/2017.
//
//

#ifndef ida_h
#define ida_h

namespace cv
{
class EigenvalueDecomposition {
private:

    // Holds the data dimension.
    int n;

    // Stores real/imag part of a complex division.
    double cdivr, cdivi;

    // Pointer to internal memory.
    double *d, *e, *ort;
    double **V, **H;

    // Holds the computed eigenvalues.
    Mat _eigenvalues;

    // Holds the computed eigenvectors.
    Mat _eigenvectors;

    // Allocates memory.
    template<typename _Tp>
    _Tp *alloc_1d(int m) ;

    // Allocates memory.
    template<typename _Tp>
    _Tp *alloc_1d(int m, _Tp val) ;

    // Allocates memory.
    template<typename _Tp>
    _Tp **alloc_2d(int m, int _n) ;

    // Allocates memory.
    template<typename _Tp>
    _Tp **alloc_2d(int m, int _n, _Tp val) ;

    void cdiv(double xr, double xi, double yr, double yi) ;

    // Nonsymmetric reduction from Hessenberg to real Schur form.

    void hqr2() ;

    // Nonsymmetric reduction to Hessenberg form.
    void orthes() ;

    // Releases all internal working memory.
    void release() ;

    // Computes the Eigenvalue Decomposition for a matrix given in H.
    void compute() ;

public:
    EigenvalueDecomposition();

    // Initializes & computes the Eigenvalue Decomposition for a general matrix src
    EigenvalueDecomposition(InputArray src) ;

    // This function computes the Eigenvalue Decomposition for a general matrix src
    void compute(InputArray src);

    ~EigenvalueDecomposition();

    // Returns the eigenvalues of the Eigenvalue Decomposition.
    Mat eigenvalues() ;
    // Returns the eigenvectors of the Eigenvalue Decomposition.
    Mat eigenvectors() ;
};

}


#endif /* ida_h */
