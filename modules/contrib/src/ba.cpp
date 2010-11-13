/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                         License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2009, PhaseSpace Inc., all rights reserved.
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
//   * The names of the copyright holders may not be used to endorse or promote products
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
#include "opencv2/calib3d/calib3d.hpp"

namespace cv {  

LevMarqSparse::LevMarqSparse()
{
    A = B = W = Vis_index = X = prevP = P = deltaP = err = JtJ_diag = S = hX = NULL;
    U = ea = V = inv_V_star = eb = Yj = NULL;    
    num_points = 0;
    num_cams = 0;
}

LevMarqSparse::~LevMarqSparse()
{
    clear();
} 

LevMarqSparse::LevMarqSparse(int npoints, // number of points
        int ncameras, // number of cameras
        int nPointParams, // number of params per one point  (3 in case of 3D points)
        int nCameraParams, // number of parameters per one camera
        int nErrParams, // number of parameters in measurement vector
                        // for 1 point at one camera (2 in case of 2D projections)
        Mat& visibility, // visibility matrix. rows correspond to points, columns correspond to cameras
                         // 1 - point is visible for the camera, 0 - invisible
        Mat& P0, // starting vector of parameters, first cameras then points
        Mat& X_, // measurements, in order of visibility. non visible cases are skipped 
        TermCriteria criteria, // termination criteria
        
        // callback for estimation of Jacobian matrices
        void (CV_CDECL * fjac)(int i, int j, Mat& point_params,
                               Mat& cam_params, Mat& A, Mat& B, void* data),
        // callback for estimation of backprojection errors
        void (CV_CDECL * func)(int i, int j, Mat& point_params,
                               Mat& cam_params, Mat& estim, void* data),
        void* data // user-specific data passed to the callbacks
        )
{
    A = B = W = Vis_index = X = prevP = P = deltaP = err = JtJ_diag = S = hX = NULL;
    U = ea = V = inv_V_star = eb = Yj = NULL;
    
    run(npoints, ncameras, nPointParams, nCameraParams, nErrParams, visibility,
        P0, X_, criteria, fjac, func, data);
}

void LevMarqSparse::clear()
{
     for( int i = 0; i < num_points; i++ )
     {
         for(int j = 0; j < num_cams; j++ )
         {  
                 CvMat* tmp = ((CvMat**)(A->data.ptr + i * A->step))[j];
                 if( tmp )
                     cvReleaseMat( &tmp );

                 tmp = ((CvMat**)(B->data.ptr + i * B->step))[j];
                 if( tmp )
                     cvReleaseMat( &tmp );
                 
                 tmp = ((CvMat**)(W->data.ptr + j * W->step))[i];
                 if( tmp )
                     cvReleaseMat( &tmp ); 
         }
     }   
     cvReleaseMat( &A );
     cvReleaseMat( &B );
     cvReleaseMat( &W );       
     cvReleaseMat( &Vis_index);

     for( int j = 0; j < num_cams; j++ )
     {
         cvReleaseMat( &U[j] );
     }
     delete U;

     for( int j = 0; j < num_cams; j++ )
     {
         cvReleaseMat( &ea[j] );
     }
     delete ea;
     
     //allocate V and inv_V_star
     for( int i = 0; i < num_points; i++ )
     {
         cvReleaseMat(&V[i]);
         cvReleaseMat(&inv_V_star[i]);
     }
     delete V;
     delete inv_V_star;

     for( int i = 0; i < num_points; i++ )
     {
         cvReleaseMat(&eb[i]);
     }
     delete eb;

     for( int i = 0; i < num_points; i++ )
     {
         cvReleaseMat(&Yj[i]);
     }   
     delete Yj;
     
    cvReleaseMat(&X);
    cvReleaseMat(&prevP);
    cvReleaseMat(&P);
    cvReleaseMat(&deltaP);

    cvReleaseMat(&err);      
    
    cvReleaseMat(&JtJ_diag);
    cvReleaseMat(&S);
    cvReleaseMat(&hX);
}

//A params correspond to  Cameras
//B params correspont to  Points

//num_cameras  - total number of cameras
//num_points   - total number of points

//num_par_per_camera - number of parameters per camera
//num_par_per_point - number of parameters per point

//num_errors - number of measurements.

void LevMarqSparse::run( int num_points_, //number of points
            int num_cams_, //number of cameras
            int num_point_param_, //number of params per one point  (3 in case of 3D points)
            int num_cam_param_, //number of parameters per one camera
            int num_err_param_, //number of parameters in measurement vector for 1 point at one camera (2 in case of 2D projections)
            Mat& visibility,   //visibility matrix . rows correspond to points, columns correspond to cameras
                               // 0 - point is visible for the camera, 0 - invisible
            Mat& P0, //starting vector of parameters, first cameras then points
            Mat& X_init, //measurements, in order of visibility. non visible cases are skipped 
            TermCriteria criteria_init,
            void (*fjac_)(int i, int j, Mat& point_params, Mat& cam_params, Mat& A, Mat& B, void* data),
            void (*func_)(int i, int j, Mat& point_params, Mat& cam_params, Mat& estim, void* data),
            void* data_
             ) //termination criteria
{
    clear();
    
    func = func_; //assign evaluation function
    fjac = fjac_; //assign jacobian
    data = data_;

    num_cams = num_cams_;
    num_points = num_points_;
    num_err_param = num_err_param_; 
    num_cam_param = num_cam_param_;
    num_point_param = num_point_param_;

    //compute all sizes
    int Aij_width = num_cam_param;
    int Aij_height = num_err_param;

    int Bij_width = num_point_param;
    int Bij_height = num_err_param;

    int U_size = Aij_width;
    int V_size = Bij_width;

    int Wij_height = Aij_width;
    int Wij_width = Bij_width;

    //allocate memory for all Aij, Bij, U, V, W
    
    //allocate num_points*num_cams matrices A
    
    //Allocate matrix A whose elements are nointers to Aij
    //if Aij is zero (point i is not visible in camera j) then A(i,j) contains NULL
    A = cvCreateMat( num_points, num_cams, CV_32S /*pointer is stored here*/ );
    B = cvCreateMat( num_points, num_cams, CV_32S /*pointer is stored here*/ );
    W = cvCreateMat( num_cams, num_points, CV_32S /*pointer is stored here*/ );
    Vis_index = cvCreateMat( num_points, num_cams, CV_32S /*integer index is stored here*/ );
    cvSetZero( A );
    cvSetZero( B );
    cvSetZero( W );
    cvSet( Vis_index, cvScalar(-1) );
    
    //fill matrices A and B based on visibility
    CvMat _vis = visibility;
    int index = 0;
    for( int i = 0; i < num_points; i++ )
    {
        for(int j = 0; j < num_cams; j++ )
        {  
            if( ((int*)(_vis.data.ptr+ i * _vis.step))[j] )
            {
                ((int*)(Vis_index->data.ptr + i * Vis_index->step))[j] = index;
                index += num_err_param;
    
                //create matrices Aij, Bij
                CvMat* tmp = cvCreateMat( Aij_height, Aij_width, CV_64F );
                ((CvMat**)(A->data.ptr + i * A->step))[j] = tmp;
    
                tmp = cvCreateMat( Bij_height, Bij_width, CV_64F );
                ((CvMat**)(B->data.ptr + i * B->step))[j] = tmp;
    
                tmp = cvCreateMat( Wij_height, Wij_width, CV_64F );
                ((CvMat**)(W->data.ptr + j * W->step))[i] = tmp;  //note indices i and j swapped
            }              
        }                
    }
    
    //allocate U
    U = new CvMat* [num_cams];
    for( int j = 0; j < num_cams; j++ )
    {
        U[j] = cvCreateMat( U_size, U_size, CV_64F );
    }
    //allocate ea
    ea = new CvMat* [num_cams];
    for( int j = 0; j < num_cams; j++ )
    {
        ea[j] = cvCreateMat( U_size, 1, CV_64F );
    }
    
    //allocate V and inv_V_star
    V = new CvMat* [num_points];
    inv_V_star = new CvMat* [num_points];
    for( int i = 0; i < num_points; i++ )
    {
        V[i] = cvCreateMat( V_size, V_size, CV_64F );
        inv_V_star[i] = cvCreateMat( V_size, V_size, CV_64F );
    }
    
    //allocate eb
    eb = new CvMat* [num_points];
    for( int i = 0; i < num_points; i++ )
    {
        eb[i] = cvCreateMat( V_size, 1, CV_64F );
    }   
    
    //allocate Yj
    Yj = new CvMat* [num_points];
    for( int i = 0; i < num_points; i++ )
    {
        Yj[i] = cvCreateMat( Wij_height, Wij_width, CV_64F );  //Yij has the same size as Wij
    }        
    
    //allocate matrix S
    S = cvCreateMat( num_cams * num_cam_param, num_cams * num_cam_param, CV_64F);
    
    JtJ_diag = cvCreateMat( num_cams * num_cam_param + num_points * num_point_param, 1, CV_64F );
    
    //set starting parameters
    CvMat _tmp_ = CvMat(P0); 
    prevP = cvCloneMat( &_tmp_ );          
    P = cvCloneMat( &_tmp_ );
    deltaP = cvCloneMat( &_tmp_ );
    
    //set measurements
    _tmp_ = CvMat(X_init);
    X = cvCloneMat( &_tmp_ );  
    //create vector for estimated measurements
    hX = cvCreateMat( X->rows, X->cols, CV_64F );
    //create error vector
    err = cvCreateMat( X->rows, X->cols, CV_64F );
                              
    ask_for_proj();
         
    //compute initial error
    cvSub(  X, hX, err );
    
    prevErrNorm = cvNorm( err, 0,  CV_L2 );
    iters = 0; 
    criteria = criteria_init;
    
    optimize();
}

void LevMarqSparse::ask_for_proj()
{
    //given parameter P, compute measurement hX
    int ind = 0;
    for( int i = 0; i < num_points; i++ )
    {
        CvMat point_mat;
        cvGetSubRect( P, &point_mat, cvRect( 0, num_cams * num_cam_param + num_point_param * i, 1, num_point_param ));
    
        for( int j = 0; j < num_cams; j++ )
        {
            CvMat* Aij = ((CvMat**)(A->data.ptr + A->step * i))[j];
            if( Aij ) //visible
            {
                CvMat cam_mat;
                cvGetSubRect( P, &cam_mat, cvRect( 0, j * num_cam_param, 1, num_cam_param ));
                CvMat measur_mat;
                cvGetSubRect( hX, &measur_mat, cvRect( 0, ind * num_err_param, 1, num_err_param ));
                Mat _point_mat(&point_mat), _cam_mat(&cam_mat), _measur_mat(&measur_mat);
                func( i, j, _point_mat, _cam_mat, _measur_mat, data );
                
                assert( ind*num_err_param == ((int*)(Vis_index->data.ptr + i * Vis_index->step))[j]);
    
                ind+=1;
                
            }  
        } 
    }
}
//iteratively asks for Jacobians for every camera_point pair
void LevMarqSparse::ask_for_projac()  //should be evaluated at point prevP 
{
    // compute jacobians Aij and Bij
    for( int i = 0; i < A->height; i++ )
    {
        CvMat point_mat;
        cvGetSubRect( prevP, &point_mat, cvRect( 0, num_cams * num_cam_param + num_point_param * i, 1, num_point_param ));


        CvMat** A_line = (CvMat**)(A->data.ptr + A->step * i);
        CvMat** B_line = (CvMat**)(B->data.ptr + B->step * i);

        for( int j = 0; j < A->width; j++ )
        {   
            CvMat* Aij = A_line[j];
            if( Aij ) //Aij is not zero
            {
                CvMat cam_mat;
                cvGetSubRect( prevP, &cam_mat, cvRect( 0, j * num_cam_param, 1, num_cam_param ));
                
                CvMat* Bij = B_line[j];
                Mat _point_mat(&point_mat), _cam_mat(&cam_mat), _Aij(Aij), _Bij(Bij);
                (*fjac)(i, j, _point_mat, _cam_mat, _Aij, _Bij, data);
            }
        }
    }
}  

void LevMarqSparse::optimize() //main function that runs minimization
{   
    bool done = false;
    
    CvMat* YWt = cvCreateMat( num_cam_param, num_cam_param, CV_64F ); //this matrix used to store Yij*Wik' 
    CvMat* E = cvCreateMat( S->height, 1 , CV_64F ); //this is right part of system with S       

    while(!done)
    { 
        // compute jacobians Aij and Bij
        ask_for_projac();
        
        //compute U_j  and  ea_j
        for( int j = 0; j < num_cams; j++ )
        {
            cvSetZero(U[j]); 
            cvSetZero(ea[j]);
            //summ by i (number of points)
            for( int i = 0; i < num_points; i++ )
            {
                //get Aij
                CvMat* Aij = ((CvMat**)(A->data.ptr + A->step * i))[j];                
                if( Aij )
                {
                    //Uj+= AijT*Aij
                    cvGEMM( Aij, Aij, 1, U[j], 1, U[j], CV_GEMM_A_T );

                    //ea_j += AijT * e_ij
                    CvMat eij;

                    int index = ((int*)(Vis_index->data.ptr + i * Vis_index->step))[j];

                    cvGetSubRect( err, &eij, cvRect( 0, index, 1, Aij->height /*width of transposed Aij*/ ) );
                    cvGEMM( Aij, &eij, 1, ea[j], 1, ea[j], CV_GEMM_A_T );
                }
            }
        } //U_j and ea_j computed for all j

        //compute V_i  and  eb_i
        for( int i = 0; i < num_points; i++ )
        {
            cvSetZero(V[i]); 
            cvSetZero(eb[i]);
            
            //summ by i (number of points)
            for( int j = 0; j < num_cams; j++ )
            {
                //get Bij
                CvMat* Bij = ((CvMat**)(B->data.ptr + B->step * i))[j];
                
                if( Bij )
                {
                    //Vi+= BijT*Bij
                    cvGEMM( Bij, Bij, 1, V[i], 1, V[i], CV_GEMM_A_T );

                    //eb_i += BijT * e_ij
                    int index = ((int*)(Vis_index->data.ptr + i * Vis_index->step))[j];

                    CvMat eij;
                    cvGetSubRect( err, &eij, cvRect( 0, index, 1, Bij->height /*width of transposed Bij*/ ) );
                    cvGEMM( Bij, &eij, 1, eb[i], 1, eb[i], CV_GEMM_A_T );
                }
            }
        } //V_i and eb_i computed for all i

        //compute W_ij
        for( int i = 0; i < num_points; i++ )
        {
            for( int j = 0; j < num_cams; j++ )
            {
                 CvMat* Aij = ((CvMat**)(A->data.ptr + A->step * i))[j];
                 if( Aij ) //visible
                 {
                     CvMat* Bij = ((CvMat**)(B->data.ptr + B->step * i))[j];
                     CvMat* Wij = ((CvMat**)(W->data.ptr + W->step * j))[i];

                     //multiply
                     cvGEMM( Aij, Bij, 1, NULL, 0, Wij, CV_GEMM_A_T );                     
                 }
            }
        } //Wij computed

        //backup diagonal of JtJ before we start augmenting it
        {               
            CvMat dia;
            CvMat subr;
            for( int j = 0; j < num_cams; j++ )
            {                          
                cvGetDiag(U[j], &dia);
                cvGetSubRect(JtJ_diag, &subr, 
                             cvRect(0, j*num_cam_param, 1, num_cam_param ));
                cvCopy( &dia, &subr );
            } 
            for( int i = 0; i < num_points; i++ )
            {
                cvGetDiag(V[i], &dia);
                cvGetSubRect(JtJ_diag, &subr, 
                             cvRect(0, num_cams*num_cam_param + i * num_point_param, 1, num_point_param ));
                cvCopy( &dia, &subr );
            }   
        } 

        if( iters == 0 )
        {
            //initialize lambda. It is set to 1e-3 * average diagonal element in JtJ
            double average_diag = 0;
            for( int j = 0; j < num_cams; j++ )
            {
                average_diag += cvTrace( U[j] ).val[0];
            }
            for( int i = 0; i < num_points; i++ )
            {
                average_diag += cvTrace( V[i] ).val[0];
            }
            average_diag /= (num_cams*num_cam_param + num_points * num_point_param );
                        
            lambda = 1e-3 * average_diag;        
        }
       
        //now we are going to find good step and make it
        for(;;)
        {
            //augmentation of diagonal
            for(int j = 0; j < num_cams; j++ )
            {
                CvMat diag;
                cvGetDiag( U[j], &diag );
#if 1
                cvAddS( &diag, cvScalar( lambda ), &diag );
#else
                cvScale( &diag, &diag, 1 + lambda );
#endif
            }
            for(int i = 0; i < num_points; i++ )
            {
                CvMat diag;
                cvGetDiag( V[i], &diag );
#if 1
                cvAddS( &diag, cvScalar( lambda ), &diag );
#else
                cvScale( &diag, &diag, 1 + lambda );
#endif
            }                              
            bool error = false;
            //compute inv(V*)
            bool inverted_ok = true;
            for(int i = 0; i < num_points; i++ )
            {
                double det = cvInvert( V[i], inv_V_star[i] );

                if( fabs(det) <= FLT_EPSILON ) 
                {                       
                    inverted_ok = false;
                    break;
                } //means we did wrong augmentation, try to choose different lambda
            }

            if( inverted_ok )
            {
                cvSetZero( E ); 
                //loop through cameras, compute upper diagonal blocks of matrix S 
                for( int j = 0; j < num_cams; j++ )
                {  
                    //compute Yij = Wij (V*_i)^-1  for all i   (if Wij exists/nonzero)
                    for( int i = 0; i < num_points; i++ )
                    {   
                        //
                        CvMat* Wij = ((CvMat**)(W->data.ptr + W->step * j))[i];
                        if( Wij )
                        {
                            cvMatMul( Wij, inv_V_star[i], Yj[i] );
                        }
                    }

                    //compute Sjk   for k>=j  (because Sjk = Skj)
                    for( int k = j; k < num_cams; k++ )
                    {
                        cvSetZero( YWt );
                        for( int i = 0; i < num_points; i++ )
                        {
                            //check that both Wij and Wik exist
                            CvMat* Wij = ((CvMat**)(W->data.ptr + W->step * j))[i];
                            CvMat* Wik = ((CvMat**)(W->data.ptr + W->step * k))[i];

                            if( Wij && Wik )
                            {
                                //multiply YWt += Yj[i]*Wik'
                                cvGEMM( Yj[i], Wik, 1, YWt, 1, YWt, CV_GEMM_B_T /*transpose Wik*/ ); 
                            }
                        }

                        //copy result to matrix S

                        CvMat Sjk;
                        //extract submat
                        cvGetSubRect( S, &Sjk, cvRect( k * num_cam_param, j * num_cam_param, num_cam_param, num_cam_param ));  
                        

                        //if j==k, add diagonal
                        if( j != k )
                        {
                            //just copy with minus
                            cvScale( YWt, &Sjk, -1 ); //if we set initial S to zero then we can use cvSub( Sjk, YWt, Sjk);
                        }
                        else
                        {
                            //add diagonal value

                            //subtract YWt from augmented Uj
                            cvSub( U[j], YWt, &Sjk );
                        }                
                    }

                    //compute right part of equation involving matrix S
                    // e_j=ea_j - \sum_i Y_ij eb_i 
                    {
                    CvMat e_j; 
                    
                    //select submat
                    cvGetSubRect( E, &e_j, cvRect( 0, j * num_cam_param, 1, num_cam_param ) ); 
                    
                    for( int i = 0; i < num_points; i++ )
                    {
                        CvMat* Wij = ((CvMat**)(W->data.ptr + W->step * j))[i];
                        if( Wij )
                            cvMatMulAdd( Yj[i], eb[i], &e_j, &e_j );
                    }

                    cvSub( ea[j], &e_j, &e_j );
                    }

                } 
                //fill below diagonal elements of matrix S
                cvCompleteSymm( S,  0 /*from upper to low*/ ); //operation may be done by nonzero blocks or during upper diagonal computation 
                
                //Solve linear system  S * deltaP_a = E
                CvMat dpa;
                cvGetSubRect( deltaP, &dpa, cvRect(0, 0, 1, S->width ) );
                int res = cvSolve( S, E, &dpa );
            
                if( res ) //system solved ok
                {   
                    //compute db_i
                    for( int i = 0; i < num_points; i++ )
                    {
                        CvMat dbi;
                        cvGetSubRect( deltaP, &dbi, cvRect( 0, dpa.height + i * num_point_param, 1, num_point_param ) );   

                        /* compute \sum_j W_ij^T da_j */
                        for( int j = 0; j < num_cams; j++ )
                        {
                            //get Wij
                            CvMat* Wij = ((CvMat**)(W->data.ptr + W->step * j))[i];

                            if( Wij )
                            {
                                //get da_j
                                CvMat daj;
                                cvGetSubRect( &dpa, &daj, cvRect( 0, j * num_cam_param, 1, num_cam_param ));  
                                cvGEMM( Wij, &daj, 1, &dbi, 1, &dbi, CV_GEMM_A_T /* transpose Wij */ ); 
                            }  
                        }
                        //finalize dbi
                        cvSub( eb[i], &dbi, &dbi );
                        cvMatMul(inv_V_star[i], &dbi, &dbi );  //here we get final dbi  
                    }  //now we computed whole deltaP

                    //add deltaP to delta 
                    cvAdd( prevP, deltaP, P );
                                        
                    //evaluate  function with new parameters
                    ask_for_proj(); // func( P, hX );

                    //compute error
                    errNorm = cvNorm( X, hX, CV_L2 );
                                        
                }
                else
                {
                    error = true;
                }                
            }
            else
            {
                error = true;
            }
            //check solution
            if( error || /* singularities somewhere */ 
                errNorm > prevErrNorm )  //step was not accepted
            {
                //increase lambda and reject change 
                lambda *= 10;

                //restore diagonal from backup
                {               
                    CvMat dia;
                    CvMat subr;
                    for( int j = 0; j < num_cams; j++ )
                    {                          
                        cvGetDiag(U[j], &dia);
                        cvGetSubRect(JtJ_diag, &subr, 
                                     cvRect(0, j*num_cam_param, 1, num_cam_param ));
                        cvCopy( &subr, &dia );
                    } 
                    for( int i = 0; i < num_points; i++ )
                    {
                        cvGetDiag(V[i], &dia);
                        cvGetSubRect(JtJ_diag, &subr, 
                                     cvRect(0, num_cams*num_cam_param + i * num_point_param, 1, num_point_param ));
                        cvCopy( &subr, &dia );
                    }   
                }                  
            }
            else  //all is ok
            {
                //accept change and decrease lambda
                lambda /= 10;
                lambda = MAX(lambda, 1e-16);
                prevErrNorm = errNorm;

                //compute new projection error vector
                cvSub(  X, hX, err );
                break;
            }
        }      
        iters++;

        double param_change_norm = cvNorm(P, prevP, CV_RELATIVE_L2);
        //check termination criteria
        if( (criteria.type&CV_TERMCRIT_ITER && iters > criteria.max_iter ) || 
            (criteria.type&CV_TERMCRIT_EPS && param_change_norm < criteria.epsilon) )
        {
            done = true;
            break;
        }  
        else
        {
            //copy new params and continue iterations
            cvCopy( P, prevP );
        }
    }   
    cvReleaseMat(&YWt); 
    cvReleaseMat(&E);
} 

//Utilities

void fjac(int /*i*/, int /*j*/, CvMat *point_params, CvMat* cam_params, CvMat* A, CvMat* B, void* /*data*/) 
{
    //compute jacobian per camera parameters (i.e. Aij)
    //take i-th point 3D current coordinates
    
    CvMat _Mi;
    cvReshape(point_params, &_Mi, 3, 1 );

    CvMat* _mp = cvCreateMat(1, 2, CV_64F ); //projection of the point

    //split camera params into different matrices
    CvMat _ri, _ti, _k;
    cvGetRows( cam_params, &_ri, 0, 3 );
    cvGetRows( cam_params, &_ti, 3, 6 );

    double intr_data[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1};
    intr_data[0] = cam_params->data.db[6];
    intr_data[4] = cam_params->data.db[7];
    intr_data[2] = cam_params->data.db[8];
    intr_data[5] = cam_params->data.db[9];

    CvMat matA = cvMat(3,3, CV_64F, intr_data ); 

    CvMat _dpdr, _dpdt, _dpdf, _dpdc, _dpdk;
    
    bool have_dk = cam_params->height - 10 ? true : false;

    cvGetCols( A, &_dpdr, 0, 3 );
    cvGetCols( A, &_dpdt, 3, 6 );
    cvGetCols( A, &_dpdf, 6, 8 );
    cvGetCols( A, &_dpdc, 8, 10 );
    
    if( have_dk )
    {
        cvGetRows( cam_params, &_k, 10, cam_params->height );
        cvGetCols( A, &_dpdk, 10, A->width );
    }
    cvProjectPoints2( &_Mi, &_ri, &_ti, &matA, have_dk ? &_k : NULL, _mp, &_dpdr, &_dpdt,
        &_dpdf, &_dpdc, have_dk ? &_dpdk : NULL, 0);   

    cvReleaseMat( &_mp );                                 

    //compute jacobian for point params
    //compute dMeasure/dPoint3D

    // x = (r11 * X + r12 * Y + r13 * Z + t1)
    // y = (r21 * X + r22 * Y + r23 * Z + t2)
    // z = (r31 * X + r32 * Y + r33 * Z + t3)

    // x' = x/z
    // y' = y/z

    //d(x') = ( dx*z - x*dz)/(z*z)
    //d(y') = ( dy*z - y*dz)/(z*z) 

    //g = 1 + k1*r_2 + k2*r_4 + k3*r_6
    //r_2 = x'*x' + y'*y'

    //d(r_2) = 2*x'*dx' + 2*y'*dy'

    //dg = k1* d(r_2) + k2*2*r_2*d(r_2) + k3*3*r_2*r_2*d(r_2) 

    //x" = x'*g + 2*p1*x'*y' + p2(r_2+2*x'_2)
    //y" = y'*g + p1(r_2+2*y'_2) + 2*p2*x'*y'
               
    //d(x") = d(x') * g + x' * d(g) + 2*p1*( d(x')*y' + x'*dy) + p2*(d(r_2) + 2*2*x'* dx')
    //d(y") = d(y') * g + y' * d(g) + 2*p2*( d(x')*y' + x'*dy) + p1*(d(r_2) + 2*2*y'* dy')  

    // u = fx*( x") + cx
    // v = fy*( y") + cy
    
    // du = fx * d(x")  = fx * ( dx*z - x*dz)/ (z*z)
    // dv = fy * d(y")  = fy * ( dy*z - y*dz)/ (z*z)

    // dx/dX = r11,  dx/dY = r12, dx/dZ = r13 
    // dy/dX = r21,  dy/dY = r22, dy/dZ = r23
    // dz/dX = r31,  dz/dY = r32, dz/dZ = r33 

    // du/dX = fx*(r11*z-x*r31)/(z*z)
    // du/dY = fx*(r12*z-x*r32)/(z*z)
    // du/dZ = fx*(r13*z-x*r33)/(z*z)

    // dv/dX = fy*(r21*z-y*r31)/(z*z)
    // dv/dY = fy*(r22*z-y*r32)/(z*z)
    // dv/dZ = fy*(r23*z-y*r33)/(z*z)

    //get rotation matrix
    double R[9], t[3], fx = intr_data[0], fy = intr_data[4];
    CvMat matR = cvMat( 3, 3, CV_64F, R );
    cvRodrigues2(&_ri, &matR);

    double X,Y,Z;
    X = point_params->data.db[0];
    Y = point_params->data.db[1];
    Z = point_params->data.db[2];

    t[0] = _ti.data.db[0];
    t[1] = _ti.data.db[1];
    t[2] = _ti.data.db[2];

    //compute x,y,z
    double x = R[0] * X + R[1] * Y + R[2] * Z + t[0];
    double y = R[3] * X + R[4] * Y + R[5] * Z + t[1];
    double z = R[6] * X + R[7] * Y + R[8] * Z + t[2]; 

#if 1    
    //compute x',y'
    double x_strike = x/z;
    double y_strike = y/z;   
    //compute dx',dy'  matrix
    //
    //    dx'/dX  dx'/dY dx'/dZ    =    
    //    dy'/dX  dy'/dY dy'/dZ

    double coeff[6] = { z, 0, -x,
                        0, z, -y };
    CvMat coeffmat = cvMat( 2, 3, CV_64F, coeff );

    CvMat* dstrike_dbig = cvCreateMat(2,3,CV_64F);
    cvMatMul(&coeffmat, &matR, dstrike_dbig);
    cvScale(dstrike_dbig, dstrike_dbig, 1/(z*z) );      
    
    if( have_dk )
    {
        double strike_[2] = {x_strike, y_strike};
        CvMat strike = cvMat(1, 2, CV_64F, strike_);       
        
        //compute r_2
        double r_2 = x_strike*x_strike + y_strike*y_strike;
        double r_4 = r_2*r_2;
        double r_6 = r_4*r_2;

        //compute d(r_2)/dbig
        CvMat* dr2_dbig = cvCreateMat(1,3,CV_64F);
        cvMatMul( &strike, dstrike_dbig, dr2_dbig);
        cvScale( dr2_dbig, dr2_dbig, 2 );

        double& k1 = _k.data.db[0];
        double& k2 = _k.data.db[1];
        double& p1 = _k.data.db[2];
        double& p2 = _k.data.db[3];          
        double k3 = 0;

        if( _k.cols*_k.rows == 5 )
        {   
            k3 = _k.data.db[4];
        }    
        //compute dg/dbig
        double dg_dr2 = k1 + k2*2*r_2 + k3*3*r_4;
        double g = 1+k1*r_2+k2*r_4+k3*r_6;

        CvMat* dg_dbig = cvCreateMat(1,3,CV_64F);
        cvScale( dr2_dbig, dg_dbig, dg_dr2 ); 

        CvMat* tmp = cvCreateMat( 2, 3, CV_64F );
        CvMat* dstrike2_dbig = cvCreateMat( 2, 3, CV_64F );
                                  
        double c[4] = { g+2*p1*y_strike+4*p2*x_strike,       2*p1*x_strike,
                        2*p2*y_strike,                 g+2*p2*x_strike + 4*p1*y_strike };

        CvMat coeffmat = cvMat(2,2,CV_64F, c );

        cvMatMul(&coeffmat, dstrike_dbig, dstrike2_dbig );

        cvGEMM( &strike, dg_dbig, 1, NULL, 0, tmp, CV_GEMM_A_T );
        cvAdd( dstrike2_dbig, tmp, dstrike2_dbig );

        double p[2] = { p2, p1 };
        CvMat pmat = cvMat(2, 1, CV_64F, p );

        cvMatMul( &pmat, dr2_dbig ,tmp);
        cvAdd( dstrike2_dbig, tmp, dstrike2_dbig );   

        cvCopy( dstrike2_dbig, B );

        cvReleaseMat(&dr2_dbig);
        cvReleaseMat(&dg_dbig);

        cvReleaseMat(&tmp);
        cvReleaseMat(&dstrike2_dbig);
        cvReleaseMat(&tmp);  
    } 
    else
    {
        cvCopy(dstrike_dbig, B);
    }
    //multiply by fx, fy
    CvMat row;
    cvGetRows( B, &row, 0, 1 );
    cvScale( &row, &row, fx );    
    
    cvGetRows( B, &row, 1, 2 );
    cvScale( &row, &row, fy );

#else

    double k = fx/(z*z);

    cvmSet( B, 0, 0, k*(R[0]*z-x*R[6]));
    cvmSet( B, 0, 1, k*(R[1]*z-x*R[7]));
    cvmSet( B, 0, 2, k*(R[2]*z-x*R[8]));
    
    k = fy/(z*z);        
    
    cvmSet( B, 1, 0, k*(R[3]*z-y*R[6]));
    cvmSet( B, 1, 1, k*(R[4]*z-y*R[7]));
    cvmSet( B, 1, 2, k*(R[5]*z-y*R[8]));
    
#endif
    
};
void func(int /*i*/, int /*j*/, CvMat *point_params, CvMat* cam_params, CvMat* estim, void* /*data*/) 
{
    //just do projections
    CvMat _Mi;
    cvReshape( point_params, &_Mi, 3, 1 );

    CvMat* _mp = cvCreateMat(1, 2, CV_64F ); //projection of the point

    //split camera params into different matrices
    CvMat _ri, _ti, _k;

    cvGetRows( cam_params, &_ri, 0, 3 );
    cvGetRows( cam_params, &_ti, 3, 6 );

    double intr_data[9] = {0, 0, 0, 0, 0, 0, 0, 0, 1};
    intr_data[0] = cam_params->data.db[6];
    intr_data[4] = cam_params->data.db[7];
    intr_data[2] = cam_params->data.db[8];
    intr_data[5] = cam_params->data.db[9];

    CvMat matA = cvMat(3,3, CV_64F, intr_data ); 

    //int cn = CV_MAT_CN(_Mi.type);

    bool have_dk = cam_params->height - 10 ? true : false;
           
    if( have_dk )
    {
        cvGetRows( cam_params, &_k, 10, cam_params->height );        
    }  
    cvProjectPoints2( &_Mi, &_ri, &_ti, &matA, have_dk ? &_k : NULL, _mp, NULL, NULL,
                                                              NULL, NULL, NULL, 0);   
    cvTranspose( _mp, estim );
    cvReleaseMat( &_mp );
};

void fjac_new(int i, int j, Mat& point_params, Mat& cam_params, Mat& A, Mat& B, void* data)
{
    CvMat _point_params = point_params, _cam_params = cam_params, matA = A, matB = B;
    fjac(i,j, &_point_params, &_cam_params, &matA, &matB, data);
};

void func_new(int i, int j, Mat& point_params, Mat& cam_params, Mat& estim, void* data) 
{
    CvMat _point_params = point_params, _cam_params = cam_params, _estim = estim;
    func(i,j,&_point_params,&_cam_params,&_estim,data);
};                                                 

void LevMarqSparse::bundleAdjust( vector<Point3d>& points, //positions of points in global coordinate system (input and output)
                  const vector<vector<Point2d> >& imagePoints, //projections of 3d points for every camera 
                  const vector<vector<int> >& visibility, //visibility of 3d points for every camera 
                  vector<Mat>& cameraMatrix, //intrinsic matrices of all cameras (input and output)
                  vector<Mat>& R, //rotation matrices of all cameras (input and output)
                  vector<Mat>& T, //translation vector of all cameras (input and output)
                  vector<Mat>& distCoeffs, //distortion coefficients of all cameras (input and output)
                  const TermCriteria& criteria)
                  //,enum{MOTION_AND_STRUCTURE,MOTION,STRUCTURE})
{     
    int num_points = (int)points.size();
    int num_cameras = (int)cameraMatrix.size();

    CV_Assert( imagePoints.size() == (size_t)num_cameras && 
               visibility.size() == (size_t)num_cameras && 
               R.size() == (size_t)num_cameras &&
               T.size() == (size_t)num_cameras &&
               (distCoeffs.size() == (size_t)num_cameras || distCoeffs.size() == 0) );                

    int numdist = distCoeffs.size() ? (distCoeffs[0].rows * distCoeffs[0].cols) : 0;

    int num_cam_param = 3 /* rotation vector */ + 3 /* translation vector */
                        + 2 /* fx, fy */ + 2 /* cx, cy */ + numdist; 

    int num_point_param = 3; 

    //collect camera parameters into vector
    Mat params( num_cameras * num_cam_param + num_points * num_point_param, 1, CV_64F );

    //fill camera params
    for( int i = 0; i < num_cameras; i++ )
    {   
        //rotation
        Mat rot_vec; Rodrigues( R[i], rot_vec );
        Mat dst = params.rowRange(i*num_cam_param, i*num_cam_param+3);
        rot_vec.copyTo(dst);

        //translation
        dst = params.rowRange(i*num_cam_param + 3, i*num_cam_param+6);
        T[i].copyTo(dst); 
        
        //intrinsic camera matrix
        double* intr_data = (double*)cameraMatrix[i].data;
        double* intr = (double*)(params.data + params.step * (i*num_cam_param+6));
        //focals
        intr[0] = intr_data[0];  //fx
        intr[1] = intr_data[4];  //fy
        //center of projection
        intr[2] = intr_data[2];  //cx
        intr[3] = intr_data[5];  //cy  

        //add distortion if exists
        if( distCoeffs.size() )
        {
            dst = params.rowRange(i*num_cam_param + 10, i*num_cam_param+10+numdist);
            distCoeffs[i].copyTo(dst); 
        }
    }  

    //fill point params
    Mat ptparams(num_points, 1, CV_64FC3, params.data + num_cameras*num_cam_param*params.step);
    Mat _points(points);
    CV_Assert(_points.size() == ptparams.size() && _points.type() == ptparams.type());
    _points.copyTo(ptparams);

    //convert visibility vectors to visibility matrix
    Mat vismat(num_points, num_cameras, CV_32S);
    for( int i = 0; i < num_cameras; i++ )
    {
        //get row
        Mat col = vismat.col(i);
        Mat((int)visibility[i].size(), 1, vismat.type(), (void*)&visibility[i][0]).copyTo( col );
    }

    int num_proj = countNonZero(vismat); //total number of points projections

    //collect measurements
    Mat X(num_proj*2,1,CV_64F); //measurement vector      
    
    int counter = 0;
    for(int i = 0; i < num_points; i++ )
    {
        for(int j = 0; j < num_cameras; j++ )
        {
            //check visibility
            if( visibility[j][i] )
            {
                //extract point and put tu vector
                Point2d p = imagePoints[j][i];
                ((double*)(X.data))[counter] = p.x;
                ((double*)(X.data))[counter+1] = p.y;
                counter+=2;
            }             
        }   
    }

    LevMarqSparse levmar( num_points, num_cameras, num_point_param, num_cam_param, 2, vismat, params, X,
                          TermCriteria(criteria), fjac_new, func_new, NULL );
    //extract results
    //fill point params
    Mat final_points(num_points, 1, CV_64FC3,
        levmar.P->data.db + num_cameras*num_cam_param *levmar.P->step);
    CV_Assert(_points.size() == final_points.size() && _points.type() == final_points.type());
    final_points.copyTo(_points);
    
    //fill camera params
    for( int i = 0; i < num_cameras; i++ )
    {   
        //rotation
        Mat rot_vec = Mat(levmar.P).rowRange(i*num_cam_param, i*num_cam_param+3);
        Rodrigues( rot_vec, R[i] );
        //translation
        T[i] = Mat(levmar.P).rowRange(i*num_cam_param + 3, i*num_cam_param+6);  

        //intrinsic camera matrix
        double* intr_data = (double*)cameraMatrix[i].data;
        double* intr = (double*)(Mat(levmar.P).data + Mat(levmar.P).step * (i*num_cam_param+6));
        //focals
        intr_data[0] = intr[0];  //fx
        intr_data[4] = intr[1];  //fy
        //center of projection
        intr_data[2] = intr[2];  //cx
        intr_data[5] = intr[3];  //cy  

        //add distortion if exists
        if( distCoeffs.size() )
        {
            params.rowRange(i*num_cam_param + 10, i*num_cam_param+10+numdist).copyTo(distCoeffs[i]);
        }
    } 
}    

}// end of namespace cv
