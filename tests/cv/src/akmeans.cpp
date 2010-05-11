/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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

#include "cvtest.h"

#if 0
                       
/* Testing parameters */
static char test_desc[] = "KMeans clustering";
static char* func_name[] = 
{
    "cvKMeans"
};

//based on Ara Nefian's implementation
float distance(float* vector_1, float *vector_2, int VecSize)
{
  int i;
  float dist;

  dist = 0.0;
  for (i = 0; i < VecSize; i++)
  {
      //printf ("%f, %f\n", vector_1[i], vector_2[i]);
      dist = dist + (vector_1[i] - vector_2[i])*(vector_1[i] - vector_2[i]);
  }
  return dist;  
}

//returns number of made iterations
int _real_kmeans( int numClusters, float **sample, int numSamples, 
                   int VecSize, int* a_class, double eps, int iter )

{                            
  int     i, k, n;
  int     *counter;
  float   minDist;
  float   *dist; 
  float   **curr_cluster;
  float   **prev_cluster;

  float   error;
  
  //printf("* numSamples = %d, numClusters = %d, VecSize = %d\n", numSamples, numClusters, VecSize);

  //memory allocation 
  dist = new float[numClusters];
  counter = new int[numClusters];

  //allocate memory for curr_cluster and prev_cluster
  curr_cluster = new float*[numClusters];
  prev_cluster = new float*[numClusters];
  for (k = 0; k < numClusters; k++){
      curr_cluster[k] = new float[VecSize]; 
      prev_cluster[k] = new float[VecSize]; 
  } 

  //pick initial cluster centers
  for (k = 0; k < numClusters; k++)
  { 
    for (n = 0; n < VecSize; n++)
    {
       curr_cluster[k][n] = sample[k*(numSamples/numClusters)][n]; 
       prev_cluster[k][n] = sample[k*(numSamples/numClusters)][n]; 
    }
  }
  

  int NumIter = 0;
  error = FLT_MAX;
  while ((error > eps) && (NumIter < iter))
  {
    NumIter++;
    //printf("NumIter = %d, error = %lf, \n", NumIter, error);

    //assign samples to clusters
    for (i = 0; i < numSamples; i++)
    { 
      for (k = 0; k < numClusters; k++)
      {
          dist[k] = distance(sample[i], curr_cluster[k], VecSize);
      }
      minDist = dist[0];
      a_class[i] = 0;
      for (k = 1; k < numClusters; k++)
      {
        if (dist[k] < minDist)
        {
           minDist = dist[k];
           a_class[i] = k;
        }
      }
    }
    
   //reset clusters and counters
   for (k = 0; k < numClusters; k++){
     counter[k] = 0; 
     for (n = 0; n < VecSize; n++){
        curr_cluster[k][n] = 0.0; 
     }
   }
    for (i = 0; i < numSamples; i++){
      for (n = 0; n < VecSize; n++){ 
          curr_cluster[a_class[i]][n] = curr_cluster[a_class[i]][n] + sample[i][n];
      }
      counter[a_class[i]]++;  
    }
   
   for (k = 0; k < numClusters; k++){  
      for (n = 0; n < VecSize; n++){
         curr_cluster[k][n] = curr_cluster[k][n]/(float)counter[k];
      }
    }

    error = 0.0;  
    for (k = 0; k < numClusters; k++){
      for (n = 0; n < VecSize; n++){
        error = error + (curr_cluster[k][n] - prev_cluster[k][n])*(curr_cluster[k][n] - prev_cluster[k][n]);
      }
    }
    //error = error/(double)(numClusters*VecSize);

    //copy curr_clusters to prev_clusters
    for (k = 0; k < numClusters; k++){
      for (n =0; n < VecSize; n++){
        prev_cluster[k][n] = curr_cluster[k][n];  
      }
    }

  } 
  
  //deallocate memory for curr_cluster and prev_cluster 
  for (k = 0; k < numClusters; k++){
      delete curr_cluster[k]; 
      delete prev_cluster[k]; 
  } 
  delete curr_cluster;
  delete prev_cluster;

  delete counter;
  delete dist;
  return NumIter;
     
}

static int fmaKMeans(void)
{
    CvTermCriteria crit;
    float** vectors;
    int*    output;
    int*    etalon_output;

    int lErrors = 0;
    int lNumVect = 0;
    int lVectSize = 0;
    int lNumClust = 0;
    int lMaxNumIter = 0;
    float flEpsilon = 0;

    int i,j;
    static int  read_param = 0;

    /* Initialization global parameters */
    if( !read_param )
    {
        read_param = 1;
        /* Read test-parameters */
        trsiRead( &lNumVect, "1000", "Number of vectors" );
        trsiRead( &lVectSize, "10", "Number of vectors" );
        trsiRead( &lNumClust, "20", "Number of clusters" );
        trsiRead( &lMaxNumIter,"100","Maximal number of iterations");
        trssRead( &flEpsilon, "0.5", "Accuracy" );
    }

    crit = cvTermCriteria( CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, lMaxNumIter, flEpsilon );
    
    //allocate vectors
    vectors = (float**)cvAlloc( lNumVect * sizeof(float*) );
    for( i = 0; i < lNumVect; i++ )
    {
        vectors[i] = (float*)cvAlloc( lVectSize * sizeof( float ) );
    }

    output = (int*)cvAlloc( lNumVect * sizeof(int) );
    etalon_output = (int*)cvAlloc( lNumVect * sizeof(int) );
    
    //fill input vectors
    for( i = 0; i < lNumVect; i++ )
    {
        ats1flInitRandom( -2000, 2000, vectors[i], lVectSize );
    }
    
    /* run etalon kmeans */
    /* actually it is the simpliest realization of kmeans */

    int ni = _real_kmeans( lNumClust, vectors, lNumVect, lVectSize, etalon_output, crit.epsilon, crit.max_iter );

    trsWrite(  ATS_CON, "%d iterations done\n",  ni );
                  
    /* Run OpenCV function */
#define _KMEANS_TIME 0

#if _KMEANS_TIME
    //timing section 
    trsTimerStart(0);
    __int64 tics = atsGetTickCount();  
#endif  

    cvKMeans( lNumClust, vectors, lNumVect, lVectSize, 
              crit, output );

#if _KMEANS_TIME
    tics = atsGetTickCount() - tics;     
    trsTimerStop(0);
    //output result
    //double dbUsecs =ATS_TICS_TO_USECS((double)tics);
    trsWrite( ATS_CON, "Tics per iteration %d\n", tics/ni );    

#endif

    //compare results
    for( j = 0; j < lNumVect; j++ )
    {
        if ( output[j] != etalon_output[j] )
        {
            lErrors++;
        }
    }

    //free memory
    for( i = 0; i < lNumVect; i++ )
    {
        cvFree( &(vectors[i]) );
    }
    cvFree(&vectors);
    cvFree(&output);
    cvFree(&etalon_output);      
   
   if( lErrors == 0 ) return trsResult( TRS_OK, "No errors fixed for this text" );
    else return trsResult( TRS_FAIL, "Detected %d errors", lErrors );

}



void InitAKMeans()
{
    /* Register test function */
    trsReg( func_name[0], test_desc, atsAlgoClass, fmaKMeans );
    
} /* InitAKMeans */

#endif
