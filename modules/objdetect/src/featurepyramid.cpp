#include "precomp.hpp"
#include "_latentsvm.h"
#include "_lsvm_resizeimg.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

/*
// Getting feature map for the selected subimage
//
// API
// int getFeatureMaps(const IplImage * image, const int k, featureMap **map);
// INPUT
// image             - selected subimage
// k                 - size of cells
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int getFeatureMaps(const IplImage* image, const int k, CvLSVMFeatureMap **map)
{
    int sizeX, sizeY;
    int p, px, stringSize;
    int height, width, numChannels;
    int i, j, kk, c, ii, jj, d;
    float  * datadx, * datady;
    
    //номер канала в цикле
    int   ch; 
    //переменные вычисления магнитуды
    float magnitude, x, y, tx, ty;
    
    IplImage * dx, * dy;
    int *nearest;
    float *w, a_x, b_x;

    // ядро для вычисления градиентов изображение по осям x и y
    float kernel[3] = {-1.f, 0.f, 1.f};
    CvMat kernel_dx = cvMat(1, 3, CV_32F, kernel);
    CvMat kernel_dy = cvMat(3, 1, CV_32F, kernel);

    // грачение градиента
    float * r;
    // новер сектора куда попало значение градиента
    //     четные иннексы не контрастное изображение
    //  не четные иннексы    контрастное изображение
    int   * alfa;
    
    // векторы границ секторов
    float boundary_x[NUM_SECTOR + 1];
    float boundary_y[NUM_SECTOR + 1];
    float max, dotProd;
    int   maxi;

    height = image->height;
    width  = image->width ;

    numChannels = image->nChannels;

    dx    = cvCreateImage(cvSize(image->width, image->height), 
                          IPL_DEPTH_32F, 3);
    dy    = cvCreateImage(cvSize(image->width, image->height), 
                          IPL_DEPTH_32F, 3);

    sizeX = width  / k;
    sizeY = height / k;
    px    = 3 * NUM_SECTOR; // контрастное и не контрастное изображение
    p     = px;
    stringSize = sizeX * p;
    allocFeatureMapObject(map, sizeX, sizeY, p);

    cvFilter2D(image, dx, &kernel_dx, cvPoint(-1, 0));
    cvFilter2D(image, dy, &kernel_dy, cvPoint(0, -1));
    
    float arg_vector;
    for(i = 0; i <= NUM_SECTOR; i++)
    {
        arg_vector    = ( (float) i ) * ( (float)(PI) / (float)(NUM_SECTOR) );
        boundary_x[i] = cosf(arg_vector);
        boundary_y[i] = sinf(arg_vector);
    }/*for(i = 0; i <= NUM_SECTOR; i++) */

    r    = (float *)malloc( sizeof(float) * (width * height));
    alfa = (int   *)malloc( sizeof(int  ) * (width * height * 2));

    for(j = 1; j < height - 1; j++)
    {
        datadx = (float*)(dx->imageData + dx->widthStep * j);
        datady = (float*)(dy->imageData + dy->widthStep * j);
        for(i = 1; i < width - 1; i++)
        {
            c = 0;
            x = (datadx[i * numChannels + c]);
            y = (datady[i * numChannels + c]);

            r[j * width + i] =sqrtf(x * x + y * y);
            for(ch = 1; ch < numChannels; ch++)
            {
                tx = (datadx[i * numChannels + ch]);
                ty = (datady[i * numChannels + ch]);
                magnitude = sqrtf(tx * tx + ty * ty);
                if(magnitude > r[j * width + i])
                {
                    r[j * width + i] = magnitude;
                    c = ch;
                    x = tx;
                    y = ty;
                }
            }/*for(ch = 1; ch < numChannels; ch++)*/
            
            max  = boundary_x[0] * x + boundary_y[0] * y;
            maxi = 0;
            for (kk = 0; kk < NUM_SECTOR; kk++) 
            {
                dotProd = boundary_x[kk] * x + boundary_y[kk] * y;
                if (dotProd > max) 
                {
                    max  = dotProd;
                    maxi = kk;
                }
                else 
                {
                    if (-dotProd > max) 
                    {
                        max  = -dotProd;
                        maxi = kk + NUM_SECTOR;
                    }
                }
            }
            alfa[j * width * 2 + i * 2    ] = maxi % NUM_SECTOR;
            alfa[j * width * 2 + i * 2 + 1] = maxi;  
        }/*for(i = 0; i < width; i++)*/
    }/*for(j = 0; j < height; j++)*/

    //подсчет весов и смещений
    nearest = (int  *)malloc(sizeof(int  ) *  k);
    w       = (float*)malloc(sizeof(float) * (k * 2));
    
    for(i = 0; i < k / 2; i++)
    {
        nearest[i] = -1;
    }/*for(i = 0; i < k / 2; i++)*/
    for(i = k / 2; i < k; i++)
    {
        nearest[i] = 1;
    }/*for(i = k / 2; i < k; i++)*/

    for(j = 0; j < k / 2; j++)
    {
        b_x = k / 2 + j + 0.5f;
        a_x = k / 2 - j - 0.5f;
        w[j * 2    ] = 1.0f/a_x * ((a_x * b_x) / ( a_x + b_x)); 
        w[j * 2 + 1] = 1.0f/b_x * ((a_x * b_x) / ( a_x + b_x));  
    }/*for(j = 0; j < k / 2; j++)*/
    for(j = k / 2; j < k; j++)
    {
        a_x = j - k / 2 + 0.5f;
        b_x =-j + k / 2 - 0.5f + k;
        w[j * 2    ] = 1.0f/a_x * ((a_x * b_x) / ( a_x + b_x)); 
        w[j * 2 + 1] = 1.0f/b_x * ((a_x * b_x) / ( a_x + b_x));  
    }/*for(j = k / 2; j < k; j++)*/


    //интерполяция
    for(i = 0; i < sizeY; i++)
    {
      for(j = 0; j < sizeX; j++)
      {
        for(ii = 0; ii < k; ii++)
        {
          for(jj = 0; jj < k; jj++)
          {
            if ((i * k + ii > 0) && 
                (i * k + ii < height - 1) && 
                (j * k + jj > 0) && 
                (j * k + jj < width  - 1))
            {
              d = (k * i + ii) * width + (j * k + jj);
              (*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]] += 
                  r[d] * w[ii * 2] * w[jj * 2];
              (*map)->map[ i * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2] * w[jj * 2];
              if ((i + nearest[ii] >= 0) && 
                  (i + nearest[ii] <= sizeY - 1))
              {
                (*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2    ]             ] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 ];
                (*map)->map[(i + nearest[ii]) * stringSize + j * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 ];
              }
              if ((j + nearest[jj] >= 0) && 
                  (j + nearest[jj] <= sizeX - 1))
              {
                (*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2    ]             ] += 
                  r[d] * w[ii * 2] * w[jj * 2 + 1];
                (*map)->map[i * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2] * w[jj * 2 + 1];
              }
              if ((i + nearest[ii] >= 0) && 
                  (i + nearest[ii] <= sizeY - 1) && 
                  (j + nearest[jj] >= 0) && 
                  (j + nearest[jj] <= sizeX - 1))
              {
                (*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2    ]             ] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
                (*map)->map[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * (*map)->numFeatures + alfa[d * 2 + 1] + NUM_SECTOR] += 
                  r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
              }
            }
          }/*for(jj = 0; jj < k; jj++)*/
        }/*for(ii = 0; ii < k; ii++)*/
      }/*for(j = 1; j < sizeX - 1; j++)*/
    }/*for(i = 1; i < sizeY - 1; i++)*/
    
    cvReleaseImage(&dx);
    cvReleaseImage(&dy);


    free(w);
    free(nearest);
    
    free(r);
    free(alfa);

    return LATENT_SVM_OK;
}

/*
// Feature map Normalization and Truncation 
//
// API
// int normalizeAndTruncate(featureMap *map, const float alfa);
// INPUT
// map               - feature map
// alfa              - truncation threshold
// OUTPUT
// map               - truncated and normalized feature map
// RESULT
// Error status
*/
int normalizeAndTruncate(CvLSVMFeatureMap *map, const float alfa)
{
    int i,j, ii;
    int sizeX, sizeY, p, pos, pp, xp, pos1, pos2;
    float * partOfNorm; // norm of C(i, j)
    float * newData;
    float   valOfNorm;

    sizeX     = map->sizeX;
    sizeY     = map->sizeY;
    partOfNorm = (float *)malloc (sizeof(float) * (sizeX * sizeY));

    p  = NUM_SECTOR;
    xp = NUM_SECTOR * 3;
    pp = NUM_SECTOR * 12;

    for(i = 0; i < sizeX * sizeY; i++)
    {
        valOfNorm = 0.0f;
        pos = i * map->numFeatures;
        for(j = 0; j < p; j++)
        {
            valOfNorm += map->map[pos + j] * map->map[pos + j];
        }/*for(j = 0; j < p; j++)*/
        partOfNorm[i] = valOfNorm;
    }/*for(i = 0; i < sizeX * sizeY; i++)*/
    
    sizeX -= 2;
    sizeY -= 2;

    newData = (float *)malloc (sizeof(float) * (sizeX * sizeY * pp));
//normalization
    for(i = 1; i <= sizeY; i++)
    {
        for(j = 1; j <= sizeX; j++)
        {
            valOfNorm = sqrtf(
                partOfNorm[(i    )*(sizeX + 2) + (j    )] +
                partOfNorm[(i    )*(sizeX + 2) + (j + 1)] +
                partOfNorm[(i + 1)*(sizeX + 2) + (j    )] +
                partOfNorm[(i + 1)*(sizeX + 2) + (j + 1)]);
            pos1 = (i  ) * (sizeX + 2) * xp + (j  ) * xp;
            pos2 = (i-1) * (sizeX    ) * pp + (j-1) * pp;
            for(ii = 0; ii < p; ii++)
            {
                newData[pos2 + ii        ] = map->map[pos1 + ii    ] / valOfNorm;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                newData[pos2 + ii + p * 4] = map->map[pos1 + ii + p] / valOfNorm;
            }/*for(ii = 0; ii < 2 * p; ii++)*/
            valOfNorm = sqrtf(
                partOfNorm[(i    )*(sizeX + 2) + (j    )] +
                partOfNorm[(i    )*(sizeX + 2) + (j + 1)] +
                partOfNorm[(i - 1)*(sizeX + 2) + (j    )] +
                partOfNorm[(i - 1)*(sizeX + 2) + (j + 1)]);
            for(ii = 0; ii < p; ii++)
            {
                newData[pos2 + ii + p    ] = map->map[pos1 + ii    ] / valOfNorm;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                newData[pos2 + ii + p * 6] = map->map[pos1 + ii + p] / valOfNorm;
            }/*for(ii = 0; ii < 2 * p; ii++)*/
            valOfNorm = sqrtf(
                partOfNorm[(i    )*(sizeX + 2) + (j    )] +
                partOfNorm[(i    )*(sizeX + 2) + (j - 1)] +
                partOfNorm[(i + 1)*(sizeX + 2) + (j    )] +
                partOfNorm[(i + 1)*(sizeX + 2) + (j - 1)]);
            for(ii = 0; ii < p; ii++)
            {
                newData[pos2 + ii + p * 2] = map->map[pos1 + ii    ] / valOfNorm;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                newData[pos2 + ii + p * 8] = map->map[pos1 + ii + p] / valOfNorm;
            }/*for(ii = 0; ii < 2 * p; ii++)*/
            valOfNorm = sqrtf(
                partOfNorm[(i    )*(sizeX + 2) + (j    )] +
                partOfNorm[(i    )*(sizeX + 2) + (j - 1)] +
                partOfNorm[(i - 1)*(sizeX + 2) + (j    )] +
                partOfNorm[(i - 1)*(sizeX + 2) + (j - 1)]);
            for(ii = 0; ii < p; ii++)
            {
                newData[pos2 + ii + p * 3 ] = map->map[pos1 + ii    ] / valOfNorm;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                newData[pos2 + ii + p * 10] = map->map[pos1 + ii + p] / valOfNorm;
            }/*for(ii = 0; ii < 2 * p; ii++)*/
        }/*for(j = 1; j <= sizeX; j++)*/
    }/*for(i = 1; i <= sizeY; i++)*/
//truncation
    for(i = 0; i < sizeX * sizeY * pp; i++)
    {
        if(newData [i] > alfa) newData [i] = alfa;
    }/*for(i = 0; i < sizeX * sizeY * pp; i++)*/
//swop data

    map->numFeatures  = pp;
    map->sizeX = sizeX;
    map->sizeY = sizeY;

    free (map->map);
    free (partOfNorm);

    map->map = newData;

    return LATENT_SVM_OK;
}
/*
// Feature map reduction
// In each cell we reduce dimension of the feature vector
// according to original paper special procedure
//
// API
// int PCAFeatureMaps(featureMap *map)
// INPUT
// map               - feature map
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int PCAFeatureMaps(CvLSVMFeatureMap *map)
{ 
    int i,j, ii, jj, k;
    int sizeX, sizeY, p,  pp, xp, yp, pos1, pos2;
    float * newData;
    float val;
    float nx, ny;
    
    sizeX = map->sizeX;
    sizeY = map->sizeY;
    p     = map->numFeatures;
    pp    = NUM_SECTOR * 3 + 4;
    yp    = 4;
    xp    = NUM_SECTOR;

    nx    = 1.0f / sqrtf((float)(xp * 2));
    ny    = 1.0f / sqrtf((float)(yp    ));

    newData = (float *)malloc (sizeof(float) * (sizeX * sizeY * pp));

    for(i = 0; i < sizeY; i++)
    {
        for(j = 0; j < sizeX; j++)
        {
            pos1 = ((i)*sizeX + j)*p;
            pos2 = ((i)*sizeX + j)*pp;
            k = 0;
            for(jj = 0; jj < xp * 2; jj++)
            {
                val = 0;
                for(ii = 0; ii < yp; ii++)
                {
                    val += map->map[pos1 + yp * xp + ii * xp * 2 + jj];
                }/*for(ii = 0; ii < yp; ii++)*/
                newData[pos2 + k] = val * ny;
                k++;
            }/*for(jj = 0; jj < xp * 2; jj++)*/
            for(jj = 0; jj < xp; jj++)
            {
                val = 0;
                for(ii = 0; ii < yp; ii++)
                {
                    val += map->map[pos1 + ii * xp + jj];
                }/*for(ii = 0; ii < yp; ii++)*/
                newData[pos2 + k] = val * ny;
                k++;
            }/*for(jj = 0; jj < xp; jj++)*/
            for(ii = 0; ii < yp; ii++)
            {
                val = 0;
                for(jj = 0; jj < 2 * xp; jj++)
                {
                    val += map->map[pos1 + yp * xp + ii * xp * 2 + jj];
                }/*for(jj = 0; jj < xp; jj++)*/
                newData[pos2 + k] = val * nx;
                k++;
            } /*for(ii = 0; ii < yp; ii++)*/           
        }/*for(j = 0; j < sizeX; j++)*/
    }/*for(i = 0; i < sizeY; i++)*/
//swop data

    map->numFeatures = pp;

    free (map->map);

    map->map = newData;

    return LATENT_SVM_OK;
}


int getPathOfFeaturePyramid(IplImage * image, 
                            float step, int numStep, int startIndex,
                            int sideLength, CvLSVMFeaturePyramid **maps)
{
    CvLSVMFeatureMap *map;
    IplImage *scaleTmp;
    float scale;
    int   i, err;
    
    for(i = 0; i < numStep; i++)
    {
        scale = 1.0f / powf(step, (float)i);
        scaleTmp = resize_opencv (image, scale);
        err = getFeatureMaps(scaleTmp, sideLength, &map);
        err = normalizeAndTruncate(map, VAL_OF_TRUNCATE);
        err = PCAFeatureMaps(map);
        (*maps)->pyramid[startIndex + i] = map;
        cvReleaseImage(&scaleTmp);
    }/*for(i = 0; i < numStep; i++)*/
    return LATENT_SVM_OK;
}

/*
// Getting feature pyramid  
//
// API
// int getFeaturePyramid(IplImage * image, const filterObject **all_F, 
                      const int n_f,
                      const int lambda, const int k, 
                      const int startX, const int startY, 
                      const int W, const int H, featurePyramid **maps);
// INPUT
// image             - image
// OUTPUT
// maps              - feature maps for all levels
// RESULT
// Error status
*/
int getFeaturePyramid(IplImage * image, CvLSVMFeaturePyramid **maps)
{
    IplImage *imgResize;
    float step;
    int   numStep;
    int   maxNumCells;
    int   W, H;
    
    if(image->depth == IPL_DEPTH_32F)
    {
        imgResize = image;
    }
    else
    {
        imgResize = cvCreateImage(cvSize(image->width , image->height) ,
                                  IPL_DEPTH_32F , 3);
        cvConvert(image, imgResize);        
    }
    
    W = imgResize->width;
    H = imgResize->height;

    step = powf(2.0f, 1.0f / ((float)LAMBDA));
    maxNumCells = W / SIDE_LENGTH;
    if( maxNumCells > H / SIDE_LENGTH )
    {
        maxNumCells = H / SIDE_LENGTH;
    }
    numStep = (int)(logf((float) maxNumCells / (5.0f)) / logf( step )) + 1;
    
    allocFeaturePyramidObject(maps, numStep + LAMBDA);

    getPathOfFeaturePyramid(imgResize, step   , LAMBDA, 0, 
                            SIDE_LENGTH / 2, maps);
    getPathOfFeaturePyramid(imgResize, step, numStep, LAMBDA, 
                            SIDE_LENGTH    , maps);
    
    if(image->depth != IPL_DEPTH_32F)
    {
        cvReleaseImage(&imgResize);
    }

    return LATENT_SVM_OK;
}