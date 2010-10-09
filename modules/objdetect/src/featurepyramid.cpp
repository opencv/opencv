#include "_latentsvm.h"
#include "_resizeimg.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

int sign(float r){
    if(r >  0.0001f) return  1;
    if(r < -0.0001f) return -1;
    return 0;
}

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
int getFeatureMaps_dp(const IplImage * image,const int k, featureMap **map)
{
    int sizeX, sizeY;
    int p, px, strsz;
    int height, width, channels;
    int i, j, kk, c, ii, jj, d;
    float  * datadx, * datady;
    float tmp, x, y, tx, ty;
    IplImage * dx, * dy;
    int *nearest_x, *nearest_y;
    float *w, a_x, b_x;

    float kernel[3] = {-1.f, 0.f, 1.f}; 
    CvMat kernel_dx = cvMat(1, 3, CV_32F, kernel);
    CvMat kernel_dy = cvMat(3, 1, CV_32F, kernel);

    float * r;
    int    * alfa;
    
    float boundary_x[CNTPARTION+1];
    float boundary_y[CNTPARTION+1];
    float max, tmp_scal;
    int    maxi;

	height = image->height;
	width  = image->width ;

    channels  = image->nChannels;

	dx    = cvCreateImage(cvSize(image->width , image->height) , IPL_DEPTH_32F , 3);
    dy    = cvCreateImage(cvSize(image->width , image->height) , IPL_DEPTH_32F , 3);

    sizeX = width  / k;
    sizeY = height / k;
    px    = CNTPARTION  + 2 * CNTPARTION; // контрастное и не контрастное изображение
    p     = px;
    strsz = sizeX * p;
    allocFeatureMapObject(map, sizeX, sizeY, p,  px);

	cvFilter2D(image, dx, &kernel_dx, cvPoint(-1, 0));
	cvFilter2D(image, dy, &kernel_dy, cvPoint(0, -1));
	
    for(i = 0; i <= CNTPARTION; i++)
    {
        boundary_x[i] = cosf((((float)i) * (((float)PI) / (float) (CNTPARTION))));
        boundary_y[i] = sinf((((float)i) * (((float)PI) / (float) (CNTPARTION))));
    }/*for(i = 0; i <= CNTPARTION; i++) */

    r    = (float *)malloc( sizeof(float) * (width * height));
    alfa = (int   *)malloc( sizeof(int  ) * (width * height * 2));

    for(j = 1; j < height-1; j++)
    {
        datadx = (float*)(dx->imageData + dx->widthStep *j);
        datady = (float*)(dy->imageData + dy->widthStep *j);
        for(i = 1; i < width-1; i++)
        {
			c = 0;
            x = (datadx[i*channels+c]);
            y = (datady[i*channels+c]);

            r[j * width + i] =sqrtf(x*x + y*y);
            for(kk = 1; kk < channels; kk++)
            {
                tx = (datadx[i*channels+kk]);
                ty = (datady[i*channels+kk]);
                tmp =sqrtf(tx*tx + ty*ty);
                if(tmp > r[j * width + i])
                {
                    r[j * width + i] = tmp;
                    c = kk;
                    x = tx;
                    y = ty;
                }
            }/*for(kk = 1; kk < channels; kk++)*/

            
            
            max  = boundary_x[0]*x + boundary_y[0]*y;
            maxi = 0;
            for (kk = 0; kk < CNTPARTION; kk++) {
                tmp_scal = boundary_x[kk]*x + boundary_y[kk]*y;
                if (tmp_scal> max) {
                    max = tmp_scal;
                    maxi = kk;
                }else if (-tmp_scal> max) {
                    max = -tmp_scal;
                    maxi = kk + CNTPARTION;
                }
            }
            alfa[j * width * 2 + i * 2    ] = maxi % CNTPARTION;
            alfa[j * width * 2 + i * 2 + 1] = maxi;  
        }/*for(i = 0; i < width; i++)*/
    }/*for(j = 0; j < height; j++)*/

    //подсчет весов и смещений
    nearest_x = (int *)malloc(sizeof(int) * k);
    nearest_y = (int *)malloc(sizeof(int) * k);
    w         = (float*)malloc(sizeof(float) * (k * 2));
    
    for(i = 0; i < k / 2; i++)
    {
        nearest_x[i] = -1;
        nearest_y[i] = -1;
    }/*for(i = 0; i < k / 2; i++)*/
    for(i = k / 2; i < k; i++)
    {
        nearest_x[i] = 1;
        nearest_y[i] = 1;
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
					if ((i * k + ii > 0) && (i * k + ii < height - 1) && (j * k + jj > 0) && (j * k + jj < width - 1))
					{
						d    =  (k*i + ii)* width + (j*k + jj);
						(*map)->Map[(i                ) * strsz + (j                ) * (*map)->p + alfa[d * 2    ]             ] += 
							r[d] * w[ii * 2    ] * w[jj * 2    ];
						(*map)->Map[(i                ) * strsz + (j                ) * (*map)->p + alfa[d * 2 + 1] + CNTPARTION] += 
							r[d] * w[ii * 2    ] * w[jj * 2    ];
						if ((i + nearest_y[ii] >= 0) && (i + nearest_y[ii] <= sizeY - 1))
						{
							(*map)->Map[(i + nearest_y[ii]) * strsz + (j                ) * (*map)->p + alfa[d * 2    ]             ] += 
								r[d] * w[ii * 2 + 1] * w[jj * 2    ];
							(*map)->Map[(i + nearest_y[ii]) * strsz + (j                ) * (*map)->p + alfa[d * 2 + 1] + CNTPARTION] += 
								r[d] * w[ii * 2 + 1] * w[jj * 2    ];
						}
						if ((j + nearest_x[jj] >= 0) && (j + nearest_x[jj] <= sizeX - 1))
						{
							(*map)->Map[(i                ) * strsz + (j + nearest_x[jj]) * (*map)->p + alfa[d * 2    ]             ] += 
								r[d] * w[ii * 2    ] * w[jj * 2 + 1];
							(*map)->Map[(i                ) * strsz + (j + nearest_x[jj]) * (*map)->p + alfa[d * 2 + 1] + CNTPARTION] += 
								r[d] * w[ii * 2    ] * w[jj * 2 + 1];
						}
						if ((i + nearest_y[ii] >= 0) && (i + nearest_y[ii] <= sizeY - 1) && (j + nearest_x[jj] >= 0) && (j + nearest_x[jj] <= sizeX - 1))
						{
							(*map)->Map[(i + nearest_y[ii]) * strsz + (j + nearest_x[jj]) * (*map)->p + alfa[d * 2    ]             ] += 
								r[d] * w[ii * 2 + 1] * w[jj * 2 + 1];
							(*map)->Map[(i + nearest_y[ii]) * strsz + (j + nearest_x[jj]) * (*map)->p + alfa[d * 2 + 1] + CNTPARTION] += 
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
    free(nearest_x);
    free(nearest_y);

    free(r);
    free(alfa);

    return LATENT_SVM_OK;
}

/*
// Feature map Normalization and Truncation 
//
// API
// int normalizationAndTruncationFeatureMaps(featureMap *map, const float alfa);
// INPUT
// map               - feature map
// alfa              - truncation threshold
// OUTPUT
// map               - truncated and normalized feature map
// RESULT
// Error status
*/
int normalizationAndTruncationFeatureMaps(featureMap *map, const float alfa)
{
    int i,j, ii;
    int sizeX, sizeY, p, pos, pp, xp, pos1, pos2;
    float * part_noma; // norm of C(i, j)
    float * new_data;
    float   norm_val;

    sizeX     = map->sizeX;
    sizeY     = map->sizeY;
    part_noma = (float *)malloc (sizeof(float) * (sizeX * sizeY));

    p = map->xp / 3;

    for(i = 0; i < sizeX * sizeY; i++)
    {
        norm_val = 0.0;
        pos = i * map->p;
        for(j = 0; j < p; j++)
        {
            norm_val += map->Map[pos + j] * map->Map[pos + j];
        }/*for(j = 0; j < p; j++)*/
        part_noma[i] = norm_val;
    }/*for(i = 0; i < sizeX * sizeY; i++)*/
	
    xp = map->xp;
    pp = xp * 4;
    sizeX -= 2;
    sizeY -= 2;

    new_data = (float *)malloc (sizeof(float) * (sizeX * sizeY * pp));
//normalization
    for(i = 1; i <= sizeY; i++)
    {
        for(j = 1; j <= sizeX; j++)
        {
            norm_val = sqrtf(
                part_noma[(i    )*(sizeX + 2) + (j    )] +
                part_noma[(i    )*(sizeX + 2) + (j + 1)] +
                part_noma[(i + 1)*(sizeX + 2) + (j    )] +
                part_noma[(i + 1)*(sizeX + 2) + (j + 1)]);
            pos1 = (i  ) * (sizeX + 2) * xp + (j  ) * xp;
            pos2 = (i-1) * (sizeX    ) * pp + (j-1) * pp;
            for(ii = 0; ii < p; ii++)
            {
                new_data[pos2 + ii        ] = map->Map[pos1 + ii    ] / norm_val;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                new_data[pos2 + ii + p * 4] = map->Map[pos1 + ii + p] / norm_val;
            }/*for(ii = 0; ii < 2 * p; ii++)*/
            norm_val = sqrtf(
                part_noma[(i    )*(sizeX + 2) + (j    )] +
                part_noma[(i    )*(sizeX + 2) + (j + 1)] +
                part_noma[(i - 1)*(sizeX + 2) + (j    )] +
                part_noma[(i - 1)*(sizeX + 2) + (j + 1)]);
            for(ii = 0; ii < p; ii++)
            {
                new_data[pos2 + ii + p    ] = map->Map[pos1 + ii    ] / norm_val;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                new_data[pos2 + ii + p * 6] = map->Map[pos1 + ii + p] / norm_val;
            }/*for(ii = 0; ii < 2 * p; ii++)*/
            norm_val = sqrtf(
                part_noma[(i    )*(sizeX + 2) + (j    )] +
                part_noma[(i    )*(sizeX + 2) + (j - 1)] +
                part_noma[(i + 1)*(sizeX + 2) + (j    )] +
                part_noma[(i + 1)*(sizeX + 2) + (j - 1)]);
            for(ii = 0; ii < p; ii++)
            {
                new_data[pos2 + ii + p * 2] = map->Map[pos1 + ii    ] / norm_val;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                new_data[pos2 + ii + p * 8] = map->Map[pos1 + ii + p] / norm_val;
            }/*for(ii = 0; ii < 2 * p; ii++)*/
            norm_val = sqrtf(
                part_noma[(i    )*(sizeX + 2) + (j    )] +
                part_noma[(i    )*(sizeX + 2) + (j - 1)] +
                part_noma[(i - 1)*(sizeX + 2) + (j    )] +
                part_noma[(i - 1)*(sizeX + 2) + (j - 1)]);
            for(ii = 0; ii < p; ii++)
            {
                new_data[pos2 + ii + p * 3 ] = map->Map[pos1 + ii    ] / norm_val;
            }/*for(ii = 0; ii < p; ii++)*/
            for(ii = 0; ii < 2 * p; ii++)
            {
                new_data[pos2 + ii + p * 10] = map->Map[pos1 + ii + p] / norm_val;
            }/*for(ii = 0; ii < 2 * p; ii++)*/
        }/*for(j = 1; j <= sizeX; j++)*/
    }/*for(i = 1; i <= sizeY; i++)*/
//truncation
    for(i = 0; i < sizeX * sizeY * pp; i++)
    {
        if(new_data [i] > alfa) new_data [i] = alfa;
    }/*for(i = 0; i < sizeX * sizeY * pp; i++)*/
//swop data

    map->p  = pp;
    map->xp = xp;
    map->sizeX = sizeX;
    map->sizeY = sizeY;

    free (map->Map);
    free (part_noma);

    map->Map = new_data;

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
int PCAFeatureMaps(featureMap *map)
{ 
    int i,j, ii, jj, k;
    int sizeX, sizeY, p,  pp, xp, yp, pos1, pos2;
    float * new_data;
    float val;
    float nx, ny;
    
    sizeX = map->sizeX;
    sizeY = map->sizeY;
    p     = map->p;
    pp    = map->xp + 4;
    yp    = 4;
    xp    = (map->xp / 3);

    nx    = 1.0f / sqrtf((float)(xp * 2));
    ny    = 1.0f / sqrtf((float)(yp    ));

    new_data = (float *)malloc (sizeof(float) * (sizeX * sizeY * pp));

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
                    val += map->Map[pos1 + yp * xp + ii * xp * 2 + jj];
                }/*for(ii = 0; ii < yp; ii++)*/
                new_data[pos2 + k] = val * ny;
                k++;
            }/*for(jj = 0; jj < xp * 2; jj++)*/
            for(jj = 0; jj < xp; jj++)
            {
                val = 0;
                for(ii = 0; ii < yp; ii++)
                {
                    val += map->Map[pos1 + ii * xp + jj];
                }/*for(ii = 0; ii < yp; ii++)*/
                new_data[pos2 + k] = val * ny;
                k++;
            }/*for(jj = 0; jj < xp; jj++)*/
            for(ii = 0; ii < yp; ii++)
            {
                val = 0;
                for(jj = 0; jj < 2 * xp; jj++)
                {
                    val += map->Map[pos1 + yp * xp + ii * xp * 2 + jj];
                }/*for(jj = 0; jj < xp; jj++)*/
                new_data[pos2 + k] = val * nx;
                k++;
            } /*for(ii = 0; ii < yp; ii++)*/           
        }/*for(j = 0; j < sizeX; j++)*/
    }/*for(i = 0; i < sizeY; i++)*/
//swop data

    map->p  = pp;
    map->xp = pp;

    free (map->Map);

    map->Map = new_data;

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
// lambda            - resize scale
// k                 - size of cells
// startX            - X coordinate of the image rectangle to search
// startY            - Y coordinate of the image rectangle to search
// W                 - width of the image rectangle to search
// H                 - height of the image rectangle to search
// OUTPUT
// maps              - feature maps for all levels
// RESULT
// Error status
*/
int getFeaturePyramid(IplImage * image,
                      const int lambda, const int k, 
                      const int startX, const int startY, 
                      const int W, const int H, featurePyramid **maps)
{
    IplImage *img2, *imgTmp, *imgResize;
    float   step, tmp;
    int      cntStep;
    int      maxcall;
    int i;
    int err;
    featureMap *map;
    
    //geting subimage
    cvSetImageROI(image, cvRect(startX, startY, W, H));
    img2 = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
    cvCopy(image, img2, NULL);
    cvResetImageROI(image);

    if(img2->depth != IPL_DEPTH_32F)
    {
        imgResize = cvCreateImage(cvSize(img2->width , img2->height) , IPL_DEPTH_32F , 3);
        cvConvert(img2, imgResize);
    }
    else
    {
        imgResize = img2;
    }
    
    step = powf(2.0f, 1.0f/ ((float)lambda));
    maxcall = W/k;
    if( maxcall > H/k )
    {
        maxcall = H/k;
    }
    cntStep = (int)(logf((float)maxcall/(5.0f))/logf(step)) + 1;
    //printf("Count step: %f %d\n", step, cntStep);

    allocFeaturePyramidObject(maps, lambda, cntStep + lambda);

    for(i = 0; i < lambda; i++)
    {
        tmp = 1.0f / powf(step, (float)i);
        imgTmp = resize_opencv (imgResize, tmp);
        //imgTmp = resize_article_dp(img2, tmp, 4);
        err = getFeatureMaps_dp(imgTmp, 4, &map);
        err = normalizationAndTruncationFeatureMaps(map, 0.2f);
        err = PCAFeatureMaps(map);
        (*maps)->pyramid[i] = map;
        //printf("%d, %d\n", map->sizeY, map->sizeX);
        cvReleaseImage(&imgTmp);
    }

    /**********************************one**************/
    for(i = 0; i <  cntStep; i++)
    {
        tmp = 1.0f / powf(step, (float)i);
        imgTmp = resize_opencv (imgResize, tmp);
        //imgTmp = resize_article_dp(imgResize, tmp, 8);
	    err = getFeatureMaps_dp(imgTmp, 8, &map);
        err = normalizationAndTruncationFeatureMaps(map, 0.2f);
        err = PCAFeatureMaps(map);
        (*maps)->pyramid[i + lambda] = map;
        //printf("%d, %d\n", map->sizeY, map->sizeX);
		cvReleaseImage(&imgTmp);
    }/*for(i = 0; i < cntStep; i++)*/

    if(img2->depth != IPL_DEPTH_32F)
    {
        cvReleaseImage(&imgResize);
    }

    cvReleaseImage(&img2);
    return LATENT_SVM_OK;
}

/*
// add zero border to feature map
//
// API
// int addBordersToFeatureMaps(featureMap *map, const int bX, const int bY);
// INPUT
// map               - feature map
// bX                - border size in x
// bY                - border size in y
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int addBordersToFeatureMaps(featureMap *map, const int bX, const int bY){
    int i,j, jj;
    int sizeX, sizeY, p, pos1, pos2;
    float * new_data;
    
    sizeX = map->sizeX;
    sizeY = map->sizeY;
    p     = map->p;

    new_data = (float *)malloc (sizeof(float) * ((sizeX + 2 * bX) * (sizeY + 2 * bY) * p));

    for(i = 0; i < ((sizeX + 2 * bX) * (sizeY + 2 * bY) * p); i++)
    {
        new_data[i] = (float)0;
    }/*for(i = 0; i < ((sizeX + 2 * bX) * (sizeY + 2 * bY) * p); i++)*/

    for(i = 0; i < sizeY; i++)
    {
        for(j = 0; j < sizeX; j++)
        {

            pos1 = ((i     )*sizeX            + (j     )) * p;
            pos2 = ((i + bY)*(sizeX + 2 * bX) + (j + bX)) * p;
            
            for(jj = 0; jj < p; jj++)
            {
                new_data[pos2 + jj] = map->Map[pos1 + jj];
            }/*for(jj = 0; jj < p; jj++)*/
        }/*for(j = 0; j < sizeX; j++)*/
    }/*for(i = 0; i < sizeY; i++)*/
    //swop data

    map->sizeX = sizeX + 2 * bX;
    map->sizeY = sizeY + 2 * bY;
    
    free (map->Map);

    map->Map = new_data;

    return LATENT_SVM_OK;
}