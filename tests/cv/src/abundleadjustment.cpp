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

#include "cvtest.h"

#if 0

#if defined WIN32 || defined _WIN32
#include "direct.h"
#else
#include <sys/stat.h>
#endif

using namespace cv;

//global variable
#ifndef MAX_PATH
#define MAX_PATH 1024
#endif

char g_filepath[MAX_PATH];

//objects for virtual environmnt

/*****************************Camera class ************************************************/
class Camera
{
public:
    Camera(void);
    virtual ~Camera(void);

    inline void SetPosition(double x, double y, double z)
    {
        position[0] = x;
        position[1] = y;
        position[2] = z;
    }    

    inline double* GetPosition() { return position; }

    inline void SetDistortion(double r1, double r2, double t1, double t2)
    {
        distortion[0] = r1;
        distortion[1] = r2;
        distortion[2] = t1;
        distortion[3] = t2;             
    }      
    inline void SetIntrinsics( double fx, double fy, double cx, double cy ) //fx,fy,cx,cy
    {
        intrinsic[0] =    fx;
        intrinsic[1] =    0;
        intrinsic[2] =    cx;
        intrinsic[3] =    0;
        intrinsic[4] =    fy;
        intrinsic[5] =    cy;
        intrinsic[6] =    0;
        intrinsic[7] =    0;
        intrinsic[8] =    1.0;  
    }

    CvPoint3D64f ConvertPoint2WCS( CvPoint3D64f pt )
    {
        double tmp[3];
        CvMat tmp_point = cvMat( 3, 1, CV_64FC1, tmp );
                
        CvMat rot = cvMat( 3, 3, CV_64FC1, rotation );
        CvMat trans = cvMat( 3, 1, CV_64FC1, translation );
        CvMat in_point = cvMat( 3, 1, CV_64FC1, &pt );

        // out = inv(R) * (in - t)

        cvSub(&in_point, &trans, &tmp_point);
        cvGEMM( &rot, &tmp_point, 1, NULL, 0, &tmp_point, CV_GEMM_A_T /* use transposed rotmat*/  );
        
        CvPoint3D64f out;
        out.x = tmp[0];
        out.y = tmp[1];
        out.z = tmp[2]; 

        return out;
    }

    inline void SetRotation( double* rotmat ) //fx,fy,cx,cy
    {
        memcpy( rotation, rotmat, 9 * sizeof(double) );  
    }        

    double* GetRotation()
    {
        return rotation;
    }

    void ComputeTranslation()
    {
        //convert camera position in WCS into translation vector of the camera 
        //t = -R*pos
        translation[0] = -(rotation[0]*position[0]+rotation[1]*position[1]+rotation[2]*position[2]);
        translation[1] = -(rotation[3]*position[0]+rotation[4]*position[1]+rotation[5]*position[2]);
        translation[2] = -(rotation[6]*position[0]+rotation[7]*position[1]+rotation[8]*position[2]);  
    }

    double* GetTranslation()
    {   
        return translation;
    }
    double* GetIntrinsics()
    {
        return intrinsic;
    }               

    double* GetDistortion()
    {
        return distortion;
    }

    void SetResolution(CvSize res)
    {
        resolution = res;
    }
    CvSize GetResolution()
    {
        return resolution;
    } 
    void saveCamParams(FILE* stream);
    void readCamParams(FILE* stream);
                

protected:
    double distortion[4]; //distortion coeffs according to OpenCV (2 radial and 2 tangential)
    double intrinsic[9];     //matrix of intrinsic parameters
    double translation[3]; //camera's translation vector 
    double rotation[9]; //camera rotation matrix (probably need to convert to camera axis vector
    double position[3]; //camera's position in WCS  

    CvSize resolution;   
};

Camera::Camera(void)
{
    //default parameters
    translation[0] = translation[1] = translation[2] = 0.0;
    rotation[0] = rotation[1] = rotation[2] =
    rotation[3] = rotation[4] = rotation[5] =
    rotation[6] = rotation[7] = rotation[8] = 0.0;
    rotation[0] = rotation[4] = rotation[8] =  1.0;

    distortion[0] = distortion[1] = distortion[2] = distortion[3] = 0.0;
}

Camera::~Camera(void)
{
}

void Camera::saveCamParams(FILE * stream)
{   
    float tmp0 = 0.0f, tmp1 = 1.0f;
    
    // printing camera distortion. 4 parameters
    fprintf(stream, "%.12f %.12f %.12f %.12f\n", distortion[0], distortion[1],
                                                 distortion[2], distortion[3]);
    
    //printing camera intrinsics matrix
    fprintf(stream, "%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n", 
        intrinsic[0], intrinsic[1], intrinsic[2],
        intrinsic[3], intrinsic[4], intrinsic[5],
        intrinsic[6], intrinsic[7], intrinsic[8]);
        
    //printing camera extrinsic transform
    fprintf(stream, "%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n",
        rotation[0], rotation[1], rotation[2], translation[0],
        rotation[3], rotation[4], rotation[5], translation[1],
        rotation[6], rotation[7], rotation[8], translation[2],
        tmp0, tmp0, tmp0, tmp1);           
}

void Camera::readCamParams(FILE* stream)
{   
    double dummy;
    
    // read camera distortion. 4 parameters
    fscanf(stream, "%lf %lf %lf %lf\n", &(distortion[0]), &(distortion[1]), &(distortion[2]), &(distortion[3]) );
    
    //read camera intrinsics matrix
    fscanf(stream, "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n", 
        &(intrinsic[0]), &(intrinsic[1]), &(intrinsic[2]),
        &(intrinsic[3]), &(intrinsic[4]), &(intrinsic[5]),
        &(intrinsic[6]), &(intrinsic[7]), &(intrinsic[8]));
        
    //read camera extrinsic transform
    fscanf(stream, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
        &(rotation[0]), &(rotation[1]), &(rotation[2]), &(translation[0]),
        &(rotation[3]), &(rotation[4]), &(rotation[5]), &(translation[1]),
        &(rotation[6]), &(rotation[7]), &(rotation[8]), &(translation[2]),
        &dummy, &dummy, &dummy, &dummy);   
}

/******************************** body**************************************************/

class RigidBody
{
    //global position of body's center
    double pos[3];
    double rotation[9]; //body's rotation matrix 
    
    int* m_ids; //points ids, these are integer numbers not necessarily zero-based indices
    CvPoint3D64f* m_points3D;  //these are points in object coordinate system
    CvPoint3D64f* m_points3D_WCS;  //these are points in WCS
    int m_numPoints;

public:

    RigidBody()
    {
        m_ids = 0;
        m_points3D = 0;
        m_points3D_WCS = 0;
        m_numPoints = 0;

        SetPosition(0,0,0);
        double t[9] = {1,0,0,0,1,0,0,0,1};
        SetRotation( t );
    }
    virtual ~RigidBody() {}

    virtual void SetPosition(double x, double y, double z )
    {
        pos[0] = x;
        pos[1] = y;
        pos[2] = z;
    }
    
    virtual void SetRotation( double* rotmat ) 
    {
        memcpy( rotation, rotmat, 9 * sizeof(double) );     
    }    

    void Save(char* fname, bool wcs /* 1 - save coordinates in wcs, 0 - save coordinates in object coordinate system*/  )
    {
        FILE* fp = fopen(fname, "w");
        fprintf(fp, "%d\n", m_numPoints );
        if( wcs )
        {
            for( int i = 0; i < m_numPoints; i++ )
            {   
                fprintf(fp, "%d %.12f %.12f %.12f\n", m_ids[i], m_points3D_WCS[i].x, m_points3D_WCS[i].y, m_points3D_WCS[i].z ); 
            }
        }
        else
        {
            for( int i = 0; i < m_numPoints; i++ )
            {   
                fprintf(fp, "%d, %.12f, %.12f, %.12f\n", m_ids[i], m_points3D[i].x, m_points3D[i].y, m_points3D[i].z ); 
            }
        }   
        fclose(fp);
    }

    void Load(char* fname)
    {   
        clear_points();
        FILE* fp = fopen(fname, "r");
        fscanf(fp, "%d\n", &m_numPoints );
        //allocate arrays
        m_points3D = new CvPoint3D64f[m_numPoints];
        m_points3D_WCS = new CvPoint3D64f[m_numPoints];
        m_ids = new int[m_numPoints];
        
        for(int i = 0; i < m_numPoints; i++ )
        {
            fscanf(fp, "%d %lf %lf %lf\n", &(m_ids[i]), &(m_points3D[i].x), &(m_points3D[i].y), &(m_points3D[i].z));
        } 
    }

    void clear_points()
    {   
        if(m_points3D)
            delete m_points3D;
        m_points3D = NULL;
        if(m_points3D_WCS)
            delete m_points3D_WCS;
        m_points3D_WCS = NULL;
        if( m_ids )
            delete m_ids;
        m_ids = NULL;
        m_numPoints = 0;
    }

    void generate_random( int n, double x_min, double x_max, 
                                 double y_min, double y_max,
                                 double z_min, double z_max )
    {
        clear_points();
        m_numPoints = n;
        m_points3D = new CvPoint3D64f[m_numPoints];
        m_points3D_WCS = new CvPoint3D64f[m_numPoints];
        m_ids = new int[m_numPoints];

        //fill ids
        for( int i = 0 ; i < n; i++ )
        {
            m_ids[i] = i;
        }    

        CvRNG rng;

        CvMat values = cvMat( m_numPoints, 1, CV_64FC3, m_points3D  );
        
        cvRandArr( &rng, &values, CV_RAND_UNI,
                   cvScalar(x_min, y_min, z_min), // min 
                   cvScalar(x_max, y_max, z_max) // deviation
                 ); 
    } 
    CvPoint3D64f* GetPoints3D(bool wcs)
    {    
        if( wcs )
        {
            //fill points in WCS accordingly to rotation matrix and board position
            for(int i = 0; i < NumPoints(); i++ )
            {                
                m_points3D_WCS[i].x = rotation[0]*m_points3D[i].x + 
                                      rotation[1]*m_points3D[i].y + 
                                      rotation[2]*m_points3D[i].z + pos[0];

                m_points3D_WCS[i].y = rotation[3]*m_points3D[i].x + 
                                  rotation[4]*m_points3D[i].y + 
                                  rotation[5]*m_points3D[i].z + pos[1];

                m_points3D_WCS[i].z = rotation[6]*m_points3D[i].x + 
                                  rotation[7]*m_points3D[i].y + 
                                  rotation[8]*m_points3D[i].z + pos[2];
            }  
            //return points in global cooordinates
            return m_points3D_WCS;

        }
        else
            return m_points3D;
    } 
        
    double* GetRotation() { return rotation; } 

    int NumPoints()
    {
        return m_numPoints;
    }

    int* GetPointsIds()
    {
        return this->m_ids;
    }

};


/********************************* environment *****************************************/
#define PT_INVISIBLE 0
#define PT_VISIBLE 1
#define PT_OUTBORDER 2

class Environment
{
public:
    Environment(void);
    virtual ~Environment(void);

    inline int NumCameras()
    {
        return (int)m_cameras.size();
    }

    inline int AddCamera(Camera* cam)
    {
        m_cameras.push_back(cam);
        return (int)m_cameras.size();
    }    
    
    RigidBody* GetBody()
    {
        return m_body;
    }

    void SetNoise(float n) 
    { 
        noise = n; 
    }

    int Capture();
    int Save(char* filename);
    
protected:

    std::vector<Camera*> m_cameras;
    RigidBody* m_body;
    std::vector<CvPoint2D64f*> m_image_points;
    std::vector<int*> m_point_visibility;
    float noise;

};

Environment::Environment(void)
{       
    m_body = new RigidBody();
}

Environment::~Environment(void)
{
    delete m_body;
}

int Environment::Capture()
{
    //clear points
    for( size_t i = 0; i < m_image_points.size(); i++ )
    {
        if( m_image_points[i]  )
            delete m_image_points[i]; 
    }
    m_image_points.clear();
    
    for( size_t i = 0; i < m_point_visibility.size(); i++ )
    {
        if( m_point_visibility[i]  )
            delete m_point_visibility[i]; 
    }
    m_point_visibility.clear();            
    
    CvPoint3D64f* board_pts = m_body->GetPoints3D(true);
    int num_points = m_body->NumPoints();        

    //loop over cameras 
    for( size_t i = 0; i < m_cameras.size(); i++ )
    {
        //get camera parameters       
        //project points onto camera image 
        
        double* rot = m_cameras[i]->GetRotation();
        double* trans = m_cameras[i]->GetTranslation();
        double* intr = m_cameras[i]->GetIntrinsics();
        double* dist = m_cameras[i]->GetDistortion();

        CvPoint2D64f* image_points = new CvPoint2D64f[num_points];
        m_image_points.push_back(image_points);

        int* points_visibility = new int[num_points];
        m_point_visibility.push_back(points_visibility);
                        
        cvProjectPointsSimple(num_points, board_pts, rot, trans, intr, dist, image_points);    

        CvRNG rng;

        if( noise > 0)
        {
            CvMat* values = cvCreateMat( num_points, 1, CV_32FC2 );

            float stdev = noise;         

            cvRandArr( &rng, values, CV_RAND_NORMAL,
                       cvScalar(0.0, 0.0), // mean
                       cvScalar(stdev, stdev) // deviation
                       );      

            //add gaussian noise to image points
            
            for( int j = 0; j < num_points; j++ )
            {
                CvPoint2D32f pt = *(CvPoint2D32f*)cvPtr1D( values, j, 0 );

                pt.x = min( pt.x, stdev);
                pt.x = max( pt.x, -stdev);
                
                pt.y = min( pt.y, stdev);
                pt.y = max( pt.y, -stdev);   
                                
                image_points[j].x += pt.x;
                image_points[j].y += pt.y;
            }                
            cvReleaseMat( &values );            
        }

        //decide if point visible to camera
        //loop over points and assign visibility flag to them
        for( int j = 0; j < num_points; j++ )
        {
            //generate random visibility of the point
            int visible = cvRandInt(&rng) % 2; //visibility 50% 
            
            //check the point is in camera FOV    (1-pixel border assumed invisible)
            if( image_points[j].x > 0 && image_points[j].x < m_cameras[i]->GetResolution().width-1 
                && image_points[j].y > 0 && image_points[j].y < m_cameras[i]->GetResolution().height-1 )
            {
                if(!visible)
                    points_visibility[j] = PT_INVISIBLE;
                else
                    points_visibility[j] = PT_VISIBLE;
            }
            else
                points_visibility[j] = PT_OUTBORDER;             
        }  
    }  

    //some points may become completely invisible for all cameras or visible only at one camera
    //we will forcely make them visible for at least 2 cameras
    for( int i = 0 ; i < num_points; i++ )
    {
        int numvis = 0;
        for( size_t j = 0; j < m_cameras.size(); j++ )
        {
            if( m_point_visibility[j][i] == PT_VISIBLE )
                numvis++;
        }

        if(numvis < 2)
        {
            for( size_t j = 0; j < m_cameras.size(); j++ )
            {
                if( m_point_visibility[j][i] == PT_INVISIBLE )
                {
                    m_point_visibility[j][i] = PT_VISIBLE;
                    numvis++;
                }
                if(numvis > 1)
                    break;
            }  
            assert(numvis > 1 );
        }    
    }

    return 0;
}

/*void Environment::TestCamera(int camind)
{
    CvPoint3D64f* board_pts = m_body->GetPoints3D(false);

    int count = 0;
    for( int j = 0; j < m_body->NumPoints(); j++ )
    {
        if( m_point_visibility[camind][j] == PT_VISIBLE )
            count++;
    }

    CvMat* object_points = cvCreateMat( 1, count, CV_64FC3 );
    CvMat* image_points  = cvCreateMat( 1, count, CV_64FC2 );
    CvMat* intrinsic_matrix  = cvCreateMatHeader( 3, 3, CV_64FC1 );
       cvSetData(intrinsic_matrix, m_cameras[camind]->GetIntrinsics(), 3*sizeof(double) );

    CvMat* distortion_coeffs  = cvCreateMat( 4, 1, CV_64FC1 );   cvSetZero(distortion_coeffs);
    CvMat* rotation_vector = cvCreateMat( 3, 1, CV_64FC1 );
    CvMat* translation_vector = cvCreateMat( 3, 1, CV_64FC1 );

    CvMat* rotation_matrix = cvCreateMat( 3, 3, CV_64FC1 );

    //loop over points and assign visibility flag to them
    int ind = 0;
    for( int j = 0; j < m_body->NumPoints(); j++ )
    {
        if( m_point_visibility[camind][j] == PT_VISIBLE)
        {
            //add to matrix
            *((CvPoint3D64f*)(object_points->data.db + ind*3)) = board_pts[j];
            *((CvPoint2D64f*)(image_points->data.db + ind*2)) = m_image_points[camind][j];
            ind++;
        }             
    }  
    

    //find extrinsic parameters of the board 
    cvFindExtrinsicCameraParams2( object_points,
                                  image_points,
                                  intrinsic_matrix,
                                  distortion_coeffs,
                                  rotation_vector,
                                  translation_vector );

    //cvRodrigues2( rotation_vector, rotation_matrix );

    
    //reproject points and see error of reprojection

    CvMat* image_points_repro = cvCloneMat( image_points );  

    cvProjectPoints2( object_points, rotation_vector, translation_vector, intrinsic_matrix, distortion_coeffs,
                          image_points_repro );

    for( int j = 0; j < image_points_repro->cols; j++ )
    {
        CvPoint2D64f pt_orig = *((CvPoint2D64f*)(image_points->data.db + j*2));
        CvPoint2D64f pt_repro = *((CvPoint2D64f*)(image_points_repro->data.db + j*2));

        CvPoint2D64f diff;
        diff.x = pt_repro.x - pt_orig.x;
        diff.y = pt_repro.y - pt_orig.y;    

    }   
}     */

int Environment::Save(char* filename)
{               
    int* ind = m_body->GetPointsIds();
    FILE* saveFile = fopen( filename, "w" );
    if(saveFile) 
    {
        for( size_t cam_index = 0; cam_index < m_cameras.size(); cam_index++ )
        {     
            CvPoint2D64f* image_points = m_image_points[cam_index];

            int numPnt = 0; //no visible points by default

            //check points visibility             
            //camera is visible, check individual points 
            for(int i = 0; i < m_body->NumPoints(); i++ )
            {
                if( m_point_visibility[cam_index][i] == PT_VISIBLE)
                {
                    //point is visible 
                    numPnt++;
                }                         
            }    
            
            if(numPnt)    //some points are visible
            {
                for( int i = 0; i < m_body->NumPoints(); i++) 
                {
                    if( m_point_visibility[cam_index][i] == PT_VISIBLE )
                    {
                        //point is visible 
                        fprintf(saveFile, "%d %d %d %.12f %.12f\n", 0 /*snapshot_id*/, (int)cam_index, ind[i], image_points[i].x, image_points[i].y ); 
                    }
                }
            }

        }
    }

    //close file 
    fclose(saveFile);
    return 0;
}



  
//input parameters for envirinment generation
struct Params
{
    int width; //width of images
    int height; //height of images
    float FOVx; //camera field of view in horizontal direction. vertical FOV is detected from aspect ratio
    float noise; //corners projection noise
    float k1, k2, k3; //radial distortion coeffs
    float p1,p2; //tangential distortion coeffs
};

int GenerateTestData2(Params& params)
{
    //create environment
    Environment* env = new Environment();
    env->SetNoise(params.noise);

    CvSize im_res = cvSize(params.width,params.height);

    double FOVx = params.FOVx; // field of view relatively to width  (in degrees)
    double fov_rad = FOVx * CV_PI / 180;
    double fx = im_res.width/2.0 / tan( fov_rad/2);
    double fy = fx;
    double cx = im_res.width/2.0-0.5;
    double cy = im_res.height/2.0-0.5;

    //model cube room of size 2x2x2 meters
    //8 cameras on the perimeter at the height 1 meter in corners and in the middle of walls
    // all they look to the center of the room
    //coordinate center in the center of the cube
    //Z coordinate is oriented up
        
    Camera* cam[8];
    for( int i = 0; i < 8; i++ )
    {
        cam[i] = new Camera();
        cam[i]->SetDistortion(params.k1, params.k2,0,0);  //only radial distortion
        cam[i]->SetIntrinsics(fx,fy, cx, cy ); //fx,fy,cx,cy
        cam[i]->SetResolution(im_res);  
        env->AddCamera(cam[i]);
    }

    //set positions
    cam[0]->SetPosition( 1.,1.,0);
    cam[1]->SetPosition( 0.,1.,0);
    cam[2]->SetPosition( -1.,1.,0);
    cam[3]->SetPosition( -1.,0.,0);
    cam[4]->SetPosition( -1.,-1.,0);
    cam[5]->SetPosition( 0.,-1.,0);
    cam[6]->SetPosition( 1.,-1.,0);
    cam[7]->SetPosition( 1.,0.,0);

    //set rotation matrices
    //they will be oriented strongly vertically and rotated only around vertical axis (Z coorinate in WCS)
        
    CvMat* camrot = cvCreateMat( 3, 3, CV_64FC1);

    for( int i = 0; i < 8; i++ )
    {
        //y of the camera is oriented along negative Z of WCS
        double yDirCamera[3] = {0,0,-1};
        double zDirCamera[3]; //oriented from camera center to WCS center
        double* pos = cam[i]->GetPosition();
        zDirCamera[0] = -pos[0];
        zDirCamera[1] = -pos[1];
        zDirCamera[2] = -pos[2]; 

        double xDirCamera[3]; //cross product Y*Z
        xDirCamera[0] = yDirCamera[1]*zDirCamera[2] - yDirCamera[2]*zDirCamera[1];
        xDirCamera[1] = yDirCamera[2]*zDirCamera[0] - yDirCamera[0]*zDirCamera[2];
        xDirCamera[2] = yDirCamera[0]*zDirCamera[1] - yDirCamera[1]*zDirCamera[0];

        //normalize z and x
        double inv_norm = 1.0/sqrt(xDirCamera[0]*xDirCamera[0] + xDirCamera[1]*xDirCamera[1] + xDirCamera[2]*xDirCamera[2]);
        xDirCamera[0]*=inv_norm;
        xDirCamera[1]*=inv_norm;
        xDirCamera[2]*=inv_norm;

        inv_norm = 1.0 / sqrt(zDirCamera[0]*zDirCamera[0] + zDirCamera[1]*zDirCamera[1] + zDirCamera[2]*zDirCamera[2]);
        zDirCamera[0] *= inv_norm;
        zDirCamera[1] *= inv_norm;
        zDirCamera[2] *= inv_norm; 

        camrot->data.db[0] = xDirCamera[0];
        camrot->data.db[3] = xDirCamera[1];
        camrot->data.db[6] = xDirCamera[2];

        camrot->data.db[1] = yDirCamera[0];
        camrot->data.db[4] = yDirCamera[1];
        camrot->data.db[7] = yDirCamera[2];
        
        camrot->data.db[2] = zDirCamera[0];
        camrot->data.db[5] = zDirCamera[1];
        camrot->data.db[8] = zDirCamera[2];    
        
        //get inverse matrix (equal to transposed)
        cvTranspose(camrot, camrot);

        cam[i]->SetRotation(camrot->data.db);
        cam[i]->ComputeTranslation();// compute translation after we set position and rotation matrix    
    }
 
#if defined WIN32 || defined _WIN32
    _mkdir(g_filepath);
#else
    mkdir(g_filepath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
    char fname[2048];  
    sprintf(fname, "%scameras.txt", g_filepath );
    FILE* clist = fopen(fname, "w");
    for(unsigned int i = 0; i < 8; i++)
    {            
        sprintf(fname, "%scamera%d.calib", g_filepath, i );
        FILE *calibFile = fopen(fname,"w");
        cam[i]->saveCamParams(calibFile);
        fclose( calibFile ); 
        fprintf(clist, "%s\n", fname);
    }                                 
    fclose(clist); 

    std::vector<CvPoint3D64f> board_pos;
    board_pos.clear();

    //generate body position, set zero for now
    CvPoint3D64f pos;
    pos.x = 0; pos.y = 0; pos.z = 0;                 
    board_pos.push_back(pos);
        
    RigidBody* body = env->GetBody();
    body->generate_random(50, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25 );
    
    CvRNG rng = cvRNG();
    int index = 0; //index of board's position to save, is used to enumerate snapshots
    
    for(size_t i = 0; i < board_pos.size(); i++ )
    {
        CvPoint3D64f point = board_pos[i];
        body->SetPosition(point.x,point.y,point.z);

        //generate random orientation of the board 
        unsigned int rn = cvRandInt(&rng);
        //scale to 90 degrees
        rn = rn%90;
        double slant = (double)rn-45-90; //angle of slant of the bord relatively to its horizontal direction
        
        rn = cvRandInt(&rng);
        //scale to 360 degrees
        rn = rn%360;
        double azimuth = rn; //azimuth


        //set slant and azimuth to 0
        slant = azimuth = 0;

        double sn = sin(azimuth*CV_PI/180);
        double cs = cos(azimuth*CV_PI/180);  

        //generate rotation matrix
        CvMat* mat_azimuth = cvCreateMat( 3, 3, CV_64FC1);
        mat_azimuth->data.db[0] = cs;
        mat_azimuth->data.db[1] = -sn;
        mat_azimuth->data.db[2] = 0;

        mat_azimuth->data.db[3] = sn;
        mat_azimuth->data.db[4] = cs;
        mat_azimuth->data.db[5] = 0;
        
        mat_azimuth->data.db[6] = 0;
        mat_azimuth->data.db[7] = 0;
        mat_azimuth->data.db[8] = 1;

        sn = sin(slant*CV_PI/180);
        cs = cos(slant*CV_PI/180);  
        CvMat* mat_slant = cvCreateMat( 3, 3, CV_64FC1);  //rotation around X axis
        mat_slant->data.db[0] = 1;
        mat_slant->data.db[1] = 0;
        mat_slant->data.db[2] = 0;

        mat_slant->data.db[3] = 0;
        mat_slant->data.db[4] = cs;
        mat_slant->data.db[5] = -sn;
        
        mat_slant->data.db[6] = 0;
        mat_slant->data.db[7] = sn;
        mat_slant->data.db[8] = cs;

        CvMat* rot = cvCreateMat( 3, 3, CV_64FC1);  //rotation around X axis
        //create complete rotation matrix
        cvMatMul(mat_azimuth, mat_slant, rot);    

        //these are coordinates of board's axis in WCS
        body->SetRotation(rot->data.db);
        
        //project images onto all cameras
        env->Capture();

        //save 3d points of corners into the file
        char fname[2048];
        sprintf(fname, "%sPoints%d.3d", g_filepath, index );
        body->Save(fname, true);
        //save shots from camera

        sprintf(fname, "%salldata.txt", g_filepath );
        env->Save(fname);
    }      
    //destroy everything
    delete env;   
    return 0;
}    

int TestLevmar2()
{
    //load data from file

    // 1. load ground truth 3D points
    RigidBody body;

    char fname[2048];
    sprintf(fname, "%sPoints0.3d", g_filepath );
    body.Load(fname);
    int num_points = body.NumPoints(); 

    // 2. load cameras
    int num_cams = 0;
    //int num_cam_param = 10;
    std::vector<Camera*> cams;

    sprintf(fname, "%scameras.txt", g_filepath );    
    FILE* fp = fopen(fname, "r" );
    while( !feof(fp) )
    {
        char fname[MAX_PATH];
        fscanf(fp, "%s\n", fname );
        Camera* newcam = new Camera();
        FILE* fp2 = fopen( fname, "r" );
        newcam->readCamParams(fp2);
        cams.push_back(newcam);
        fclose(fp2);
        num_cams++;
    }
    fclose(fp);   

    // 2. Load projection data  
    CvMat* m = cvCreateMat( body.NumPoints(), num_cams, CV_64FC2 );
    //all invisible point will have (-1,-1) projection
    cvSet( m, cvScalar(-1,-1) );

    sprintf(fname, "%salldata.txt", g_filepath );    
    fp = fopen( fname, "r");
    
    int counter = 0;
    while( !feof(fp) )
    {
        //read line
        int snapid, cameraid, pointid;
        double x,y;
        fscanf(fp, "%d %d %d %lf %lf\n", &snapid, &cameraid, &pointid, &x, &y);
        cvSet2D(m, pointid, cameraid, cvScalar(x,y) );         
        counter++;
    }

    vector<vector<int> > visibility;
    
    //transform this matrix to measurement vector
    vector<vector<Point2d> > imagePoints;
    
    for(int j = 0; j < num_cams; j++ )
    {
        vector<Point2d> vec;
        vector<int> visvec;
        for(int i = 0; i < num_points; i++ )
        {          
            CvScalar s = cvGet2D( m, i, j ); 
            if( s.val[0] != -1 ) //point is visible
            {
                vec.push_back(Point2d(s.val[0],s.val[1]));
                visvec.push_back(1);                
            } 
            else
            {
                vec.push_back(Point2d(DBL_MAX,DBL_MAX));            
                visvec.push_back(0);      
            }
        } 
        imagePoints.push_back(vec);
        visibility.push_back(visvec);
    }
        
    //form initial params
    vector<Mat> cameraMatrix; //intrinsic matrices of all cameras (input and output)
    vector<Mat> R; //rotation matrices of all cameras (input and output)
    vector<Mat> T; //translation vector of all cameras (input and output)
    vector<Mat> distCoeffs; //distortion coefficients of all cameras (input and output)
    
    //store original camera positions
    vector<Mat> T_backup;
    //store original camera rotations
    vector<Mat> R_backup;
    //store original intrinsic matrix
    vector<Mat> cameraMatrix_backup;
    //store original distortion
    vector<Mat> distCoeffs_backup;
    
    for( int i = 0; i < (int)cams.size(); i++ )
    {
        //fill params
        Camera* c = cams[i]; 
        //rotation
        double* rotmat = c->GetRotation();
        Mat rm(3,3,CV_64F,rotmat);
        R_backup.push_back(rm);
        //generate small random rotation
        Mat rodr; Rodrigues(rm, rodr);
        double n = norm(rodr);
        //add small angle 
        //add about 5 degrees to rotation 
        rodr *= ((n+0.1)/n);
        Rodrigues(rodr, rm);        
        R.push_back(rm); 
        
        //translation
        double* tr = c->GetTranslation();
        Mat tv(Size(1,3), CV_64F, tr);
        T_backup.push_back(tv);
        //add random translation within 1 cm
        Mat t_(3,1,CV_64F);
        randu(t_, -0.01, 0.01);
        tv+=t_;

        T.push_back(tv);
        //intrinsic matrix
        double* intr = c->GetIntrinsics();
        cameraMatrix_backup.push_back(Mat(Size(3,3), CV_64F, intr));
        cameraMatrix.push_back(Mat(Size(3,3), CV_64F, intr));  

        //distortion         
        Mat d(4, 1, CV_64F, c->GetDistortion() );
        distCoeffs_backup.push_back(d); 

        //variate distortion by 5%
        d*=1.05;
        distCoeffs.push_back(d);
        
    } 
      
    //form input points
    const Point3d* ptptr = (const Point3d*)body.GetPoints3D(true);
    vector<Point3d> points(ptptr, ptptr + num_points);
    //store points
    vector<Point3d> points_backup(num_points);
    std::copy(points.begin(), points.end(), points_backup.begin());
    
    //variate initial points
    CvRNG rng;
    CvMat* values = cvCreateMat( num_points*3, 1, CV_64F );          
    cvRandArr( &rng, values, CV_RAND_NORMAL,
               cvScalar(0.0), // mean
               cvScalar(0.02) // deviation (in meters)
               );  
    CvMat tmp = cvMat(values->rows, values->cols, values->type, &points[0] );
    cvAdd( &tmp, values, &tmp );      
    cvReleaseMat( &values );   
        
    LevMarqSparse::bundleAdjust( points, //positions of points in global coordinate system (input and output)
                  imagePoints, //projections of 3d points for every camera 
                  visibility, //visibility of 3d points for every camera 
                  cameraMatrix, //intrinsic matrices of all cameras (input and output)
                  R, //rotation matrices of all cameras (input and output)
                  T, //translation vector of all cameras (input and output)
                  distCoeffs, //distortion coefficients of all cameras (input and output)
                  TermCriteria(TermCriteria::COUNT|TermCriteria::EPS, 3000, DBL_EPSILON ));
                  //,enum{MOTION_AND_STRUCTURE,MOTION,STRUCTURE})

    //compare input points and found points
    double maxdist = 0;
    for( size_t i = 0; i < points.size(); i++ )
    {
        Point3d in = points_backup[i];
        Point3d out = points[i];
        double dist = sqrt(in.dot(out)); 
        if(dist > maxdist)
            maxdist = dist;
    }
    printf("Maximal distance between points: %.12lf\n", maxdist );    

    //compare camera positions
    maxdist = 0;
    for( size_t i = 0; i < T.size(); i++ )
    {
        double dist = norm(T[i], T_backup[i]);
         
        if(dist > maxdist)
            maxdist = dist;         
    }   
    printf("Maximal distance between cameras centers: %.12lf\n", maxdist ); 

    //compare rotation matrices
    maxdist = 0;
    for( size_t i = 0; i < R.size(); i++ )
    {
        double dist = norm(R[i], R_backup[i], NORM_INF);
         
        if(dist > maxdist)
            maxdist = dist;         
    }    
    printf("Maximal difference in rotation matrices elements: %.12lf\n", maxdist ); 

    //compare intrinsic matrices
    maxdist = 0;
    double total_diff = 0;
    for( size_t i = 0; i < cameraMatrix.size(); i++ )
    {
        double fx_ratio = cameraMatrix[i].at<double>(0,0)/
                          cameraMatrix_backup[i].at<double>(0,0);
        double fy_ratio = cameraMatrix[i].at<double>(1,1)/
                          cameraMatrix_backup[i].at<double>(1,1);
        double cx_diff = cameraMatrix[i].at<double>(0,2) -
                          cameraMatrix_backup[i].at<double>(0,2);
        double cy_diff = cameraMatrix[i].at<double>(1,2) -
                          cameraMatrix_backup[i].at<double>(1,2);
        total_diff += fabs(fx_ratio - 1) + fabs(fy_ratio - 1) + fabs(cx_diff) + fabs(cy_diff);
    }
    //ts->printf(CvTS::LOG, "total diff = %g\n", total_diff);

    return 1;
}


class CV_BundleAdjustmentTest : public CvTest
{

public:
    CV_BundleAdjustmentTest();
    ~CV_BundleAdjustmentTest();
    void clear();
    //int write_default_params(CvFileStorage* fs);

protected:
    //int read_params( CvFileStorage* fs );
    int compare(double* val, double* ref_val, int len,
                double eps, const char* param_name);

    void run(int);
};


CV_BundleAdjustmentTest::CV_BundleAdjustmentTest():
    CvTest( "bundleadjust", "bundleAdjust" )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}


CV_BundleAdjustmentTest::~CV_BundleAdjustmentTest()
{
    clear();
}


void CV_BundleAdjustmentTest::clear()
{
    CvTest::clear();
}


int CV_BundleAdjustmentTest::compare(double* val, double* ref_val, int len,
                                      double eps, const char* param_name )
{
    return cvTsCmpEps2_64f( ts, val, ref_val, len, eps, param_name );
}

void CV_BundleAdjustmentTest::run( int start_from )
{
    int code = CvTS::OK;
    char            filepath[100];
    char            filename[100];
    
    CvSize          imageSize;
    int             numImages;
    CvSize          etalonSize;
    
    CvPoint2D64f*   imagePoints;
    CvPoint3D64f*   objectPoints;
    CvPoint2D64f*   reprojectPoints;

    double*         transVects;
    double*         rotMatrs;

    double*         goodTransVects;
    double*         goodRotMatrs;

    double          cameraMatrix[3*3];
    double          distortion[4];

    double          goodDistortion[4];

    int*            numbers;
    FILE*           file = 0;
    FILE*           datafile = 0; 
    int             i,j;
    int             currImage;
    int             currPoint;

    int             calibFlags;
    char            i_dat_file[100];
    int             numPoints;
    int numTests;
    int currTest;

    imagePoints     = 0;
    objectPoints    = 0;
    reprojectPoints = 0;
    numbers         = 0;

    transVects      = 0;
    rotMatrs        = 0;
    goodTransVects  = 0;
    goodRotMatrs    = 0;
    int progress = 0;

    //generate test data
    Params params;
    //set default params
    
    params.FOVx = 30;
    params.height = 480;
    params.width = 640;
    //params.noise = 0.25f;
    params.k1 = -0.5f;
    params.k2 = -0.5f;
    params.k3 = 0.0f;
    params.noise = 0.0f;
    params.p1 = 0;
    params.p2 = 0;  

    sprintf( g_filepath, "%sSBA/", ts->get_data_path() );
   
    GenerateTestData2(params);
    TestLevmar2();
    
    sprintf( filepath, "%scameracalibration/", ts->get_data_path() );
    sprintf( filename, "%sdatafiles.txt", filepath );
    datafile = fopen( filename, "r" );
    if( datafile == 0 ) 
    {
        ts->printf( CvTS::LOG, "Could not open file with list of test files: %s\n", filename );
        code = CvTS::FAIL_MISSING_TEST_DATA;
        goto _exit_;
    }

    fscanf(datafile,"%d",&numTests);

    for( currTest = start_from; currTest < numTests; currTest++ )
    {
        progress = update_progress( progress, currTest, numTests, 0 );
        fscanf(datafile,"%s",i_dat_file);
        sprintf(filename, "%s%s", filepath, i_dat_file);
        file = fopen(filename,"r");

        ts->update_context( this, currTest, true );

        if( file == 0 )
        {
            ts->printf( CvTS::LOG,
                "Can't open current test file: %s\n",filename);
            if( numTests == 1 )
            {
                code = CvTS::FAIL_MISSING_TEST_DATA;
                goto _exit_;
            }
            continue; // if there is more than one test, just skip the test
        }

        fscanf(file,"%d %d\n",&(imageSize.width),&(imageSize.height));
        if( imageSize.width <= 0 || imageSize.height <= 0 )
        {
            ts->printf( CvTS::LOG, "Image size in test file is incorrect\n" );
            code = CvTS::FAIL_INVALID_TEST_DATA;
            goto _exit_;
        }

        /* Read etalon size */
        fscanf(file,"%d %d\n",&(etalonSize.width),&(etalonSize.height));
        if( etalonSize.width <= 0 || etalonSize.height <= 0 )
        {
            ts->printf( CvTS::LOG, "Pattern size in test file is incorrect\n" );
            code = CvTS::FAIL_INVALID_TEST_DATA;
            goto _exit_;
        }

        numPoints = etalonSize.width * etalonSize.height;

        /* Read number of images */
        fscanf(file,"%d\n",&numImages);
        if( numImages <=0 )
        {
            ts->printf( CvTS::LOG, "Number of images in test file is incorrect\n");
            code = CvTS::FAIL_INVALID_TEST_DATA;
            goto _exit_;
        }

        /* Need to allocate memory */
        imagePoints     = (CvPoint2D64f*)cvAlloc( numPoints *
                                                    numImages * sizeof(CvPoint2D64f));
        
        objectPoints    = (CvPoint3D64f*)cvAlloc( numPoints *
                                                    numImages * sizeof(CvPoint3D64f));

        reprojectPoints = (CvPoint2D64f*)cvAlloc( numPoints *
                                                    numImages * sizeof(CvPoint2D64f));

        /* Alloc memory for numbers */
        numbers = (int*)cvAlloc( numImages * sizeof(int));

        /* Fill it by numbers of points of each image*/
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            numbers[currImage] = etalonSize.width * etalonSize.height;
        }

        /* Allocate memory for translate vectors and rotmatrixs*/
        transVects     = (double*)cvAlloc(3 * 1 * numImages * sizeof(double));
        rotMatrs       = (double*)cvAlloc(3 * 3 * numImages * sizeof(double));

        goodTransVects = (double*)cvAlloc(3 * 1 * numImages * sizeof(double));
        goodRotMatrs   = (double*)cvAlloc(3 * 3 * numImages * sizeof(double));

        /* Read object points */
        i = 0;/* shift for current point */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( currPoint = 0; currPoint < numPoints; currPoint++ )
            {
                double x,y,z;
                fscanf(file,"%lf %lf %lf\n",&x,&y,&z);

                (objectPoints+i)->x = x;
                (objectPoints+i)->y = y;
                (objectPoints+i)->z = z;
                i++;
            }
        }

        /* Read image points */
        i = 0;/* shift for current point */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( currPoint = 0; currPoint < numPoints; currPoint++ )
            {
                double x,y;
                fscanf(file,"%lf %lf\n",&x,&y);

                (imagePoints+i)->x = x;
                (imagePoints+i)->y = y;
                i++;
            }
        }

        /* Read good data computed before */

        /* Focal lengths */
        double goodFcx,goodFcy;
        fscanf(file,"%lf %lf",&goodFcx,&goodFcy);

        /* Principal points */
        double goodCx,goodCy;
        fscanf(file,"%lf %lf",&goodCx,&goodCy);

        /* Read distortion */

        fscanf(file,"%lf",goodDistortion+0);
        fscanf(file,"%lf",goodDistortion+1);
        fscanf(file,"%lf",goodDistortion+2);
        fscanf(file,"%lf",goodDistortion+3);

        /* Read good Rot matrixes */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( i = 0; i < 3; i++ )
                for( j = 0; j < 3; j++ )
                    fscanf(file, "%lf", goodRotMatrs + currImage * 9 + j * 3 + i);
        }

        /* Read good Trans vectors */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( i = 0; i < 3; i++ )
                fscanf(file, "%lf", goodTransVects + currImage * 3 + i);
        }
        
        calibFlags = 
                     //CV_CALIB_FIX_PRINCIPAL_POINT +
                     //CV_CALIB_ZERO_TANGENT_DIST +
                     //CV_CALIB_FIX_ASPECT_RATIO +
                     //CV_CALIB_USE_INTRINSIC_GUESS + 
                     0;
        memset( cameraMatrix, 0, 9*sizeof(cameraMatrix[0]) );
        cameraMatrix[0] = cameraMatrix[4] = 807.;
        cameraMatrix[2] = (imageSize.width - 1)*0.5;
        cameraMatrix[5] = (imageSize.height - 1)*0.5;
        cameraMatrix[8] = 1.;

        /* Now we can calibrate camera */
        cvCalibrateCamera_64f(  numImages,
                                numbers,
                                imageSize,
                                imagePoints,
                                objectPoints,
                                distortion,
                                cameraMatrix,
                                transVects,
                                rotMatrs,
                                calibFlags );

        /* ---- Reproject points to the image ---- */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            int numPoints = etalonSize.width * etalonSize.height;
            cvProjectPointsSimple(  numPoints,
                                    objectPoints + currImage * numPoints,
                                    rotMatrs + currImage * 9,
                                    transVects + currImage * 3,
                                    cameraMatrix,
                                    distortion,
                                    reprojectPoints + currImage * numPoints);
        }


        /* ----- Compute reprojection error ----- */
        i = 0;
        double dx,dy;
        double rx,ry;
        double meanDx,meanDy;
        double maxDx = 0.0;
        double maxDy = 0.0;

        meanDx = 0;
        meanDy = 0;
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            for( currPoint = 0; currPoint < etalonSize.width * etalonSize.height; currPoint++ )
            {
                rx = reprojectPoints[i].x;
                ry = reprojectPoints[i].y;
                dx = rx - imagePoints[i].x;
                dy = ry - imagePoints[i].y;

                meanDx += dx;
                meanDy += dy;

                dx = fabs(dx);
                dy = fabs(dy);

                if( dx > maxDx )
                    maxDx = dx;
                
                if( dy > maxDy )
                    maxDy = dy;
                i++;
            }
        }

        meanDx /= numImages * etalonSize.width * etalonSize.height;
        meanDy /= numImages * etalonSize.width * etalonSize.height;

        /* ========= Compare parameters ========= */

        /* ----- Compare focal lengths ----- */
        code = compare(cameraMatrix+0,&goodFcx,1,0.01,"fx");
        if( code < 0 )
            goto _exit_;

        code = compare(cameraMatrix+4,&goodFcy,1,0.01,"fy");
        if( code < 0 )
            goto _exit_;

        /* ----- Compare principal points ----- */
        code = compare(cameraMatrix+2,&goodCx,1,0.01,"cx");
        if( code < 0 )
            goto _exit_;

        code = compare(cameraMatrix+5,&goodCy,1,0.01,"cy");
        if( code < 0 )
            goto _exit_;

        /* ----- Compare distortion ----- */
        code = compare(distortion,goodDistortion,4,0.01,"[k1,k2,p1,p2]");
        if( code < 0 )
            goto _exit_;

        /* ----- Compare rot matrixs ----- */
        code = compare(rotMatrs,goodRotMatrs, 9*numImages,0.05,"rotation matrices");
        if( code < 0 )
            goto _exit_;

        /* ----- Compare rot matrixs ----- */
        code = compare(transVects,goodTransVects, 3*numImages,0.05,"translation vectors");
        if( code < 0 )
            goto _exit_;

        if( maxDx > 1.0 )
        {
            ts->printf( CvTS::LOG,
                      "Error in reprojection maxDx=%f > 1.0\n",maxDx);
            code = CvTS::FAIL_BAD_ACCURACY; goto _exit_;
        }

        if( maxDy > 1.0 )
        {
            ts->printf( CvTS::LOG,
                      "Error in reprojection maxDy=%f > 1.0\n",maxDy);
            code = CvTS::FAIL_BAD_ACCURACY; goto _exit_;
        }

        cvFree(&imagePoints);
        cvFree(&objectPoints);
        cvFree(&reprojectPoints);
        cvFree(&numbers);

        cvFree(&transVects);
        cvFree(&rotMatrs);
        cvFree(&goodTransVects);
        cvFree(&goodRotMatrs);

        fclose(file);
        file = 0;
    }

_exit_:

    if( file )
        fclose(file);

    if( datafile )
        fclose(datafile);

    /* Free all allocated memory */
    cvFree(&imagePoints);
    cvFree(&objectPoints);
    cvFree(&reprojectPoints);
    cvFree(&numbers);

    cvFree(&transVects);
    cvFree(&rotMatrs);
    cvFree(&goodTransVects);
    cvFree(&goodRotMatrs);

    if( code < 0 )
        ts->set_failed_test_info( code );
}

//CV_BundleAdjustmentTest bundleadjustment_test;

#endif

