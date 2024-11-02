// C++
#include <iostream>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/3d.hpp>
#include <opencv2/features.hpp>
// PnP Tutorial
#include "Mesh.h"
#include "Model.h"
#include "PnPProblem.h"
#include "RobustMatcher.h"
#include "ModelRegistration.h"
#include "Utils.h"

using namespace cv;
using namespace std;

/**  GLOBAL VARIABLES  **/

// Boolean the know if the registration it's done
bool end_registration = false;

// Intrinsic camera parameters: UVC WEBCAM
const double f = 45; // focal length in mm
const double sx = 22.3, sy = 14.9;
const double width = 2592, height = 1944;
const double params_CANON[] = { width*f/sx,   // fx
                                height*f/sy,  // fy
                                width/2,      // cx
                                height/2};    // cy

// Setup the points to register in the image
// In the order of the *.ply file and starting at 1
const int n = 8;
const int pts[] = {1, 2, 3, 4, 5, 6, 7, 8}; // 3 -> 4

/*
 * CREATE MODEL REGISTRATION OBJECT
 * CREATE OBJECT MESH
 * CREATE OBJECT MODEL
 * CREATE PNP OBJECT
 */
ModelRegistration registration;
Model model;
Mesh mesh;
PnPProblem pnp_registration(params_CANON);

/**********************************************************************************************************/
static void help()
{
    cout
            << "--------------------------------------------------------------------------"   << endl
            << "This program shows how to create your 3D textured model. "                    << endl
            << "Usage:"                                                                       << endl
            << "./cpp-tutorial-pnp_registration"                                              << endl
            << "--------------------------------------------------------------------------"   << endl
            << endl;
}

// Mouse events for model registration
static void onMouseModelRegistration( int event, int x, int y, int, void* )
{
    if  ( event == EVENT_LBUTTONUP )
    {
        bool is_registrable = registration.is_registrable();
        if (is_registrable)
        {
            int n_regist = registration.getNumRegist();
            int n_vertex = pts[n_regist];

            Point2f point_2d = Point2f((float)x,(float)y);
            Point3f point_3d = mesh.getVertex(n_vertex-1);

            registration.registerPoint(point_2d, point_3d);
            if( registration.getNumRegist() == registration.getNumMax() ) end_registration = true;
        }
    }
}

/**  Main program  **/
int main(int argc, char *argv[])
{
    help();

    const String keys =
            "{help h        |      | print this message                                                 }"
            "{image i       |      | path to input image                                                }"
            "{model         |      | path to output yml model                                           }"
            "{mesh          |      | path to ply mesh                                                   }"
            "{keypoints k   |2000  | number of keypoints to detect (only for ORB)                       }"
            "{feature       |ORB   | feature name (ORB, KAZE, AKAZE, BRISK, SIFT, SURF, BINBOOST, VGG)  }"
            ;
    CommandLineParser parser(argc, argv, keys);

    string img_path = samples::findFile("samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/resized_IMG_3875.JPG");  // image to register
    string ply_read_path = samples::findFile("samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/box.ply");          // object mesh
    string write_path = samples::findFile("samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/cookies_ORB.yml");     // output file
    int numKeyPoints = 2000;
    string featureName = "ORB";

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    else
    {
        img_path = parser.get<string>("image").size() > 0 ? parser.get<string>("image") : img_path;
        ply_read_path = parser.get<string>("mesh").size() > 0 ? parser.get<string>("mesh") : ply_read_path;
        write_path = parser.get<string>("model").size() > 0 ? parser.get<string>("model") : write_path;
        numKeyPoints = parser.has("keypoints") ? parser.get<int>("keypoints") : numKeyPoints;
        featureName = parser.has("feature") ? parser.get<string>("feature") : featureName;
    }

    std::cout << "Input image: " << img_path << std::endl;
    std::cout << "CAD model: " << ply_read_path << std::endl;
    std::cout << "Output training file: " << write_path << std::endl;
    std::cout << "Feature: " << featureName << std::endl;
    std::cout << "Number of keypoints for ORB: " << numKeyPoints << std::endl;

    // load a mesh given the *.ply file path
    mesh.load(ply_read_path);

    //Instantiate robust matcher: detector, extractor, matcher
    RobustMatcher rmatcher;
    Ptr<Feature2D> detector, descriptor;
    createFeatures(featureName, numKeyPoints, detector, descriptor);
    rmatcher.setFeatureDetector(detector);
    rmatcher.setDescriptorExtractor(descriptor);


    /**  GROUND TRUTH OF THE FIRST IMAGE  **/

    // Create & Open Window
    namedWindow("MODEL REGISTRATION", WINDOW_KEEPRATIO);

    // Set up the mouse events
    setMouseCallback("MODEL REGISTRATION", onMouseModelRegistration, 0);

    // Open the image to register
    Mat img_in = imread(img_path, IMREAD_COLOR);
    Mat img_vis;

    if (img_in.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Set the number of points to register
    int num_registrations = n;
    registration.setNumMax(num_registrations);

    cout << "Click the box corners ..." << endl;
    cout << "Waiting ..." << endl;

    // Some basic colors
    const Scalar red(0, 0, 255);
    const Scalar green(0,255,0);
    const Scalar blue(255,0,0);
    const Scalar yellow(0,255,255);

    // Loop until all the points are registered
    while ( waitKey(30) < 0 )
    {
        // Refresh debug image
        img_vis = img_in.clone();

        // Current registered points
        vector<Point2f> list_points2d = registration.get_points2d();
        vector<Point3f> list_points3d = registration.get_points3d();

        // Draw current registered points
        drawPoints(img_vis, list_points2d, list_points3d, red);

        // If the registration is not finished, draw which 3D point we have to register.
        // If the registration is finished, breaks the loop.
        if (!end_registration)
        {
            // Draw debug text
            int n_regist = registration.getNumRegist();
            int n_vertex = pts[n_regist];
            Point3f current_poin3d = mesh.getVertex(n_vertex-1);

            drawQuestion(img_vis, current_poin3d, green);
            drawCounter(img_vis, registration.getNumRegist(), registration.getNumMax(), red);
        }
        else
        {
            // Draw debug text
            drawText(img_vis, "END REGISTRATION", green);
            drawCounter(img_vis, registration.getNumRegist(), registration.getNumMax(), green);
            break;
        }

        // Show the image
        imshow("MODEL REGISTRATION", img_vis);
    }

    /** COMPUTE CAMERA POSE **/

    cout << "COMPUTING POSE ..." << endl;

    // The list of registered points
    vector<Point2f> list_points2d = registration.get_points2d();
    vector<Point3f> list_points3d = registration.get_points3d();

    // Estimate pose given the registered points
    bool is_correspondence = pnp_registration.estimatePose(list_points3d, list_points2d, SOLVEPNP_ITERATIVE);
    if ( is_correspondence )
    {
        cout << "Correspondence found" << endl;

        // Compute all the 2D points of the mesh to verify the algorithm and draw it
        vector<Point2f> list_points2d_mesh = pnp_registration.verify_points(&mesh);
        draw2DPoints(img_vis, list_points2d_mesh, green);
    } else {
        cout << "Correspondence not found" << endl << endl;
    }

    // Show the image
    imshow("MODEL REGISTRATION", img_vis);

    // Show image until ESC pressed
    waitKey(0);


    /** COMPUTE 3D of the image Keypoints **/

    // Containers for keypoints and descriptors of the model
    vector<KeyPoint> keypoints_model;
    Mat descriptors;

    // Compute keypoints and descriptors
    rmatcher.computeKeyPoints(img_in, keypoints_model);
    rmatcher.computeDescriptors(img_in, keypoints_model, descriptors);

    // Check if keypoints are on the surface of the registration image and add to the model
    for (unsigned int i = 0; i < keypoints_model.size(); ++i) {
        Point2f point2d(keypoints_model[i].pt);
        Point3f point3d;
        bool on_surface = pnp_registration.backproject2DPoint(&mesh, point2d, point3d);
        if (on_surface)
        {
            model.add_correspondence(point2d, point3d);
            model.add_descriptor(descriptors.row(i));
            model.add_keypoint(keypoints_model[i]);
        }
        else
        {
            model.add_outlier(point2d);
        }
    }

    model.set_trainingImagePath(img_path);
    // save the model into a *.yaml file
    model.save(write_path);

    // Out image
    img_vis = img_in.clone();

    // The list of the points2d of the model
    vector<Point2f> list_points_in = model.get_points2d_in();
    vector<Point2f> list_points_out = model.get_points2d_out();

    // Draw some debug text
    string num = IntToString((int)list_points_in.size());
    string text = "There are " + num + " inliers";
    drawText(img_vis, text, green);

    // Draw some debug text
    num = IntToString((int)list_points_out.size());
    text = "There are " + num + " outliers";
    drawText2(img_vis, text, red);

    // Draw the object mesh
    drawObjectMesh(img_vis, &mesh, &pnp_registration, blue);

    // Draw found keypoints depending on if are or not on the surface
    draw2DPoints(img_vis, list_points_in, green);
    draw2DPoints(img_vis, list_points_out, red);

    // Show the image
    imshow("MODEL REGISTRATION", img_vis);

    // Wait until ESC pressed
    waitKey(0);

    // Close and Destroy Window
    destroyWindow("MODEL REGISTRATION");

    cout << "GOODBYE" << endl;
}
