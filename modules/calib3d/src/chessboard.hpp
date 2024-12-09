// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef CHESSBOARD_HPP_
#define CHESSBOARD_HPP_

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include <vector>
#include <set>
#include <map>

namespace cv {
namespace details{
/**
 * \brief Fast point sysmetric cross detector based on a localized radon transformation
 */
class FastX : public cv::Feature2D
{
    public:
        struct Parameters
        {
            float strength;       //!< minimal strength of a valid junction in dB
            float resolution;     //!< angle resolution in radians
            int branches;         //!< the number of branches
            int min_scale;        //!< scale level [0..8]
            int max_scale;        //!< scale level [0..8]
            bool filter;          //!< post filter feature map to improve impulse response
            bool super_resolution; //!< up-sample

            Parameters()
            {
                strength = 40;
                resolution = float(CV_PI*0.25);
                branches = 2;
                min_scale = 2;
                max_scale = 5;
                super_resolution = true;
                filter = true;
            }
        };

    public:
        FastX(const Parameters &config = Parameters());
        virtual ~FastX(){}

        void reconfigure(const Parameters &para);

        //declaration to be wrapped by rbind
        void detect(cv::InputArray image,std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask=cv::Mat())override
        {cv::Feature2D::detect(image.getMat(),keypoints,mask.getMat());}

        virtual void detectAndCompute(cv::InputArray image,
                                              cv::InputArray mask,
                                              std::vector<cv::KeyPoint>& keypoints,
                                              cv::OutputArray descriptors,
                                              bool useProvidedKeyPoints = false)override;

        void detectImpl(const cv::Mat& image,
                               std::vector<cv::KeyPoint>& keypoints,
                               std::vector<cv::Mat> &feature_maps,
                               const cv::Mat& mask=cv::Mat())const;

        void detectImpl(const cv::Mat& image,
                                std::vector<cv::Mat> &rotated_images,
                                std::vector<cv::Mat> &feature_maps,
                                const cv::Mat& mask=cv::Mat())const;

        void findKeyPoints(const std::vector<cv::Mat> &feature_map,
                                   std::vector<cv::KeyPoint>& keypoints,
                                   const cv::Mat& mask = cv::Mat())const;

        std::vector<std::vector<float> > calcAngles(const std::vector<cv::Mat> &rotated_images,
                                                            std::vector<cv::KeyPoint> &keypoints)const;
        // define pure virtual methods
        virtual int descriptorSize()const override{return 0;}
        virtual int descriptorType()const override{return 0;}
        virtual void operator()( cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool useProvidedKeypoints=false )const
        {
            descriptors.clear();
            detectImpl(image.getMat(),keypoints,mask);
            if(!useProvidedKeypoints)        // suppress compiler warning
                return;
            return;
        }

    protected:
        virtual void computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)const
        {
            descriptors = cv::Mat();
            detectImpl(image,keypoints);
        }

    private:
        void detectImpl(const cv::Mat& _src, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask)const;
        virtual void detectImpl(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask=cv::noArray())const;

        void rotate(float angle,cv::InputArray img,cv::Size size,cv::OutputArray out)const;
        void calcFeatureMap(const cv::Mat &images,cv::Mat& out)const;

    private:
        Parameters parameters;
};

/**
 * \brief Ellipse class
 */
class Ellipse
{
    public:
        Ellipse();
        Ellipse(const cv::Point2f &center, const cv::Size2f &axes, float angle);

        void draw(cv::InputOutputArray img,const cv::Scalar &color = cv::Scalar::all(120))const;
        bool contains(const cv::Point2f &pt)const;
        cv::Point2f getCenter()const;
        const cv::Size2f &getAxes()const;

    private:
        cv::Point2f center;
        cv::Size2f axes;
        float angle,cosf,sinf;
};

/**
 * \brief Chessboard corner detector
 *
 * The detectors tries to find all chessboard corners of an imaged
 * chessboard and returns them as an ordered vector of KeyPoints.
 * Thereby, the left top corner has index 0 and the bottom right
 * corner n*m-1.
 */
class Chessboard: public cv::Feature2D
{
    public:
        static const int DUMMY_FIELD_SIZE = 100;  // in pixel

        /**
         * \brief Configuration of a chessboard corner detector
         *
         */
        struct Parameters
        {
            cv::Size chessboard_size; //!< size of the chessboard
            int min_scale;            //!< scale level [0..8]
            int max_scale;            //!< scale level [0..8]
            int max_points;           //!< maximal number of points regarded
            int max_tests;            //!< maximal number of tested hypothesis
            bool super_resolution;    //!< use super-repsolution for chessboard detection
            bool larger;              //!< indicates if larger boards should be returned
            bool marker;              //!< indicates that valid boards must have a white and black cirlce marker used for orientation

            Parameters()
            {
                chessboard_size = cv::Size(9,6);
                min_scale = 3;
                max_scale = 4;
                super_resolution = true;
                max_points = 200;
                max_tests = 50;
                larger = false;
                marker = false;
            }

            Parameters(int scale,int _max_points):
                min_scale(scale),
                max_scale(scale),
                max_points(_max_points)
            {
                chessboard_size = cv::Size(9,6);
            }
        };


        /**
         * \brief Gets the 3D objects points for the chessboard assuming the
         * left top corner is located at the origin.
         *
         * \param[in] pattern_size Number of rows and cols of the pattern
         * \param[in] cell_size Size of one cell
         *
         * \returns Returns the object points as CV_32FC3
         */
        static cv::Mat getObjectPoints(const cv::Size &pattern_size,float cell_size);

        /**
         * \brief Class for searching and storing chessboard corners.
         *
         * The search is based on a feature map having strong pixel
         * values at positions where a chessboard corner is located.
         *
         * The board must be rectangular but supports empty cells
         *
         */
        class Board
        {
            public:
                /**
                 * \brief Estimates the position of the next point on a line using cross ratio constrain
                 *
                 * cross ratio:
                 * d12/d34 = d13/d24
                 *
                 * point order on the line:
                 * p0 --> p1 --> p2 --> p3
                 *
                 * \param[in] p0 First point coordinate
                 * \param[in] p1 Second point coordinate
                 * \param[in] p2 Third point coordinate
                 * \param[out] p3 Forth point coordinate
                 *
                 */
                static bool estimatePoint(const cv::Point2f &p0,const cv::Point2f &p1,const cv::Point2f &p2,cv::Point2f &p3);

                // using 1D homography
                static bool estimatePoint(const cv::Point2f &p0,const cv::Point2f &p1,const cv::Point2f &p2,const cv::Point2f &p3, cv::Point2f &p4);

                /**
                 * \brief Checks if all points of a row or column have a valid cross ratio constraint
                 *
                 * cross ratio:
                 * d12/d34 = d13/d24
                 *
                 * point order on the row/column:
                 * pt1 --> pt2 --> pt3 --> pt4
                 *
                 * \param[in] points THe points of the row/column
                 *
                 */
                static bool checkRowColumn(const std::vector<cv::Point2f> &points);

                /**
                 * \brief Estimates the search area for the next point on the line using cross ratio
                 *
                 * point order on the line:
                 * (p0) --> p1 --> p2 --> p3 --> search area
                 *
                 * \param[in] p1 First point coordinate
                 * \param[in] p2 Second point coordinate
                 * \param[in] p3 Third point coordinate
                 * \param[in] p Percentage of d34 used for the search area width and height [0..1]
                 * \param[out] ellipse The search area
                 * \param[in] p0 optional point to improve accuracy
                 *
                 * \return Returns false if no search area can be calculated
                 *
                 */
                static bool estimateSearchArea(const cv::Point2f &p1,const cv::Point2f &p2,const cv::Point2f &p3,float p,
                                                       Ellipse &ellipse,const cv::Point2f *p0 =NULL);

                /**
                 * \brief Estimates the search area for a specific point based on the given homography
                 *
                 * \param[in] H homography describing the transformation from ideal board to real one
                 * \param[in] row Row of the point
                 * \param[in] col Col of the point
                 * \param[in] p Percentage [0..1]
                 *
                 * \return Returns false if no search area can be calculated
                 *
                 */
                static Ellipse estimateSearchArea(cv::Mat H,int row, int col,float p,int field_size = DUMMY_FIELD_SIZE);

                /**
                 * \brief Searches for the maximum in a given search area
                 *
                 * \param[in] map feature map
                 * \param[in] ellipse search area
                 * \param[in] min_val Minimum value of the maximum to be accepted as maximum
                 *
                 * \return Returns a negative value if all points are outside the ellipse
                 *
                 */
                static float findMaxPoint(cv::flann::Index &index,const cv::Mat &data,const Ellipse &ellipse,float white_angle,float black_angle,cv::Point2f &pt);

                /**
                 * \brief Searches for the next point using cross ratio constrain
                 *
                 * \param[in] index flann index
                 * \param[in] data extended flann data
                 * \param[in] pt1
                 * \param[in] pt2
                 * \param[in] pt3
                 * \param[in] white_angle
                 * \param[in] black_angle
                 * \param[in] min_response
                 * \param[out] point The resulting point
                 *
                 * \return Returns false if no point could be found
                 *
                 */
                static bool findNextPoint(cv::flann::Index &index,const cv::Mat &data,
                                                  const cv::Point2f &pt1,const cv::Point2f &pt2, const cv::Point2f &pt3,
                                                  float white_angle,float black_angle,float min_response,cv::Point2f &point);

                /**
                 * \brief Creates a new Board object
                 *
                 */
                Board(float white_angle=0,float black_angle=0);
                Board(const cv::Size &size, const std::vector<cv::Point2f> &points,float white_angle=0,float black_angle=0);
                Board(const Chessboard::Board &other);
                virtual ~Board();

                Board& operator=(const Chessboard::Board &other);

                /**
                 * \brief Draws the corners into the given image
                 *
                 * \param[in] m The image
                 * \param[out] out The resulting image
                 * \param[in] H optional homography to calculate search area
                 *
                 */
                void draw(cv::InputArray m,cv::OutputArray out,cv::InputArray H=cv::Mat())const;

                /**
                 * \brief Estimates the pose of the chessboard
                 *
                 */
                bool estimatePose(const cv::Size2f &real_size,cv::InputArray _K,cv::OutputArray rvec,cv::OutputArray tvec)const;

                /**
                 * \brief Clears all internal data of the object
                 *
                 */
                void clear();

                /**
                 * \brief Returns the angle of the black diagnonale
                 *
                 */
                float getBlackAngle()const;

                /**
                 * \brief Returns the angle of the black diagnonale
                 *
                 */
                float getWhiteAngle()const;

                /**
                 * \brief Initializes a 3x3 grid from 9 corner coordinates
                 *
                 * All points must be ordered:
                 * p0 p1 p2
                 * p3 p4 p5
                 * p6 p7 p8
                 *
                 * \param[in] points vector of points
                 *
                 * \return Returns false if the grid could not be initialized
                 */
                bool init(const std::vector<cv::Point2f> points);

                /**
                 * \brief Returns true if the board is empty
                 *
                 */
                bool isEmpty() const;

                /**
                 * \brief Returns all board corners as ordered vector
                 *
                 * The left top corner has index 0 and the bottom right
                 * corner rows*cols-1. All corners which only belong to
                 * empty cells are returned as NaN.
                 */
                std::vector<cv::Point2f> getCorners(bool ball=true) const;

                /**
                 * \brief Returns all board corners as ordered vector of KeyPoints
                 *
                 * The left top corner has index 0 and the bottom right
                 * corner rows*cols-1.
                 *
                 * \param[in] ball if set to false only non empty points are returned
                 *
                 */
                std::vector<cv::KeyPoint> getKeyPoints(bool ball=true) const;

                /**
                 * \brief Returns the centers of the chessboard cells
                 *
                 * The left top corner has index 0 and the bottom right
                 * corner (rows-1)*(cols-1)-1.
                 *
                 */
                std::vector<cv::Point2f> getCellCenters() const;

                /**
                 * \brief Returns all cells as mats of four points each describing their corners.
                 *
                 * The left top cell has index 0
                 *
                 */
                std::vector<cv::Mat> getCells(float shrink_factor = 1.0,bool bwhite=true,bool bblack = true) const;

                /**
                 * \brief Estimates the homography between an ideal board
                 * and reality based on the already recovered points
                 *
                 * \param[in] rect selecting a subset of the already recovered points
                 * \param[in] field_size The field size of the ideal board
                 *
                 */
                cv::Mat estimateHomography(cv::Rect rect,int field_size = DUMMY_FIELD_SIZE)const;

                /**
                 * \brief Estimates the homography between an ideal board
                 * and reality based on the already recovered points
                 *
                 * \param[in] field_size The field size of the ideal board
                 *
                 */
                cv::Mat estimateHomography(int field_size = DUMMY_FIELD_SIZE)const;

                /**
                 * \brief Warp image to match ideal checkerboard
                 *
                 */
                cv::Mat warpImage(cv::InputArray image)const;

                /**
                 * \brief Returns the size of the board
                 *
                 */
                cv::Size getSize() const;

                /**
                 * \brief Returns the number of cols
                 *
                 */
                size_t colCount() const;

                /**
                 * \brief Returns the number of rows
                 *
                 */
                size_t rowCount() const;

                /**
                 * \brief Returns the inner contour of the board including only valid corners
                 *
                 * \info the contour might be non squared if not all points of the board are defined
                 *
                 */
                std::vector<cv::Point2f> getContour()const;

                /**
                 * \brief Masks the found board in the given image
                 *
                 */
                void maskImage(cv::InputOutputArray img,const cv::Scalar &color=cv::Scalar::all(0))const;

                /**
                 * \brief Grows the board in all direction until no more corners are found in the feature map
                 *
                 * \param[in] data CV_32FC1 data of the flann index
                 * \param[in] flann_index flann index
                 *
                 * \returns the number of grows
                 */
                int grow(const cv::Mat &data,cv::flann::Index &flann_index);

                /**
                 * \brief Validates all corners using guided search based on the given homography
                 *
                 * \param[in] data CV_32FC1 data of the flann index
                 * \param[in] flann_index flann index
                 * \param[in] h Homography describing the transformation from ideal board to the real one
                 * \param[in] min_response Min response
                 *
                 * \returns the number of valid corners
                 */
                int validateCorners(const cv::Mat &data,cv::flann::Index &flann_index,const cv::Mat &h,float min_response=0);

                /**
                 * \brief check that no corner is used more than once
                 *
                 * \returns Returns false if a corner is used more than once
                 */
                 bool checkUnique()const;

                 /**
                  * \brief Returns false if the angles of the contour are smaller than 35°
                  *
                  */
                 bool validateContour()const;


                 /**
                   \brief delete left column of the board
                   */
                 bool shrinkLeft();

                 /**
                   \brief delete right column of the board
                   */
                 bool shrinkRight();

                 /**
                   \brief shrink first row of the board
                   */
                 bool shrinkTop();

                 /**
                   \brief delete last row of the board
                   */
                 bool shrinkBottom();

                /**
                 * \brief Grows the board to the left by adding one column.
                 *
                 * \param[in] map CV_32FC1 feature map
                 *
                 * \returns Returns false if the feature map has no maxima at the requested positions
                 */
                bool growLeft(const cv::Mat &map,cv::flann::Index &flann_index);
                void growLeft();

                /**
                 * \brief Grows the board to the top by adding one row.
                 *
                 * \param[in] map CV_32FC1 feature map
                 *
                 * \returns Returns false if the feature map has no maxima at the requested positions
                 */
                bool growTop(const cv::Mat &map,cv::flann::Index &flann_index);
                void growTop();

                /**
                 * \brief Grows the board to the right by adding one column.
                 *
                 * \param[in] map CV_32FC1 feature map
                 *
                 * \returns Returns false if the feature map has no maxima at the requested positions
                 */
                bool growRight(const cv::Mat &map,cv::flann::Index &flann_index);
                void growRight();

                /**
                 * \brief Grows the board to the bottom by adding one row.
                 *
                 * \param[in] map CV_32FC1 feature map
                 *
                 * \returns Returns false if the feature map has no maxima at the requested positions
                 */
                bool growBottom(const cv::Mat &map,cv::flann::Index &flann_index);
                void growBottom();

                /**
                 * \brief Adds one column on the left side
                 *
                 * \param[in] points The corner coordinates
                 *
                 */
                void addColumnLeft(const std::vector<cv::Point2f> &points);

                /**
                 * \brief Adds one column at the top
                 *
                 * \param[in] points The corner coordinates
                 *
                 */
                void addRowTop(const std::vector<cv::Point2f> &points);

                /**
                 * \brief Adds one column on the right side
                 *
                 * \param[in] points The corner coordinates
                 *
                 */
                void addColumnRight(const std::vector<cv::Point2f> &points);

                /**
                 * \brief Adds one row at the bottom
                 *
                 * \param[in] points The corner coordinates
                 *
                 */
                void addRowBottom(const std::vector<cv::Point2f> &points);

                /**
                 * \brief Rotates the board 90° degrees to the left
                 */
                void rotateLeft();

                /**
                 * \brief Rotates the board 90° degrees to the right
                 */
                void rotateRight();

                /**
                 * \brief Flips the board along its local x(width) coordinate direction
                 */
                void flipVertical();

                /**
                 * \brief Flips the board along its local y(height) coordinate direction
                 */
                void flipHorizontal();

                /**
                 * \brief Flips and rotates the board so that the angle of
                 * either the black or white diagonal is bigger than the x
                 * and y axis of the board and from a right handed
                 * coordinate system
                 */
                void normalizeOrientation(bool bblack=true);

                /**
                 * \brief Flips and rotates the board so that the marker
                 * is normalized
                 */
                bool normalizeMarkerOrientation();

                /**
                 * \brief Exchanges the stored board with the board stored in other
                 */
                void swap(Chessboard::Board &other);

                bool operator==(const Chessboard::Board& other) const {return rows*cols == other.rows*other.cols;}
                bool operator< (const Chessboard::Board& other) const {return rows*cols < other.rows*other.cols;}
                bool operator> (const Chessboard::Board& other) const {return rows*cols > other.rows*other.cols;}
                bool operator>= (const cv::Size& size)const { return rows*cols >= size.width*size.height; }

                /**
                 * \brief Returns a specific corner
                 *
                 * \info raises runtime_error if row col does not exists
                 */
                cv::Point2f& getCorner(int row,int col);

                /**
                 * \brief Returns true if the cell is empty meaning at least one corner is NaN
                 */
                bool isCellEmpty(int row,int col);

                /**
                 * \brief Returns the mapping from all corners idx to only valid corners idx
                 */
                std::map<int,int> getMapping()const;

                /**
                 * \brief Returns true if the cell is black
                 *
                 */
                 bool isCellBlack(int row,int col)const;

                /**
                 * \brief Returns true if the cell has a round marker at its
                 * center
                 *
                 */
                 bool hasCellMarker(int row,int col);

                /**
                 * \brief Detects round markers in the chessboard fields based
                 * on the given image and the already recoverd board corners
                 *
                 * \returns Returns the number of found markes
                 *
                 */
                 int detectMarkers(cv::InputArray image);

                 /**
                  * \brief Calculates the average edge sharpness for the chessboard
                  *
                  * \param[in] image The image where the chessboard was detected
                  * \param[in] rise_distance Rise distance 0.8 means 10% ... 90%
                  * \param[in] vertical by default only edge response for horiontal lines are calculated
                  *
                  * \returns Scalar(sharpness, average min_val, average max_val)
                  *
                  * \author aduda@krakenrobotik.de
                  */
                 cv::Scalar calcEdgeSharpness(cv::InputArray image,float rise_distance=0.8,bool vertical=false,cv::OutputArray sharpness=cv::noArray());


                 /**
                  * \brief Gets the 3D objects points for the chessboard
                  * assuming the left top corner is located at the origin. In
                  * case the board as a marker, the white marker cell is at position zero
                  *
                  * \param[in] cell_size Size of one cell
                  *
                  * \returns Returns the object points as CV_32FC3
                  */
                 cv::Mat getObjectPoints(float cell_size)const;


                 /**
                  * \brief Returns the angle the board is rotated agains the x-axis of the image plane
                  * \returns Returns the object points as CV_32FC3
                  */
                 float getAngle()const;

                 /**
                  * \brief Returns true if the main direction of the board is close to the image x-axis than y-axis
                  */
                 bool isHorizontal()const;

                 /**
                  * \brief Updates the search angles
                  */
                 void setAngles(float white,float black);

            private:
                // stores one cell
                // in general a cell is initialized by the Board so that:
                // * all corners are always pointing to a valid cv::Point2f
                // * depending on the position left,top,right and bottom might be set to NaN
                // * A cell is empty if at least one corner is NaN
                struct Cell
                {
                    cv::Point2f *top_left,*top_right,*bottom_right,*bottom_left; // corners
                    Cell *left,*top,*right,*bottom;         // neighbouring cells
                    bool black;                             // set to true if cell is black
                    bool marker;                            // set to true if cell has a round marker in its center
                    Cell();
                    bool empty()const;                      // indicates if the cell is empty (one of its corners has NaN)
                    int getRow()const;
                    int getCol()const;
                    cv::Point2f getCenter()const;
                    bool isInside(const cv::Point2f &pt)const;  // check if point is inside the cell
                };

                // corners
                enum CornerIndex
                {
                    TOP_LEFT,
                    TOP_RIGHT,
                    BOTTOM_RIGHT,
                    BOTTOM_LEFT
                };

                Cell* getCell(int row,int column); // returns a specific cell
                const Cell* getCell(int row,int column)const; // returns a specific cell
                void drawEllipses(const std::vector<Ellipse> &ellipses);

                // Iterator for iterating over board corners
                class PointIter
                {
                    public:
                        PointIter(Cell *cell,CornerIndex corner_index);
                        PointIter(const PointIter &other);
                        void operator=(const PointIter &other);
                        bool valid() const;                   // returns if the pointer is pointing to a cell

                        bool left(bool check_empty=false);    // moves one corner to the left or returns false
                        bool right(bool check_empty=false);   // moves one corner to the right or returns false
                        bool bottom(bool check_empty=false);  // moves one corner to the bottom or returns false
                        bool top(bool check_empty=false);     // moves one corner to the top or returns false
                        bool checkCorner()const;              // returns true if the current corner belongs to at least one
                                                              // none empty cell
                        bool isNaN()const;                    // returns true if the current corner is NaN

                        const cv::Point2f* operator*() const;  // current corner coordinate
                        cv::Point2f* operator*();              // current corner coordinate
                        const cv::Point2f* operator->() const; // current corner coordinate
                        cv::Point2f* operator->();             // current corner coordinate

                        Cell *getCell();                 // current cell
                    private:
                        CornerIndex corner_index;
                        Cell *cell;
                };

                std::vector<Cell*> cells;          // storage for all board cells
                std::vector<cv::Point2f*> corners; // storage for all corners
                Cell *top_left;                    // pointer to the top left corner of the board in its local coordinate system
                int rows;                          // number of inner pattern rows
                int cols;                          // number of inner pattern cols
                float white_angle,black_angle;
        };
    public:

        /**
         * \brief Creates a chessboard corner detectors
         *
         * \param[in] config Configuration used to detect chessboard corners
         *
         */
        Chessboard(const Parameters &config = Parameters());
        virtual ~Chessboard();
        void reconfigure(const Parameters &config = Parameters());
        Parameters getPara()const;

        /*
         * \brief Detects chessboard corners in the given image.
         *
         * The detectors tries to find all chessboard corners of an imaged
         * chessboard and returns them as an ordered vector of KeyPoints.
         * Thereby, the left top corner has index 0 and the bottom right
         * corner n*m-1.
         *
         * \param[in] image The image
         * \param[out] keypoints The detected corners as a vector of ordered KeyPoints
         * \param[in] mask Currently not supported
         *
         */
        void detect(cv::InputArray image,std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask=cv::Mat())override
        {cv::Feature2D::detect(image.getMat(),keypoints,mask.getMat());}

        virtual void detectAndCompute(cv::InputArray image,cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints,cv::OutputArray descriptors,
                                              bool useProvidedKeyPoints = false)override;

        /*
         * \brief Detects chessboard corners in the given image.
         *
         * The detectors tries to find all chessboard corners of an imaged
         * chessboard and returns them as an ordered vector of KeyPoints.
         * Thereby, the left top corner has index 0 and the bottom right
         * corner n*m-1.
         *
         * \param[in] image The image
         * \param[out] keypoints The detected corners as a vector of ordered KeyPoints
         * \param[out] feature_maps The feature map generated by LRJT and used to find the corners
         * \param[in] mask Currently not supported
         *
         */
        void detectImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,std::vector<cv::Mat> &feature_maps,const cv::Mat& mask)const;
        Chessboard::Board detectImpl(const cv::Mat& image,std::vector<cv::Mat> &feature_maps,const cv::Mat& mask)const;

        // define pure virtual methods
        virtual int descriptorSize()const override{return 0;}
        virtual int descriptorType()const override{return 0;}
        virtual void operator()( cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool useProvidedKeypoints=false )const
        {
            descriptors.clear();
            detectImpl(image.getMat(),keypoints,mask);
            if(!useProvidedKeypoints)        // suppress compiler warning
                return;
            return;
        }

    protected:
        virtual void computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)const
        {
            descriptors = cv::Mat();
            detectImpl(image,keypoints);
        }

        // indicates why a board could not be initialized for a certain keypoint
        enum BState
        {
            MISSING_POINTS = 0,       // at least 5 points are needed
            MISSING_PAIRS = 1,        // at least two pairs are needed
            WRONG_PAIR_ANGLE = 2,     // angle between pairs is too small
            WRONG_CONFIGURATION = 3,  // point configuration is wrong and does not belong to a board
            FOUND_BOARD = 4           // board was found
        };

        void findKeyPoints(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,std::vector<cv::Mat> &feature_maps,
                           std::vector<std::vector<float> > &angles ,const cv::Mat& mask)const;
        cv::Mat buildData(const std::vector<cv::KeyPoint>& keypoints)const;
        std::vector<cv::KeyPoint> getInitialPoints(cv::flann::Index &flann_index,const cv::Mat &data,const cv::KeyPoint &center,float white_angle,float black_angle, float min_response = 0)const;
        BState generateBoards(cv::flann::Index &flann_index,const cv::Mat &data, const cv::KeyPoint &center,
                             float white_angle,float black_angle,float min_response,const cv::Mat &img,
                             std::vector<Chessboard::Board> &boards)const;

    private:
        void detectImpl(const cv::Mat&,std::vector<cv::KeyPoint>&, const cv::Mat& mast =cv::Mat())const;
        virtual void detectImpl(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask=cv::noArray())const;

    private:
        Parameters parameters; // storing the configuration of the detector
};
}}  // end namespace details and cv

#endif
