// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include <iostream>
#include "volume_impl.hpp"
#include "tsdf_functions.hpp"
#include "hash_tsdf_functions.hpp"
#include "color_tsdf_functions.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/3d.hpp"

namespace cv
{


// For any edge, if one vertex is inside of the surface and the other is outside of the surface
//  then the edge intersects the surface
// For each of the 8 vertices of the cube can be two possible states : either inside or outside of the surface
// For any cube the are 2^8=256 possible sets of vertex states
// This table lists the edges intersected by the surface for all 256 possible vertex states
// There are 12 edges.  For each entry in the table, if edge #n is intersected, then bit #n is set to 1
int edgeTable[256] =
        {
                0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
                0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
                0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
                0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
                0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
                0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
                0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
                0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
                0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
                0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
                0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
                0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
                0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
                0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
                0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
                0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000};

//  For each of the possible vertex states listed in aiCubeEdgeFlags there is a specific triangulation
//  of the edge intersection points.  a2iTriangleConnectionTable lists all of them in the form of
//  0-5 edge triples with the list terminated by the invalid value -1.
//  For example: a2iTriangleConnectionTable[3] list the 2 triangles formed when corner[0]
//  and corner[1] are inside of the surface, but the rest of the cube is not.
int triTable[256][16] =
        {
                {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
                {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
                {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
                {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
                {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
                {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
                {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
                {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
                {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
                {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
                {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
                {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
                {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
                {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
                {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
                {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
                {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
                {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
                {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
                {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
                {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
                {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
                {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
                {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
                {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
                {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
                {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
                {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
                {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
                {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
                {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
                {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
                {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
                {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
                {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
                {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
                {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
                {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
                {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
                {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
                {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
                {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
                {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
                {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
                {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
                {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
                {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
                {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
                {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
                {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
                {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
                {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
                {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
                {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
                {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
                {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
                {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
                {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
                {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
                {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
                {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
                {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
                {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
                {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
                {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
                {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
                {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
                {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
                {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
                {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
                {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
                {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
                {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
                {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
                {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
                {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
                {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
                {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
                {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
                {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
                {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
                {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
                {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
                {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
                {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
                {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
                {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
                {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
                {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
                {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
                {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
                {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
                {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
                {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
                {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
                {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
                {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
                {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
                {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
                {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
                {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
                {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
                {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
                {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
                {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
                {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
                {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
                {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
                {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
                {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
                {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
                {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
                {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
                {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
                {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
                {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
                {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
                {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
                {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
                {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
                {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
                {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
                {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
                {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
                {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
                {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
                {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
                {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
                {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
                {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
                {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
                {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
                {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
                {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
                {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
                {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
                {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
                {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
                {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
                {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
                {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
                {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
                {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
                {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
                {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
                {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
                {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
                {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
                {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
                {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
                {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
                {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
                {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
                {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
                {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
                {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
                {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
                {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
                {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
                {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
                {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
                {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
                {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
                {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
                {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
                {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
                {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
                {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
                {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
                {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
                {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
                {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
                {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
                {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
                {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
                {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
                {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
                {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
                {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
                {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
                {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
                {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
                {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
                {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
                {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
                {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
                {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
                {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
                {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};


Volume::Impl::Impl(const VolumeSettings& _settings) :
    settings(_settings)
#ifdef HAVE_OPENCL
    , useGPU(ocl::useOpenCL())
#endif
{}

// probably useless
void Volume::Impl::saveMesh(const std::string& path) const {
    Mat vertices, faces;
    marchCubes(vertices, faces);
    cv::saveMesh(path, vertices, cv::noArray(), faces);
}


// TSDF

TsdfVolume::TsdfVolume(const VolumeSettings& _settings) :
    Volume::Impl(_settings)
{
    Vec3i volResolution;
    settings.getVolumeResolution(volResolution);
#ifndef HAVE_OPENCL
    volume = Mat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
#else
    if (useGPU)
        gpu_volume = UMat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
    else
        cpu_volume = Mat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<TsdfVoxel>());
#endif

    reset();
}
TsdfVolume::~TsdfVolume() {}

void TsdfVolume::integrate(const OdometryFrame& frame, InputArray _cameraPose)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENCL
    Mat depth;
#else
    UMat depth;
#endif
    frame.getDepth(depth);
    integrate(depth, _cameraPose);
}

void TsdfVolume::integrate(InputArray _depth, InputArray _cameraPose)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENCL
    Mat depth = _depth.getMat();
#else
    UMat depth = _depth.getUMat();
#endif
    CV_Assert(!depth.empty());

    Matx33f intr;
    settings.getCameraIntegrateIntrinsics(intr);
    Intr intrinsics(intr);
    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
#ifndef HAVE_OPENCL
        preCalculationPixNorm(depth.size(), intrinsics, pixNorms);
#else
        if (useGPU)
            ocl_preCalculationPixNorm(depth.size(), intrinsics, gpu_pixNorms);
        else
            preCalculationPixNorm(depth.size(), intrinsics, cpu_pixNorms);
#endif
    }
    const Matx44f cameraPose = _cameraPose.getMat();

#ifndef HAVE_OPENCL
    integrateTsdfVolumeUnit(settings, cameraPose, depth, pixNorms, volume);
#else
    if (useGPU)
        ocl_integrateTsdfVolumeUnit(settings, cameraPose, depth, gpu_pixNorms, gpu_volume);
    else
        integrateTsdfVolumeUnit(settings, cameraPose, depth, cpu_pixNorms, cpu_volume);
#endif
}
void TsdfVolume::integrate(InputArray, InputArray, InputArray)
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
}


void TsdfVolume::raycast(InputArray cameraPose, OutputArray points, OutputArray normals, OutputArray colors)  const
{
    raycast(cameraPose, settings.getRaycastHeight(), settings.getRaycastWidth(), points, normals, colors);
}


void TsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    if (_colors.needed())
        CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");

    CV_Assert(height > 0);
    CV_Assert(width > 0);

    const Matx44f cameraPose = _cameraPose.getMat();
#ifndef HAVE_OPENCL
    raycastTsdfVolumeUnit(settings, cameraPose, height, width, volume, _points, _normals);
#else
    if (useGPU)
        ocl_raycastTsdfVolumeUnit(settings, cameraPose, height, width, gpu_volume, _points, _normals);
    else
        raycastTsdfVolumeUnit(settings, cameraPose, height, width, cpu_volume, _points, _normals);
#endif
}

void TsdfVolume::fetchNormals(InputArray points, OutputArray normals) const
{
#ifndef HAVE_OPENCL
    fetchNormalsFromTsdfVolumeUnit(settings, volume, points, normals);
#else
    if (useGPU)
        ocl_fetchNormalsFromTsdfVolumeUnit(settings, gpu_volume, points, normals);
    else
        fetchNormalsFromTsdfVolumeUnit(settings, cpu_volume, points, normals);
#endif
}

void TsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const
{
#ifndef HAVE_OPENCL
    fetchPointsNormalsFromTsdfVolumeUnit(settings, volume, points, normals);
#else
    if (useGPU)
        ocl_fetchPointsNormalsFromTsdfVolumeUnit(settings, gpu_volume, points, normals);
    else
        fetchPointsNormalsFromTsdfVolumeUnit(settings, cpu_volume, points, normals);

#endif
}

void TsdfVolume::fetchPointsNormalsColors(OutputArray, OutputArray, OutputArray) const
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
}

void TsdfVolume::reset()
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENCL
    //TODO: use setTo(Scalar(0, 0))
    volume.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
        {
            TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
            v.tsdf = floatToTsdf(0.0f); v.weight = 0;
        });
#else
    if (useGPU)
        gpu_volume.setTo(Scalar(0, 0));
    else
        //TODO: use setTo(Scalar(0, 0))
        cpu_volume.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
            {
                TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
                v.tsdf = floatToTsdf(0.0f); v.weight = 0;
            });
#endif
}
int TsdfVolume::getVisibleBlocks() const { return 1; }
size_t TsdfVolume::getTotalVolumeUnits() const { return 1; }


template <typename T, size_t K>
bool vec_cmp(const Vec<T, K>& a, const Vec<T, K>& b)
{
    return std::lexicographical_compare(&a[0], &a[0] + K, &b[0], &b[0] + K);
}

template <typename T, size_t K>
bool unordered_vec_cmp(const Vec<T, K>& a, const Vec<T, K>& b)
{
    Vec<T, K> ta = a, tb = b;
    std::sort(&ta[0], &ta[0] + K);
    std::sort(&tb[0], &tb[0] + K);
    return std::lexicographical_compare(&ta[0], &ta[0] + K, &tb[0], &tb[0] + K);
}

struct MarchCubesInvoker : ParallelLoopBody
{
    MarchCubesInvoker(const TsdfVolume& _volume,
                      const VolumeSettings& _settings,
                      std::vector<Vec3f>& _meshPoints,
                      std::vector<Vec3i>& _meshFaces) :
            volume(_volume),
            settings(_settings),
            mcNeighbourPts{
                    Point3f(0.f, 0.f, 0.f),
                    Point3f(0.f, 0.f, 1.f),
                    Point3f(0.f, 1.f, 1.f),
                    Point3f(0.f, 1.f, 0.f),
                    Point3f(1.f, 0.f, 0.f),
                    Point3f(1.f, 0.f, 1.f),
                    Point3f(1.f, 1.f, 1.f),
                    Point3f(1.f, 1.f, 0.f)
            },
            mcNeighbourCoords([&_settings]{
                const Vec4i volDims;
                _settings.getVolumeDimensions(volDims);
                return Vec8i(
                         volDims.dot(Vec4i(0, 0, 0)),
                         volDims.dot(Vec4i(0, 0, 1)),
                         volDims.dot(Vec4i(0, 1, 1)),
                         volDims.dot(Vec4i(0, 1, 0)),
                         volDims.dot(Vec4i(1, 0, 0)),
                         volDims.dot(Vec4i(1, 0, 1)),
                         volDims.dot(Vec4i(1, 1, 1)),
                         volDims.dot(Vec4i(1, 1, 0))
                        );
            }()),
            volData(_volume.volume.ptr<TsdfVoxel>()),
            meshPoints(_meshPoints),
            meshFaces(_meshFaces)
            {}

    Point3f interpolate(Point3f p1, Point3f p2, float v1, float v2) const
    {
        float dV = 0.5f;
        if (abs(v1 - v2) > 0.0001f)
            dV = v1 / (v1 - v2);

        Point3f p = p1 + dV * (p2 - p1);
        return p;
    }

    virtual void operator()(const Range &range) const override
    {
        Vec4i volDims;
        settings.getVolumeDimensions(volDims);

        Vec3i resolution;
        settings.getVolumeResolution(resolution);
        const Point3i volResolution = Point3i(resolution);

        Matx44f _pose;
        settings.getVolumePose(_pose);
        const Affine3f pose = Affine3f(_pose);

        float voxelSize = settings.getVoxelSize();

        std::map<Vec3f, size_t, decltype(&vec_cmp<float, 3>)> point_id(vec_cmp<float, 3>);
        std::vector<Vec3f> points;
        std::set<Vec3i, decltype(&unordered_vec_cmp<int, 3>)> faces(unordered_vec_cmp<int, 3>);
        for (int x = range.start; x < range.end; x++)
        {
            int coordBaseX = x * volDims[0];
            for (int y = 0; y < volResolution.y - 1; y++)
            {
                int coordBaseY = coordBaseX + y * volDims[1];
                for (int z = 0; z < volResolution.z - 1; z++)
                {
                    int coordBase = coordBaseY + z * volDims[2];

                    if (volData[coordBase].weight == 0)
                        continue;

                    uint8_t cubeIndex = 0;
                    float tsdfValues[8]{};
                    for (int i = 0; i < 8; i++)
                    {
                        if (volData[mcNeighbourCoords[i] + coordBase].weight == 0)
                            continue;

                        tsdfValues[i] = tsdfToFloat(volData[mcNeighbourCoords[i] + coordBase].tsdf);
                        if (tsdfValues[i] <= 0)
                            cubeIndex |= (1 << i);
                    }

                    if (edgeTable[cubeIndex] == 0)
                        continue;

                    Point3f vertices[12]{};
                    Point3f basePt((float)x, (float)y, (float)z);


                    if (edgeTable[cubeIndex] & 1)
                        vertices[0] = basePt + interpolate(mcNeighbourPts[0], mcNeighbourPts[1],
                                                           tsdfValues[0], tsdfValues[1]);
                    if (edgeTable[cubeIndex] & 2)
                        vertices[1] = basePt + interpolate(mcNeighbourPts[1], mcNeighbourPts[2],
                                                           tsdfValues[1], tsdfValues[2]);
                    if (edgeTable[cubeIndex] & 4)
                        vertices[2] = basePt + interpolate(mcNeighbourPts[2], mcNeighbourPts[3],
                                                           tsdfValues[2], tsdfValues[3]);
                    if (edgeTable[cubeIndex] & 8)
                        vertices[3] = basePt + interpolate(mcNeighbourPts[3], mcNeighbourPts[0],
                                                           tsdfValues[3], tsdfValues[0]);
                    if (edgeTable[cubeIndex] & 16)
                        vertices[4] = basePt + interpolate(mcNeighbourPts[4], mcNeighbourPts[5],
                                                           tsdfValues[4], tsdfValues[5]);
                    if (edgeTable[cubeIndex] & 32)
                        vertices[5] = basePt + interpolate(mcNeighbourPts[5], mcNeighbourPts[6],
                                                           tsdfValues[5], tsdfValues[6]);
                    if (edgeTable[cubeIndex] & 64)
                        vertices[6] = basePt + interpolate(mcNeighbourPts[6], mcNeighbourPts[7],
                                                           tsdfValues[6], tsdfValues[7]);
                    if (edgeTable[cubeIndex] & 128)
                        vertices[7] = basePt + interpolate(mcNeighbourPts[7], mcNeighbourPts[4],
                                                           tsdfValues[7], tsdfValues[4]);
                    if (edgeTable[cubeIndex] & 256)
                        vertices[8] = basePt + interpolate(mcNeighbourPts[0], mcNeighbourPts[4],
                                                           tsdfValues[0], tsdfValues[4]);
                    if (edgeTable[cubeIndex] & 512)
                        vertices[9] = basePt + interpolate(mcNeighbourPts[1], mcNeighbourPts[5],
                                                           tsdfValues[1], tsdfValues[5]);
                    if (edgeTable[cubeIndex] & 1024)
                        vertices[10] = basePt + interpolate(mcNeighbourPts[2], mcNeighbourPts[6],
                                                            tsdfValues[2], tsdfValues[6]);
                    if (edgeTable[cubeIndex] & 2048)
                        vertices[11] = basePt + interpolate(mcNeighbourPts[3], mcNeighbourPts[7],
                                                            tsdfValues[3], tsdfValues[7]);

                    for (int i = 0; triTable[cubeIndex][i] != -1; i += 3)
                    {
                        Vec<size_t, 3> face;
                        for (size_t k = 0; k < 3; ++k)
                        {
                            Vec3f vertex = pose * (vertices[triTable[cubeIndex][i + k]] * voxelSize);
                            if (!point_id.count(vertex))
                            {
                                point_id[vertex] = points.size();
                                points.push_back(vertex);
                            }
                            face[k] = point_id[vertex];
                        }
                        faces.insert(face);
                    }
                }
            }
        }

        if(!points.empty())
        {
            AutoLock al(m);
            Vec3i shift = Vec3i::all(meshPoints.size());
            meshPoints.insert(meshPoints.end(), points.begin(), points.end());
            for (const auto& face : faces)
            {
                meshFaces.emplace_back(face + shift);
            }
        }
    }

    const TsdfVolume& volume;
    const VolumeSettings& settings;
    const Point3f mcNeighbourPts[8];
    const Vec8i mcNeighbourCoords;
    const TsdfVoxel* const volData;

    std::vector<Vec3f>& meshPoints;
    std::vector<Vec3i>& meshFaces;
    mutable Mutex m;
};

void TsdfVolume::marchCubes(OutputArray vertices, OutputArray faces) const
{
    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);

    std::vector<Vec3f> meshPoints;
    std::vector<Vec3i> meshFaces;
    MarchCubesInvoker mci(*this, settings, meshPoints, meshFaces);
    Range range(0, volResolution.x - 1);
    parallel_for_(range, mci);

    if (vertices.needed())
        Mat((int)meshPoints.size(), 1, CV_32FC3, &meshPoints[0]).copyTo(vertices);

    if (faces.needed())
        Mat((int)meshFaces.size(), 3, CV_32SC1, &meshFaces[0]).copyTo(faces);
}


// HASH_TSDF

HashTsdfVolume::HashTsdfVolume(const VolumeSettings& _settings) :
    Volume::Impl(_settings)
{
    Vec3i resolution;
    settings.getVolumeResolution(resolution);
    const Point3i volResolution = Point3i(resolution);
    volumeUnitDegree = calcVolumeUnitDegree(volResolution);

#ifndef HAVE_OPENCL
    volUnitsData = cv::Mat(VOLUMES_SIZE, resolution[0] * resolution[1] * resolution[2], rawType<TsdfVoxel>());
    reset();
#else
    if (useGPU)
    {
        reset();
    }
    else
    {
        cpu_volUnitsData = cv::Mat(VOLUMES_SIZE, resolution[0] * resolution[1] * resolution[2], rawType<TsdfVoxel>());
        reset();
    }
#endif
}

HashTsdfVolume::~HashTsdfVolume() {}

void HashTsdfVolume::integrate(const OdometryFrame& frame, InputArray _cameraPose)
{
    CV_TRACE_FUNCTION();
#ifndef HAVE_OPENCL
    Mat depth;
#else
    UMat depth;
#endif
    frame.getDepth(depth);
    integrate(depth, _cameraPose);
}

void HashTsdfVolume::integrate(InputArray _depth, InputArray _cameraPose)
{
#ifndef HAVE_OPENCL
    Mat depth = _depth.getMat();
#else
    UMat depth = _depth.getUMat();
#endif
    const Matx44f cameraPose = _cameraPose.getMat();
    Matx33f intr;
    settings.getCameraIntegrateIntrinsics(intr);
    Intr intrinsics(intr);
    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
#ifndef HAVE_OPENCL
        preCalculationPixNorm(depth.size(), intrinsics, pixNorms);
#else
        if (useGPU)
            ocl_preCalculationPixNorm(depth.size(), intrinsics, gpu_pixNorms);
        else
            preCalculationPixNorm(depth.size(), intrinsics, cpu_pixNorms);
#endif
    }
#ifndef HAVE_OPENCL
    integrateHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, lastFrameId, volumeUnitDegree, depth, pixNorms, volUnitsData, volumeUnits);
    lastFrameId++;
#else
    if (useGPU)
    {
        ocl_integrateHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, lastFrameId, bufferSizeDegree, volumeUnitDegree, depth, gpu_pixNorms, lastVisibleIndices, volUnitsDataCopy, gpu_volUnitsData, hashTable, isActiveFlags);
    }
    else
    {
        integrateHashTsdfVolumeUnit(settings, cameraPose, lastVolIndex, lastFrameId, volumeUnitDegree, depth, cpu_pixNorms, cpu_volUnitsData, cpu_volumeUnits);
        lastFrameId++;
    }
#endif
}

void HashTsdfVolume::integrate(InputArray, InputArray, InputArray)
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
}


void HashTsdfVolume::raycast(InputArray cameraPose, OutputArray points, OutputArray normals, OutputArray colors)  const
{
    raycast(cameraPose, settings.getRaycastHeight(), settings.getRaycastWidth(), points, normals, colors);
}


void HashTsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    if (_colors.needed())
        CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");

    const Matx44f cameraPose = _cameraPose.getMat();

#ifndef HAVE_OPENCL
    raycastHashTsdfVolumeUnit(settings, cameraPose, height, width, volumeUnitDegree, volUnitsData, volumeUnits, _points, _normals);
#else
    if (useGPU)
        ocl_raycastHashTsdfVolumeUnit(settings, cameraPose, height, width, volumeUnitDegree, hashTable, gpu_volUnitsData, _points, _normals);
    else
        raycastHashTsdfVolumeUnit(settings, cameraPose, height, width, volumeUnitDegree, cpu_volUnitsData, cpu_volumeUnits, _points, _normals);
#endif
}

void HashTsdfVolume::fetchNormals(InputArray points, OutputArray normals) const
{
#ifndef HAVE_OPENCL
    fetchNormalsFromHashTsdfVolumeUnit(settings, volUnitsData, volumeUnits, volumeUnitDegree, points, normals);
#else
    if (useGPU)
        olc_fetchNormalsFromHashTsdfVolumeUnit(settings, volumeUnitDegree, gpu_volUnitsData, volUnitsDataCopy, hashTable, points, normals);
    else
        fetchNormalsFromHashTsdfVolumeUnit(settings, cpu_volUnitsData, cpu_volumeUnits, volumeUnitDegree, points, normals);

#endif
}
void HashTsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const
{
#ifndef HAVE_OPENCL
    fetchPointsNormalsFromHashTsdfVolumeUnit(settings, volUnitsData, volumeUnits, volumeUnitDegree, points, normals);
#else
    if (useGPU)
        ocl_fetchPointsNormalsFromHashTsdfVolumeUnit(settings, volumeUnitDegree, gpu_volUnitsData, volUnitsDataCopy, hashTable, points, normals);
    else
        fetchPointsNormalsFromHashTsdfVolumeUnit(settings, cpu_volUnitsData, cpu_volumeUnits, volumeUnitDegree, points, normals);
#endif
}

void HashTsdfVolume::fetchPointsNormalsColors(OutputArray, OutputArray, OutputArray) const
{
    CV_Error(cv::Error::StsBadFunc, "This volume doesn't support vertex colors");
};

void HashTsdfVolume::reset()
{
    CV_TRACE_FUNCTION();
    lastVolIndex = 0;
    lastFrameId = 0;
#ifndef HAVE_OPENCL
    volUnitsData.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
        {
            TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
            v.tsdf = floatToTsdf(0.0f); v.weight = 0;
        });
    volumeUnits = VolumeUnitIndexes();
#else
    if (useGPU)
    {
        Vec3i resolution;
        settings.getVolumeResolution(resolution);

        bufferSizeDegree = 15;
        int buff_lvl = (int)(1 << bufferSizeDegree);
        int volCubed = resolution[0] * resolution[1] * resolution[2];

        volUnitsDataCopy = cv::Mat(buff_lvl, volCubed, rawType<TsdfVoxel>());
        gpu_volUnitsData = cv::UMat(buff_lvl, volCubed, CV_8UC2);
        lastVisibleIndices = cv::UMat(buff_lvl, 1, CV_32S);
        isActiveFlags = cv::UMat(buff_lvl, 1, CV_8U);
        hashTable = CustomHashSet();
        frameParams = Vec6f();
        gpu_pixNorms = UMat();
    }
    else
    {
        cpu_volUnitsData.forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
            {
                TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
                v.tsdf = floatToTsdf(0.0f); v.weight = 0;
            });
        cpu_volumeUnits = VolumeUnitIndexes();
    }
#endif
}

int HashTsdfVolume::getVisibleBlocks() const { return 1; }
size_t HashTsdfVolume::getTotalVolumeUnits() const { return 1; }

void HashTsdfVolume::marchCubes(OutputArray, OutputArray) const
{
    CV_Error(CV_StsNotImplemented, "Marching cubes is not implemented for HashTsdfVolume yet.");
}

// COLOR_TSDF

ColorTsdfVolume::ColorTsdfVolume(const VolumeSettings& _settings) :
    Volume::Impl(_settings)
{
    Vec3i volResolution;
    settings.getVolumeResolution(volResolution);
    volume = Mat(1, volResolution[0] * volResolution[1] * volResolution[2], rawType<RGBTsdfVoxel>());
    reset();
}

ColorTsdfVolume::~ColorTsdfVolume() {}

void ColorTsdfVolume::integrate(const OdometryFrame& frame, InputArray cameraPose)
{
    CV_TRACE_FUNCTION();
    Mat depth;
    frame.getDepth(depth);
    Mat rgb;
    frame.getImage(rgb);

    integrate(depth, rgb, cameraPose);
}

void ColorTsdfVolume::integrate(InputArray, InputArray)
{
    CV_Error(cv::Error::StsBadFunc, "Color data should be passed for this volume type");
}

void ColorTsdfVolume::integrate(InputArray _depth, InputArray _image, InputArray _cameraPose)
{
    Mat depth = _depth.getMat();
    Colors image = _image.getMat();
    const Matx44f cameraPose = _cameraPose.getMat();
    Matx33f intr;
    settings.getCameraIntegrateIntrinsics(intr);
    Intr intrinsics(intr);
    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
        preCalculationPixNorm(depth.size(), intrinsics, pixNorms);
    }
    integrateColorTsdfVolumeUnit(settings, cameraPose, depth, image, pixNorms, volume);
}

void ColorTsdfVolume::raycast(InputArray cameraPose, OutputArray points, OutputArray normals, OutputArray colors)  const
{
    raycast(cameraPose, settings.getRaycastHeight(), settings.getRaycastWidth(), points, normals, colors);
}

void ColorTsdfVolume::raycast(InputArray _cameraPose, int height, int width, OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    const Matx44f cameraPose = _cameraPose.getMat();
    raycastColorTsdfVolumeUnit(settings, cameraPose, height, width, volume, _points, _normals, _colors);
}

void ColorTsdfVolume::fetchNormals(InputArray points, OutputArray normals) const
{
    fetchNormalsFromColorTsdfVolumeUnit(settings, volume, points, normals);
}

void ColorTsdfVolume::fetchPointsNormals(OutputArray points, OutputArray normals) const
{
    fetchPointsNormalsFromColorTsdfVolumeUnit(settings, volume, points, normals);
}

void ColorTsdfVolume::fetchPointsNormalsColors(OutputArray points, OutputArray normals, OutputArray colors) const
{
    fetchPointsNormalsColorsFromColorTsdfVolumeUnit(settings, volume, points, normals, colors);
}

void ColorTsdfVolume::reset()
{
    CV_TRACE_FUNCTION();

    volume.forEach<VecRGBTsdfVoxel>([](VecRGBTsdfVoxel& vv, const int* /* position */)
        {
            RGBTsdfVoxel& v = reinterpret_cast<RGBTsdfVoxel&>(vv);
            v.tsdf = floatToTsdf(0.0f); v.weight = 0;
        });
}

int ColorTsdfVolume::getVisibleBlocks() const { return 1; }
size_t ColorTsdfVolume::getTotalVolumeUnits() const { return 1; }

void ColorTsdfVolume::marchCubes(OutputArray, OutputArray) const
{
    CV_Error(CV_StsNotImplemented, "Marching cubes is not implemented for ColorTsdfVolume yet.");
}

}
