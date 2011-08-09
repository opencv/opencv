#include <opencv2/opencv.hpp>
#include <iostream>

namespace cv
{

class CV_EXPORTS_W Subdiv2D
{
public:
    
    enum
    {
        PTLOC_ERROR = -2,
        PTLOC_OUTSIDE_RECT = -1,
        PTLOC_INSIDE = 0,
        PTLOC_VERTEX = 1,
        PTLOC_ON_EDGE = 2
    };
    
    enum
    {
        NEXT_AROUND_ORG   = 0x00,
        NEXT_AROUND_DST   = 0x22,
        PREV_AROUND_ORG   = 0x11,
        PREV_AROUND_DST   = 0x33,
        NEXT_AROUND_LEFT  = 0x13,
        NEXT_AROUND_RIGHT = 0x31,
        PREV_AROUND_LEFT  = 0x20,
        PREV_AROUND_RIGHT = 0x02
    };
    
    CV_WRAP Subdiv2D();
    CV_WRAP Subdiv2D(Rect rect);
    CV_WRAP void initDelaunay(Rect rect);
    
    CV_WRAP int insert(Point2f pt);
    CV_WRAP void insert(const vector<Point2f>& ptvec);
    CV_WRAP int locate(Point2f pt, CV_OUT int& edge, CV_OUT int& vertex);
    
    CV_WRAP int findNearest(Point2f pt, CV_OUT Point2f* nearestPt=0);
    CV_WRAP void getTriangleList(CV_OUT vector<Vec6f>& triangleList);
    CV_WRAP void getVoronoiFacetList(const vector<int>& idx, CV_OUT vector<vector<Point2f> >& facetList);
    
    CV_WRAP Point2f getVertex(int vertex, CV_OUT int* firstEdge=0) const;
    
    CV_WRAP int getEdge( int edge, int nextEdgeType ) const;
    CV_WRAP int nextEdge(int edge) const;
    CV_WRAP int rotateEdge(int edge, int rotate) const;
    CV_WRAP int symEdge(int edge) const;
    CV_WRAP int edgeOrg(int edge, CV_OUT Point2f* orgpt=0) const;
    CV_WRAP int edgeDst(int edge, CV_OUT Point2f* dstpt=0) const;
    
protected:
    int newEdge();
    void deleteEdge(int edge);
    int newPoint(Point2f pt, bool isvirtual, int firstEdge=0);
    void deletePoint(int vtx);
    void setEdgePoints( int edge, int orgPt, int dstPt );
    void splice( int edgeA, int edgeB );
    int connectEdges( int edgeA, int edgeB );
    void swapEdges( int edge );
    int isRightOf(Point2f pt, int edge) const;
    void calcVoronoi();
    void clearVoronoi();
    
    struct CV_EXPORTS Vertex
    {
        Vertex();
        Vertex(Point2f pt, bool _isvirtual, int _firstEdge=0);
        bool isvirtual() const;
        bool isfree() const;
        int firstEdge;
        int type;
        Point2f pt;
    };
    struct CV_EXPORTS QuadEdge
    {
        QuadEdge();
        QuadEdge(int edgeidx);
        bool isfree() const;
        int next[4];
        int pt[4];
    };
    
    vector<Vertex> vtx;
    vector<QuadEdge> qedges;
    int freeQEdge;
    int freePoint;
    bool validGeometry;
    
    int recentEdge;
    Point2f topLeft;
    Point2f bottomRight;
};


int Subdiv2D::nextEdge(int edge) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    return qedges[edge >> 2].next[edge & 3];
}

int Subdiv2D::rotateEdge(int edge, int rotate) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    return (edge & ~3) + ((edge + rotate) & 3);
}

int Subdiv2D::symEdge(int edge) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    return edge ^ 2;
}

int Subdiv2D::getEdge(int edge, int nextEdgeType) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    edge = qedges[edge >> 2].next[(edge + nextEdgeType) & 3];
    return (edge & ~3) + ((edge + (nextEdgeType >> 4)) & 3);
}
    
int Subdiv2D::edgeOrg(int edge, CV_OUT Point2f* orgpt) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    int vidx = qedges[edge >> 2].pt[edge & 3];
    if( orgpt )
    {
        CV_DbgAssert((size_t)vidx < vtx.size());
        *orgpt = vtx[vidx].pt;
    }
    return vidx;
}
    
int Subdiv2D::edgeDst(int edge, CV_OUT Point2f* dstpt) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    int vidx = qedges[edge >> 2].pt[(edge + 2) & 3];
    if( dstpt )
    {
        CV_DbgAssert((size_t)vidx < vtx.size());
        *dstpt = vtx[vidx].pt;
    }
    return vidx;
}

    
Point2f Subdiv2D::getVertex(int vertex, CV_OUT int* firstEdge) const
{
    CV_DbgAssert((size_t)vertex < vtx.size());
    if( firstEdge )
        *firstEdge = vtx[vertex].firstEdge;
    return vtx[vertex].pt;
}
    
    
Subdiv2D::Subdiv2D()
{
    validGeometry = false;
    freeQEdge = 0;
    freePoint = 0;
    recentEdge = 0;
}
    
Subdiv2D::Subdiv2D(Rect rect)
{
    validGeometry = false;
    freeQEdge = 0;
    freePoint = 0;
    recentEdge = 0;
    
    initDelaunay(rect);
}
    

Subdiv2D::QuadEdge::QuadEdge()
{
    next[0] = next[1] = next[2] = next[3] = 0;
    pt[0] = pt[1] = pt[2] = pt[3] = 0;
}
    
Subdiv2D::QuadEdge::QuadEdge(int edgeidx)
{
    next[0] = edgeidx;
    next[1] = edgeidx+3;
    next[2] = edgeidx+2;
    next[3] = edgeidx+1;
    
    pt[0] = pt[1] = pt[2] = pt[3] = 0;
}

bool Subdiv2D::QuadEdge::isfree() const
{
    return next[0] <= 0;
}
    
Subdiv2D::Vertex::Vertex()
{
    firstEdge = 0;
    type = -1;
}
    
Subdiv2D::Vertex::Vertex(Point2f _pt, bool _isvirtual, int _firstEdge)
{
    firstEdge = _firstEdge;
    type = (int)_isvirtual;
    pt = _pt;
}

bool Subdiv2D::Vertex::isvirtual() const
{
    return type > 0;
}
    
bool Subdiv2D::Vertex::isfree() const
{
    return firstEdge <= 0;
}
    
void Subdiv2D::splice( int edgeA, int edgeB )
{
    int& a_next = qedges[edgeA >> 2].next[edgeA & 3];
    int& b_next = qedges[edgeB >> 2].next[edgeB & 3];
    int a_rot = rotateEdge(a_next, 1);
    int b_rot = rotateEdge(b_next, 1);
    int& a_rot_next = qedges[a_rot >> 2].next[a_rot & 3];
    int& b_rot_next = qedges[b_rot >> 2].next[b_rot & 3];
    std::swap(a_next, b_next);
    std::swap(a_rot_next, b_rot_next);
}

void Subdiv2D::setEdgePoints(int edge, int orgPt, int dstPt)
{
    qedges[edge >> 2].pt[edge & 3] = orgPt;
    qedges[edge >> 2].pt[(edge + 2) & 3] = dstPt;
}
    
int Subdiv2D::connectEdges( int edgeA, int edgeB )
{
    int edge = newEdge();
    
    splice(edge, getEdge(edgeA, NEXT_AROUND_LEFT));
    splice(symEdge(edge), edgeB);
    
    setEdgePoints(edge, edgeDst(edgeA), edgeOrg(edgeB));
    return edge;
}
    
void Subdiv2D::swapEdges( int edge )
{
    int sedge = symEdge(edge);
    int a = getEdge(edge, PREV_AROUND_ORG);
    int b = getEdge(sedge, PREV_AROUND_ORG);
    
    splice(edge, a);
    splice(sedge, b);
    
    setEdgePoints(edge, edgeDst(a), edgeDst(b));

    splice(edge, getEdge(a, NEXT_AROUND_LEFT));
    splice(sedge, getEdge(b, NEXT_AROUND_LEFT));
}
   
int Subdiv2D::isRightOf(Point2f pt, int edge) const
{
    Point2f org, dst;
    edgeOrg(edge, &org);
    edgeDst(edge, &dst);
    double cw_area = cvTriangleArea( pt, dst, org );
    
    return (cw_area > 0) - (cw_area < 0);
}

    
int Subdiv2D::newEdge()
{
    if( freeQEdge == 0 )
    {
        qedges.push_back(QuadEdge());
        freeQEdge = (int)(qedges.size()-1);
    }
    int edge = freeQEdge*4;
    freeQEdge = qedges[edge >> 2].next[1];
    qedges[edge >> 2] = QuadEdge(edge);
    return edge;
}
    
void Subdiv2D::deleteEdge(int edge)
{
    CV_DbgAssert((size_t)(edge >> 2) < (size_t)qedges.size());
    splice( edge, getEdge(edge, PREV_AROUND_ORG) );
    int sedge = symEdge(edge);
    splice(sedge, getEdge(sedge, PREV_AROUND_ORG) );

    edge >>= 2;
    qedges[edge].next[0] = -1;
    qedges[edge].next[1] = freeQEdge;
    freeQEdge = edge;
}
    
int Subdiv2D::newPoint(Point2f pt, bool isvirtual, int firstEdge)
{
    if( freePoint == 0 )
    {
        vtx.push_back(Vertex());
        freePoint = (int)(vtx.size()-1);
        vtx[freePoint].type = -1;
        vtx[freePoint].firstEdge = 0;
    }
    int vidx = freePoint;
    freePoint = vtx[vidx].firstEdge;
    vtx[vidx] = Vertex(pt, isvirtual, firstEdge);
    
    return vidx;
}

void Subdiv2D::deletePoint(int vidx)
{
    CV_DbgAssert( (size_t)vidx < vtx.size() );
    vtx[vidx].firstEdge = freePoint;
    vtx[vidx].type = -1;
    freePoint = vidx;
}
    
int Subdiv2D::locate(Point2f pt, int& _edge, int& _vertex)
{
    int vertex = 0;
        
    int i, maxEdges = (int)(qedges.size() * 4);
    int edge = recentEdge;
    
    CV_Assert(edge > 0);
    
    if( pt.x < topLeft.x || pt.y < topLeft.y || pt.x >= bottomRight.x || pt.y >= bottomRight.y )
        CV_Error( CV_StsOutOfRange, "" );
    
    int location = PTLOC_ERROR;
    
    int right_of_curr = isRightOf(pt, edge);
    if( right_of_curr > 0 )
    {
        edge = symEdge(edge);
        right_of_curr = -right_of_curr;
    }
    
    for( i = 0; i < maxEdges; i++ )
    {
        int onext_edge = nextEdge( edge );
        int dprev_edge = getEdge( edge, PREV_AROUND_DST );
        
        int right_of_onext = isRightOf( pt, onext_edge );
        int right_of_dprev = isRightOf( pt, dprev_edge );
        
        if( right_of_dprev > 0 )
        {
            if( right_of_onext > 0 || (right_of_onext == 0 && right_of_curr == 0) )
            {
                location = PTLOC_INSIDE;
                break;
            }
            else
            {
                right_of_curr = right_of_onext;
                edge = onext_edge;
            }
        }
        else
        {
            if( right_of_onext > 0 )
            {
                if( right_of_dprev == 0 && right_of_curr == 0 )
                {
                    location = PTLOC_INSIDE;
                    break;
                }
                else
                {
                    right_of_curr = right_of_dprev;
                    edge = dprev_edge;
                }
            }
            else if( right_of_curr == 0 &&
                    isRightOf( vtx[edgeDst(onext_edge)].pt, edge ) >= 0 )
            {
                edge = symEdge( edge );
            }
            else
            {
                right_of_curr = right_of_onext;
                edge = onext_edge;
            }
        }
    }
    
    recentEdge = edge;
    
    if( location == PTLOC_INSIDE )
    {
        Point2f org_pt, dst_pt;
        edgeOrg(edge, &org_pt);
        edgeDst(edge, &dst_pt);
        
        double t1 = fabs( pt.x - org_pt.x );
        t1 += fabs( pt.y - org_pt.y );
        double t2 = fabs( pt.x - dst_pt.x );
        t2 += fabs( pt.y - dst_pt.y );
        double t3 = fabs( org_pt.x - dst_pt.x );
        t3 += fabs( org_pt.y - dst_pt.y );
        
        if( t1 < FLT_EPSILON )
        {
            location = PTLOC_VERTEX;
            vertex = edgeOrg( edge );
            edge = 0;
        }
        else if( t2 < FLT_EPSILON )
        {
            location = PTLOC_VERTEX;
            vertex = edgeDst( edge );
            edge = 0;
        }
        else if( (t1 < t3 || t2 < t3) &&
                fabs( cvTriangleArea( pt, org_pt, dst_pt )) < FLT_EPSILON )
        {
            location = PTLOC_ON_EDGE;
            vertex = 0;
        }
    }
    
    if( location == PTLOC_ERROR )
    {
        edge = 0;
        vertex = 0;
    }
    
    _edge = edge;
    _vertex = vertex;
    
    return location;
}
    
    
inline bool
isPtInCircle3( Point2f pt, Point2f a, Point2f b, Point2f c)
{
    const double eps = FLT_EPSILON*0.125;
    double val = ((double)a.x * a.x + (double)a.y * a.y) * cvTriangleArea( b, c, pt );
    val -= ((double)b.x * b.x + (double)b.y * b.y) * cvTriangleArea( a, c, pt );
    val += ((double)c.x * c.x + (double)c.y * c.y) * cvTriangleArea( a, b, pt );
    val -= ((double)pt.x * pt.x + (double)pt.y * pt.y) * cvTriangleArea( a, b, c );
    
    return val > eps ? 1 : val < -eps ? -1 : 0;
}
    

int Subdiv2D::insert(Point2f pt)
{
    int curr_point = 0, curr_edge = 0, deleted_edge = 0;
    int location = locate( pt, curr_edge, curr_point );
    
    if( location == PTLOC_ERROR )
        CV_Error( CV_StsBadSize, "" );
        
    if( location == PTLOC_OUTSIDE_RECT )
        CV_Error( CV_StsOutOfRange, "" );
    
    if( location == PTLOC_VERTEX )
        return curr_point;
    
    if( location == PTLOC_ON_EDGE )
    {
        deleted_edge = curr_edge;
        recentEdge = curr_edge = getEdge( curr_edge, PREV_AROUND_ORG );
        deleteEdge(deleted_edge);
    }
    else if( location == PTLOC_INSIDE )
        ;
    else
        CV_Error_(CV_StsError, ("Subdiv2D::locate returned invalid location = %d", location) );
        
    assert( curr_edge != 0 );
    validGeometry = false;
    
    curr_point = newPoint(pt, false);
    int base_edge = newEdge();
    int first_point = edgeOrg(curr_edge);
    setEdgePoints(base_edge, first_point, curr_point);
    splice(base_edge, curr_edge);
    
    do
    {
        base_edge = connectEdges( curr_edge, symEdge(base_edge) );
        curr_edge = getEdge(base_edge, PREV_AROUND_ORG);
    }
    while( edgeDst(curr_edge) != first_point );
    
    curr_edge = getEdge( base_edge, PREV_AROUND_ORG );
    
    int i, max_edges = qedges.size()*4;
    
    for( i = 0; i < max_edges; i++ )
    {
        int temp_dst = 0, curr_org = 0, curr_dst = 0;
        int temp_edge = getEdge( curr_edge, PREV_AROUND_ORG );
        
        temp_dst = edgeDst( temp_edge );
        curr_org = edgeOrg( curr_edge );
        curr_dst = edgeDst( curr_edge );
        
        if( isRightOf( vtx[temp_dst].pt, curr_edge ) > 0 &&
            isPtInCircle3( vtx[curr_org].pt, vtx[temp_dst].pt,
                           vtx[curr_dst].pt, vtx[curr_point].pt ) < 0 )
        {
            swapEdges( curr_edge );
            curr_edge = getEdge( curr_edge, PREV_AROUND_ORG );
        }
        else if( curr_org == first_point )
            break;
        else
            curr_edge = getEdge( nextEdge( curr_edge ), PREV_AROUND_LEFT );
    }
    
    return curr_point;
}

void Subdiv2D::insert(const vector<Point2f>& ptvec)
{
    for( size_t i = 0; i < ptvec.size(); i++ )
        insert(ptvec[i]);
}
    
void Subdiv2D::initDelaunay( Rect rect )
{
    float big_coord = 3.f * MAX( rect.width, rect.height );
    float rx = (float)rect.x;
    float ry = (float)rect.y;
    
    vtx.clear();
    qedges.clear();
    
    recentEdge = 0;
    validGeometry = false;
    
    topLeft = Point2f( rx, ry );
    bottomRight = Point2f( rx + rect.width, ry + rect.height );
    
    Point2f ppA( rx + big_coord, ry );
    Point2f ppB( rx, ry + big_coord );
    Point2f ppC( rx - big_coord, ry - big_coord );
    
    vtx.push_back(Vertex());
    qedges.push_back(QuadEdge());
    
    freeQEdge = 0;
    freePoint = 0;
    
    int pA = newPoint(ppA, false);
    int pB = newPoint(ppB, false);
    int pC = newPoint(ppC, false);
    
    int edge_AB = newEdge();
    int edge_BC = newEdge();
    int edge_CA = newEdge();
    
    setEdgePoints( edge_AB, pA, pB );
    setEdgePoints( edge_BC, pB, pC );
    setEdgePoints( edge_CA, pC, pA );
    
    splice( edge_AB, symEdge( edge_CA ));
    splice( edge_BC, symEdge( edge_AB ));
    splice( edge_CA, symEdge( edge_BC ));
    
    recentEdge = edge_AB;
}


void Subdiv2D::clearVoronoi()
{
    size_t i, total = qedges.size();
    
    for( i = 0; i < total; i++ )
        qedges[i].pt[1] = qedges[i].pt[3] = 0;
    
    total = vtx.size();
    for( i = 0; i < total; i++ )
    {
        if( vtx[i].isvirtual() )
            deletePoint((int)i);
    }
    
    validGeometry = false;
}

    
static Point2f computeVoronoiPoint(Point2f org0, Point2f dst0, Point2f org1, Point2f dst1)
{
    double a0 = dst0.x - org0.x;
    double b0 = dst0.y - org0.y;
    double c0 = -0.5*(a0 * (dst0.x + org0.x) + b0 * (dst0.y + org0.y));
    
    double a1 = dst1.x - org1.x;
    double b1 = dst1.y - org1.y;
    double c1 = -0.5*(a1 * (dst1.x + org1.x) + b1 * (dst1.y + org1.y));
    
    double det = a0 * b1 - a1 * b0;
    
    if( det != 0 )
    {
        det = 1. / det;
        return Point2f((float) ((b0 * c1 - b1 * c0) * det),
                       (float) ((a1 * c0 - a0 * c1) * det));
    }
    
    return Point2f(FLT_MAX, FLT_MAX);
}
    

void Subdiv2D::calcVoronoi()
{
    // check if it is already calculated
    if( validGeometry )
        return;
    
    clearVoronoi();
    int i, total = (int)qedges.size();
    
    // loop through all quad-edges, except for the first 3 (#1, #2, #3 - 0 is reserved for "NULL" pointer)
    for( i = 4; i < total; i++ )
    {
        QuadEdge& quadedge = qedges[i];
        
        if( quadedge.isfree() )
            continue;
        
        int edge0 = (int)(i*4);
        Point2f org0, dst0, org1, dst1;
        
        if( !quadedge.pt[3] )
        {
            int edge1 = getEdge( edge0, NEXT_AROUND_LEFT );
            int edge2 = getEdge( edge1, NEXT_AROUND_LEFT );
            
            edgeOrg(edge0, &org0);
            edgeDst(edge0, &dst0);
            edgeOrg(edge1, &org1);
            edgeDst(edge1, &dst1);
            
            Point2f virt_point = computeVoronoiPoint(org0, dst0, org1, dst1);
            
            if( fabs( virt_point.x ) < FLT_MAX * 0.5 &&
                fabs( virt_point.y ) < FLT_MAX * 0.5 )
            {
                quadedge.pt[3] = qedges[edge1 >> 2].pt[3 - (edge1 & 2)] =
                    qedges[edge2 >> 2].pt[3 - (edge2 & 2)] = newPoint(virt_point, true);
            }
        }
        
        if( !quadedge.pt[1] )
        {
            int edge1 = getEdge( edge0, NEXT_AROUND_RIGHT );
            int edge2 = getEdge( edge1, NEXT_AROUND_RIGHT );
            
            edgeOrg(edge0, &org0);
            edgeDst(edge0, &dst0);
            edgeOrg(edge1, &org1);
            edgeDst(edge1, &dst1);
            
            Point2f virt_point = computeVoronoiPoint(org0, dst0, org1, dst1);
            
            if( fabs( virt_point.x ) < FLT_MAX * 0.5 &&
                fabs( virt_point.y ) < FLT_MAX * 0.5 )
            {
                quadedge.pt[1] = qedges[edge1 >> 2].pt[1 + (edge1 & 2)] = 
                qedges[edge2 >> 2].pt[1 + (edge2 & 2)] = newPoint(virt_point, true);
            }
        }
    }
    
    validGeometry = true;
}


static int
isRightOf2( const Point2f& pt, const Point2f& org, const Point2f& diff )
{
    double cw_area = ((double)org.x - pt.x)*diff.y - ((double)org.y - pt.y)*diff.x;
    return (cw_area > 0) - (cw_area < 0);
}


int Subdiv2D::findNearest(Point2f pt, Point2f* nearestPt)
{
    if( !validGeometry )
        calcVoronoi();
    
    int vertex = 0, edge = 0;
    int loc = locate( pt, edge, vertex );
    
    if( loc != PTLOC_ON_EDGE && loc != PTLOC_INSIDE )
        return vertex;
    
    vertex = 0;
    
    Point2f start;
    edgeOrg(edge, &start);
    Point2f diff = pt - start;
    
    edge = rotateEdge(edge, 1);
    
    int i, total = (int)vtx.size();
    
    for( i = 0; i < total; i++ )
    {
        Point2f t;
        
        for(;;)
        {
            CV_Assert( edgeDst(edge, &t) > 0 );
            if( isRightOf2( t, start, diff ) >= 0 )
                break;
            
            edge = getEdge( edge, NEXT_AROUND_LEFT );
        }
        
        for(;;)
        {
            CV_Assert( edgeOrg( edge, &t ) > 0 );
            
            if( isRightOf2( t, start, diff ) < 0 )
                break;
            
            edge = getEdge( edge, PREV_AROUND_LEFT );
        }
        
        Point2f tempDiff;
        edgeDst(edge, &tempDiff);
        edgeOrg(edge, &t);
        tempDiff -= t;
        
        if( isRightOf2( pt, t, tempDiff ) >= 0 )
        {
            vertex = edgeOrg(rotateEdge( edge, 3 ));
            break;
        }
        
        edge = symEdge( edge );
    }
    
    if( nearestPt && vertex > 0 )
        *nearestPt = vtx[vertex].pt;
    
    return vertex;
}

void Subdiv2D::getTriangleList(vector<Vec6f>& triangleList)
{
    vector<bool> processed(vtx.size(), false);
    processed[0] = true;
    
    calcVoronoi();
    triangleList.clear();
    
    for( size_t i = 4; i < qedges.size(); i++ )
    {
        if( qedges[i].isfree() )
            continue;
        int e0 = (int)(i*4), e1 = rotateEdge(e0, 1), e;
        int vidx0 = edgeOrg(e1), vidx1 = edgeDst(e1);
        Point2f a, b, c;
        if( !processed[vidx0] )
        {
            edgeOrg(e0, &a);
            edgeDst(e0, &b);
            e = getEdge(e0, NEXT_AROUND_LEFT);
            edgeDst(e, &c);
            triangleList.push_back(Vec6f(a.x, a.y, b.x, b.y, c.x, c.y));
            processed[vidx0] = true;
        }
        if( !processed[vidx1] )
        {
            edgeDst(e0, &a);
            edgeOrg(e0, &b);
            e = getEdge(e0, PREV_AROUND_RIGHT);
            edgeOrg(e, &c);
            triangleList.push_back(Vec6f(a.x, a.y, b.x, b.y, c.x, c.y));
            processed[vidx1] = true;
        }
    }
}

void Subdiv2D::getVoronoiFacetList(const vector<int>& idx, CV_OUT vector<vector<Point2f> >& facetList)
{
    calcVoronoi();
    facetList.clear();
    
    vector<Point2f> buf;
    
    size_t i, total;
    if( idx.empty() )
        i = 4, total = vtx.size();
    else
        i = 0, total = idx.size();
    
    for( ; i < total; i++ )
    {
        int k = idx.empty() ? (int)i : idx[i];
        
        if( vtx[k].isvirtual() )
            continue;
        int edge = rotateEdge(vtx[k].firstEdge, 1), t = edge;       

        // gather points
        buf.clear();
        do
        {
            buf.push_back(vtx[edgeOrg(t)].pt);
            t = getEdge( t, NEXT_AROUND_LEFT );
        }
        while( t != edge );
        
        facetList.push_back(buf);
    }
}
        
}

using namespace cv;
using namespace std;

static void help()
{
	cout << "\nThis program demostrates iterative construction of\n"
           "delaunay triangulation and voronoi tesselation.\n"
           "It draws a random set of points in an image and then delaunay triangulates them.\n"
           "Usage: \n"
           "./delaunay \n"
           "\nThis program builds the traingulation interactively, you may stop this process by\n"
           "hitting any key.\n";
}


static void draw_subdiv_point( Mat& img, Point2f fp, Scalar color )
{
    circle( img, fp, 3, color, CV_FILLED, 8, 0 );
}

static void draw_subdiv( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color )
{
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    
    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
        line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
        line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
        line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
    }
}

static void locate_point( Mat& img, Subdiv2D& subdiv, Point2f fp, Scalar active_color )
{
    int e0=0, vertex=0;
    
    subdiv.locate(fp, e0, vertex);
    
    if( e0 > 0 )
    {
        int e = e0;
        do
        {
            Point2f org, dst;
            if( subdiv.edgeOrg(e, &org) > 0 && subdiv.edgeDst(e, &dst) > 0 )
                line( img, org, dst, active_color, 3, CV_AA, 0 );
                
            e = subdiv.getEdge(e, Subdiv2D::NEXT_AROUND_LEFT);
        }
        while( e != e0 );
    }
    
    draw_subdiv_point( img, fp, active_color );
}


void paint_voronoi( Mat& img, Subdiv2D& subdiv )
{
    vector<vector<Point2f> > facets;
    subdiv.getVoronoiFacetList(vector<int>(), facets);
    
    vector<Point> ifacet;
    vector<vector<Point> > ifacets(1);
    
    for( size_t i = 0; i < facets.size(); i++ )
    {
        ifacet.resize(facets[i].size());
        for( size_t j = 0; j < facets[i].size(); j++ )
            ifacet[j] = facets[i][j];
    
        Scalar color;
        color[0] = rand() & 256;
        color[1] = rand() & 256;
        color[2] = rand() & 256;
        fillConvexPoly(img, ifacet, color, 8, 0);
        
        ifacets[0] = ifacet;
        polylines(img, ifacets, true, Scalar(), 1, CV_AA, 0);
    }
}


int main( int, char** )
{
    help();
    
    Scalar active_facet_color(0, 0, 255), delaunay_color(255,255,255);
    Rect rect(0, 0, 600, 600);
    
    Subdiv2D subdiv(rect);
    Mat img(rect.size(), CV_8UC3);
    
    img = Scalar::all(0);
    string win = "Delaunay Demo";
    imshow(win, img);
    
    for( int i = 0; i < 200; i++ )
    {
        Point2f fp( (float)(rand()%(rect.width-10)+5),
                    (float)(rand()%(rect.height-10)+5));
        
        locate_point( img, subdiv, fp, active_facet_color );
        imshow( win, img );
        
        if( waitKey( 100 ) >= 0 )
            break;
        
        subdiv.insert(fp);
        img = Scalar::all(0);
        draw_subdiv( img, subdiv, delaunay_color );
        imshow( win, img );
        
        if( waitKey( 100 ) >= 0 )
            break;
    }
    
    img = Scalar::all(0);
    paint_voronoi( img, subdiv );
    imshow( win, img );
        
    waitKey(0);
    
    return 0;
}
