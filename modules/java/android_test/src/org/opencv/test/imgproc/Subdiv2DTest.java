package org.opencv.test.imgproc;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Subdiv2D;
import org.opencv.test.OpenCVTestCase;

public class Subdiv2DTest extends OpenCVTestCase {

    protected void setUp() throws Exception {
        super.setUp();
    }

    public void testEdgeDstInt() {
        fail("Not yet implemented");
    }

    public void testEdgeDstIntPoint() {
        fail("Not yet implemented");
    }

    public void testEdgeOrgInt() {
        fail("Not yet implemented");
    }

    public void testEdgeOrgIntPoint() {
        fail("Not yet implemented");
    }

    public void testFindNearestPoint() {
        fail("Not yet implemented");
    }

    public void testFindNearestPointPoint() {
        fail("Not yet implemented");
    }

    public void testGetEdge() {
        fail("Not yet implemented");
    }

    public void testGetEdgeList() {
        fail("Not yet implemented");
    }

    public void testGetTriangleList() {
    	Subdiv2D s2d = new Subdiv2D( new Rect(0, 0, 50, 50) );
    	s2d.insert( new Point(10, 10) );
    	s2d.insert( new Point(20, 10) );
    	s2d.insert( new Point(20, 20) );
    	s2d.insert( new Point(10, 20) );
    	Mat triangles = new Mat();
    	s2d.getTriangleList(triangles);
    	assertEquals(10, triangles.rows());
    	/*
    	int cnt = triangles.rows();
    	float buff[] = new float[cnt*6];
    	triangles.get(0, 0, buff);
    	for(int i=0; i<cnt; i++)
    		Log.d("*****", "["+i+"]: " + // (a.x, a.y) -> (b.x, b.y) -> (c.x, c.y)
    				"("+buff[6*i+0]+","+buff[6*i+1]+")" + "->" +
    				"("+buff[6*i+2]+","+buff[6*i+3]+")" + "->" +
    				"("+buff[6*i+4]+","+buff[6*i+5]+")"
    				);
    	*/
    }

    public void testGetVertexInt() {
        fail("Not yet implemented");
    }

    public void testGetVertexIntIntArray() {
        fail("Not yet implemented");
    }

    public void testGetVoronoiFacetList() {
        fail("Not yet implemented");
    }

    public void testInitDelaunay() {
        fail("Not yet implemented");
    }

    public void testInsertListOfPoint() {
        fail("Not yet implemented");
    }

    public void testInsertPoint() {
        fail("Not yet implemented");
    }

    public void testLocate() {
        fail("Not yet implemented");
    }

    public void testNextEdge() {
        fail("Not yet implemented");
    }

    public void testRotateEdge() {
        fail("Not yet implemented");
    }

    public void testSubdiv2D() {
        fail("Not yet implemented");
    }

    public void testSubdiv2DRect() {
        fail("Not yet implemented");
    }

    public void testSymEdge() {
        fail("Not yet implemented");
    }

}
