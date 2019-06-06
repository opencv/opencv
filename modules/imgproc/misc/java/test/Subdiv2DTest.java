package org.opencv.test.imgproc;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.opencv.core.MatOfFloat6;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Subdiv2D;
import org.opencv.test.NotYetImplemented;
import org.opencv.test.OpenCVTestCase;

public class Subdiv2DTest extends OpenCVTestCase {

    @Test
    @NotYetImplemented
    public void testEdgeDstInt() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testEdgeDstIntPoint() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testEdgeOrgInt() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testEdgeOrgIntPoint() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testFindNearestPoint() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testFindNearestPointPoint() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testGetEdge() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testGetEdgeList() {
        fail("Not yet implemented");
    }

    @Test
    public void testGetTriangleList() {
        Subdiv2D s2d = new Subdiv2D( new Rect(0, 0, 50, 50) );
        s2d.insert( new Point(10, 10) );
        s2d.insert( new Point(20, 10) );
        s2d.insert( new Point(20, 20) );
        s2d.insert( new Point(10, 20) );
        MatOfFloat6 triangles = new MatOfFloat6();
        s2d.getTriangleList(triangles);
        assertEquals(2, triangles.rows());
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

    @Test
    @NotYetImplemented
    public void testGetVertexInt() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testGetVertexIntIntArray() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testGetVoronoiFacetList() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testInitDelaunay() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testInsertListOfPoint() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testInsertPoint() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testLocate() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testNextEdge() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testRotateEdge() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testSubdiv2D() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testSubdiv2DRect() {
        fail("Not yet implemented");
    }

    @Test
    @NotYetImplemented
    public void testSymEdge() {
        fail("Not yet implemented");
    }

}
