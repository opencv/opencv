package org.opencv.test.core;

import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.test.OpenCVTestCase;

public class RotatedRectTest extends OpenCVTestCase {
	
	private Point center;
	private Size size;
	private double angle;
	
    @Override
    protected void setUp() throws Exception {
        super.setUp();
        
        center = new Point(matSize/2, matSize/2);
        size = new Size(matSize/4, matSize/2);
        angle = 40;
    }
    
	public void test_1() {
		super.test_1("core.RotatedRect");
	}
	
	public void testBoundingRect() {
		assertEquals(size.height, size.width);
		double length = size.height;
		
		angle = 45;
		RotatedRect rr = new RotatedRect(center, size, angle);		
		
		Rect r = rr.boundingRect();		
		double halfDiagonal = length * Math.sqrt(2)/2;
		
		assertTrue((r.x == Math.floor(center.x - halfDiagonal)) && 
				   (r.y == Math.floor(center.y - halfDiagonal)));
		
		assertTrue((r.br().x >= Math.ceil(center.x + halfDiagonal)) && 
				   (r.br().y >= Math.ceil(center.y + halfDiagonal)));
		
		assertTrue((r.br().x - Math.ceil(center.x + halfDiagonal)) <= 1 && 
				   (r.br().y - Math.ceil(center.y + halfDiagonal)) <= 1);
	}
	
	public void testClone() {
		RotatedRect rrect = new RotatedRect(center, size, angle);
		RotatedRect clone = rrect.clone();
		
		assertTrue(clone != null);
		assertTrue(rrect.center.equals(clone.center));
		assertTrue(rrect.size.equals(clone.size));
		assertTrue(rrect.angle == clone.angle);
	}

	public void testEqualsObject() {
		Point center2 = new Point(matSize/3, matSize/1.5);
		Size size2 = new Size(matSize/2, matSize/4);
		double angle2 = 0;

		RotatedRect rrect1 = new RotatedRect(center, size, angle);
		RotatedRect rrect2 = new RotatedRect(center2, size2, angle2);
		RotatedRect rrect3 = rrect1;
		RotatedRect clone1 = rrect1.clone();
		RotatedRect clone2 = rrect2.clone();

		assertTrue(rrect1.equals(rrect3));
		assertTrue(!rrect1.equals(rrect2));

		assertTrue(rrect2.equals(clone2));
		clone2.angle = 10;
		assertTrue(!rrect2.equals(clone2));
		
		assertTrue(rrect1.equals(clone1));
		
		clone1.center.x += 1;
		assertTrue(!rrect1.equals(clone1));		

		clone1.center.x -= 1;
		assertTrue(rrect1.equals(clone1));
		
		clone1.size.width += 1;
		assertTrue(!rrect1.equals(clone1));
		
		assertTrue(!rrect1.equals(size));
	}

	public void testPoints() {
		RotatedRect rrect = new RotatedRect(center, size, angle);
		
		Point p[] = new Point[4];		
		rrect.points(p);
		
		boolean is_p0_irrational = (100 * p[0].x != (int)(100 * p[0].x)) && (100 * p[0].y != (int)(100 * p[0].y));
		boolean is_p1_irrational = (100 * p[1].x != (int)(100 * p[1].x)) && (100 * p[1].y != (int)(100 * p[1].y));
		boolean is_p2_irrational = (100 * p[2].x != (int)(100 * p[2].x)) && (100 * p[2].y != (int)(100 * p[2].y));
		boolean is_p3_irrational = (100 * p[3].x != (int)(100 * p[3].x)) && (100 * p[3].y != (int)(100 * p[3].y));
		
		assertTrue(is_p0_irrational && is_p1_irrational && is_p2_irrational && is_p3_irrational);
		
		assertTrue("Symmetric points 0 and 2", 
				Math.abs((p[0].x + p[2].x)/2 - center.x) + Math.abs((p[0].y + p[2].y)/2 - center.y) < EPS);
		
		assertTrue("Symmetric points 1 and 3", 
				Math.abs((p[1].x + p[3].x)/2 - center.x) + Math.abs((p[1].y + p[3].y)/2 - center.y) < EPS);
		
		assertTrue("Orthogonal vectors 01 and 12", 
				Math.abs((p[1].x - p[0].x) * (p[2].x - p[1].x) + (p[1].y - p[0].y) * (p[2].y - p[1].y) ) < EPS);
		
		assertTrue("Orthogonal vectors 12 and 23", 
				Math.abs((p[2].x - p[1].x) * (p[3].x - p[2].x) + (p[2].y - p[1].y) * (p[3].y - p[2].y) ) < EPS);

		assertTrue("Orthogonal vectors 23 and 30", 
				Math.abs((p[3].x - p[2].x) * (p[0].x - p[3].x) + (p[3].y - p[2].y) * (p[0].y - p[3].y) ) < EPS);

		assertTrue("Orthogonal vectors 30 and 01", 
				Math.abs((p[0].x - p[3].x) * (p[1].x - p[0].x) + (p[0].y - p[3].y) * (p[1].y - p[0].y) ) < EPS);
		
		assertTrue("Length of the vector 01", 
				Math.abs((p[1].x - p[0].x) * (p[1].x - p[0].x) + (p[1].y - p[0].y) * (p[1].y - p[0].y) - size.height * size.height) < EPS);
		
		assertTrue("Length of the vector 21", 
				Math.abs((p[1].x - p[2].x) * (p[1].x - p[2].x) + (p[1].y - p[2].y) * (p[1].y - p[2].y) - size.width * size.width ) < EPS);

		assertTrue("Angle of the vector 21 with the axes", 
				Math.abs((p[2].x - p[1].x) / size.width - Math.cos(angle * Math.PI / 180)) < EPS);

	}

	public void testRotatedRect() {
		RotatedRect rr = new RotatedRect();
		assertTrue(rr != null);
	}

	public void testRotatedRectPointSizeDouble() {
		RotatedRect rr = new RotatedRect(center, size, 40);
		assertTrue(rr != null);
	}

}
