package org.opencv.test.core;

import org.opencv.Point;
import org.opencv.Rect;
import org.opencv.RotatedRect;
import org.opencv.Size;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

public class RotatedRectTest extends OpenCVTestCase {
	
	public void testBoundingRect() {
		Point center = new Point(matSize/2, matSize/2);
		double length1 = matSize/4;
		Size size = new Size(length1, length1);
		double angle = 45;

		RotatedRect rr = new RotatedRect(center, size, angle);
		
		Rect r = rr.boundingRect();
		
		OpenCVTestRunner.Log("testBoundingRect: r="+r.toString());
		OpenCVTestRunner.Log("testBoundingRect: center.x + length1*Math.sqrt(2)/2="+ (center.x + length1*Math.sqrt(2)/2));
		OpenCVTestRunner.Log("testBoundingRect: length1*Math.sqrt(2)="+ (length1*Math.sqrt(2)));
		
		assertTrue( 
				(r.x == Math.floor(center.x - length1*Math.sqrt(2)/2)) 
				&& 
				(r.y == Math.floor(center.y - length1*Math.sqrt(2)/2)));
		
		assertTrue( 
				(r.br().x >= Math.ceil(center.x + length1*Math.sqrt(2)/2)) 
				&& 
				(r.br().y >= Math.ceil(center.y + length1*Math.sqrt(2)/2)));
		
		assertTrue( 
				(r.br().x - Math.ceil(center.x + length1*Math.sqrt(2)/2)) <= 1 
				&& 
				(r.br().y - Math.ceil(center.y + length1*Math.sqrt(2)/2)) <= 1);
	}

	
	public void testClone() {
		Point center = new Point(matSize/2, matSize/2);
		Size size = new Size(matSize/4, matSize/2);
		double angle = 40;
		
		RotatedRect rr1 = new RotatedRect(center, size, angle);
		RotatedRect rr1c = rr1.clone();
		
		assertTrue(rr1c != null);
		assertTrue(rr1.center.equals(rr1c.center));
		assertTrue(rr1.size.equals(rr1c.size));
		assertTrue(rr1.angle == rr1c.angle);
	}

	public void testEqualsObject() {
		Point center = new Point(matSize/2, matSize/2);
		Size size = new Size(matSize/4, matSize/2);
		double angle = 40;
		Point center2 = new Point(matSize/3, matSize/1.5);
		Size size2 = new Size(matSize/2, matSize/4);
		double angle2 = 0;

		RotatedRect rr1 = new RotatedRect(center, size, angle);
		RotatedRect rr2 = new RotatedRect(center2, size2, angle2);
		RotatedRect rr1c = rr1.clone();
		RotatedRect rr3 = rr2.clone();
		RotatedRect rr11=rr1;
		rr3.angle=10;
		
		assertTrue(rr1.equals(rr11));
		assertTrue(!rr1.equals(rr2));
		assertTrue(rr1.equals(rr1c));
		assertTrue(!rr2.equals(rr3));
		
		rr1c.center.x+=1;
		assertTrue(!rr1.equals(rr1c));
		
		rr1c.center.x-=1;
		assertTrue(rr1.equals(rr1c));
		
		rr1c.size.width+=1;
		assertTrue(!rr1.equals(rr1c));
		
		assertTrue(! rr1.equals(size));
	}

	public void testPoints() {
		Point center = new Point(matSize/2, matSize/2);
		Size size = new Size(matSize/4, matSize/2);
		double angle = 40;
		RotatedRect rr = new RotatedRect(center, size, angle);
		Point p[] = new Point[4];
		
		rr.points(p);
		
		boolean is_p0_irrational = (100*p[0].x!=(int)(100*p[0].x)) && (100*p[0].y!=(int)(100*p[0].y));
		boolean is_p1_irrational = (100*p[1].x!=(int)(100*p[1].x)) && (100*p[1].y!=(int)(100*p[1].y));
		boolean is_p2_irrational = (100*p[2].x!=(int)(100*p[2].x)) && (100*p[2].y!=(int)(100*p[2].y));
		boolean is_p3_irrational = (100*p[3].x!=(int)(100*p[3].x)) && (100*p[3].y!=(int)(100*p[3].y));
		
		assertTrue(is_p0_irrational && is_p1_irrational && is_p2_irrational && is_p3_irrational);
		
		assertTrue("Symmetric points 0 and 2", 
				Math.abs((p[0].x + p[2].x)/2 - center.x) + Math.abs((p[0].y + p[2].y)/2 - center.y) < 0.001);
		
		assertTrue("Symmetric points 1 and 3", 
				Math.abs((p[1].x + p[3].x)/2 - center.x) + Math.abs((p[1].y + p[3].y)/2 - center.y) < 0.001);
		
		assertTrue("Orthogonal vectors 01 and 12", 
				Math.abs( (p[1].x - p[0].x) * (p[2].x - p[1].x) + (p[1].y - p[0].y) * (p[2].y - p[1].y) ) < 0.001);
		
		assertTrue("Orthogonal vectors 12 and 23", 
				Math.abs( (p[2].x - p[1].x) * (p[3].x - p[2].x) + (p[2].y - p[1].y) * (p[3].y - p[2].y) ) < 0.001);

		assertTrue("Orthogonal vectors 23 and 30", 
				Math.abs( (p[3].x - p[2].x) * (p[0].x - p[3].x) + (p[3].y - p[2].y) * (p[0].y - p[3].y) ) < 0.001);

		assertTrue("Orthogonal vectors 30 and 01", 
				Math.abs( (p[0].x - p[3].x) * (p[1].x - p[0].x) + (p[0].y - p[3].y) * (p[1].y - p[0].y) ) < 0.001);
		
		assertTrue("Length of the vector 01", 
				Math.abs( 
						(p[1].x - p[0].x) * (p[1].x - p[0].x)  + (p[1].y - p[0].y)*(p[1].y - p[0].y)
						- 
						size.height * size.height
						) < 0.001);
		
		assertTrue("Length of the vector 21", 
				Math.abs( 
						(p[1].x - p[2].x) * (p[1].x - p[2].x)  + (p[1].y - p[2].y)*(p[1].y - p[2].y)
						- 
						size.width * size.width
						) < 0.001);

		assertTrue("Angle of the vector 21 with the axes", 
				Math.abs( 
						(p[2].x - p[1].x) / size.width
						- 
						Math.cos(angle * Math.PI / 180)
						) < 0.001);

	}

	public void testRotatedRect() {
		RotatedRect rr = new RotatedRect();
		assertTrue(rr != null);
	}

	public void testRotatedRectPointSizeDouble() {
		RotatedRect rr = new RotatedRect(new Point(matSize/2, matSize/2), new Size(matSize/4, matSize/2), 45);
		assertTrue(rr != null);
	}

}
