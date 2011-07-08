package org.opencv;

//javadoc:Rect_
public class Rect {
	
	int x, y, width, height;

	public Rect(int x, int y, int width, int height) {
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
	}

	public Rect() {
		this(0, 0, 0, 0);
	}
	
	public Rect(Point p1, Point p2) {
	    x = (int) (p1.x < p2.x ? p1.x : p2.x); 
	    y = (int) (p1.y < p2.y ? p1.y : p2.y);
	    width = (int) (p1.x > p2.x ? p1.x : p2.x) - x; 
	    height = (int) (p1.y > p2.y ? p1.y : p2.y) - y;
	}

	public Rect(Point p, Size s) {
		this((int)p.x, (int)p.y, s.width, s.height);
	}
	
	public Rect clone() {
		return new Rect(x, y, width, height);
	}
	
	public Point tl() {
		return new Point(x, y);
	}

	public Point br() {
		return new Point(x + width, y + height);
	}

	public Size size() {
		return new Size(width, height);
	}

	public double area() {
		return width * height;
	}

	public boolean contains(Point p) {
		return x <= p.x && p.x < x + width && y <= p.y && p.y < y + height;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(height);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(width);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(x);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(y);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (!(obj instanceof Rect)) return false;
		Rect it = (Rect) obj;
		return x == it.x && y == it.y && width == it.width && height == it.height;
	}
}
