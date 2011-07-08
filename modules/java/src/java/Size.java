package org.opencv;

//javadoc:Size_
public class Size {

	public int width, height;

	public Size(int width, int height) {
		this.width = width;
		this.height = height;
	}
	
	public Size() {
		this(0, 0);
	}
	
	public Size(Point p) {
		width = (int) p.x;
		height = (int) p.y;
	}
	
	public double area() {
		return width * height;
	}

	public Size clone() {
		return new Size(width, height);
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
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (!(obj instanceof Size)) return false;
		Size it = (Size) obj;
		return width == it.width && height == it.height;
	}

}
