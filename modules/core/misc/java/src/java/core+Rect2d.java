package org.opencv.core;

//javadoc:Rect2d_
public class Rect2d {

    public double x, y, width, height;

    public Rect2d(double x, double y, double width, double height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public Rect2d() {
        this(0, 0, 0, 0);
    }

    public Rect2d(Point p1, Point p2) {
        x = (double) (p1.x < p2.x ? p1.x : p2.x);
        y = (double) (p1.y < p2.y ? p1.y : p2.y);
        width = (double) (p1.x > p2.x ? p1.x : p2.x) - x;
        height = (double) (p1.y > p2.y ? p1.y : p2.y) - y;
    }

    public Rect2d(Point p, Size s) {
        this((double) p.x, (double) p.y, (double) s.width, (double) s.height);
    }

    public Rect2d(double[] vals) {
        set(vals);
    }

    public void set(double[] vals) {
        if (vals != null) {
            x = vals.length > 0 ? (double) vals[0] : 0;
            y = vals.length > 1 ? (double) vals[1] : 0;
            width = vals.length > 2 ? (double) vals[2] : 0;
            height = vals.length > 3 ? (double) vals[3] : 0;
        } else {
            x = 0;
            y = 0;
            width = 0;
            height = 0;
        }
    }

    public Rect2d clone() {
        return new Rect2d(x, y, width, height);
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
        if (!(obj instanceof Rect2d)) return false;
        Rect2d it = (Rect2d) obj;
        return x == it.x && y == it.y && width == it.width && height == it.height;
    }

    @Override
    public String toString() {
        return "{" + x + ", " + y + ", " + width + "x" + height + "}";
    }
}
