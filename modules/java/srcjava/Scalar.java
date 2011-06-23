package org.opencv;

public class Scalar {
	
	public double v0, v1, v2, v3;

	public Scalar(double v0, double v1, double v2, double v3) {
		this.v0 = v0;
		this.v1 = v1;
		this.v2 = v2;
		this.v3 = v3;
	}

	public Scalar(double v0, double v1, double v2) {
		this(v0, v1, v2, 0);
	}
	
	public Scalar(double v0, double v1) {
		this(v0, v1, 0, 0);
	}
	
	public Scalar(double v0) {
		this(v0, 0, 0, 0);
	}
	
	public static Scalar all(double v) {
		return new Scalar(v, v, v, v);
	}
	
	public Scalar clone() {
		return new Scalar(v0, v1, v2, v3);
	}

	public Scalar mul(Scalar it, double scale) {
		return new Scalar( v0 * it.v0 * scale, v1 * it.v1 * scale, 
						   v2 * it.v2 * scale, v3 * it.v3 * scale );
	}

	public Scalar mul(Scalar it) {
		return mul(it, 1);
	}
	public Scalar conj() {
		return new Scalar(v0, -v1, -v2, -v3);
	}

	public boolean isReal() {
		return v1 == 0 && v2 == 0 && v3 == 0;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(v0);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(v1);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(v2);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(v3);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (!(obj instanceof Scalar)) return false;
		Scalar it = (Scalar) obj;
		return v0 == it.v0 && v1 == it.v1 && v2 == it.v2 && v3 == it.v3;
	}
}
