package org.opencv.core;

//javadoc:Scalar_
public class Scalar {

    public double val[];

    public Scalar(double v0, double v1, double v2, double v3) {
        val = new double[] { v0, v1, v2, v3 };
    }

    public Scalar(double v0, double v1, double v2) {
        val = new double[] { v0, v1, v2, 0 };
    }

    public Scalar(double v0, double v1) {
        val = new double[] { v0, v1, 0, 0 };
    }

    public Scalar(double v0) {
        val = new double[] { v0, 0, 0, 0 };
    }

    public Scalar(double[] vals) {
        if (vals != null && vals.length == 4)
            val = vals.clone();
        else {
            val = new double[4];
            set(vals);
        }
    }

    public void set(double[] vals) {
        if (vals != null) {
            val[0] = vals.length > 0 ? vals[0] : 0;
            val[1] = vals.length > 1 ? vals[1] : 0;
            val[2] = vals.length > 2 ? vals[2] : 0;
            val[3] = vals.length > 3 ? vals[3] : 0;
        } else
            val[0] = val[1] = val[2] = val[3] = 0;
    }

    public static Scalar all(double v) {
        return new Scalar(v, v, v, v);
    }

    public Scalar clone() {
        return new Scalar(val);
    }

    public Scalar mul(Scalar it, double scale) {
        return new Scalar(val[0] * it.val[0] * scale, val[1] * it.val[1] * scale,
                val[2] * it.val[2] * scale, val[3] * it.val[3] * scale);
    }

    public Scalar mul(Scalar it) {
        return mul(it, 1);
    }

    public Scalar conj() {
        return new Scalar(val[0], -val[1], -val[2], -val[3]);
    }

    public boolean isReal() {
        return val[1] == 0 && val[2] == 0 && val[3] == 0;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + java.util.Arrays.hashCode(val);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Scalar)) return false;
        Scalar it = (Scalar) obj;
        if (!java.util.Arrays.equals(val, it.val)) return false;
        return true;
    }

    @Override
    public String toString() {
        return "[" + val[0] + ", " + val[1] + ", " + val[2] + ", " + val[3] + "]";
    }

}
