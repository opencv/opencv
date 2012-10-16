package org.opencv.core;

//javadoc:TermCriteria
public class TermCriteria {

    /**
     * The maximum number of iterations or elements to compute
     */
    public static final int COUNT = 1;
    /**
     * The maximum number of iterations or elements to compute
     */
    public static final int MAX_ITER = COUNT;
    /**
     * The desired accuracy threshold or change in parameters at which the iterative algorithm is terminated.
     */
    public static final int EPS = 2;

    public int type;
    public int maxCount;
    public double epsilon;

    /**
     * Termination criteria for iterative algorithms.
     *
     * @param type
     *            the type of termination criteria: COUNT, EPS or COUNT + EPS.
     * @param maxCount
     *            the maximum number of iterations/elements.
     * @param epsilon
     *            the desired accuracy.
     */
    public TermCriteria(int type, int maxCount, double epsilon) {
        this.type = type;
        this.maxCount = maxCount;
        this.epsilon = epsilon;
    }

    /**
     * Termination criteria for iterative algorithms.
     */
    public TermCriteria() {
        this(0, 0, 0.0);
    }

    public TermCriteria(double[] vals) {
        set(vals);
    }

    public void set(double[] vals) {
        if (vals != null) {
            type = vals.length > 0 ? (int) vals[0] : 0;
            maxCount = vals.length > 1 ? (int) vals[1] : 0;
            epsilon = vals.length > 2 ? (double) vals[2] : 0;
        } else {
            type = 0;
            maxCount = 0;
            epsilon = 0;
        }
    }

    public TermCriteria clone() {
        return new TermCriteria(type, maxCount, epsilon);
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        long temp;
        temp = Double.doubleToLongBits(type);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(maxCount);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(epsilon);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof TermCriteria)) return false;
        TermCriteria it = (TermCriteria) obj;
        return type == it.type && maxCount == it.maxCount && epsilon == it.epsilon;
    }

    @Override
    public String toString() {
        if (this == null) return "null";
        return "{ type: " + type + ", maxCount: " + maxCount + ", epsilon: " + epsilon + "}";
    }
}
