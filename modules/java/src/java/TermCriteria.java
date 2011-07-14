package org.opencv;

//javadoc:TermCriteria
public class TermCriteria {

    public int type;
    public int maxCount;
    public double epsilon;

    public TermCriteria(int t, int c, double e) {
        this.type = t;
        this.maxCount = c;
        this.epsilon = e;
    }

    public TermCriteria() {
        this(0, 0, 0.0);
    }

    public TermCriteria(double[] vals) {
    	this();
    	if(vals!=null) {
    		type      = vals.length>0 ? (int)vals[0]    : 0;
    		maxCount  = vals.length>1 ? (int)vals[1]    : 0;
    		epsilon   = vals.length>2 ? (double)vals[2] : 0;
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
        return type == it.type && maxCount == it.maxCount && epsilon== it.epsilon;
    }

    @Override
    public String toString() {
        if (this == null) return "null";
        return "{ type: " + type + ", maxCount: " + maxCount + ", epsilon: " + epsilon + "}";
    }
}
