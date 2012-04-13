package org.opencv.core;

import java.util.Arrays;
import java.util.List;

public class MatOfDouble extends Mat {
	// 64FC(x)
	private static final int _depth = CvType.CV_64F;
	private final int _channels;

    public MatOfDouble(int channels) {
        super();
        _channels = channels;
    }

    public MatOfDouble() {
        this(1);
    }

    public MatOfDouble(int channels, long addr) {
        super(addr);
        _channels = channels;
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfDouble(int channels, Mat m) {
    	super(m, Range.all());
        _channels = channels;
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfDouble(int channels, double...a) {
        super();
        _channels = channels;
        fromArray(a);
    }

    public void alloc(int elemNumber) {
        if(elemNumber>0)
            super.create(elemNumber, 1, CvType.makeType(_depth, _channels));
    }

    public void fromArray(double...a) {
        if(a==null || a.length==0)
            return;
        int num = a.length / _channels;
        alloc(num);
        put(0, 0, a); //TODO: check ret val!
    }
    
    public double[] toArray() {
        int num = (int) total();
        double[] a = new double[num * _channels];
        if(num == 0)
            return a;
        get(0, 0, a); //TODO: check ret val!
        return a;
    }

    public void fromList(List<Double> lb) {
    	if(lb==null || lb.size()==0)
    		return;
    	Double ab[] = lb.toArray(null);
    	double a[] = new double[ab.length];
    	for(int i=0; i<ab.length; i++)
    		a[i] = ab[i];
    	fromArray(a);
    }
    
    public List<Double> toList() {
    	double[] a = toArray();
    	Double ab[] = new Double[a.length];
    	for(int i=0; i<a.length; i++)
    		ab[i] = a[i];
    	return Arrays.asList(ab); 
    }
}
