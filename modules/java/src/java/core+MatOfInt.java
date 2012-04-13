package org.opencv.core;

import java.util.Arrays;
import java.util.List;


public class MatOfInt extends Mat {
	// 32SC(x)
	private static final int _depth = CvType.CV_32S;
	private final int _channels;

    public MatOfInt(int channels) {
        super();
        _channels = channels;
    }

    public MatOfInt() {
        this(1);
    }

    public MatOfInt(int channels, long addr) {
        super(addr);
        _channels = channels;
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfInt(int channels, Mat m) {
    	super(m, Range.all());
        _channels = channels;
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfInt(int channels, int...a) {
        super();
        _channels = channels;
        fromArray(a);
    }

    public void alloc(int elemNumber) {
        if(elemNumber>0)
            super.create(elemNumber, 1, CvType.makeType(_depth, _channels));
    }

    public void fromArray(int...a) {
        if(a==null || a.length==0)
            return;
        int num = a.length / _channels;
        alloc(num);
        put(0, 0, a); //TODO: check ret val!
    }
    
    public int[] toArray() {
        int num = (int) total();
        int[] a = new int[num * _channels];
        if(num == 0)
            return a;
        get(0, 0, a); //TODO: check ret val!
        return a;
    }

    public void fromList(List<Integer> lb) {
    	if(lb==null || lb.size()==0)
    		return;
    	Integer ab[] = lb.toArray(null);
    	int a[] = new int[ab.length];
    	for(int i=0; i<ab.length; i++)
    		a[i] = ab[i];
    	fromArray(a);
    }
    
    public List<Integer> toList() {
    	int[] a = toArray();
    	Integer ab[] = new Integer[a.length];
    	for(int i=0; i<a.length; i++)
    		ab[i] = a[i];
    	return Arrays.asList(ab); 
    }
}
