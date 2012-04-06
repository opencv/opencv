package org.opencv.core;

import java.util.Arrays;
import java.util.List;

public class MatOfFloat extends Mat {
	// 32FC(x)
	private static final int _depth = CvType.CV_32F;
	private final int _channels;

    public MatOfFloat(int channels) {
        super();
        _channels = channels;
    }

    public MatOfFloat() {
        this(1);
    }

    public MatOfFloat(int channels, long addr) {
        super(addr);
        _channels = channels;
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfFloat(int channels, Mat m) {
    	super(m, Range.all());
        _channels = channels;
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfFloat(int channels, float...a) {
        super();
        _channels = channels;
        fromArray(a);
    }

    public void alloc(int elemNumber) {
        if(elemNumber>0)
            super.create(elemNumber, 1, CvType.makeType(_depth, _channels));
    }

    public void fromArray(float...a) {
        if(a==null || a.length==0)
            return;
        int num = a.length / _channels;
        alloc(num);
        put(0, 0, a); //TODO: check ret val!
    }
    
    public float[] toArray() {
        int num = (int) total();
        float[] a = new float[num * _channels];
        if(num == 0)
            return a;
        get(0, 0, a); //TODO: check ret val!
        return a;
    }

    public void fromList(List<Float> lb) {
    	if(lb==null || lb.size()==0)
    		return;
    	Float ab[] = lb.toArray(null);
    	float a[] = new float[ab.length];
    	for(int i=0; i<ab.length; i++)
    		a[i] = ab[i];
    	fromArray(a);
    }
    
    public List<Float> toList() {
    	float[] a = toArray();
    	Float ab[] = new Float[a.length];
    	for(int i=0; i<a.length; i++)
    		ab[i] = a[i];
    	return Arrays.asList(ab); 
    }
}
