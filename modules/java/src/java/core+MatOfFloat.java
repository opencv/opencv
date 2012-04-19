package org.opencv.core;

import java.util.Arrays;
import java.util.List;

public class MatOfFloat extends Mat {
    // 32FC1
    private static final int _depth = CvType.CV_32F;
    private static final int _channels = 1;

    public MatOfFloat() {
        super();
    }

    protected MatOfFloat(long addr) {
        super(addr);
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }
    
    public static MatOfFloat fromNativeAddr(long addr) {
		return new MatOfFloat(addr);
	}

    public MatOfFloat(Mat m) {
        super(m, Range.all());
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfFloat(float...a) {
        super();
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
        int num = checkVector(_channels, _depth);
        if(num < 0)
        	throw new RuntimeException("Native Mat has unexpected type or size: " + toString());
        float[] a = new float[num * _channels];
        if(num == 0)
            return a;
        get(0, 0, a); //TODO: check ret val!
        return a;
    }

    public void fromList(List<Float> lb) {
        if(lb==null || lb.size()==0)
            return;
        Float ab[] = lb.toArray(new Float[0]);
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
