package org.opencv.core;

import java.util.Arrays;
import java.util.List;


public class MatOfInt4 extends Mat {
    // 32SC4
    private static final int _depth = CvType.CV_32S;
    private static final int _channels = 4;

    public MatOfInt4() {
        super();
    }

    protected MatOfInt4(long addr) {
        super(addr);
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public static MatOfInt4 fromNativeAddr(long addr) {
        return new MatOfInt4(addr);
    }

    public MatOfInt4(Mat m) {
        super(m, Range.all());
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfInt4(int...a) {
        super();
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
        int num = checkVector(_channels, _depth);
        if(num < 0)
        	throw new RuntimeException("Native Mat has unexpected type or size: " + toString());
        int[] a = new int[num * _channels];
        if(num == 0)
            return a;
        get(0, 0, a); //TODO: check ret val!
        return a;
    }

    public void fromList(List<Integer> lb) {
        if(lb==null || lb.size()==0)
            return;
        Integer ab[] = lb.toArray(new Integer[0]);
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
