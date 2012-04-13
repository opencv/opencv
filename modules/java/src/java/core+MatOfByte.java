package org.opencv.core;

import java.util.Arrays;
import java.util.List;

public class MatOfByte extends Mat {
	// 8UC(x)
	private static final int _depth = CvType.CV_8U;
	private final int _channels;

    public MatOfByte(int channels) {
        super();
        _channels = channels;
    }

    public MatOfByte() {
        this(1);
    }

    public MatOfByte(int channels, long addr) {
        super(addr);
        _channels = channels;
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfByte(int channels, Mat m) {
    	super(m, Range.all());
        _channels = channels;
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfByte(int channels, byte...a) {
        super();
        _channels = channels;
        fromArray(a);
    }

    public void alloc(int elemNumber) {
        if(elemNumber>0)
            super.create(elemNumber, 1, CvType.makeType(_depth, _channels));
    }

    public void fromArray(byte...a) {
        if(a==null || a.length==0)
            return;
        int num = a.length / _channels;
        alloc(num);
        put(0, 0, a); //TODO: check ret val!
    }
    
    public byte[] toArray() {
        int num = (int) total();
        byte[] a = new byte[num * _channels];
        if(num == 0)
            return a;
        get(0, 0, a); //TODO: check ret val!
        return a;
    }

    public void fromList(List<Byte> lb) {
    	if(lb==null || lb.size()==0)
    		return;
    	Byte ab[] = lb.toArray(null);
    	byte a[] = new byte[ab.length];
    	for(int i=0; i<ab.length; i++)
    		a[i] = ab[i];
    	fromArray(a);
    }
    
    public List<Byte> toList() {
    	byte[] a = toArray();
    	Byte ab[] = new Byte[a.length];
    	for(int i=0; i<a.length; i++)
    		ab[i] = a[i];
    	return Arrays.asList(ab); 
    }
}
