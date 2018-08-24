package org.opencv.core;

import java.util.Arrays;
import java.util.List;

import org.opencv.core.RotatedRect;



public class MatOfRotatedRect extends Mat {
    // 32FC5
    private static final int _depth = CvType.CV_32F;
    private static final int _channels = 5;

    public MatOfRotatedRect() {
        super();
    }

    protected MatOfRotatedRect(long addr) {
        super(addr);
        if( !empty() && checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incompatible Mat");
        //FIXME: do we need release() here?
    }

    public static MatOfRotatedRect fromNativeAddr(long addr) {
        return new MatOfRotatedRect(addr);
    }

    public MatOfRotatedRect(Mat m) {
        super(m, Range.all());
        if( !empty() && checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incompatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfRotatedRect(RotatedRect...a) {
        super();
        fromArray(a);
    }

    public void alloc(int elemNumber) {
        if(elemNumber>0)
            super.create(elemNumber, 1, CvType.makeType(_depth, _channels));
    }

    public void fromArray(RotatedRect...a) {
        if(a==null || a.length==0)
            return;
        int num = a.length;
        alloc(num);
        float buff[] = new float[num * _channels];
        for(int i=0; i<num; i++) {
            RotatedRect r = a[i];
            buff[_channels*i+0] = (float) r.center.x;
            buff[_channels*i+1] = (float) r.center.y;
            buff[_channels*i+2] = (float) r.size.width;
            buff[_channels*i+3] = (float) r.size.height;
            buff[_channels*i+4] = (float) r.angle;
        }
        put(0, 0, buff); //TODO: check ret val!
    }

    public RotatedRect[] toArray() {
        int num = (int) total();
        RotatedRect[] a = new RotatedRect[num];
        if(num == 0)
            return a;
        float buff[] = new float[_channels];
        for(int i=0; i<num; i++) {
            get(i, 0, buff); //TODO: check ret val!
            a[i] = new RotatedRect(new Point(buff[0],buff[1]),new Size(buff[2],buff[3]),buff[4]);
        }
        return a;
    }

    public void fromList(List<RotatedRect> lr) {
        RotatedRect ap[] = lr.toArray(new RotatedRect[0]);
        fromArray(ap);
    }

    public List<RotatedRect> toList() {
        RotatedRect[] ar = toArray();
        return Arrays.asList(ar);
    }
}
