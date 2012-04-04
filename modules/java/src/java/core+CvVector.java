package org.opencv.core;

public class CvVector extends Mat {
    protected int depth;
    protected int channels;

    protected CvVector(int d, int ch) {
        super();
        depth = d;
        channels = ch;
    }

    protected CvVector(int d, int ch, long addr) {
        super(addr);
        depth = d;
        channels = ch;
        if( !empty() && checkVector(channels, depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    protected CvVector(int d, int ch, Mat m) {
        super(m, Range.all());
        depth = d;
        channels = ch;
        if( !empty() && checkVector(channels, depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    protected void create(int cnt) {
        if(cnt>0)
            super.create(cnt, 1, CvType.makeType(depth, channels));
    }
}
