package org.opencv.core;

import java.util.Arrays;
import java.util.List;

import org.opencv.features2d.DMatch;

public class MatOfDMatch extends Mat {
	// 32FC4
	private static final int _depth = CvType.CV_32F;
	private static final int _channels = 4;

    public MatOfDMatch() {
        super();
    }

    public MatOfDMatch(long addr) {
        super(addr);
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfDMatch(Mat m) {
    	super(m, Range.all());
        if(checkVector(_channels, _depth) < 0 )
            throw new IllegalArgumentException("Incomatible Mat");
        //FIXME: do we need release() here?
    }

    public MatOfDMatch(DMatch...ap) {
        super();
        fromArray(ap);
    }
    
    public void alloc(int elemNumber) {
        if(elemNumber>0)
            super.create(elemNumber, 1, CvType.makeType(_depth, _channels));
    }


    public void fromArray(DMatch...a) {
        if(a==null || a.length==0)
            return;
        int num = a.length;
        alloc(num);
        float buff[] = new float[num * _channels];
        for(int i=0; i<num; i++) {
            DMatch m = a[i];
            buff[_channels*i+0] = m.queryIdx;
            buff[_channels*i+1] = m.trainIdx;
            buff[_channels*i+2] = m.imgIdx;
            buff[_channels*i+3] = m.distance;
        }
        put(0, 0, buff); //TODO: check ret val!
    }

    public DMatch[] toArray() {
        int num = (int) total();
        DMatch[] a = new DMatch[num];
        if(num == 0)
            return a;
        float buff[] = new float[num * _channels];
        get(0, 0, buff); //TODO: check ret val!
        for(int i=0; i<num; i++)
            a[i] = new DMatch((int) buff[_channels*i+0], (int) buff[_channels*i+1], (int) buff[_channels*i+2], buff[_channels*i+3]);
        return a;
    }

    public void fromList(List<DMatch> ldm) {
    	DMatch adm[] = ldm.toArray(null);
    	fromArray(adm);
    }
    
    public List<DMatch> toList() {
    	DMatch[] adm = toArray();
    	return Arrays.asList(adm); 
    }
}
