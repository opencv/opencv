package org.opencv.core;

import org.opencv.features2d.DMatch;

public class CvVectorDMatch extends CvVectorFloat {
    private static final int _ch = 4; //xxxC4

    public CvVectorDMatch() {
        super(_ch);
    }

    public CvVectorDMatch(long addr) {
        super(_ch, addr);
    }

    public CvVectorDMatch(Mat m) {
        super(_ch, m);
    }

    public CvVectorDMatch(DMatch...a) {
        super(_ch);
        if(a==null || a.length==0)
            return;
        int cnt = a.length;
        create(cnt);
        float buff[] = new float[_ch * cnt];
        for(int i=0; i<cnt; i++) {
        	DMatch m = a[i];
            buff[_ch*i+0] = m.queryIdx;
            buff[_ch*i+1] = m.trainIdx;
            buff[_ch*i+2] = m.imgIdx;
            buff[_ch*i+3] = m.distance;
        }
        put(0, 0, buff); //TODO: check ret val!
    }

    public DMatch[] toArray(DMatch[] a) {
    	float buff[] = super.toArray(null);
        if(buff.length == 0)
            return new DMatch[0]; //null;
        int cnt = buff.length / _ch;
        DMatch[] res = a;
        if(a==null || a.length<cnt)
            res = new DMatch[cnt];
        for(int i=0; i<cnt; i++)
            res[i] = new DMatch((int) buff[_ch*i+0], (int) buff[_ch*i+1], (int) buff[_ch*i+2], buff[_ch*i+3]);
        return res;
    }
}
