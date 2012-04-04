package org.opencv.core;

import org.opencv.features2d.KeyPoint;

public class CvVectorKeyPoint extends CvVectorFloat {
    private static final int _ch = 7; //xxxC7

    public CvVectorKeyPoint() {
        super(_ch);
    }

    public CvVectorKeyPoint(long addr) {
        super(_ch, addr);
    }

    public CvVectorKeyPoint(Mat m) {
        super(_ch, m);
    }

    public CvVectorKeyPoint(KeyPoint...a) {
        super(_ch);
        if(a==null || a.length==0)
            return;
        int cnt = a.length;
        create(cnt);
        float buff[] = new float[_ch * cnt];
        for(int i=0; i<cnt; i++) {
        	KeyPoint kp = a[i];
            buff[_ch*i+0] = (float) kp.pt.x;
            buff[_ch*i+1] = (float) kp.pt.y;
            buff[_ch*i+2] = kp.size;
            buff[_ch*i+3] = kp.angle;
            buff[_ch*i+4] = kp.response;
            buff[_ch*i+5] = kp.octave;
            buff[_ch*i+6] = kp.class_id;
        }
        put(0, 0, buff); //TODO: check ret val!
    }

    public KeyPoint[] toArray(KeyPoint[] a) {
    	float buff[] = super.toArray(null);
        if(buff.length == 0)
            return new KeyPoint[0]; //null;
        int cnt = buff.length / _ch;
        KeyPoint[] res = a;
        if(a==null || a.length<cnt)
            res = new KeyPoint[cnt];
        for(int i=0; i<cnt; i++)
            res[i] = new KeyPoint( buff[_ch*i+0], buff[_ch*i+1], buff[_ch*i+2], buff[_ch*i+3],
                    			   buff[_ch*i+4], (int) buff[_ch*i+5], (int) buff[_ch*i+6] );
        return res;
    }
}
