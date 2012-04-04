package org.opencv.core;

public class CvVectorPoint3f extends CvVectorFloat {
    private static final int _ch = 3; //xxxC2

    public CvVectorPoint3f() {
        super(_ch);
    }

    public CvVectorPoint3f(long addr) {
        super(_ch, addr);
    }

    public CvVectorPoint3f(Mat m) {
        super(_ch, m);
    }

    public CvVectorPoint3f(Point3...a) {
        super(_ch);
        if(a==null || a.length==0)
            return;
        int cnt = a.length;
        create(cnt);
        float buff[] = new float[_ch * cnt];
        for(int i=0; i<cnt; i++) {
        	Point3 p = a[i];
            buff[_ch*i]   = (float) p.x;
            buff[_ch*i+1] = (float) p.y;
            buff[_ch*i+2] = (float) p.z;
        }
        put(0, 0, buff); //TODO: check ret val!
    }

    public Point3[] toArray(Point3[] a) {
        float buff[] = super.toArray(null);
        if(buff.length == 0)
            return new Point3[0]; //null;
        int cnt = buff.length / _ch;
        Point3[] res = a;
        if(a==null || a.length<cnt)
            res = new Point3[cnt];
        for(int i=0; i<cnt; i++)
            res[i] = new Point3(buff[i*_ch], buff[i*_ch+1], buff[i*_ch+2]);
        return res;
    }
}
