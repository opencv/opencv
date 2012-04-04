package org.opencv.core;


public class CvVectorRect extends CvVectorInt {
    private static final int _ch = 4; //xxxC4

    public CvVectorRect() {
        super(_ch);
    }

    public CvVectorRect(long addr) {
        super(_ch, addr);
    }

    public CvVectorRect(Mat m) {
        super(_ch, m);
    }

    public CvVectorRect(Rect...a) {
        super(_ch);
        if(a==null || a.length==0)
            return;
        int cnt = a.length;
        create(cnt);
        int buff[] = new int[_ch * cnt];
        for(int i=0; i<cnt; i++) {
        	Rect r = a[i];
            buff[_ch*i]   = r.x;
            buff[_ch*i+1] = r.y;
            buff[_ch*i+2] = r.width;
            buff[_ch*i+3] = r.height;
        }
        put(0, 0, buff); //TODO: check ret val!
    }

    public Rect[] toArray(Rect[] a) {
        int buff[] = super.toArray(null);
        if(buff.length == 0)
            return new Rect[0]; //null;
        int cnt = buff.length / _ch;
        Rect[] res = a;
        if(a==null || a.length<cnt)
            res = new Rect[cnt];
        for(int i=0; i<cnt; i++)
            res[i] = new Rect(buff[i*_ch], buff[i*_ch+1], buff[i*_ch+2], buff[i*_ch+3]);
        return res;
    }
}
