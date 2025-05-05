package org.opencv.videoio;

public interface IStreamReader {
    public long read(byte[] buffer);
    public long seek(long offset, long origin);
}
