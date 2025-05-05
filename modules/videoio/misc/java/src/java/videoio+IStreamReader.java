package org.opencv.videoio;

public interface IStreamReader {
    public long read(byte[] buffer, long size);
    public long seek(long offset, long origin);
}
