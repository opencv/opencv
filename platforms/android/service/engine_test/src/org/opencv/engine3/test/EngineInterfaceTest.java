package org.opencv.engine3.test;

import org.opencv.engine3.OpenCVEngineInterface;
import org.opencv.engine3.OpenCVEngineService;

import android.content.Intent;
import android.os.IBinder;
import android.os.RemoteException;
import android.test.ServiceTestCase;

public class EngineInterfaceTest extends ServiceTestCase<OpenCVEngineService>
{
    public EngineInterfaceTest()
    {
        super(OpenCVEngineService.class);
        // TODO Auto-generated constructor stub
    }

    public void testVersion() throws RemoteException
    {
        IBinder ServiceBinder = bindService(new Intent("org.opencv.engine3.BIND"));
        assertNotNull(ServiceBinder);
        OpenCVEngineInterface ServiceObj = OpenCVEngineInterface.Stub.asInterface(ServiceBinder);
        assertNotNull(ServiceObj);
        int ServiceVersion = ServiceObj.getEngineVersion();
        assertEquals(1, ServiceVersion);
    }

    public void testInstallVersion() throws RemoteException
    {
        IBinder ServiceBinder = bindService(new Intent("org.opencv.engine3"));
        assertNotNull(ServiceBinder);
        OpenCVEngineInterface ServiceObj = OpenCVEngineInterface.Stub.asInterface(ServiceBinder);
        assertNotNull(ServiceObj);
        assertTrue(ServiceObj.installVersion("2.4"));
    }

    public void testGetPathForExistVersion() throws RemoteException
    {
        IBinder ServiceBinder = bindService(new Intent("org.opencv.engine3"));
        assertNotNull(ServiceBinder);
        OpenCVEngineInterface ServiceObj = OpenCVEngineInterface.Stub.asInterface(ServiceBinder);
        assertNotNull(ServiceObj);
        assertEquals("/data/data/org.opencv.lib_v240_tegra3/lib", ServiceObj.getLibPathByVersion("2.4"));
    }

    public void testGetPathForUnExistVersion() throws RemoteException
    {
        IBinder ServiceBinder = bindService(new Intent("org.opencv.engine3"));
        assertNotNull(ServiceBinder);
        OpenCVEngineInterface ServiceObj = OpenCVEngineInterface.Stub.asInterface(ServiceBinder);
        assertNotNull(ServiceObj);
        assertEquals("", ServiceObj.getLibPathByVersion("2.5"));
    }

    public void testInstallAndGetVersion() throws RemoteException
    {
        IBinder ServiceBinder = bindService(new Intent("org.opencv.engine3"));
        assertNotNull(ServiceBinder);
        OpenCVEngineInterface ServiceObj = OpenCVEngineInterface.Stub.asInterface(ServiceBinder);
        assertNotNull(ServiceObj);
        assertTrue(ServiceObj.installVersion("2.4"));
        assertEquals("/data/data/org.opencv.lib_v240_tegra3/lib", ServiceObj.getLibPathByVersion("2.4"));
    }
}
