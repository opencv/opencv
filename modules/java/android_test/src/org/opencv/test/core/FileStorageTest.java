package org.opencv.test.core;

import org.opencv.core.FileStorage;
import org.opencv.test.OpenCVTestCase;

public class FileStorageTest extends OpenCVTestCase {
	
	private FileStorage fs;
	
    @Override
    protected void setUp() throws Exception {
        super.setUp();
        
        fs = null;
    }
    
	public void test_1() {
		super.test_1("CORE.FileStorage");
	}

	public void testFileStorage() {
		fs = new FileStorage();
		assertTrue(null != fs);
	}

	public void testFileStorageLong() {
		fail("Not yet implemented");
	}

	public void testFileStorageStringInt() {
		fs = new FileStorage("test.yml", FileStorage.WRITE);
		assertTrue(null != fs);
	}

	public void testFileStorageStringIntString() {
		fail("Not yet implemented");
	}

	public void testIsOpened() {
		fs = new FileStorage();
		assertFalse(fs.isOpened());
		
		fs = new FileStorage("test.yml", FileStorage.WRITE);
		assertTrue(fs.isOpened());
	}

	public void testOpenStringInt() {
		fail("Not yet implemented");
	}

	public void testOpenStringIntString() {
		fail("Not yet implemented");
	}

	public void testRelease() {
		fail("Not yet implemented");
	}

}
