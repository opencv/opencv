import cv2
import numpy as np
import gc
import tracemalloc

def test_subtract_memory_leak(iterations=5000):
    """
    Tests for memory leaks when using cv2.subtract in a loop.
    Optimized with in-place operations and reused arrays.
    """

    tracemalloc.start()

    # Pre-allocate arrays to avoid continuous memory allocation
    img1 = np.empty((512, 512, 3), dtype=np.float32)
    result = np.empty_like(img1)

    for i in range(iterations):
        # Reuse the existing arrays
        np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8, out=img1)
        img2 = np.random.randint(0, 256, (3), dtype=np.uint8).astype(np.float32)

        # In-place operation to prevent new allocations
        cv2.subtract(img1, img2, dst=result)

        # Display memory usage every 100 iterations
        if (i + 1) % 100 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Iteration {i+1}: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    tracemalloc.stop()

# Run the memory leak test
test_subtract_memory_leak()
