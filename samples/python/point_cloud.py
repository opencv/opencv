import numpy as np
import cv2 as cv

vertices, _ = cv.loadPointCloud("../data/teapot.obj")
vertices = np.squeeze(vertices, axis=1)
print(vertices)

color = [1.0, 1.0, 0.0]
colors = np.tile(color, (vertices.shape[0], 1))
obj_pts = np.concatenate((vertices, colors), axis=1)
obj_pts= np.float32(obj_pts)

cv.viz3d.showPoints("window", "points", obj_pts)
cv.viz3d.setGridVisible("window", True)

cv.waitKey(0)

vertices, indices = cv.loadMesh("../data/teapot.obj")
vertices = np.squeeze(vertices, axis=1)

cv.viz3d.showMesh("window", "mesh", vertices, indices)

cv.waitKey(0)
