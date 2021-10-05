import numpy as np
import cv2

vertices, _ = cv2.loadPointCloud("../data/teapot.obj")
vertices = np.squeeze(vertices, axis=1)
print(vertices)

color = [1.0, 1.0, 0.0]
colors = np.tile(color, (vertices.shape[0], 1))
obj_pts = np.concatenate((vertices, colors), axis=1)
obj_pts= np.float32(obj_pts)

cv2.viz3d.showPoints("window", "points", obj_pts)
cv2.viz3d.setGridVisible("window", True)

cv2.waitKey(0)

vertices, _, indices = cv2.loadMesh("../data/teapot.obj")
vertices = np.squeeze(vertices, axis=1)

cv2.viz3d.showMesh("window", "mesh", vertices, indices)

cv2.waitKey(0)
