import rsaitehu_geometry as geom
import numpy as np

points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
print(geom.fit_plane_svd(points))
# (-0.0, -1.0, -0.0, 0.5, 1.0)