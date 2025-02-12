import rsaitehu_geometry as geom
import numpy as np

points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
print(geom.get_centroid_of_points(points))
# array([0.33333333, 0.33333333, 0.        ])