import rsaitehu_geometry as geom
import numpy as np

plane = np.array([0, 0, 1, -5])  # z = 5 plane
point = np.array([1, 2, 3])
closest_point = geom.get_point_of_plane_closest_to_given_point(plane, point)
print(closest_point)
# array([1., 2., 5.])