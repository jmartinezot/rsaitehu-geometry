import rsaitehu_geometry as geom
import numpy as np

plane = np.array([0, 0, 1, -3])  # Plane: z = 3
point = np.array([1, 1, 1])
polygon = geom.get_a_polygon_from_plane_equation_and_point(plane, point)
print(polygon)
# [[2. 2. 3.]
#  [0. 2. 3.]
#  [0. 0. 3.]
#  [2. 0. 3.]]
