import rsaitehu_geometry as geom
import numpy as np

plane = np.array([0, 0, 1, -3])  # Plane: z = 3
perpendicular1, perpendicular2 = geom.get_two_perpendicular_unit_vectors_in_plane(plane)
print(perpendicular1)
# array([1., 0., 0.])
print(perpendicular2)
# array([0., 1., 0.])