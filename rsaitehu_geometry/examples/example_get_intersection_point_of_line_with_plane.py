import rsaitehu_geometry as geom
import numpy as np

line = np.array([[0, 0, 0], [1, 1, 1]])
plane = np.array([0, 0, 1, -3])  # Plane: z = 3
intersection_point = geom.get_intersection_point_of_line_with_plane(line, plane)
print(intersection_point)
# array([3., 3., 3.])