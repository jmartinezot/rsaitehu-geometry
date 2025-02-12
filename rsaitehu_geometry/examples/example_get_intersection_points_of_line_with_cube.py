import rsaitehu_geometry as geom
import numpy as np

line = np.array([[0, 0, 0], [1, 1, 1]])
cube_min = np.array([-2, -2, -1])
cube_max = np.array([1, 2, 2])
intersection_points = geom.get_intersection_points_of_line_with_cube(line, cube_min, cube_max)
print(intersection_points)
# array([[ 1.,  1.,  1.],
#       [-1., -1., -1.]])