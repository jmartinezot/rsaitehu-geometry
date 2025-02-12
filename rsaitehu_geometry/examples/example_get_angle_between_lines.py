import rsaitehu_geometry as geom
import numpy as np

l1 = np.array([[0, 0, 0], [1, 1, 1]])
l2 = np.array([[0, 0, 0], [-1, -1, -1]])
print(geom.get_angle_between_lines(l1, l2))
# 3.141592653589793  # 180 degrees in radians