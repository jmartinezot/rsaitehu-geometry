import rsaitehu_geometry as geom
import numpy as np

# Single point example
point = (1, 2, 3)
plane = (1, -1, 1, -10)  # Plane: x - y + z - 10 = 0
distance = geom.get_distance_from_points_to_plane(point, plane)
print(distance)
# 4.618802153517007
# Multiple points example
points = np.array([[1, 2, 3], [4, 5, 6]])
distance = geom.get_distance_from_points_to_plane(points, plane)
print(distance)
# [4.61880215 2.88675135]