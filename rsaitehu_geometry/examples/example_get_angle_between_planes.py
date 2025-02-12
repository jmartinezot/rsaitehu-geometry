import rsaitehu_geometry as geom
import numpy as np

plane1 = np.array([0, 0, 1, -3])  # Plane: z = 3
plane2 = np.array([0, 1, 1, -4])  # Some inclined plane
angle = geom.get_angle_between_planes(plane1, plane2)
print(angle)
# 0.7853981633974484  # Approximately 45 degrees in radians