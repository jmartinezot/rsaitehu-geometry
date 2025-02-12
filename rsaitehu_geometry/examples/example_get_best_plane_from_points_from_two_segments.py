import rsaitehu_geometry as geom
import numpy as np

segment_1 = np.array([[0, 0, 0], [1, 0, 0]])
segment_2 = np.array([[0, 1, 0], [1, 1, 0]])
print(geom.get_best_plane_from_points_from_two_segments(segment_1, segment_2))
# (array([ 0.,  0.,  1., -0.]), 0.0)
segment_1 = np.array([[1, 2, 3], [4, 5, 6]])
segment_2 = np.array([[7, 8, 9], [10, 11, 12]])
print(geom.get_best_plane_from_points_from_two_segments(segment_1, segment_2))
#(array([ 0.81649658, -0.40824829, -0.40824829,  1.22474487]), 1.0107280348144214e-29)