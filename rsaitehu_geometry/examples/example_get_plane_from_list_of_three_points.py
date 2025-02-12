import rsaitehu_geometry as geom

points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
plane = geom.get_plane_from_list_of_three_points(points)
print(plane)
# array([0., 0., 1., 0.])