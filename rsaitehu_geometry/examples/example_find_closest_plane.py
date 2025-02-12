import rsaitehu_geometry as geom

points = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
plane = geom.find_closest_plane(points)
print(plane)
# (-0.0, -1.0, -0.0, 0.5)