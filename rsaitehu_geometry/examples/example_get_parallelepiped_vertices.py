import rsaitehu_geometry as geom

center = [0, 0, 0]
normal1 = [1, 0, 0]
normal2 = [0, 1, 0]
normal3 = [0, 0, 1]
lengths = [2, 2, 2]
print(geom.get_parallelepiped_vertices(center, [normal1, normal2, normal3], lengths))
# [[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0]]