import rsaitehu_geometry as geom

center = [0, 0, 0]
normal1 = [1, 0, 0]
normal2 = [0, 1, 0]
lengths = [2, 4]
print(geom.get_parallelogram_vertices(center, [normal1, normal2], lengths))
# [[1.0, 2.0, 0.0], [-1.0, 2.0, 0.0], [-1.0, -2.0, 0.0], [1.0, -2.0, 0.0]]