import rsaitehu_geometry as geom

normal1 = [1, 0, 0]
normal2 = [0, 1, 0]
point = [0, 0, 1]
print(geom.get_plane_equation(normal1, normal2, point))
# (0.0, 0.0, 1.0, -1.0)  # z = 1 plane