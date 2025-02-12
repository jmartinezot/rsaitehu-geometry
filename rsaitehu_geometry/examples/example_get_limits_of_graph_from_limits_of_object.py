import rsaitehu_geometry as geom

# 2D Example
print(geom.get_limits_of_graph_from_limits_of_object(-5, 10, -3, 8))
# (-10.0, 10.0, -10.0, 10.0)
# 3D Example
print(geom.get_limits_of_graph_from_limits_of_object(-5, 10, -3, 8, -7, 15))
# (-15.0, 15.0, -15.0, 15.0, -15.0, 15.0)