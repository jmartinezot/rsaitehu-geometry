'''
Geometry related functions.
'''
import numpy as np
from typing import List, Union, Tuple, Optional
from scipy.linalg import svd
import math

def get_plane_from_list_of_three_points(points: List[List[float]]) -> Union[np.ndarray, None]:
    """
    Calculate the equation of a plane in the form Ax + By + Cz + D = 0 from three points in 3D space.

    :param points: A list containing exactly three points, where each point is a list of three coordinates [x, y, z].
    :type points: List[List[float]]
    :return: A numpy array [A, B, C, D] representing the plane equation or None if the points are collinear or invalid.
    :rtype: Union[np.ndarray, None]

    :raises ValueError: If the number of points is not three or if any point does not have three coordinates.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_plane_from_list_of_three_points.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_plane_from_list_of_three_points.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        >>> plane = geom.get_plane_from_list_of_three_points(points)
        >>> plane
        array([0., 0., 1., 0.])
    """
    if len(points) != 3:
        raise ValueError("Exactly three points are required to define a plane.")
    
    for point in points:
        if len(point) != 3:
            raise ValueError("Each point must have exactly three coordinates [x, y, z].")
    
    # Convert points to numpy arrays
    p1, p2, p3 = map(np.array, points)

    # Compute direction vectors
    v1 = p2 - p1
    v2 = p3 - p1

    # Check for collinearity or invalid vectors
    if np.allclose(v1, 0) or np.allclose(v2, 0) or np.allclose(np.cross(v1, v2), 0):
        raise ValueError("The provided points are collinear or overlapping; a plane cannot be defined.")

    # Compute the normal vector to the plane
    normal = np.cross(v1, v2)

    # Compute the plane's D coefficient
    d = -np.dot(normal, p1)

    # Return the plane as a numpy array
    return np.array([*normal, d], dtype=float)

def find_closest_plane(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """
    Find the closest plane to a set of points using Singular Value Decomposition (SVD).

    This function determines the plane that minimizes the sum of squared distances
    to a set of 3D points. The plane is defined by the equation Ax + By + Cz + D = 0.

    :param points: A list of points, where each point is a list of three coordinates [x, y, z].
    :type points: List[List[float]]
    :return: A tuple (A, B, C, D) representing the plane equation coefficients.
    :rtype: Tuple[float, float, float, float]

    :raises ValueError: If the input points are not valid (e.g., fewer than 3 points, incorrect dimensions).

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_find_closest_plane.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_find_closest_plane.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> points = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
        >>> plane = geom.find_closest_plane(points)
        >>> plane
        (-0.0, -1.0, -0.0, 0.5)
    """
    if len(points) < 3:
        raise ValueError("At least three points are required to define a plane.")

    points_array = np.array(points)
    if points_array.shape[1] != 3:
        raise ValueError("Each point must have exactly three coordinates [x, y, z].")

    # Step 1: Center the points around the origin
    centroid = np.mean(points_array, axis=0)
    centered_points = points_array - centroid

    # Step 2: Compute the singular value decomposition (SVD)
    _, _, vh = svd(centered_points)

    # Step 3: Extract the plane normal from the last row of V^T
    plane_normal = vh[-1]

    # Step 4: Normalize the plane normal vector
    plane_normal /= np.linalg.norm(plane_normal)

    # Step 5: Compute the D coefficient using the centroid
    D = -np.dot(plane_normal, centroid)

    # Return the plane equation coefficients as a tuple
    return (*plane_normal, D)

def get_point_of_plane_closest_to_given_point(plane: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Compute the point on a plane closest to a given point in 3D space.

    The plane is described by the equation Ax + By + Cz + D = 0. The closest point
    is determined by projecting the given point onto the plane along the plane's normal.

    :param plane: A numpy array [A, B, C, D] representing the plane equation coefficients.
    :type plane: np.ndarray
    :param point: A numpy array [x, y, z] representing the point in 3D space.
    :type point: np.ndarray
    :return: A numpy array [x, y, z] representing the point on the plane closest to the given point.
    :rtype: np.ndarray

    :raises ValueError: If the plane or point input is not properly formatted.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_point_of_plane_closest_to_given_point.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_point_of_plane_closest_to_given_point.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> import numpy as np
        >>> plane = np.array([0, 0, 1, -5])  # z = 5 plane
        >>> point = np.array([1, 2, 3])
        >>> closest_point = geom.get_point_of_plane_closest_to_given_point(plane, point)
        >>> closest_point
        array([1., 2., 5.])
    """
    if plane.shape != (4,):
        raise ValueError("Plane must be a numpy array with exactly 4 coefficients [A, B, C, D].")
    
    if point.shape != (3,):
        raise ValueError("Point must be a numpy array with exactly 3 coordinates [x, y, z].")

    # Step 1: Extract the normal vector of the plane
    normal = plane[:3]

    # Step 2: Project the point onto the plane
    # t is the scalar that determines the projection distance along the normal
    t = - (np.dot(normal, point) + plane[3]) / np.dot(normal, normal)

    # Step 3: Compute the closest point by adding the projection to the original point
    closest_point = point + t * normal

    return closest_point

def get_plane_equation(normal1: List[float], normal2: List[float], point: List[float]) -> Tuple[float, float, float, float]:
    """
    Calculate the equation of a plane defined by two normal vectors and a point in 3D space.

    The plane equation is represented in the form Ax + By + Cz + D = 0, where (A, B, C) is
    the normal vector to the plane, and D is the distance from the origin along the normal.

    :param normal1: The first normal vector to the plane, as a list of three coordinates [x, y, z].
    :type normal1: List[float]
    :param normal2: The second normal vector to the plane, as a list of three coordinates [x, y, z].
    :type normal2: List[float]
    :param point: A point on the plane, as a list of three coordinates [x, y, z].
    :type point: List[float]
    :return: A tuple (A, B, C, D) representing the plane equation coefficients.
    :rtype: Tuple[float, float, float, float]

    :raises ValueError: If the inputs are not valid vectors with three components each.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_plane_equation.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_plane_equation.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> normal1 = [1, 0, 0]
        >>> normal2 = [0, 1, 0]
        >>> point = [0, 0, 1]
        >>> geom.get_plane_equation(normal1, normal2, point)
        (0.0, 0.0, 1.0, -1.0)
    """
    normal1 = np.array(normal1, dtype=float)
    normal2 = np.array(normal2, dtype=float)
    point = np.array(point, dtype=float)

    # Validate input dimensions
    if normal1.shape != (3,) or normal2.shape != (3,) or point.shape != (3,):
        raise ValueError("All inputs must be vectors of three components [x, y, z].")

    # Normalize the input normal vectors
    normal1 /= np.linalg.norm(normal1)
    normal2 /= np.linalg.norm(normal2)

    # Compute the normal vector of the plane using the cross product
    plane_normal = np.cross(normal1, normal2)

    # Normalize the resulting normal vector
    plane_normal /= np.linalg.norm(plane_normal)

    # Compute the D coefficient using the given point
    D = -np.dot(plane_normal, point)

    # Extract plane coefficients
    A, B, C = plane_normal

    return A, B, C, D

def get_distance_from_points_to_plane(points: Union[Tuple[float, float, float], np.ndarray],plane: Tuple[float, float, float, float]
) -> Union[float, np.ndarray]:
    """
    Calculate the perpendicular distance(s) from one or more points to a plane in 3D space.

    The plane is defined by the equation Ax + By + Cz + D = 0, and the distance is computed
    using the formula: \|Ax + By + Cz + D\| / sqrt(A^2 + B^2 + C^2).

    :param points: A single point as a tuple (x, y, z) or an array of points with shape (N, 3).
    :type points: Union[Tuple[float, float, float], np.ndarray]
    :param plane: A tuple (A, B, C, D) representing the plane equation coefficients.
    :type plane: Tuple[float, float, float, float]
    :return: The distance(s) from the point(s) to the plane. Returns a float for a single point
             and a numpy array for multiple points.
    :rtype: Union[float, np.ndarray]

    :raises ValueError: If the input points are not in a valid format.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_distance_from_points_to_plane.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_distance_from_points_to_plane.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> # Single point example
        >>> point = (1, 2, 3)
        >>> plane = (1, -1, 1, -10)  # Plane: x - y + z - 10 = 0
        >>> distance = geom.get_distance_from_points_to_plane(point, plane)
        >>> distance
        4.618802153517007
        >>> # Multiple points example
        >>> points = np.array([[1, 2, 3], [4, 5, 6]])
        >>> distance = geom.get_distance_from_points_to_plane(points, plane)
        >>> distance
        array([4.61880215, 2.88675135])
    """
    A, B, C, D = plane

    if isinstance(points, tuple):
        if len(points) != 3:
            raise ValueError("Point must be a tuple with exactly three elements (x, y, z).")
        x, y, z = points
        distance = abs(A * x + B * y + C * z + D) / math.sqrt(A**2 + B**2 + C**2)
        return distance

    elif isinstance(points, np.ndarray):
        if points.shape[1] != 3:
            raise ValueError("Points array must have shape (N, 3).")
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        distance = abs(A * x + B * y + C * z + D) / math.sqrt(A**2 + B**2 + C**2)
        return distance

    else:
        raise ValueError("Input must be a tuple (x, y, z) or a numpy array with shape (N, 3).")

def fit_plane_svd(points: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Fit a plane to a set of 3D points using Singular Value Decomposition (SVD).

    This function calculates the plane that minimizes the sum of squared perpendicular
    distances to the provided points. The plane is represented by the equation Ax + By + Cz + D = 0.

    :param points: A numpy array of shape (N, 3), where N is the number of points. Each row represents a point (x, y, z).
    :type points: np.ndarray
    :return: A tuple containing:
             - A, B, C (float): Normal vector components of the plane.
             - D (float): Offset of the plane from the origin.
             - SSE (float): Sum of squared errors between the plane and the points.
    :rtype: Tuple[float, float, float, float, float]

    :raises ValueError: If the input array does not have the required shape.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_fit_plane_svd.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_fit_plane_svd.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> import numpy as np
        >>> points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        >>> geom.fit_plane_svd(points)
        (-0.0, -1.0, -0.0, 0.5, 1.0)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input points must be a numpy array of shape (N, 3), where N is the number of points.")

    # Step 1: Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Step 2: Shift points to center them around the origin
    shifted_points = points - centroid

    # Step 3: Perform Singular Value Decomposition (SVD)
    _, _, Vt = np.linalg.svd(shifted_points)

    # Step 4: The normal vector to the plane is the last row of V^T
    normal = Vt[-1]

    # Step 5: Calculate the D coefficient using the centroid
    d = -np.dot(normal, centroid)

    # Step 6: Compute the sum of squared errors (SSE)
    errors = np.abs(np.dot(points, normal) + d)
    sse = np.sum(errors ** 2)

    return normal[0], normal[1], normal[2], d, sse

def get_intersection_point_of_line_with_plane(line: np.ndarray, plane: np.ndarray) -> Optional[np.ndarray]:
    '''
    Calculate the intersection point of a line with a plane in 3D space.

    The line is represented by two points, and the plane is defined by the equation Ax + By + Cz + D = 0.
    If the line is parallel to the plane (no intersection), the function returns None.

    :param line: A numpy array of shape (2, 3), where each row represents a point [x, y, z] on the line.
    :type line: np.ndarray
    :param plane: A numpy array [A, B, C, D] representing the plane equation coefficients.
    :type plane: np.ndarray
    :return: The intersection point as a numpy array [x, y, z], or None if the line is parallel to the plane.
    :rtype: Optional[np.ndarray]

    :raises ValueError: If the line or plane input is not in the correct format.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_intersection_point_of_line_with_plane.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_intersection_point_of_line_with_plane.py

    |drawing_draw_line_extension_to_plane_example|

    .. |drawing_draw_line_extension_to_plane_example| image:: ../../doc/source/_static/images/drawing_draw_line_extension_to_plane_example.png


    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> import numpy as np
        >>> line = np.array([[0, 0, 0], [1, 1, 1]])
        >>> plane = np.array([0, 0, 1, -3])  # Plane: z = 3
        >>> intersection_point = geom.get_intersection_point_of_line_with_plane(line, plane)
        >>> intersection_point
        array([3., 3., 3.])
    '''
    if line.shape != (2, 3):
        raise ValueError("Line must be a numpy array with shape (2, 3), representing two points in 3D space.")
    if plane.shape != (4,):
        raise ValueError("Plane must be a numpy array with shape (4,), representing [A, B, C, D].")

    # Step 1: Calculate the direction vector of the line
    direction = line[1] - line[0]

    # Step 2: Extract the normal vector of the plane
    normal = plane[:3]

    # Step 3: Check if the line is parallel to the plane
    dot_product = np.dot(direction, normal)
    if np.isclose(dot_product, 0, atol=1e-6):  # Parallel case
        return None

    # Step 4: Calculate the parameter t for the line's parametric equation
    t = - (np.dot(normal, line[0]) + plane[3]) / dot_product

    # Step 5: Compute the intersection point
    intersection_point = line[0] + t * direction

    return intersection_point

def get_two_perpendicular_unit_vectors_in_plane(plane: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two perpendicular unit vectors that lie in a given plane in 3D space.

    The plane is described by the equation Ax + By + Cz + D = 0, and the function returns
    two unit vectors that are perpendicular to each other and lie within the plane.

    :param plane: A numpy array [A, B, C, D] representing the plane equation coefficients.
    :type plane: np.ndarray
    :return: Two perpendicular unit vectors in the plane as numpy arrays.
    :rtype: Tuple[np.ndarray, np.ndarray]

    :raises ValueError: If the plane's normal vector is invalid (e.g., zero vector).

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_two_perpendicular_unit_vectors_in_plane.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_two_perpendicular_unit_vectors_in_plane.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> import numpy as np
        >>> plane = np.array([0, 0, 1, -3])  # Plane: z = 3
        >>> perpendicular1, perpendicular2 = geom.get_two_perpendicular_unit_vectors_in_plane(plane)
        >>> perpendicular1
        array([1., 0., 0.])
        >>> perpendicular2
        array([0., 1., 0.])

    """
    if plane.shape != (4,):
        raise ValueError("Plane must be a numpy array with shape (4,), representing [A, B, C, D].")

    # Step 1: Extract the normal vector of the plane
    normal = plane[:3]

    if np.allclose(normal, 0, atol=1e-6):
        raise ValueError("The plane's normal vector cannot be the zero vector.")

    # Step 2: Calculate the first perpendicular vector
    if np.isclose(normal[0], 0) and np.isclose(normal[1], 0):
        # If the normal vector points along the z-axis, choose x-axis for perpendicular1
        perpendicular1 = np.array([1.0, 0.0, 0.0])
    else:
        # Otherwise, construct a perpendicular vector in the x-y plane
        perpendicular1 = np.array([normal[1], -normal[0], 0.0], dtype=float)

    # Step 3: Calculate the second perpendicular vector using the cross product
    perpendicular2 = np.cross(normal, perpendicular1)

    # Step 4: Normalize both perpendicular vectors
    perpendicular1 /= np.linalg.norm(perpendicular1)
    perpendicular2 /= np.linalg.norm(perpendicular2)

    return perpendicular1, perpendicular2

def get_best_plane_from_points_from_two_segments(segment_1: np.ndarray, segment_2: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the best fitting plane to the four points defined by two line segments in 3D space.

    This function uses Singular Value Decomposition (SVD) to determine the plane that minimizes
    the sum of squared errors (SSE) for the four points derived from the two segments. The plane
    is represented by the equation Ax + By + Cz + D = 0.

    :param segment_1: A numpy array of shape (2, 3) representing the first segment's endpoints [P1, P2].
    :type segment_1: np.ndarray
    :param segment_2: A numpy array of shape (2, 3) representing the second segment's endpoints [P3, P4].
    :type segment_2: np.ndarray
    :return: A tuple containing:
             - best_plane: A numpy array [A, B, C, D] representing the best fitting plane equation.
             - sse: The sum of squared errors (SSE) for the fitted plane.
    :rtype: Tuple[np.ndarray, float]

    :raises ValueError: If either segment is not a numpy array of shape (2, 3).

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_best_plane_from_points_from_two_segments.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_best_plane_from_points_from_two_segments.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> import numpy as np
        >>> segment_1 = np.array([[0, 0, 0], [1, 0, 0]])
        >>> segment_2 = np.array([[0, 1, 0], [1, 1, 0]])
        >>> geom.get_best_plane_from_points_from_two_segments(segment_1, segment_2)
        (array([ 0.,  0.,  1., -0.]), 0.0)
        >>>
        >>> segment_1 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> segment_2 = np.array([[7, 8, 9], [10, 11, 12]])
        >>> geom.get_best_plane_from_points_from_two_segments(segment_1, segment_2)
        (array([ 0.81649658, -0.40824829, -0.40824829,  1.22474487]), 1.0107280348144214e-29)
    """
    if segment_1.shape != (2, 3) or segment_2.shape != (2, 3):
        raise ValueError("Both segments must be numpy arrays of shape (2, 3), representing two 3D points each.")

    # Combine the four points from the two segments
    points = np.array([segment_1[0], segment_1[1], segment_2[0], segment_2[1]])

    # Fit the best plane to the points using SVD
    a, b, c, d, sse = fit_plane_svd(points)

    # Construct the plane equation
    best_plane = np.array([a, b, c, d])

    return best_plane, sse

def get_a_polygon_from_plane_equation_and_point(plane: np.ndarray, point: np.ndarray, scale: float = 1.0)->np.ndarray:
    """
    Generate a quadrilateral polygon lying in a given plane and centered around a specific point.

    The polygon is defined in the plane specified by its equation (Ax + By + Cz + D = 0),
    with vertices scaled relative to the given point and aligned with the plane's normal vector.

    :param plane: A numpy array [A, B, C, D] representing the plane equation coefficients.
    :type plane: np.ndarray
    :param point: A numpy array [x, y, z] representing the point in 3D space.
    :type point: np.ndarray
    :param scale: A float value indicating the scaling factor for the polygon's size. Default is 1.0.
    :type scale: float
    :return: A numpy array of shape (4, 3), representing the vertices of the polygon.
    :rtype: np.ndarray

    :raises ValueError: If the input plane or point is not in the correct format.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_a_polygon_from_plane_equation_and_point.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_a_polygon_from_plane_equation_and_point.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> import numpy as np
        >>> plane = np.array([0, 0, 1, -3])  # Plane: z = 3
        >>> point = np.array([1, 1, 1])
        >>> polygon = geom.get_a_polygon_from_plane_equation_and_point(plane, point)
        >>> polygon
        array([[2., 2., 3.],
               [0., 2., 3.],
               [0., 0., 3.],
               [2., 0., 3.]])
    """
    if plane.shape != (4,):
        raise ValueError("Plane must be a numpy array with shape (4,), representing [A, B, C, D].")
    if point.shape != (3,):
        raise ValueError("Point must be a numpy array with shape (3,), representing [x, y, z].")

    # Step 1: Extract the normal vector of the plane
    normal = plane[:3]

    # Step 2: Project the point onto the plane to find the closest point
    t = - (np.dot(normal, point) + plane[3]) / np.dot(normal, normal)
    closest_point = point + t * normal

    # Step 3: Calculate two perpendicular unit vectors in the plane
    perpendicular1, perpendicular2 = get_two_perpendicular_unit_vectors_in_plane(plane)

    # Step 4: Compute the vertices of the quadrilateral polygon
    vertex1 = closest_point + perpendicular1 * scale + perpendicular2 * scale
    vertex2 = closest_point - perpendicular1 * scale + perpendicular2 * scale
    vertex3 = closest_point - perpendicular1 * scale - perpendicular2 * scale
    vertex4 = closest_point + perpendicular1 * scale - perpendicular2 * scale

    # Step 5: Return the vertices as a numpy array
    return np.array([vertex1, vertex2, vertex3, vertex4])

def get_limits_of_graph_from_limits_of_object(
    min_x: float, max_x: float, min_y: float, max_y: float, min_z: Union[float, None] = None, max_z: Union[float, None] = None
) -> Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]]:
    """
    Compute the limits of a graph from the bounding limits of an object.

    Ensures the graph is centered at the origin (0, 0, [0]) and that the object is fully visible.
    For 3D graphs, the visible zone is cubic, and for 2D graphs, it is square.

    :param min_x: Minimum x-coordinate of the object.
    :type min_x: float
    :param max_x: Maximum x-coordinate of the object.
    :type max_x: float
    :param min_y: Minimum y-coordinate of the object.
    :type min_y: float
    :param max_y: Maximum y-coordinate of the object.
    :type max_y: float
    :param min_z: (Optional) Minimum z-coordinate of the object for 3D graphs.
    :type min_z: float or None
    :param max_z: (Optional) Maximum z-coordinate of the object for 3D graphs.
    :type max_z: float or None
    :return: The limits of the graph. For 2D graphs, returns a tuple (min_x, max_x, min_y, max_y).
             For 3D graphs, returns a tuple (min_x, max_x, min_y, max_y, min_z, max_z).
    :rtype: Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]]

    :raises ValueError: If only one of min_z or max_z is provided.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_limits_of_graph_from_limits_of_object.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_limits_of_graph_from_limits_of_object.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> # 2D Example
        >>> geom.get_limits_of_graph_from_limits_of_object(-5, 10, -3, 8)
        (-10.0, 10.0, -10.0, 10.0)
        >>> # 3D Example
        >>> geom.get_limits_of_graph_from_limits_of_object(-5, 10, -3, 8, -7, 15)
        (-15.0, 15.0, -15.0, 15.0, -15.0, 15.0)
    """
    if (min_z is None) != (max_z is None):
        raise ValueError("Both min_z and max_z must be provided for 3D graphs, or neither for 2D graphs.")

    # Determine the maximum absolute limit
    if min_z is not None and max_z is not None:  # 3D case
        limit = float(max(abs(min_x), abs(max_x), abs(min_y), abs(max_y), abs(min_z), abs(max_z)))
        return -limit, limit, -limit, limit, -limit, limit
    else:  # 2D case
        limit = float(max(abs(min_x), abs(max_x), abs(min_y), abs(max_y)))
        return -limit, limit, -limit, limit

def get_parallelogram_vertices(center: List[float], normals: List[List[float]], lengths: List[float]) -> List[List[float]]:
    """
    Compute the vertices of a parallelogram in 2D or 3D space.

    The parallelogram is defined by its center, two normal vectors, and their corresponding lengths.

    :param center: A list of 2 or 3 floats representing the center of the parallelogram.
    :type center: List[float]
    :param normals: A list containing two normal vectors, each with 2 or 3 components.
    :type normals: List[List[float]]
    :param lengths: A list of two floats representing the lengths along each normal vector.
    :type lengths: List[float]
    :return: A list of vertices defining the parallelogram.
    :rtype: List[List[float]]

    :raises ValueError: If the input dimensions are inconsistent.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_parallelogram_vertices.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_parallelogram_vertices.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> center = [0, 0, 0]
        >>> normal1 = [1, 0, 0]
        >>> normal2 = [0, 1, 0]
        >>> lengths = [2, 4]
        >>> geom.get_parallelogram_vertices(center, [normal1, normal2], lengths)
        [[1.0, 2.0, 0.0], [-1.0, 2.0, 0.0], [-1.0, -2.0, 0.0], [1.0, -2.0, 0.0]]
    """
    if len(normals) != 2 or len(lengths) != 2:
        raise ValueError("Two normal vectors and two lengths are required.")

    center = np.array(center)
    normal1 = np.array(normals[0])
    normal2 = np.array(normals[1])
    length1 = lengths[0]
    length2 = lengths[1]

    vertex1 = center + normal1 * length1 / 2 + normal2 * length2 / 2
    vertex2 = center - normal1 * length1 / 2 + normal2 * length2 / 2
    vertex3 = center - normal1 * length1 / 2 - normal2 * length2 / 2
    vertex4 = center + normal1 * length1 / 2 - normal2 * length2 / 2

    return [vertex1.tolist(), vertex2.tolist(), vertex3.tolist(), vertex4.tolist()]

def get_parallelepiped_vertices(center: List[float], normals: List[List[float]], lengths: List[float]) -> List[List[float]]:
    """
    Compute the vertices of a parallelepiped in 3D space.

    The parallelepiped is defined by its center, three normal vectors, and their corresponding lengths.

    :param center: A list of three floats representing the center of the parallelepiped.
    :type center: List[float]
    :param normals: A list containing three normal vectors, each with three components.
    :type normals: List[List[float]]
    :param lengths: A list of three floats representing the lengths along each normal vector.
    :type lengths: List[float]
    :return: A list of vertices defining the parallelepiped.
    :rtype: List[List[float]]

    :raises ValueError: If the input dimensions are inconsistent.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_parallelepiped_vertices.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_parallelepiped_vertices.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> center = [0, 0, 0]
        >>> normal1 = [1, 0, 0]
        >>> normal2 = [0, 1, 0]
        >>> normal3 = [0, 0, 1]
        >>> lengths = [2, 2, 2]
        >>> geom.get_parallelepiped_vertices(center, [normal1, normal2, normal3], lengths)
        [[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0]]
    """
    if len(normals) != 3 or len(lengths) != 3:
        raise ValueError("Three normal vectors and three lengths are required.")

    center = np.array(center)
    normal1 = np.array(normals[0])
    normal2 = np.array(normals[1])
    normal3 = np.array(normals[2])
    length1 = lengths[0]
    length2 = lengths[1]
    length3 = lengths[2]

    vertex1 = center + normal1 * length1 / 2 + normal2 * length2 / 2 + normal3 * length3 / 2
    vertex2 = center - normal1 * length1 / 2 + normal2 * length2 / 2 + normal3 * length3 / 2
    vertex3 = center - normal1 * length1 / 2 - normal2 * length2 / 2 + normal3 * length3 / 2
    vertex4 = center + normal1 * length1 / 2 - normal2 * length2 / 2 + normal3 * length3 / 2
    vertex5 = center + normal1 * length1 / 2 + normal2 * length2 / 2 - normal3 * length3 / 2
    vertex6 = center - normal1 * length1 / 2 + normal2 * length2 / 2 - normal3 * length3 / 2
    vertex7 = center - normal1 * length1 / 2 - normal2 * length2 / 2 - normal3 * length3 / 2
    vertex8 = center + normal1 * length1 / 2 - normal2 * length2 / 2 - normal3 * length3 / 2

    return [vertex1.tolist(), vertex2.tolist(), vertex3.tolist(), vertex4.tolist(),
            vertex5.tolist(), vertex6.tolist(), vertex7.tolist(), vertex8.tolist()]

def get_intersection_points_of_line_with_cube(line: np.ndarray, cube_min: np.ndarray, cube_max: np.ndarray) -> np.ndarray:
    """
    Compute the intersection points of a line with a cube in 3D space.

    The line is defined by two points, and the cube is defined by its minimum and maximum
    corner points. The function determines all intersection points between the line and the cube's faces.

    :param line: A numpy array of shape (2, 3) representing two points on the line.
    :type line: np.ndarray
    :param cube_min: A numpy array [x_min, y_min, z_min] representing the minimum corner of the cube.
    :type cube_min: np.ndarray
    :param cube_max: A numpy array [x_max, y_max, z_max] representing the maximum corner of the cube.
    :type cube_max: np.ndarray
    :return: A numpy array of intersection points with shape (N, 3), where N is the number of intersection points.
    :rtype: np.ndarray

    :raises ValueError: If the inputs are not in the correct format or dimensions.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_intersection_points_of_line_with_cube.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_intersection_points_of_line_with_cube.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> import numpy as np
        >>> line = np.array([[0, 0, 0], [1, 1, 1]])
        >>> cube_min = np.array([-2, -2, -1])
        >>> cube_max = np.array([1, 2, 2])
        >>> intersection_points = geom.get_intersection_points_of_line_with_cube(line, cube_min, cube_max)
        >>> intersection_points
        array([[ 1.,  1.,  1.],
               [-1., -1., -1.]])
    """
    if line.shape != (2, 3):
        raise ValueError("Line must be a numpy array with shape (2, 3), representing two points in 3D space.")
    if cube_min.shape != (3,) or cube_max.shape != (3,):
        raise ValueError("cube_min and cube_max must be numpy arrays with shape (3,), representing 3D coordinates.")

    # Initialize a list to store intersection points.
    intersection_points = []

    # Define the planes for the cube faces in the form (A, B, C, D).
    planes = [
        (1, 0, 0, -cube_min[0]),  # Front face
        (-1, 0, 0, cube_max[0]),  # Back face
        (0, 1, 0, -cube_min[1]),  # Top face
        (0, -1, 0, cube_max[1]),  # Bottom face
        (0, 0, 1, -cube_min[2]),  # Left face
        (0, 0, -1, cube_max[2])   # Right face
    ]

    # Check intersection with each plane.
    for plane in planes:
        intersection_point = get_intersection_point_of_line_with_plane(line, np.array(plane))
        if intersection_point is not None:
            # Ensure the point is within the cube bounds.
            if np.all(cube_min <= intersection_point) and np.all(intersection_point <= cube_max):
                # Avoid duplicates by checking against existing points.
                if not any(np.allclose(intersection_point, point) for point in intersection_points):
                    intersection_points.append(intersection_point)

    return np.array(intersection_points)

def get_angle_between_vectors(v1: List[float], v2: List[float]) -> float:
    """
    Calculate the angle between two vectors in radians.

    The angle is computed using the dot product formula:
    angle = arccos((v1 • v2) / (\|v1\| * \|v2\|)).

    :param v1: A list representing the first vector [x, y, z].
    :type v1: List[float]
    :param v2: A list representing the second vector [x, y, z].
    :type v2: List[float]
    :return: The angle between the two vectors in radians.
    :rtype: float

    :raises ValueError: If either vector has zero magnitude.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_angle_between_vectors.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_angle_between_vectors.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> geom.get_angle_between_vectors([1, 0, 0], [0, 1, 0])
        1.5707963267948966
    """
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Vectors must have non-zero magnitude.")

    dot_product = np.dot(v1, v2)
    angle = np.arccos(dot_product / (norm_v1 * norm_v2))
    return angle

def get_angle_between_lines(l1: np.ndarray, l2: np.ndarray) -> float:
    """
    Calculate the angle between two lines in 3D space in radians.

    The lines are defined by two points each, and the angle is computed
    using the vectors derived from these points.

    :param l1: A numpy array of shape (2, 3) representing two points on the first line.
    :type l1: np.ndarray
    :param l2: A numpy array of shape (2, 3) representing two points on the second line.
    :type l2: np.ndarray
    :return: The angle between the two lines in radians.
    :rtype: float

    :raises ValueError: If either line's direction vector has zero magnitude.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_angle_between_lines.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_angle_between_lines.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> import numpy as np
        >>> l1 = np.array([[0, 0, 0], [1, 1, 1]])
        >>> l2 = np.array([[0, 0, 0], [-1, -1, -1]])
        >>> geom.get_angle_between_lines(l1, l2)
        3.141592653589793
    """
    if l1.shape != (2, 3) or l2.shape != (2, 3):
        raise ValueError("Each line must be defined by two points with shape (2, 3).")

    v1 = l1[1] - l1[0]
    v2 = l2[1] - l2[0]

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Lines must have non-zero direction vectors.")

    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    return angle

def get_centroid_of_points(points: np.ndarray) -> np.ndarray:
    """
    Calculate the centroid of a set of points in 2D or 3D space.

    The centroid is computed as the mean position of all the points, representing the geometric center.

    :param points: A numpy array of shape (N, D), where N is the number of points, and D is the dimensionality (2 or 3).
    :type points: np.ndarray
    :return: A numpy array representing the centroid [x, y, (z)].
    :rtype: np.ndarray

    :raises ValueError: If the input array is not a 2D array or has incorrect dimensions.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_centroid_of_points.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_centroid_of_points.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> import numpy as np
        >>> points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        >>> geom.get_centroid_of_points(points)
        array([0.33333333, 0.33333333, 0.        ])
    """
    if points.ndim != 2:
        raise ValueError("Input points must be a 2D numpy array with shape (N, D).")
    if points.shape[1] not in [2, 3]:
        raise ValueError("Each point must have 2 or 3 coordinates.")

    return np.mean(points, axis=0)

def get_angle_between_planes(plane1: np.ndarray, plane2: np.ndarray) -> float:
    """
    Calculate the acute angle between two planes in 3D space in radians.

    Each plane is represented by a numpy array [A, B, C, D] corresponding to the
    plane equation Ax + By + Cz + D = 0. The angle between the planes is defined
    as the angle between their normal vectors. Since a normal vector can be reversed
    without changing the plane, the acute angle is returned.

    :param plane1: A numpy array [A, B, C, D] representing the first plane.
    :type plane1: np.ndarray
    :param plane2: A numpy array [A, B, C, D] representing the second plane.
    :type plane2: np.ndarray
    :return: The acute angle between the two planes in radians.
    :rtype: float

    :raises ValueError: If either plane is not a numpy array of shape (4,) or if a plane's normal vector is zero.

    :Example:

    .. literalinclude:: ../../rsaitehu_geometry/examples/example_get_angle_between_planes.py
       :language: python
       :linenos:
       :caption: Interactive Example from example_get_angle_between_planes.py

    Automated test:

    .. doctest::

        >>> import rsaitehu_geometry as geom
        >>> import numpy as np
        >>> plane1 = np.array([0, 0, 1, -3])  # Plane: z = 3
        >>> plane2 = np.array([0, 1, 1, -4])  # Some inclined plane
        >>> angle = geom.get_angle_between_planes(plane1, plane2)
        >>> angle
        0.7853981633974484
    """
    if plane1.shape != (4,):
        raise ValueError("plane1 must be a numpy array with shape (4,), representing [A, B, C, D].")
    if plane2.shape != (4,):
        raise ValueError("plane2 must be a numpy array with shape (4,), representing [A, B, C, D].")

    # Extract normal vectors from the plane equations
    normal1 = plane1[:3]
    normal2 = plane2[:3]

    norm1 = np.linalg.norm(normal1)
    norm2 = np.linalg.norm(normal2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("The normal vector of a plane cannot be the zero vector.")

    # Compute the dot product of the normals
    dot_product = np.dot(normal1, normal2)

    # Compute the acute angle between the two normals (and hence the planes)
    # Use absolute value to always get the acute angle
    cosine_angle = abs(dot_product) / (norm1 * norm2)

    # Clip cosine_angle to avoid numerical issues outside the range [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)

    return angle

