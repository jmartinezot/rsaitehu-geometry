Introduction to the Geometry Module
-----------------------------------

The Geometry Module provides a comprehensive set of functions for processing and analyzing geometric objects in 2D and 3D space. This module is ideal for tasks involving planes, lines, parallelograms, and other geometric shapes, enabling users to perform mathematical and spatial computations efficiently. The module is particularly useful in applications such as 3D modeling, computer graphics, and computational geometry.

Core Functionalities
--------------------

1. **Plane Operations:**

   - **Get Plane from Points:**
     - Function: `get_plane_from_list_of_three_points(points: List[List[float]])`
     - Description: Calculates the equation of a plane (Ax + By + Cz + D = 0) given three non-collinear points in 3D space.

   - **Find Closest Plane:**
     - Function: `find_closest_plane(points: List[List[float]])`
     - Description: Determines the best-fitting plane to a set of 3D points using Euclidean distance and Singular Value Decomposition (SVD).

   - **Point Projection onto Plane:**
     - Function: `get_point_of_plane_closest_to_given_point(plane: np.ndarray, point: np.ndarray)`
     - Description: Finds the point on a plane closest to a given point in 3D space.
     
   - **Get Plane Equation:**
     - Function: `get_plane_equation(normal1: List[float], normal2: List[float], point: List[float])`
     - Description: Computes the equation of a plane (Ax + By + Cz + D = 0) defined by two normal vectors and a point in 3D space. The normal vectors are used to determine the orientation of the plane, and the point ensures proper positioning in space. 
     
   - **Get Distance from Points to Plane:**
     - Function: `get_distance_from_points_to_plane(points: Union[Tuple[float, float, float], np.ndarray], plane: Tuple[float, float, float, float])`
     - Description: Calculates the perpendicular distance(s) from one or more points to a plane in 3D space using the plane equation. Supports both a single point as a tuple or multiple points in a numpy array.

   - **Fit Plane Using SVD:**
     - Function: `fit_plane_svd(points: np.ndarray)`
     - Description: Fits a plane to a set of 3D points by minimizing the sum of squared perpendicular distances using Singular Value Decomposition (SVD). Returns the plane equation and the sum of squared errors (SSE). 
     
   - **Get Intersection Point of Line with Plane:**
     - Function: `get_intersection_point_of_line_with_plane(line: np.ndarray, plane: np.ndarray)`
     - Description: Calculates the intersection point of a line with a plane in 3D space. If the line is parallel to the plane, it returns None. The line is represented by two points, and the plane is defined by its equation coefficients.
     
   - **Get Two Perpendicular Unit Vectors in Plane:**
     - Function: `get_two_perpendicular_unit_vectors_in_plane(plane: np.ndarray)`
     - Description: Computes two unit vectors that are perpendicular to each other and lie within the given plane. Useful for constructing shapes or determining directions within the plane.

   - **Get Best Plane from Two Line Segments:**
     - Function: `get_best_plane_from_points_from_two_segments(segment_1: np.ndarray, segment_2: np.ndarray)`
     - Description: Determines the best-fitting plane for four points derived from two line segments. Uses Singular Value Decomposition (SVD) to minimize the sum of squared errors (SSE) and returns the plane equation along with the error.
     
   - **Generate a Polygon from Plane Equation and Point:**
     - Function: `get_a_polygon_from_plane_equation_and_point(plane: np.ndarray, point: np.ndarray, scale: float = 1.0)`
     - Description: Generates a quadrilateral polygon centered around a specific point and lying in the specified plane. The polygon's vertices are scaled relative to the point and aligned with the plane's orientation.   

2. **Graph and Plotting Limits:**

   - **Graph Limits:**
     - Function: `get_limits_of_graph_from_limits_of_object(min_x: float, max_x: float, min_y: float, max_y: float, min_z: Union[float, None] = None, max_z: Union[float, None] = None
) -> Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float]]:)`
     - Description: Computes graph limits to ensure visibility of a 2D or 3D object centered at the origin.

3. **Geometric Shapes:**

   - **Parallelogram Vertices:**
     - Function: `get_parallelogram_vertices(center: List[float], normals: List[List[float]], lengths: List[float])`
     - Description: Computes the vertices of a 2D or 3D parallelogram given its center, two normal vectors, and their corresponding side lengths. The function ensures the output vertices define the parallelogram in the same dimensional space as the input.

   - **Parallelepiped Vertices:**
     - Function: `get_parallelepiped_vertices(center: List[float], normals: List[List[float]], lengths: List[float])`
     - Description: Computes the vertices of a 3D parallelepiped given its center, three normal vectors, and their corresponding side lengths. The function returns the eight vertices defining the parallelepiped in 3D space.

4. **Line and Vector Operations:**

   - **Intersection of Line and Cube:**
     - Function: `get_intersection_points_of_line_with_cube(line: np.ndarray, cube_min: np.ndarray, cube_max: np.ndarray)`
     - Description: Computes the intersection points of a line with a cube.

   - **Angle Computations:**
     - Function: `get_angle_between_vectors(v1: List[float], v2: List[float])`
     - Description: Calculates the angle between two vectors in radians.

     - Function: `get_angle_between_lines(l1: np.ndarray, l2: np.ndarray)`
     - Description: Computes the angle between two lines in 3D space.
     
5. **Point Set Operations:**

   - **Centroid of Points:**
     - Function: `get_centroid_of_points(points: np.ndarray)`
     - Description: Calculates the centroid (geometric center) of a set of points in 2D or 3D space.

Use Cases
---------

- **3D Modeling and Analysis:** Calculate and manipulate planes, lines, and shapes for 3D applications.
- **Geometric Calculations:** Perform advanced spatial computations for engineering, architecture, or scientific research.
- **Graphing and Visualization:** Determine optimal graph limits for visualizing geometric data in 2D or 3D space.

This module provides a versatile foundation for working with geometric objects, simplifying complex mathematical computations while offering high precision and reliability.


