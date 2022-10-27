import numpy as np

# Point class
class Point:
    # Class constructor
    def __init__(self, x, y, z=None):
        self.x = x # X coordinate of the point
        self.y = y # Y coordinate of the point
        self.z = z # Z coordinate of the point


    # Overloading of the + operator
    def __add__(self, point):
        x = self.x + point.x
        y = self.y + point.y
        
        if self.z is not None and point.z is not None:
            z = self.z + point.z
            return Point(x, y, z)
        else:
            return Point(x, y)


    # Overloading of the - operator
    def __sub__(self, point):
        x = self.x - point.x
        y = self.y - point.y

        if self.z is not None and point.z is not None:
            z = self.z - point.z
            return Point(x, y, z)
        else:
            return Point(x, y)


    # Overloading of the * operator
    def __mult__(self, point):
        x = self.x * point.x
        y = self.y * point.y

        if self.z is not None and point.z is not None:
            z = self.z * point.z
            return Point(x, y, z)
        else:
            return Point(x, y)


    # Overloading of the / operator
    def __truediv__(self, divisor):
        x = self.x / divisor
        y = self.y / divisor

        if self.z is not None:
            z = self.z / divisor
            return Point(x, y, z)
        else:
            return Point(x, y)


    # Overloading of the print function
    def __repr__(self):
        if self.z is not None:
            return f'({self.x}, {self.y}, {self.z})'
        else:
            return f'({self.x}, {self.y})'


    # Function to compute the euclidean distance from another point
    def distance(self, point):
        if self.z is not None:
            return np.sqrt((self.x - point.x)**2 + (self.y - point.y)**2 + (self.z - point.z)**2) 
        else:
            return np.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)


    # Function to plot the point
    def draw(self, ax, s=4, c='k'):
        if self.z is not None:
            ax.scatter(self.x, self.y, self.z, s=s, c=c)
        else:
            ax.scatter(self.x, self.y, s=s, c=c)


# Line class
class Line:
    # Class constructor
    def __init__(self, p1, p2):
        self.p1 = p1 # First point of the segment
        self.p2 = p2 # Second point of the segment

        # Slope and intercept of the line
        self.m = self.p2.y - self.p1.y / self.p2.x - self.p1.x
        self.q = self.p1.y - self.m * self.p1.x


    # Function to compute the intersection point of two lines
    def LineIntersection(self, line):
        # Line AB represented as a1x + b1y = c1
        a1 = self.p2.y - self.p1.y
        b1 = self.p1.x - self.p2.x
        c1 = a1*(self.p1.x) + b1*(self.p1.y)

        # Line CD represented as a2x + b2y = c2
        a2 = line.p2.y - line.p1.y
        b2 = line.p1.x - line.p2.x
        c2 = a2*(line.p1.x) + b2*(line.p1.y)

        determinant = a1 * b2 - a2 * b1

        if (determinant == 0):
            return None
        else:
            x = (b2*c1 - b1*c2) / determinant
            y = (a1*c2 - a2*c1) / determinant

            return Point(x, y)


    # Function to compute the intersection point with a segment
    def SegmentIntersection(self, segment):
        # Computing the side of the points of the segment w.r.t. the line
        side_p1 = ((self.p2.x - self.p1.x)*(segment.p1.y - self.p1.y) - (self.p2.y - self.p1.y)*(segment.p1.x - self.p1.x))
        side_p2 = ((self.p2.x - self.p1.x)*(segment.p2.y - self.p1.y) - (self.p2.y - self.p1.y)*(segment.p2.x - self.p1.x))

        if side_p1 * side_p2 > 0:
            # There is no intersection
            return None
        else:
            intersection = self.LineIntersection(Line(segment.p1, segment.p2))
            return intersection
    

    # Function to draw the line
    def draw(self, ax, c='k', lw=1, **kwargs):
        ax.axline((self.p1.x, self.p1.y), (self.p2.x, self.p2.y), color=c, lw=lw, **kwargs)



# Segment class
class Segment:
    # Class constructor
    def __init__(self, p1, p2):
        self.p1 = p1 # First point of the segment
        self.p2 = p2 # Second point of the segment


    # Function to compute the intersection point between two lines segments
    def SegmentIntersection(self, segment):
        det = (segment.p2.y - segment.p1.y) * (self.p2.x - self.p1.x) - (segment.p2.x - segment.p1.x) * (self.p2.y - self.p1.y)

        if det == 0: # parallel
            return None

        ua = ((segment.p2.x - segment.p1.x) * (self.p1.y - segment.p1.y) - (segment.p2.y - segment.p1.y) * (self.p1.x - segment.p1.x)) / det
        if ua < 0 or ua > 1: # out of range
            return None

        ub = ((self.p2.x - self.p1.x) * (self.p1.y - segment.p1.y) - (self.p2.y - self.p1.y) * (self.p1.x - segment.p1.x)) / det
        if ub < 0 or ub > 1: # out of range
            return None

        # Computing the coordinates of the intersection point
        x = self.p1.x + ua * (self.p2.x - self.p1.x)
        y = self.p1.y + ua * (self.p2.y - self.p1.y)

        return Point(x, y)


    # Function to compute the intersection points between a line segment and a polygon
    def PolygonIntersectionPoints(self, polygon):
        points = []
        # Iterating over the vertices of the polygon
        for i in range(polygon.num_vertices):
            # Extracting the i-th side of the polygon
            segment = Segment(polygon.vertices[i], polygon.vertices[(i+1) % polygon.num_vertices])
            
            # Computing the intersection point between the segment and the current side of the polygon
            intersection = self.intersectionPoint(segment)
            if intersection is not None:
                points.append(intersection)

        return points


    # Function to plot the line segment
    def draw(self, ax, c='k', lw=1, **kwargs):
        ax.plot([self.p1.x, self.p2.x], [self.p1.y, self.p2.y], c=c, lw=lw, **kwargs)



# Polygon class
class Polygon:
    # Class constructor
    def __init__(self, vertices):
        self.vertices = vertices # Vertices of the polygon
        self.num_vertices = len(vertices) # Number of vertices of the polygon

        # Computing the reference point to sort the vertices in clockwise order
        self.reference_point = self.__referencePoint()

        # Sorting the vertices clockwise
        self.vertices = sorted(self.vertices, key=self.__clockwiseOrder)

        # Computing the area of the polygon
        self.area = self.computeArea()

        # Computing the centroid of the polygon
        self.centroid = self.computeCentroid()


    # Function to compute the centroid of the polygon
    def computeCentroid(self):
        sum_x, sum_y = 0, 0
        # Iterating over the vertices of the polygon
        for i in range(self.num_vertices):
            # Extracting the i-th side of the polygon
            a, b = self.vertices[i], self.vertices[(i+1) % self.num_vertices]

            sum_x -= (a.x + b.x) * (a.x * b.y - b.x * a.y)
            sum_y -= (a.y + b.y) * (a.x * b.y - b.x * a.y)

        # Computing the coordinates of the centroid of the polygon
        x = (1 / (6 * self.area)) * sum_x if self.area > 0 else self.reference_point.x
        y = (1 / (6 * self.area)) * sum_y if self.area > 0 else self.reference_point.y

        return Point(x, y)


    # Shoelace algorithm to compute the area of the polygon
    def computeArea(self):
        area = 0.0
        # Iterating over the vertices of the polygon
        for i in range(self.num_vertices):
            area += self.vertices[i].x * self.vertices[(i+1) % self.num_vertices].y
            area -= self.vertices[(i+1) % self.num_vertices].x * self.vertices[i].y
            
        return np.abs(area) / 2.0


    # Winding number algorithm to check if a point belongs to the polygon
    def containsPoint(self, point):
        general_side = None
        # Iterating over the vertices of the polygon
        for i in range(self.num_vertices):
            # Extracting the i-th side of the polygon
            a, b = self.vertices[i], self.vertices[(i+1) % self.num_vertices]

            # Computing the side of the point w.r.t. the segment
            side = (point.y - a.y) * (b.x - a.x) - (point.x - a.x) * (b.y - a.y)

            # The point lies on the current side of the polygon
            if side == 0:
                continue

            # Initializing the side of the polygon
            if general_side is None:
                general_side = side
                continue

            # If the side differs from the general one the point does not belong to the polygon
            if general_side * side < 0:
                return False

        return True


    # Function to extract the intersection polygon of two polygons
    def intersectionPolygon(self, polygon):
        points = []

        # Adding the points of the first polygon belonging to the second one
        for vertex in self.vertices:
            if polygon.containsPoint(vertex):
                points.append(vertex)

        # Adding the points of the second polygon belonging to the first one
        for vertex in polygon.vertices:
            if self.containsPoint(vertex):
                points.append(vertex)

        # Adding the intersection points between the polygons
        for i in range(self.num_vertices):
            side = Segment(self.vertices[i], self.vertices[(i+1) % self.num_vertices])
            points += side.intersectionPoints(polygon)

        return Polygon(points) if len(points) > 0 else None

    
    # Function to plot the polygon
    def draw(self, ax, c='k', lw=1, **kwargs):
        pts = self.vertices
        pts.append(Point(pts[0].x, pts[0].y))

        ax.plot([pt.x for pt in pts], [pt.y for pt in pts], c=c, lw=lw, **kwargs)

    
    # Function to compute the reference point of the polygon
    def __referencePoint(self):
        # Computing the sum of the vertices of the polygon
        sum = self.vertices[0]
        for i in range(1, self.num_vertices):
            sum += self.vertices[i]

        return sum / self.num_vertices


    # Function to define a clockwise order of the vertices of the polygon
    def __clockwiseOrder(self, point):
        # Defining an origin and a reference vector
        origin = self.reference_point
        reference_vector = [0, -1]

        # Vector between point and the origin: v = p - o
        vector = [point.x - origin.x, point.y - origin.y]
        
        # Length of vector
        len_vector =  np.linalg.norm(vector)

        # If length is zero there is no angle
        if len_vector == 0:
            return - np.pi, 0

        # Normalize vector: v/||v||
        normalized = [vector[0] / len_vector, vector[1] / len_vector]
        dotprod  = normalized[0] * reference_vector[0] + normalized[1] * reference_vector[1]
        diffprod = reference_vector[1] * normalized[0] - reference_vector[0] * normalized[1]
        angle = np.arctan2(diffprod, dotprod)

        # Negative angles represent counter-clockwise angles so we need to subtract them 
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2 * np.pi + angle, len_vector

        return angle, len_vector



# Rectangle class
class Rectangle(Polygon):
    # Class constructor
    def __init__(self, center, width, height):
        self.center = center # Center of the rectangle
        self.width = width # Width of the rectangle
        self.height = height # Height of the rectangle
        self.west = center.x - width / 2 # West coordinate
        self.east = center.x + width / 2 # East coordinate
        self.north = center.y + height / 2 # North coordinate
        self.south = center.y - height / 2 # South coordinate

        # Initializing the parent class
        super().__init__([
            Point(self.west, self.south), 
            Point(self.west, self.north), 
            Point(self.east, self.north), 
            Point(self.east, self.south)
        ])