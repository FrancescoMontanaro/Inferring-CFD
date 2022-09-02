import vtk
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy


'### GLOBAL VARIABLES AND CONSTANTS ###'

free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream
sensor_distance_range = np.array([10, 100]) # minimum and maximum distance of the sensor from the origin
sensor_angle_range = np.array([0, 2]) # Range of the angle of the sensor (0pi, 2pi)
sensor_length = 5.0 # Length of the side of the sensor
n_sensors = 20 # Nmber of sensors to generate
bins_count = 128 # Numer of bins



"### FUNCTIONS ###"

# Function to cut the space in order to extract the target section
def extractSection(poly_data):
    # Creating the cutting planes
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, 0.5)
    plane.SetNormal(0, 0, 1) #Orthogonal to the z axis

    # Cutting the space in the first direction
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(poly_data)
    cutter.Update()

    # Extracting the first target section
    target_section = cutter.GetOutput()

    return target_section


# Function to extract the cells from the space
def extractCells(target_section):
    # Extracting the flow quantities
    p = vtk_to_numpy(target_section.GetCellData().GetArray('p'))
    U = vtk_to_numpy(target_section.GetCellData().GetArray('U'))
    U[:,2] = 0.0 # Removing the z component of the velocity

    # Computing the magnitude of the velocity vector
    U = np.array([np.linalg.norm(u) for u in U])

    # Normalizing the velocity w.r.t. the free stream velocity magnitude
    U /= free_stream__velocity_magnitude

    # Extracting the number of cells
    n_cells = target_section.GetNumberOfCells()

    cells = []
    # Iterating over the cells of the section
    for idx in range(n_cells):
        # Extracting the i-th cell
        cell = target_section.GetCell(idx)

        # Extracting the points of the cell
        cell_points = np.array(vtk_to_numpy(cell.GetPoints().GetData()))
        cell_points = cell_points[:, :2] # Removing the z coordinate 
        cell_points = [Point(x, y) for x, y in cell_points]

        # Extracting the flow quantites of the cell
        cell_p = p[idx]
        cell_U = U[idx]

        # Adding to the main list a cell object
        cells.append(Cell(cell_points, cell_p, cell_U))

    return cells



"### CLASSES ###"

# Point class
class Point:
    # Class constructor
    def __init__(self, x, y):
        self.x = x # X coordinate of the point
        self.y = y # Y coordinate of the point


    # Overloading of the + operator
    def __add__(self, point):
        x = self.x + point.x
        y = self.y + point.y

        return Point(x, y)

    # Overloading of the - operator
    def __sub__(self, point):
        x = self.x - point.x
        y = self.y - point.y

        return Point(x, y)

    # Overloading of the * operator
    def __mult__(self, point):
        x = self.x * point.x
        y = self.y * point.y

        return Point(x, y)


    # Overloading of the / operator
    def __truediv__(self, divisor):
        x = self.x / divisor
        y = self.y / divisor

        return Point(x, y)

    # Overloading of the print function
    def __repr__(self):
        return f'({self.x}, {self.y})'

    # Function to plot the point
    def draw(self, ax, s=4, c='k'):
        ax.scatter(self.x, self.y, s=s, c=c)



# Segment class
class Segment():
    # Class constructor
    def __init__(self, p1, p2):
        self.p1 = p1 # First point of the segment
        self.p2 = p2 # Second point of the segment


    # Function to compute the intersection point between two lines segments
    def intersectionPoint(self, segment):
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
    def intersectionPoints(self, polygon):
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
        self.west = center.x - width # West coordinate
        self.east = center.x + width # East coordinate
        self.north = center.y + height # North coordinate
        self.south = center.y - height # South coordinate

        # Initializing the parent class
        super().__init__([
            Point(self.west, self.south), 
            Point(self.west, self.north), 
            Point(self.east, self.north), 
            Point(self.east, self.south)
        ])


    # Function to check if a cell belongs to the rectangle
    def containsCell(self, cell):
        # Extracting the coordinates of the centroid of the cell
        x = cell.centroid.x
        y = cell.centroid.y

        return self.containsPoint(Point(x, y))


    # Function to check if a rectangle intersects the current one
    def intersects(self, range):
        return not (range.west > self.east or range.east < self.west or range.south > self.north or range.north < self.south)



# Quadtree class
class QuadTree:
    # Class constructor
    def __init__(self, boundary, capacity = 4):
        self.boundary = boundary # Boundary of the Quad Tree
        self.capacity = capacity # Number of childs of the quad tree
        self.cells = [] # Cells of the Quad Tree
        self.divided = False # Flag to indicate if the Quad Tree has been divided


    # Function to inset a cell into the quad tree
    def insert(self, cell):
        # Checking if the cell is within the QuadTree boundaries
        if not self.boundary.containsCell(cell):
            return False

        # Adding the point if the capacity has not been reached
        if len(self.cells) < self.capacity:
            self.cells.append(cell)
            return True

        # Dividing the quad tree
        if not self.divided:
            self.divide()

        # Inserting the cell into the correct child
        if self.nw.insert(cell):
            return True
        elif self.ne.insert(cell):
            return True
        elif self.sw.insert(cell):
            return True
        elif self.se.insert(cell):
            return True

        return False


    # Function to query the cells belonging to a specified range of values
    def queryRange(self, range):
        cells_found = []

        # Checking if the quad tree intersects the specified range
        if not self.boundary.intersects(range):
            return []

        # Adding the cells belonging to the range
        for cell in self.cells:
            if range.containsCell(cell):
                cells_found.append(cell)

        # Adding the cells belonging to the child trees
        if self.divided:
            cells_found.extend(self.nw.queryRange(range))
            cells_found.extend(self.ne.queryRange(range))
            cells_found.extend(self.sw.queryRange(range))
            cells_found.extend(self.se.queryRange(range))

        return cells_found


    # Function to divide the quad tree
    def divide(self):
        # Center of the child trees
        center_x = self.boundary.center.x
        center_y = self.boundary.center.y

        # Computing the width and height of the child trees
        new_width = self.boundary.width / 2
        new_height = self.boundary.height / 2

        # Creating the child trees
        nw = Rectangle(Point(center_x - new_width, center_y + new_height), new_width, new_height)
        self.nw = QuadTree(nw)

        ne = Rectangle(Point(center_x + new_width, center_y + new_height), new_width, new_height)
        self.ne = QuadTree(ne)

        sw = Rectangle(Point(center_x - new_width, center_y - new_height), new_width, new_height)
        self.sw = QuadTree(sw)

        se = Rectangle(Point(center_x + new_width, center_y - new_height), new_width, new_height)
        self.se = QuadTree(se)

        # Flagging the current tree as divided
        self.divided = True


    # Overloading of the len() function
    def __len__(self):
        # Initilalizing the counter to the cells of the current quad tree
        count = len(self.cells)

        # Adding to the counter the cumber of cells belonging to the child trees
        if self.divided:
            count += len(self.nw) + len(self.ne) + len(self.sw) + len(self.se)

        return count


    # Function to plot the quad tree and its childs
    def draw(self, ax):
        self.boundary.draw(ax)

        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.sw.draw(ax)
            self.se.draw(ax)


# Cell class
class Cell(Polygon):
    # Class constructor
    def __init__(self, vertices, p, U):
        # Initializing parent class
        super().__init__(vertices)

        self.p = p # Cell Pressure value
        self.U = U # Cell Velocity value



# Sensor class
class Sensor(Rectangle):
    # Class constructor
    def __init__(self, length, r=None, theta=None):
        # Extracting the radius and the angle of the sensor w.r.t. the origin
        self.r = np.random.uniform(np.min(sensor_distance_range), np.max(sensor_distance_range)) if r is None else r # Distance of the sensor
        self.theta = np.pi * np.random.uniform(np.min(sensor_angle_range), np.max(sensor_angle_range)) if theta is None else theta # Angle of the sensor

        # Computing the coordinates of the center of the sensor
        x = self.r * np.cos(self.theta)
        y = self.r * np.sin(self.theta)

        # Initializing the parent class
        super().__init__(Point(x, y), length / 2, length / 2)

        # List of the cells belonging to the sensor 
        self.cells = []

        # Signal of the flow fields
        self.signal = {}

    
    # Function to generate the signal of the flow fields for the sensor
    def generateSignal(self, bins_count):
        # Binning operation
        bins = self.__surfaceBinning(bins_count)

        # Upsampling the signal in order to obtain the values of the empty bins
        bins = self.__upsample(bins)

        self.signal = {
            "p": np.array([bin["p"] for bin in bins]), 
            "U": np.array([bin["U"] for bin in bins]),
        }

        return self.signal


    # Function to plot the signal of the specified flow field
    def displaySignal(self, field_name):
        data = self.signal[field_name]
        plt.plot(data)
        plt.show()


    # Function to plot the sensor and its cells
    def displaySensor(self):
        # Plotting the sensor's cells
        plt.figure(figsize=(700/72, 500/72), dpi=72)
        ax = plt.subplot()
        ax.set_xlim(self.west-10, self.east+10)
        ax.set_ylim(self.south-10, self.north+10)
        self.draw(ax)
        for cell in self.cells:
            cell.draw(ax)
        plt.tight_layout()
        plt.show()

    
    # Function to perform the binning operation by averaging the values of the cells fully belonging and
    # intersecting each bin.
    def __surfaceBinning(self, bins_count):
        # Computing the bounds of the bins
        bins_bounds = np.linspace(self.south, self.north, num=bins_count+1)

        # Sorting the cells according to their maximum Y coordinate: to speed up the algorithm
        self.cells = sorted(self.cells, key=lambda cell: np.min([vertex.y for vertex in cell.vertices]))   

        bins = []
        # Iterating over the bins
        for idx in range(len(bins_bounds) - 1):
            # Extracting the minimum and maximum y coordinate of the bin
            bin__min_y = np.min([bins_bounds[idx], bins_bounds[idx+1]])
            bin__max_y = np.max([bins_bounds[idx], bins_bounds[idx+1]])

            # Creating a polygon object whose vertices are the ones of the bin
            bin = Polygon([
                Point(self.west, bin__min_y),
                Point(self.west, bin__max_y),
                Point(self.east, bin__max_y),
                Point(self.east, bin__min_y)
            ])

            bin_cells = []
            # Iterating over the cells of the sensor
            for cell in self.cells:
                # Stopping the execution if the cells are above the upper bound
                if np.min([vertex.y for vertex in cell.vertices]) > bin__max_y:
                    break

                # Extracting the cells that fully belongs to the polygon
                fully_belongs = True
                for vertex in cell.vertices:
                    if not bin.containsPoint(vertex):
                        fully_belongs = False

                if fully_belongs:
                    bin_cells.append(cell)
                    continue

                # Extracting the cells that partially belong to the current bin
                intersection_polygon = bin.intersectionPolygon(cell)
                if intersection_polygon is not None:
                    bin_cells.append(Cell(intersection_polygon.vertices, cell.p, cell.U))
            
            if len(bin_cells) > 0:
                # Extracting the flow quantities of the cells belonging to the current bin
                cells_p = np.array([cell.p for cell in bin_cells])
                cells_U = np.array([cell.U for cell in bin_cells])

                # Extracting the surfaces of the cells belonging to the current bin
                cells_areas = np.array([cell.area for cell in bin_cells])

                # Computing the sum of the cells belonging to current bin
                sum_areas = np.sum(cells_areas)

                # Computing the average of the flow quantities of the cells belonging to the current bin
                bin_p = np.sum(cells_areas * cells_p) / sum_areas if sum_areas > 0.0 else 0.0
                bin_U = np.sum(cells_areas * cells_U) / sum_areas if sum_areas > 0.0 else 0.0

            else:
                bin_p = None
                bin_U = None

            bins.append({"p": bin_p, "U": bin_U})

        return bins


    # Function to perform the binning operation by averaging the values of the cells whose centroid belongs
    # to the i-t bin.
    def __centroidBinning(self, bins_count):
        # Computing the bounds of the bins
        bins_bounds = np.linspace(self.south, self.north, num=bins_count+1)

        bins = []
        # Iterating over the bins
        for idx in range(len(bins_bounds) - 1):
            # Extracting the minimum and maximum y coordinate of the bin
            bin__min_y = np.min([bins_bounds[idx], bins_bounds[idx+1]])
            bin_max_y = np.max([bins_bounds[idx], bins_bounds[idx+1]])

            # Creating a polygon object whose vertices are the ones of the bin
            bin = Polygon([
                Point(self.west, bin__min_y),
                Point(self.west, bin_max_y),
                Point(self.east, bin_max_y),
                Point(self.east, bin__min_y)
            ])

            # Extracting the cells belonging to the current bin
            bin_cells = [cell for cell in self.cells if bin.containsPoint(cell.centroid)]

            # Computing the pressure and velocity field associated to the i-th bin
            bin_p = float(np.mean([cell.p for cell in bin_cells])) if len(bin_cells) > 0 else None
            bin_U = float(np.mean([cell.U for cell in bin_cells])) if len(bin_cells) > 0 else None

            bins.append({"p": bin_p, "U": bin_U})

        return bins


    @staticmethod
    # Function to upsample the signal by assigning to the empty bins the interpolation of the
    # values of the closest one
    def __upsample(bins):
        # Setting the values of the first and last bin to the closest ones, if thay are empty.
        if(bins[0]["p"] is None or bins[0]["U"] is None):
            i, upper_bin = 0, None
            while(upper_bin is None and i < len(bins)):
                if(bins[i]["p"] is not None and bins[i]["U"] is not None):
                    upper_bin = bins[i]
                i += 1

            bins[0]["p"] = float(upper_bin["p"])
            bins[0]["U"] = float(upper_bin["U"])

        if(bins[-1]["p"] is None or bins[-1]["U"] is None):
            i, lower_bin = len(bins) - 1, None
            while(lower_bin is None and i > 0):
                if(bins[i]["p"] is not None and bins[i]["U"] is not None):
                    lower_bin = bins[i]
                i -= 1

            bins[-1]["p"] = float(lower_bin["p"])
            bins[-1]["U"] = float(lower_bin["U"])   

        # For each empty bin, obtains its values by interpolating the values of the adjacent ones.
        for i in range(1, len(bins)-1):
            if(bins[i]["p"] is None or bins[i]["U"] is None):
                j, lower_bin = i, None
                while(lower_bin is None and j >= 0):
                    if(bins[j]["p"] is not None and bins[j]["U"] is not None):
                        lower_bin = bins[j]
                        lower_weight = 1 / i - j
                    j -= 1

                j, upper_bin = i, None
                while(upper_bin is None and j < len(bins)):
                    if(bins[j]["p"] is not None and bins[j]["U"] is not None):
                        upper_bin = bins[j]
                        upper_weight = 1 / j - i
                    j += 1

                # Weighted average of the values of the closest bins
                bins[i]["p"] = np.average([lower_bin["p"], upper_bin["p"]], weights=[lower_weight, upper_weight])
                bins[i]["U"] = np.average([lower_bin["U"], upper_bin["U"]], weights=[lower_weight, upper_weight])

        return bins



# Function to generate the signal of the flow fields for n sensors located at a random position
# in te space
def sensorSignal(reader):
    # Extracting the spatial data
    poly_data = reader.GetOutput()

    # Cutting the space to extract the target section
    target_section = extractSection(poly_data)

    # Extracting the X and Y boundaries of the space
    bounds = target_section.GetBounds()
    bounds = bounds[:4]

    # Computing the width and the height of the space 
    width = np.abs(np.max(bounds[:2]) - np.min(bounds[:2]))
    height = np.abs(np.max(bounds[-2:]) - np.min(bounds[-2:]))

    # Creating the space domain of the QuadTree
    domain = Rectangle(Point(0, 0), width, height)

    # Creating the QuadTree
    quad_tree = QuadTree(domain)

    # Extract the cells of the space
    cells = extractCells(target_section)

    # Inserting the cells into the quad tree
    for cell in cells:
        quad_tree.insert(cell)
    
    signals = []
    # Iterating over the number of sensors to extract
    for _ in range(n_sensors):
        sensor = Sensor(sensor_length)

        # Extracting the cells belonging to the current sensor
        sensor.cells = quad_tree.queryRange(sensor)
   
        # Plotting the sensor
        sensor.displaySensor()

        # Generating the signal
        signal = sensor.generateSignal(bins_count)

        # Plotting the signal
        #sensor.displaySignal("p")

        # Adding the data into the main list
        signal_data = {"p": [], "U": [], "r": None, "theta": None}
        signal_data["p"] = signal["p"]
        signal_data["U"] = signal["U"]
        signal_data["r"] = sensor.r
        signal_data["theta"] = sensor.theta

        signals.append(signal_data)

    return signals