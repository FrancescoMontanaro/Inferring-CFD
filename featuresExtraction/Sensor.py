import vtk
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy


# Global variables and constants
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream
sensor_distance_range = np.array([10, 100]) # minimum and maximum distance of the sensor from the origin
sensor_angle_range = np.array([0, 2]) # Range of the angle of the sensor (0pi, 2pi)
sensor_length = 5.0 # Length of the side of the sensor
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

        cells.append(Cell(cell_points, cell_p, cell_U))

    return cells



"### CLASSES ###"
# Point class
class Point:
    # Class constructor
    def __init__(self, x, y):
        self.x = x
        self.y = y


    def __add__(self, point):
        x = self.x + point.x
        y = self.y + point.y

        return Point(x, y)


    def __sub__(self, point):
        x = self.x - point.x
        y = self.y - point.y

        return Point(x, y)

    def __mult__(self, point):
        x = self.x * point.x
        y = self.y * point.y

        return Point(x, y)

    
    def __truediv__(self, divisor):
        x = self.x / divisor
        y = self.y / divisor

        return Point(x, y)


    def __repr__(self):
        return f'({self.x}, {self.y})'


    def draw(self, ax, s=4, c='k'):
        ax.scatter(self.x, self.y, s=s, c=c)



class Segment():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


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


    def intersectionPoints(self, polygon):
        points = []
        for i in range(polygon.num_vertices):
            segment = Segment(polygon.vertices[i], polygon.vertices[(i+1) % polygon.num_vertices])
            
            intersection = self.intersectionPoint(segment)
            if intersection is not None:
                points.append(intersection)

        return points


    def draw(self, ax, c='k', lw=1, **kwargs):
        ax.plot([self.p1.x, self.p2.x], [self.p1.y, self.p2.y], c=c, lw=lw, **kwargs)



class Polygon:
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


    def computeCentroid(self):
        sum_x, sum_y = 0, 0
        for i in range(self.num_vertices):
            # Extracting the i-th side of the polygon
            a, b = self.vertices[i], self.vertices[(i+1) % self.num_vertices]

            sum_x -= (a.x + b.x) * (a.x * b.y - b.x * a.y)
            sum_y -= (a.y + b.y) * (a.x * b.y - b.x * a.y)

        # Computing the coordinates of the centroid of the polygon
        x = (1 / (6 * self.area)) * sum_x if self.area > 0 else self.reference_point.x
        y = (1 / (6 * self.area)) * sum_y if self.area > 0 else self.reference_point.y

        return Point(x, y)


    # Shoelance algorithm
    def computeArea(self):
        area = 0.0
        for i in range(self.num_vertices):
            area += self.vertices[i].x * self.vertices[(i+1) % self.num_vertices].y
            area -= self.vertices[(i+1) % self.num_vertices].x * self.vertices[i].y
            
        return np.abs(area) / 2.0


    # Winding number algorithm
    def containsPoint(self, point):
        general_side = None
        for i in range(self.num_vertices):
            a, b = self.vertices[i], self.vertices[(i+1) % self.num_vertices]

            # Computing the side of the point w.r.t. the segment
            side = (point.y - a.y) * (b.x - a.x) - (point.x - a.x) * (b.y - a.y)

            if side == 0:
                continue

            if general_side is None:
                general_side = side
                continue

            if general_side * side < 0:
                return False

        return True


    def intersectionPolygon(self, polygon):
        points = []

        for vertex in self.vertices:
            if polygon.containsPoint(vertex):
                points.append(vertex)

        for vertex in polygon.vertices:
            if self.containsPoint(vertex):
                points.append(vertex)

        for i in range(self.num_vertices):
            side = Segment(self.vertices[i], self.vertices[(i+1) % self.num_vertices])
            points += side.intersectionPoints(polygon)

        return Polygon(points) if len(points) > 0 else None


    def draw(self, ax, c='k', lw=1, **kwargs):
        pts = self.vertices
        pts.append(Point(pts[0].x, pts[0].y))

        ax.plot([pt.x for pt in pts], [pt.y for pt in pts], c=c, lw=lw, **kwargs)

    
    def __referencePoint(self):
        sum = self.vertices[0]
        for i in range(1, self.num_vertices):
            sum += self.vertices[i]

        return sum / self.num_vertices


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
    def __init__(self, center, width, height):
        self.center = center # Center of the rectangle
        self.width = width # Width of the rectangle
        self.height = height # Height of the rectangle
        self.west = center.x - width # West coordinate
        self.east = center.x + width # East coordinate
        self.north = center.y + height # North coordinate
        self.south = center.y - height # South coordinate

        # Instantiating parent class
        super().__init__([
            Point(self.west, self.south), 
            Point(self.west, self.north), 
            Point(self.east, self.north), 
            Point(self.east, self.south)
        ])


    def containsCell(self, cell):
        # Extracting the coordinates of the centroid of the cell
        x = cell.centroid.x
        y = cell.centroid.y

        return self.containsPoint(Point(x, y))


    def intersects(self, range):
        return not (range.west > self.east or range.east < self.west or range.south > self.north or range.north < self.south)



# Quadtree class
class QuadTree:
    def __init__(self, boundary, capacity = 4):
        self.boundary = boundary
        self.capacity = capacity
        self.cells = []
        self.divided = False


    def insert(self, cell):
        # Checking if the cell is within the QuadTree boundaries
        if not self.boundary.containsCell(cell):
            return False

        # Adding the point if the capacity has not been reached
        if len(self.cells) < self.capacity:
            self.cells.append(cell)
            return True

        if not self.divided:
            self.divide()

        if self.nw.insert(cell):
            return True
        elif self.ne.insert(cell):
            return True
        elif self.sw.insert(cell):
            return True
        elif self.se.insert(cell):
            return True

        return False


    def queryRange(self, range):
        cells_found = []

        if not self.boundary.intersects(range):
            return []

        for cell in self.cells:
            if range.containsCell(cell):
                cells_found.append(cell)

        if self.divided:
            cells_found.extend(self.nw.queryRange(range))
            cells_found.extend(self.ne.queryRange(range))
            cells_found.extend(self.sw.queryRange(range))
            cells_found.extend(self.se.queryRange(range))

        return cells_found


    def divide(self):
        center_x = self.boundary.center.x
        center_y = self.boundary.center.y
        new_width = self.boundary.width / 2
        new_height = self.boundary.height / 2

        nw = Rectangle(Point(center_x - new_width, center_y + new_height), new_width, new_height)
        self.nw = QuadTree(nw)

        ne = Rectangle(Point(center_x + new_width, center_y + new_height), new_width, new_height)
        self.ne = QuadTree(ne)

        sw = Rectangle(Point(center_x - new_width, center_y - new_height), new_width, new_height)
        self.sw = QuadTree(sw)

        se = Rectangle(Point(center_x + new_width, center_y - new_height), new_width, new_height)
        self.se = QuadTree(se)

        self.divided = True


    def __len__(self):
        count = len(self.cells)
        if self.divided:
            count += len(self.nw) + len(self.ne) + len(self.sw) + len(self.se)

        return count


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
    def __init__(self, length, bins_count, r=None, theta=None):
        # Extracting the radius and the angle of the sensor w.r.t. the origin
        self.r = np.random.uniform(np.min(sensor_distance_range), np.max(sensor_distance_range)) if r is None else r # Distance of the sensor
        self.theta = np.pi * np.random.uniform(np.min(sensor_angle_range), np.max(sensor_angle_range)) if theta is None else theta # Angle of the sensor

        # Computing the coordinates of the center of the sensor
        x = self.r * np.cos(self.theta)
        y = self.r * np.sin(self.theta)

        # Initializing the parent class
        super().__init__(Point(x, y), length / 2, length / 2)

        # Number of bins
        self.bins_count = bins_count

        # List of the cells belonging to the sensor 
        self.cells = []

        # Signal of the flow fields
        self.signal = {}

    
    def generateSignal(self):
        # Binning operation
        bins = self.__surfaceBinning()

        # Upsampling the signal in order to obtain the values of the empty bins
        bins = self.__upsample(bins)

        self.signal = {
            "p": np.array([bin["p"] for bin in bins]), 
            "U": np.array([bin["U"] for bin in bins]),
        }

        return self.signal


    def displaySignal(self, field_name):
        data = self.signal[field_name]

        plt.plot(data)
        plt.show()

    
    def __surfaceBinning(self):
        # Computing the bounds of the bins
        bins_bounds = np.linspace(self.south, self.north, num=self.bins_count+1)

        # Sorting the cells according to their maximum Y coordinate: to speed up the algorithm
        self.cells = sorted(self.cells, key=lambda cell: np.min([vertex.y for vertex in cell.vertices]))   

        bins = []
        for idx in range(len(bins_bounds) - 1):
            bin__min_y = np.min([bins_bounds[idx], bins_bounds[idx+1]])
            bin__max_y = np.max([bins_bounds[idx], bins_bounds[idx+1]])

            bin = Polygon([
                Point(self.west, bin__min_y),
                Point(self.west, bin__max_y),
                Point(self.east, bin__max_y),
                Point(self.east, bin__min_y)
            ])

            bin_cells = []
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


    def __centroidBinning(self):
        # Computing the bounds of the bins
        bins_bounds = np.linspace(self.south, self.north, num=self.bins_count+1)

        bins = []
        for idx in range(len(bins_bounds) - 1):
            bin__min_y = np.min([bins_bounds[idx], bins_bounds[idx+1]])
            bin_max_y = np.max([bins_bounds[idx], bins_bounds[idx+1]])

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

                bins[i]["p"] = np.average([lower_bin["p"], upper_bin["p"]], weights=[lower_weight, upper_weight])
                bins[i]["U"] = np.average([lower_bin["U"], upper_bin["U"]], weights=[lower_weight, upper_weight])

        return bins



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

    print("QUADTREE CREATED")

    # Extract the cells of the space
    cells = extractCells(target_section)

    print("CELLS EXTRACTED")

    for cell in cells:
        quad_tree.insert(cell)

    print("POINTS INSERTED")
    
    signals = []
    for _ in range(20):
        sensor = Sensor(sensor_length, bins_count)

        print(f"SENSOR CREATED: r={sensor.r} | theta={sensor.theta}")

        sensor.cells = quad_tree.queryRange(sensor)

        print(f"SENSOR CELLS EXTRACTED: {len(sensor.cells)}")

        # Plotting the sensor's cells
        plt.figure(figsize=(700/72, 500/72), dpi=72)
        ax = plt.subplot()
        ax.set_xlim(sensor.west-10, sensor.east+10)
        ax.set_ylim(sensor.south-10, sensor.north+10)
        sensor.draw(ax)
        for cell in sensor.cells:
            cell.draw(ax)
        #plt.tight_layout()
        plt.show()
   
        # Generating the signal
        signal = sensor.generateSignal()
        signals.append(signal)

        print("SIGNAL GENERATED")

        # Plotting the signal
        sensor.displaySignal("p")