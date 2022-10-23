import vtk
import Utilities
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy

plt.style.use('seaborn')


'### GLOBAL VARIABLES AND CONSTANTS ###'

denoise = False # Flag to denoise or not the signal (Only for 1D signals)
n_sensors = 100 # Number of sensors to generate
signal_type = "1D" # Type of the signal (1D or 2D)
sensor_width = 1.0 # Width of the sensor (in units of c)
sensor_height = 256.0 # Height of the senor (in units of c)
normal_vector = (1, 0, 0) # Vector normal to the sensor: (1, 0, 0): Orthogonal to the flow | (0, 0, 1): Parallel to the flow
vertical_resolution = 1024 # Vertical Resolution of the signal (number of bins) 
horizontal_resolution = 10 # Horizontal Resolution of the signal (number of bins) | Used only for 2D signals
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream


"### FUNCTIONS ###"

# Function to extract a section of the space
def cut(mesh, origin, normal_vector):
    # Creating the cutting plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal_vector)

    # Cutting the space
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(mesh)
    cutter.Update()

    # Extracting the cut section
    cut_section = cutter.GetOutput()

    return cut_section


# Function to extract a clip of the space
def clip(mesh, origin, normal_vector):
    # Creating the cutting plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal_vector)

    # Clipping the space
    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(plane)
    clipper.SetInputData(mesh)
    clipper.Update()

    # Extracting the clipped section
    clipped_mesh = clipper.GetOutput()

    return clipped_mesh


# Function to extract the cells from the space
def extractCells(mesh, mask, transpose=False):
    # Extracting the flow quantities
    p = vtk_to_numpy(mesh.GetCellData().GetArray('p'))
    U = vtk_to_numpy(mesh.GetCellData().GetArray('U'))
    U[:,2] = 0.0 # Removing the z component of the velocity

    # Computing the magnitude of the velocity vector
    U = np.array([np.linalg.norm(u) for u in U])

    # Normalizing the velocity w.r.t. the free stream velocity magnitude
    U /= free_stream__velocity_magnitude

    # Extracting the number of cells
    n_cells = mesh.GetNumberOfCells()

    cells = []
    # Iterating over the cells of the section
    for idx in range(n_cells):
        # Extracting the i-th cell
        cell = mesh.GetCell(idx)

        # Extracting the points of the cell
        cell_points = np.array(vtk_to_numpy(cell.GetPoints().GetData()))
        cell_points = cell_points[:, mask] # Removing the useless coordinates
        
        cell_points = [Point(a, b) for a, b in cell_points] if not transpose else [Point(b, a) for a, b in cell_points]

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


    # Function to plot the point
    def draw(self, ax, s=4, c='k'):
        if self.z is not None:
            ax.scatter(self.x, self.y, self.z, s=s, c=c)
        else:
            ax.scatter(self.x, self.y, s=s, c=c)



# Segment class
class Segment:
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



# Cell class
class Cell(Polygon):
    # Class constructor
    def __init__(self, vertices, p, U):
        # Initializing parent class
        super().__init__(vertices)

        self.p = p # Cell Pressure value
        self.U = U # Cell Velocity value



# Signal class
class Signal:
    # Class constructor
    def __init__(self, p, U):
        self.p = p # Pressure field of the signal
        self.U = U # Velocity field of the signal


    # Function to plot the signal of the specified flow field
    def display(self, field_name):
        data = getattr(self, field_name)

        if data.ndim == 1:
            plt.xlabel("bins")
            plt.ylabel(field_name)
            plt.plot(data)
        elif data.ndim == 2:
            plt.imshow(np.rot90(data), cmap="gray")
            plt.grid(False)

        plt.title(field_name)
        plt.show()


    # Function to upsample the signal by assigning to the empty bins the interpolation of the
    # values of the closest one
    def upsample(self, field_name):
        # Extracting the signal to upsample
        signal = getattr(self, field_name)

        # Setting the values of the first and last bin to the closest ones, if thay are empty.
        if signal[0] is None:
            i, upper_bin = 0, None
            while(upper_bin is None and i < len(signal)):
                if signal[i] is not None:
                    upper_bin = signal[i]
                i += 1

            signal[0] = float(upper_bin) if upper_bin is not None else 0.0

        if signal[-1] is None:
            i, lower_bin = len(signal) - 1, None
            while(lower_bin is None and i > 0):
                if signal[i] is not None:
                    lower_bin = signal[i]
                i -= 1

            signal[-1] = float(lower_bin) if lower_bin is not None else 0.0

        # For each empty bin, obtains its values by interpolating the values of the adjacent ones.
        for i in range(1, len(signal)-1):
            if signal[i] is None:
                j, lower_bin = i, None
                while(lower_bin is None and j >= 0):
                    if signal[j] is not None:
                        lower_bin = signal[j]
                        lower_weight = 1 / i - j
                    j -= 1

                j, upper_bin = i, None
                while(upper_bin is None and j < len(signal)):
                    if signal[j] is not None:
                        upper_bin = signal[j]
                        upper_weight = 1 / j - i
                    j += 1

                upper_bin = upper_bin if upper_bin is not None else 0.0
                lower_bin = lower_bin if lower_bin is not None else 0.0

                # Weighted average of the values of the closest bins
                signal[i] = np.average([lower_bin, upper_bin], weights=[lower_weight, upper_weight])

        # Updating the signal with the upsampled one
        setattr(self, field_name, signal)

        return self


    # Function to denoise the signal through a Low Pass filter
    def denoise(self):
        fs = 16.0
        order = 4
        nyq = 0.5 * fs
        cutoff = 2 / nyq

        # Get the filter coefficients 
        b, a = signal.butter(order, cutoff, 'low', analog=False)

        def displayFilter(b, a):
            w, h = signal.freqs(b, a)
            plt.semilogx(w, 20 * np.log10(abs(h)))
            plt.xlabel('Frequency [radians / second]')
            plt.ylabel('Amplitude [dB]')
            plt.margins(0, 0.1)
            plt.grid(which='both', axis='both')
            plt.axvline(cutoff, color='green')
            plt.show()

        self.p = signal.filtfilt(b, a, self.p)
        self.U = signal.filtfilt(b, a, self.U)

        return self



# Sensor class
class Sensor(Rectangle):
    # Class constructor
    def __init__(self, origin, normal_vector, width, height, mesh):
        self.origin = origin
        self.normal_vector = normal_vector

        # Extracting the section of interest
        self.mesh = cut(mesh, (origin.x, origin.y, origin.z), normal_vector)

        # Computing the bounds of the section
        bounds = self.mesh.GetBounds()
        x_length = np.abs(bounds[1] - bounds[0]) 
        y_length = np.abs(bounds[2] - bounds[3]) 
        z_length = np.abs(bounds[4] - bounds[5])

        if normal_vector == (0, 0, 1):
            # Correcting width and height for incorrect values
            if width >= x_length: width = x_length
            if height >= y_length: height = y_length
            
            # Initializing the parent class
            super().__init__(Point(origin.x, origin.y), width, height)

            # Clipping the space to extract the sensor according to its dimensions and position
            self.mesh = clip(self.mesh, (self.west, 0, 0), (1, 0, 0))
            self.mesh = clip(self.mesh, (self.east, 0, 0), (-1, 0, 0))
            self.mesh = clip(self.mesh, (0, self.south, 0), (0, 1, 0))
            self.mesh = clip(self.mesh, (0, self.north, 0), (0, -1, 0))

            coordinates_mask = [True, True, False]
            self.cells = extractCells(self.mesh, coordinates_mask, transpose=False) if self.mesh.GetNumberOfCells() > 0 else []
        
        if normal_vector == (1, 0, 0):
            # Correcting width and height for incorrect values
            if width >= z_length: width = z_length
            if height >= y_length: height = y_length

            # Initializing the parent class
            super().__init__(Point(origin.z, origin.y), width, height)

            # Clipping the space to extract the sensor according to its dimensions and position
            self.mesh = clip(self.mesh, (0, 0, self.west), (0, 0, 1))
            self.mesh = clip(self.mesh, (0, 0, self.east), (0, 0, -1))
            self.mesh = clip(self.mesh, (0, self.south, 0), (0, 1, 0))
            self.mesh = clip(self.mesh, (0, self.north, 0), (0, -1, 0))

            coordinates_mask = [False, True, True]
            self.cells = extractCells(self.mesh, coordinates_mask, transpose=True) if self.mesh.GetNumberOfCells() > 0 else []
            

    # Function to generate the signal of the flow fields for the sensor
    def generateSignal(self, resolution, type, denoise):
        # Binning operation
        if type == "1D":
            self.__1D_slowBinning(resolution[1])

            if denoise:
                self.signal.denoise()

        elif type == "2D":
            self.__2D_slowBinning(*resolution)

        return self.signal


    # Function to plot the sensor and its cells
    def display(self):
        # Plotting the sensor's cells
        ax = plt.subplot()
        ax.set_xlim(self.west-2, self.east+2)
        ax.set_ylim(self.south-2, self.north+2)
        self.draw(ax, c='r')
        for cell in self.cells:
            cell.draw(ax)
        plt.show()

    
    # Function to perform the binning operation by averaging the values of the cells fully belonging and
    # intersecting each bin.
    def __1D_slowBinning(self, r):
        # Computing the bounds of the bins
        bins_bounds = np.linspace(self.south, self.north, num=r+1)

        # Sorting the cells according to their maximum Y coordinate: to speed up the algorithm
        self.cells = sorted(self.cells, key=lambda cell: np.min([vertex.y for vertex in cell.vertices]))   

        self.signal = Signal(
            p = np.zeros(r),
            U = np.zeros(r)
        )

        # Iterating over the bins
        for idx in range(r):
            # Extracting the minimum and maximum y coordinate of the bin
            south = np.min([bins_bounds[idx], bins_bounds[idx+1]])
            north = np.max([bins_bounds[idx], bins_bounds[idx+1]])

            # Creating a polygon object whose vertices are the ones of the bin
            bin = Polygon([
                Point(self.west, south),
                Point(self.west, north),
                Point(self.east, north),
                Point(self.east, south)
            ])

            bin_cells = []
            # Iterating over the cells of the sensor
            for cell in self.cells:
                # Stopping the execution if the cells are above the upper bound
                if np.min([vertex.y for vertex in cell.vertices]) > north:
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
            
            # Extracting the flow quantities of the cells belonging to the current bin
            cells_p = np.array([cell.p for cell in bin_cells])
            cells_U = np.array([cell.U for cell in bin_cells])

            # Extracting the surfaces of the cells belonging to the current bin
            cells_areas = np.array([cell.area for cell in bin_cells])

            # Computing the sum of the cells belonging to current bin
            sum_areas = np.sum(cells_areas)

            # Computing the average of the flow quantities of the cells belonging to the current bin
            self.signal.p[idx] = np.sum(cells_areas * cells_p) / sum_areas if sum_areas > 0.0 else 0.0
            self.signal.U[idx] = np.sum(cells_areas * cells_U) / sum_areas if sum_areas > 0.0 else 0.0


    def __2D_slowBinning(self, rh, rv):
        self.signal = Signal(
            p = np.zeros((rh, rv)),
            U = np.zeros((rh, rv))
        )

        for i in range(rh):
            for j in range(rv):
                bin = Polygon([
                    Point((self.west + i*(self.width/rh)), (self.south + j*(self.height/rv))),
                    Point((self.west + i*(self.width/rh)), (self.south + (j+1)*(self.height/rv))),
                    Point((self.west + (i+1)*(self.width/rh)), (self.south + (j+1)*(self.height/rv))),
                    Point((self.west + (i+1)*(self.width/rh)), (self.south + j*(self.height/rv))),
                ])

                bin_cells = []
                for cell in self.cells:
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

                # Extracting the flow quantities of the cells belonging to the current bin
                cells_p = np.array([cell.p for cell in bin_cells])
                cells_U = np.array([cell.U for cell in bin_cells])

                # Extracting the surfaces of the cells belonging to the current bin
                cells_areas = np.array([cell.area for cell in bin_cells])

                # Computing the sum of the cells belonging to current bin
                sum_areas = np.sum(cells_areas)

                # Computing the average of the flow quantities of the cells belonging to the current bin
                self.signal.p[i,j] = np.sum(cells_areas * cells_p) / sum_areas if sum_areas > 0.0 else 0.0
                self.signal.U[i,j] = np.sum(cells_areas * cells_U) / sum_areas if sum_areas > 0.0 else 0.0


    # Function to perform the binning operation by averaging the values of the cells whose centroid belongs
    # to the i-t bin.
    def __1D__fastBinning(self, r):
        # Computing the bounds of the bins
        bins_bounds = np.linspace(self.south, self.north, num=r+1)

        self.signal = Signal(
            p = np.zeros(r),
            U = np.zeros(r)
        )

        # Iterating over the bins
        for idx in range(r):
            # Extracting the minimum and maximum y coordinate of the bin
            south = np.min([bins_bounds[idx], bins_bounds[idx+1]])
            north = np.max([bins_bounds[idx], bins_bounds[idx+1]])

            # Creating a polygon object whose vertices are the ones of the bin
            bin = Polygon([
                Point(self.west, south),
                Point(self.west, north),
                Point(self.east, north),
                Point(self.east, south)
            ])

            # Extracting the cells belonging to the current bin
            bin_cells = [cell for cell in self.cells if bin.containsPoint(cell.centroid)]

            # Computing the pressure and velocity field associated to the i-th bin
            self.signal.p[idx] = float(np.mean([cell.p for cell in bin_cells])) if len(bin_cells) > 0 else None
            self.signal.p[idx] = float(np.mean([cell.U for cell in bin_cells])) if len(bin_cells) > 0 else None
        
        # Upsampling the singla to fill empty bins
        self.signal.upsample("p")
        self.signal.upsample("U")



'### MAIN FUNCTION ###'

# Function to generate the signal of the flow fields for n sensors located at a random position
# in te space
def sensorSignal(reader):
    # Extracting the spatial data
    mesh = reader.GetOutput()

    # Extracting the X and Y boundaries of the space
    bounds = mesh.GetBounds()
    bounds = bounds[:4]
    
    signals = []
    # Iterating over the number of sensors to extract
    for _ in range(n_sensors):
        x = np.random.uniform(10.0 + np.sqrt(sensor_width), np.max(bounds[:2]) - np.sqrt(sensor_width))
        y = np.random.uniform(np.min(bounds[-2:]) + np.sqrt(sensor_height), np.max(bounds[-2:]) - np.sqrt(sensor_height))

        x = 2.0
        y = 0.0
        
        # Creating the sensor object
        origin = Point(x, y, 0.5)
        sensor = Sensor(origin, normal_vector, sensor_width, sensor_height, mesh)

        # Generating the signal
        sensor.generateSignal(resolution=(horizontal_resolution, vertical_resolution), type=signal_type, denoise=denoise)

        # Plotting the mesh of the sensor and the signal
        sensor.display()
        sensor.signal.display(field_name="p")

        # Adding the signal into the main list
        signals.append({
            "p": sensor.signal.p,
            "U": sensor.signal.U,
            "y": sensor.origin.x, 
            "x": sensor.origin.y
        })

        # Printing status
        if n_sensors > 1:
            print(f'{_+1} / {n_sensors} sensors generated | x: {sensor.origin.x} | y: {sensor.origin.y}')

    return signals