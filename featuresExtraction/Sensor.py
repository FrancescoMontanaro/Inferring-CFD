import vtk
import Utilities
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy
from Geometry import Point, Polygon, Rectangle

plt.style.use('seaborn')


'### GLOBAL VARIABLES AND CONSTANTS ###'

denoise = False # Flag to denoise or not the signal (Only for 1D signals)
n_sensors = 100 # Number of sensors to generate
resolution = 1024 # Resolution of the signal: Scalar for 1D signal, tuple (rx, ry) for 2D signals
sensor_width = 1.0 # Width of the sensor (in units of c)
sensor_height = 256.0 # Height of the senor (in units of c)
normal_vector = (1, 0, 0) # Vector normal to the sensor: (1, 0, 0): Orthogonal to the flow | (0, 0, 1): Parallel to the flow
sensor_origin = (2.0, 0.0) # x, y origin of the sensor | (None, None) for random position (x > 0).
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
        if type(resolution) is tuple:
            self.__2D_slowBinning(*resolution)

        else:
            self.__1D_slowBinning(resolution)

            if denoise:
                self.signal.denoise()

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
                intersection_polygon = bin.PolygonIntersectionPoints(cell)
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
                    intersection_polygon = bin.PolygonIntersectionPoints(cell)
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
        x = sensor_origin[0] if sensor_origin[0] is not None else np.random.uniform(0.0 + np.sqrt(sensor_width), np.max(bounds[:2]) - np.sqrt(sensor_width))
        y = sensor_origin[1] if sensor_origin[1] is not None else np.random.uniform(np.min(bounds[-2:]) + np.sqrt(sensor_height), np.max(bounds[-2:]) - np.sqrt(sensor_height))
        
        # Creating the sensor object
        origin = Point(x, y, 0.5)
        sensor = Sensor(origin, normal_vector, sensor_width, sensor_height, mesh)

        # Generating the signal
        sensor.generateSignal(resolution=resolution, denoise=denoise)

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