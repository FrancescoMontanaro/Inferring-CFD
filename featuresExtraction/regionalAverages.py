import vtk
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy

plt.style.use('seaborn')


'### GLOBAL VARIABLES AND CONSTANTS ###'

section_x = 2.0 # X coordinates of the cutting section
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream
regions__y_bounds = np.array([-500, -10, -1, -0.1, 0, 0.1, 1, 10, 500]) # Y boundaries of each region


'### FUNCTIONS ###'

# Function to cut the space in order to extract the target section
def extractSection(poly_data, section_distance):
    # Creating the cutting plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(section_distance, 0, 0.5)
    plane.SetNormal(1, 0, 0) #Orthogonal to the x axis

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
        cell_points = cell_points[:, 1:] # Removing the x coordinate
        cell_points = [Point(z, y) for y, z in cell_points]

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



# Cell class
class Cell(Polygon):
    # Class constructor
    def __init__(self, vertices, p, U):
        # Initializing parent class
        super().__init__(vertices)

        self.p = p # Cell Pressure value
        self.U = U # Cell Velocity value



# Section class
class Section:
    # Class constructor
    def __init__(self, cells):
        self.cells = cells # Cells of the section 

    
    # Function to compute the regional averages of the flow fields at the specified section
    def computeRegionalAverages(self, y_bounds):
        # Computing the regional averages
        regional_averages = self.__surfaceRegionalAverages(y_bounds)

        return {
            "p": np.array([region["p"] for region in regional_averages]),
            "U": np.array([region["U"] for region in regional_averages]),
        }


    # Function to compute the regional averages by averaging the values of the cells fully belonging and
    # intersecting each region.
    def __surfaceRegionalAverages(self, y_bounds):
        # Extracting the minimum and maximum x coordinate of the bins
        x_coords = np.array([[vertex.x for vertex in cell.vertices] for cell in self.cells])
        x_coords = x_coords.flatten()

        region__min_x, region__max_x = np.min(x_coords), np.max(x_coords)

        regional_averages = []
        # Iterating over the regions of a section
        for r in range(len(regions__y_bounds) - 1):
            # Extracting the minimum and maximum y coordinate of the region
            region__min_y = np.min([y_bounds[r], y_bounds[r+1]])
            region_max_y = np.max([y_bounds[r], y_bounds[r+1]])

            # Creating a polygon object whose vertices are the ones of the current region
            region = Polygon([
                Point(region__min_x, region__min_y),
                Point(region__min_x, region_max_y),
                Point(region__max_x, region_max_y),
                Point(region__max_x, region__min_y)
            ])

            region_cells = []
            # Iterating over the cells of the section
            for cell in self.cells:
                # Extracting the cells that fully belongs to the polygon
                fully_belongs = True
                for vertex in cell.vertices:
                    if not region.containsPoint(vertex):
                        fully_belongs = False

                if fully_belongs:
                    region_cells.append(cell)
                    continue

                # Extracting the cells that partially belong to the current bin
                intersection_polygon = region.intersectionPolygon(cell)
                if intersection_polygon is not None:
                    region_cells.append(Cell(intersection_polygon.vertices, cell.p, cell.U))

            # Plotting the region and its cells
            #self.displayRegion(region, region_cells)

            # Extracting the flow quantities of the cells belonging to the current region
            cells_p = np.array([cell.p for cell in region_cells])
            cells_U = np.array([cell.U for cell in region_cells])

            # Extracting the surfaces of the cells belonging to the current bin
            cells_areas = np.array([cell.area for cell in region_cells])

            # Computing the sum of the cells belonging to current bin
            sum_areas = np.sum(cells_areas)

            # Computing the average of the flow quantities of the cells belonging to the current bin
            region_p = np.sum(cells_areas * cells_p) / sum_areas if sum_areas > 0.0 else 0.0
            region_U = np.sum(cells_areas * cells_U) / sum_areas if sum_areas > 0.0 else 0.0

            regional_averages.append({"p": region_p, "U": region_U})

        return regional_averages


    # Function to compute the regional averages by averaging the values of the cells whose centroind belongs 
    # to the i-th region
    def __centroidRegionalAverages(self, y_bounds):
        # Extracting the minimum and maximum x coordinate of the bins
        x_coords = np.array([[vertex.x for vertex in cell.vertices] for cell in self.cells])
        x_coords = x_coords.flatten()

        region__min_x, region__max_x = np.min(x_coords), np.max(x_coords)

        regional_averages = []
        # Iterating over the regions of a section
        for r in range(len(regions__y_bounds) - 1):
            # Extracting the minimum and maximum y coordinate of the region
            region__min_y = np.min([y_bounds[r], y_bounds[r+1]])
            region_max_y = np.max([y_bounds[r], y_bounds[r+1]])

            # Creating a polygon object whose vertices are the ones of the current region
            region = Polygon([
                Point(region__min_x, region__min_y),
                Point(region__min_x, region_max_y),
                Point(region__max_x, region_max_y),
                Point(region__max_x, region__min_y)
            ])

            # Extracting the cells belonging to the current region
            region_cells = [cell for cell in self.cells if region.containsPoint(cell.centroid)]

            # Plottinghte region and its cells
            #self.displayRegion(region, region_cells)

            # Extracting the flow quantities of the cells belonging to the current region
            cells_p = np.array([cell.p for cell in region_cells])
            cells_U = np.array([cell.U for cell in region_cells])

            # Extracting the surfaces of the cells belonging to the current bin
            cells_areas = np.array([cell.area for cell in region_cells])

            # Computing the sum of the cells belonging to current bin
            sum_areas = np.sum(cells_areas)

            # Computing the average of the flow quantities of the cells belonging to the current bin
            region_p = np.sum(cells_areas * cells_p) / sum_areas if sum_areas > 0.0 else 0.0
            region_U = np.sum(cells_areas * cells_U) / sum_areas if sum_areas > 0.0 else 0.0

            regional_averages.append({"p": region_p, "U": region_U})

        return regional_averages


    # Function to plot the regional averages of a specified flow field
    def drawRegions(self, regional_averages, field_name):
        data = regional_averages[field_name]
        plt.plot(data)
        plt.show()


    @staticmethod
    # FUnction to plot a region and its cells
    def displayRegion(region, region_cells):
        plt.figure()
        ax = plt.subplot()
        region.draw(ax)
        for cell in region_cells:
            cell.draw(ax)
        plt.show()



# Function to generate the regional averages of the flow fields at the specified section
def regionalAverages(reader):
    # Extracting the data of the grid
    poly_data = reader.GetOutput()

    # Extracting the section of interest
    target_section = extractSection(poly_data, section_x)

    # Extracting the cells and the values of the flow quantities associated
    cells = extractCells(target_section)

    # Instantiating the section object
    section = Section(cells)

    # Computing the regional averages of the flow quantities
    regional_averages = section.computeRegionalAverages(regions__y_bounds)

    # Plotting the regional averages of the flow fields
    #section.drawRegions(regional_averages, "U")

    return [{
        "p": regional_averages["p"],
        "U": regional_averages["U"]
    }]