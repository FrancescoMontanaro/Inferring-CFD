import vtk
import numpy as np
import matplotlib.pyplot as plt
from Geometry import Point, Polygon
from vtk.util.numpy_support import vtk_to_numpy

plt.style.use('seaborn')


'### GLOBAL VARIABLES AND CONSTANTS ###'

section_x = -1.0 # X coordinates of the cutting section
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

        ax = plt.subplot()

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
                intersection_polygon = region.PolygonIntersectionPoints(cell)
                if intersection_polygon is not None:
                    region_cells.append(Cell(intersection_polygon.vertices, cell.p, cell.U))

            # Plotting the region and its cells
            #self.displayRegion(region, region_cells)

            region.draw(ax)
            for cell in region_cells:
                cell.draw(ax)

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

        plt.show()

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