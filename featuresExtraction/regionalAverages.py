import vtk
import Utils
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy


# Global variables and constants
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream
sections__x_distances = np.array([-1, 2, 11]) # X coordinates of the cutting sections
regions__y_bounds = np.array([-500, -10, -1, -0.1, 0, 0.1, 1, 10, 500]) # Y boundaries of each region


"""
Given the cells belonging to a region and its boundaries, displays the region
"""
def __displayRegion(cells, lower_bound, upper_bound):
    for cell in cells:
        # Adding the first point on the front of the array
        cell["points"] = np.vstack((cell["points"], cell["points"][0]))

        plt.plot(cell["points"][:,1], cell["points"][:,0], color="black")

    plt.axhline(lower_bound, color='red')
    plt.axhline(upper_bound, color='red')
    plt.show()


"""
Given a set of coordinates, computes the surface of the triangle
"""
def __computeSurface(points):
    # Extracting the vertices of the triangle
    v0 = points[0]
    v1 = points[1]
    v2 = points[2]

    # Computing the length of the segments of the triangle
    a = np.linalg.norm(v0-v1)
    b = np.linalg.norm(v1-v2)
    c = np.linalg.norm(v2-v0)

    # Computing the semiperimeter
    s = (a + b + c)/2

    # Computing the surface of the triangle
    surface = (s*(s-a)*(s-b)*(s-c)) ** 0.5

    return surface


"""
Given a polygon, splits it into triangles
"""
def __splitInTriangles(pts):
    # Cplitting the polygon into triangles
    triangles = [(pts[0],b,c) for b,c in zip(pts[1:],pts[2:])]

    return np.array(triangles)


"""
Given the coordinates of two segments, computes their intersection point
"""
def __intersectionPoint(segment1, segment2):
    x1, y1 = segment1[0]
    x2, y2 = segment1[1]
    x3, y3 = segment2[0]
    x4, y4 = segment2[1]

    det = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if det == 0: # parallel
        return None

    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / det
    if ua < 0 or ua > 1: # out of range
        return None

    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / det
    if ub < 0 or ub > 1: # out of range
        return None

    # Computing the coordinates of the intersection point
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)

    return (x, y)


"""
Given a VTK poly_data object cuts the mesh in the selected sections
"""
def __extractSection(poly_data, section_distance):
    # Creating the cutting planes
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


"""
Given the data of a section, extracts the flow quantities and the points 
associated to each cell.
"""
def __cellsValues(target_section):
    # Extracting the flow quantities
    p = vtk_to_numpy(target_section.GetCellData().GetArray('p'))
    U = vtk_to_numpy(target_section.GetCellData().GetArray('U'))
    U[:,2] = 0.0 # Removing the z component of the velocity

    # Computing the magnitude of the velocity vector
    U = np.array([np.linalg.norm(u) for u in U])

    # Normalizing the velocity w.r.t. the free stream velocity magnitude
    U /= free_stream__velocity_magnitude

    cells = []
    # Iterating over the cells of the section
    for idx in range(target_section.GetNumberOfCells()):
        # Extracting the i-th cell
        cell = target_section.GetCell(idx)

        # Extracting the points of the cell
        cell_points = np.array(vtk_to_numpy(cell.GetPoints().GetData()))
        cell_points = cell_points[:, 1:] # Removing the x coordinate

        # Extracting the flow quantites of the cell
        cell_p = p[idx]
        cell_U = U[idx]

        cells.append({"points": cell_points, "p": cell_p, "U": cell_U})

    return cells


"""
Given the cells of a sectionh with their corresponding flow quantities, computes the regional averages
of the flow quantities for the selected regions. The average is computed by selecting the cells whose 
centroid belongs to the selected region.
"""
def __centroidsRegionalAverages(cells):
    # Computing the centroid of each cells
    for cell in cells:
        # Getting coordinates of the vertices of the triangles
        v0 = cell["points"][0]
        v1 = cell["points"][1]
        v2 = cell["points"][2]

        # Computing centroids of the cell
        cell["centroid"] = (v0 + v1 + v2) / 3

        # Computing the surface of the cell
        cell["surface"] = __computeSurface(cell["points"])

    # Iterating over the regions of a section
    regional_averages = []
    for r in range(len(regions__y_bounds) - 1):
        # Extracting the lower and upper bounds of the current bin
        lower_bound = np.min([regions__y_bounds[r], regions__y_bounds[r+1]])
        upper_bound = np.max([regions__y_bounds[r], regions__y_bounds[r+1]])

        # Extracting the cells belonging to the current region
        region_cells = [cell for cell in cells if cell["centroid"][0] >= lower_bound and cell["centroid"][0] <= upper_bound]

        # Extracting the flow quantities of the cells belonging to the current region
        cells_p = np.array([cell["p"] for cell in region_cells])
        cells_U = np.array([cell["U"] for cell in region_cells])

        # Extracting the surface of the cells belonging to the current region
        cells_surfaces = np.array([cell["surface"] for cell in region_cells])

        # Computing the sum of the surfaces of the polygons belonging to the i-th region
        sum_surfaces = np.sum(cells_surfaces)

        # Computing the regional averages of pressure and velocity
        region_p = np.sum(cells_surfaces * cells_p) / sum_surfaces if sum_surfaces > 0.0 else 0.0
        region_U = np.sum(cells_surfaces * cells_U) / sum_surfaces if sum_surfaces > 0.0 else 0.0

        # Adding the results to the main array
        regional_averages.append({"p": region_p, "U": region_U})

    regional_averages = {
        "p": np.array([regional_average["p"] for regional_average in regional_averages]),
        "U": np.array([regional_average["U"] for regional_average in regional_averages])
    }

    return regional_averages


"""
Given a vtk reader, the sections coordinates and the bounds of each region, 
computes the Regional Averages of the flow quantitites.
"""
def __surfacesRegionalAverages(cells):
    # Iterating over the regions of a section
    regional_averages = []
    for r in range(len(regions__y_bounds) - 1):
        # Extracting the lower and upper bounds of the current bin
        lower_bound = np.min([regions__y_bounds[r], regions__y_bounds[r+1]])
        upper_bound = np.max([regions__y_bounds[r], regions__y_bounds[r+1]])

        region_cells = []
        for cell in cells:
            # Extracting the cells that fully belongs to the current bin
            full_shape = True
            for point in cell["points"]:
                if point[0] < lower_bound or point[0] > upper_bound:
                    full_shape = False
                    break

            if full_shape:
                region_cells.append(cell)
                continue

            # Extracting the cells which partially belongs to the current bin
            partial_shape = False
            if np.min(cell["points"][:,0]) < lower_bound and np.max(cell["points"][:,0]) > upper_bound:
                partial_shape = True
            else:
                for point in cell["points"]:
                    if point[0] >= lower_bound and point[0] <= upper_bound:
                        partial_shape = True
                        break

            if partial_shape:
                # Extracting the coordinates of the edges of the cell
                edges = np.array([
                    [cell["points"][0], cell["points"][1]],
                    [cell["points"][1], cell["points"][2]],
                    [cell["points"][2], cell["points"][0]]
                ])

                # Extracting the coordinates of bin boundaries
                region_bounds__segments = np.array([
                    [
                        (lower_bound, np.min(cell["points"][:,1])),
                        (lower_bound, np.max(cell["points"][:,1]))
                    ],
                    [
                        (upper_bound, np.min(cell["points"][:,1])),
                        (upper_bound, np.max(cell["points"][:,1])),
                    ]
                ])

                # Adding the points belonging to the bin boundaries
                inner_points = np.array([point for point in cell["points"] if point[0] >= lower_bound and point[0] <= upper_bound])

                # Computing the coordinates of the points of the cell intersecting the boundaries of current the bin
                for edge in edges:
                    for region_bounds__segment in region_bounds__segments:
                        intersection = __intersectionPoint(edge, region_bounds__segment)
                        if intersection is not None:
                            inner_points = np.vstack((inner_points, intersection)) if len(inner_points) > 0 else intersection

                if len(inner_points) > 3:
                    # Splitting the polygon into triangles
                    triangles = __splitInTriangles(inner_points)
                    
                    # Iterating over the triangles
                    for triangle in triangles:
                        region_cells.append({"points": triangle, "p": cell["p"], "U": cell["U"]})

                else:
                    region_cells.append({"points": inner_points, "p": cell["p"], "U": cell["U"]})

        # Computing the surfaces of the cells belongign to the current bin
        for cell in region_cells:
            cell["surface"] = __computeSurface(cell["points"])

        # Extracting the flow quantities of the cells belonging to the current region
        cells_p = np.array([cell["p"] for cell in region_cells])
        cells_U = np.array([cell["U"] for cell in region_cells])

        # Extracting the surfaces of the cells belonging to the current region
        cells_surfaces = np.array([cell["surface"] for cell in region_cells])

        # Computing the sum of the cells belonging to current region
        sum_surfaces = np.sum(cells_surfaces)

        # Computing the regional averages of pressure and velocity
        region_p = np.sum(cells_surfaces * cells_p) / sum_surfaces if sum_surfaces > 0.0 else 0.0
        region_U = np.sum(cells_surfaces * cells_U) / sum_surfaces if sum_surfaces > 0.0 else 0.0

        #__displayRegion(region_cells, lower_bound, upper_bound)

        # Adding the results to the main array
        regional_averages.append({"p": region_p, "U": region_U})

    regional_averages = {
        "p": np.array([regional_average["p"] for regional_average in regional_averages]),
        "U": np.array([regional_average["U"] for regional_average in regional_averages])
    }

    return regional_averages


def regionalAverages(reader):
    # Extracting the data of the grid
    poly_data = reader.GetOutput()

    regional_averages = {
        "p": np.zeros((len(regions__y_bounds) - 1, len(sections__x_distances))), 
        "U": np.zeros((len(regions__y_bounds) - 1, len(sections__x_distances)))
    }

    # Iterating over the X sections
    for idx in range(len(sections__x_distances)):
        # Extracting the section of interest
        target_section = __extractSection(poly_data, sections__x_distances[idx])

        # Extracting the cells and the values of the flow quantities associated
        cells = __cellsValues(target_section)

        # Computing the regional averages of the flow quantities
        section_regional_averages = __surfacesRegionalAverages(cells)

        # Appending the signal of the current section to the nd signal
        for key in regional_averages.keys():
            regional_averages[key][:, idx] = section_regional_averages[key]
            regional_averages[key][:, idx] = section_regional_averages[key]

    return regional_averages