import vtk
import Utils
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy


# Global variables and constants
bins_count = 512 # Numer of bins
sections_length = 256.0 # Y length of the cutting sections
free_stream__gradient = 0.17453 # Gradient of the free stream
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream
section__x_distances = np.array([-1, 2, 11]) # X coordinates in which the signal is generated


"""
Given the cells belonging to a bin and its boundaries, displays the bin
"""
def __displayBin(cells, lower_bound, upper_bound):
    for cell in cells:
        # Adding the first point on the front of the array
        cell["points"] = np.vstack((cell["points"], cell["points"][0]))

        plt.plot(cell["points"][:,1], cell["points"][:,0], color="blue")

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
Given a sets of bins, upsamples the signals by interpolating the values of the bins closest
to the empty ones.
"""
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


"""
Given the cells of a section with their corresponding flow quantities and the number of bins, 
performs the binning operation to generate a 1D signal. The binning operation is performed by
selecting the cells whose centroid belongs to the selected bin.
"""
def __centroidsBinning(cells):
    # Extracting the boundaries of the bins
    bins_bounds = np.linspace(-sections_length/2, +sections_length/2, num=bins_count+1)

    # Computing the centroid of each cells
    for cell in cells:
        # Getting coordinates of the vertices of the triangles
        v0 = cell["points"][0]
        v1 = cell["points"][1]
        v2 = cell["points"][2]

        # Computing centroids
        cell["centroid"] = (v0 + v1 + v2) / 3

    bins = []
    # Iterating over the total number of bins
    for idx in range(len(bins_bounds) - 1):
        # Extracting the lower and upper bounds of the current bin
        lower_bound = np.min([bins_bounds[idx], bins_bounds[idx+1]])
        upper_bound = np.max([bins_bounds[idx], bins_bounds[idx+1]])

        # Extracting the points of the mesh belonging to the i-th bin
        bin_cells = [cell for cell in cells if cell["centroid"][0] >= lower_bound and cell["centroid"][0] < upper_bound]

        # Computing the pressure and velocity field associated to the i-th bin
        bin_p = float(np.mean([cell["p"] for cell in bin_cells])) if len(bin_cells) > 0 else None
        bin_U = float(np.mean([cell["U"] for cell in bin_cells])) if len(bin_cells) > 0 else None

        bins.append({"p": bin_p, "U": bin_U})

    # Upsampling the signal in order to obtain the values of the empty bins
    bins = __upsample(bins)

    bins = {
        "p": np.array([bin["p"] for bin in bins]), 
        "U": np.array([bin["U"] for bin in bins]),
    }

    return bins


"""
Given the cells of a section with their corresponding flow quantities and the number of bins, 
performs the binning operation to generate a 1D signal. The binning operation is performed 
by computing the weighte average of the values of the cells belonging to the i-th bin. 
The weights are the surface of the cells.
"""
def __surfacesBinning(cells):
    # Extracting the boundaries of the bins
    bins_bounds = np.linspace(-sections_length/2, +sections_length/2, num=bins_count+1)

    # Sorting the cells according to their maximum Y coordinate: to speed up the algorithm
    cells = sorted(cells, key=lambda cell: np.min(cell["points"][:,0]))    

    bins = []
    # Iterating over the total number of bins
    for idx in range(len(bins_bounds) - 1):
        # Extracting the lower and upper bounds of the current bin
        lower_bound = np.min([bins_bounds[idx], bins_bounds[idx+1]])
        upper_bound = np.max([bins_bounds[idx], bins_bounds[idx+1]])

        # Iterating over the cells of the current section
        bin_cells = []
        for cell in cells:
            # Stopping the execution if the cells are above the upper bound
            if np.min(cell["points"][:,0]) > upper_bound:
                break

            # Extracting the cells that fully belongs to the current bin
            if np.min(cell["points"][:,0]) >= lower_bound and np.max(cell["points"][:,0]) <= upper_bound:
                bin_cells.append(cell)
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
                bin_bounds__segments = np.array([
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
                    for bin_bounds__segment in bin_bounds__segments:
                        intersection = __intersectionPoint(edge, bin_bounds__segment)
                        if intersection is not None:
                            inner_points = np.vstack((inner_points, intersection)) if len(inner_points) > 0 else intersection

                if len(inner_points) > 3:
                    # Splitting the polygon into triangles
                    triangles = __splitInTriangles(inner_points)
                    
                    # Iterating over the triangles
                    for triangle in triangles:
                        bin_cells.append({"points": triangle, "p": cell["p"], "U": cell["U"]})

                else:
                    bin_cells.append({"points": inner_points, "p": cell["p"], "U": cell["U"]})

        # Computing the surfaces of the cells belongign to the current bin
        for cell in bin_cells:
            cell["surface"] = __computeSurface(cell["points"])

        if len(bin_cells) > 0:
            # Extracting the flow quantities of the cells belonging to the current bin
            cells_p = np.array([cell["p"] for cell in bin_cells])
            cells_U = np.array([cell["U"] for cell in bin_cells])

            # Extracting the surfaces of the cells belonging to the current bin
            cells_surfaces = np.array([cell["surface"] for cell in bin_cells])

            # Computing the sum of the cells belonging to current bin
            sum_surfaces = np.sum(cells_surfaces)

            # Computing the average of the flow quantities of the cells belonging to the current bin
            bin_p = np.sum(cells_surfaces * cells_p) / sum_surfaces if sum_surfaces > 0.0 else 0.0
            bin_U = np.sum(cells_surfaces * cells_U) / sum_surfaces if sum_surfaces > 0.0 else 0.0

            #displayBin(bin_cells, lower_bound, upper_bound)
            
        else:
            bin_p = None
            bin_U = None

        bins.append({"p": bin_p, "U": bin_U})

    # Upsampling the signal in order to obtain the values of the empty bins
    bins = __upsample(bins)

    bins = {
        "p": np.array([bin["p"] for bin in bins]), 
        "U": np.array([bin["U"] for bin in bins]),
    }

    return bins


"""
Given a vtk reader and the number of bins, extracts the 1D signal
associated to the flow quantities.
"""
def flowSignals(reader):
    # Extracting the data of the grid
    poly_data = reader.GetOutput()

    flow_signals = {
        "p": np.zeros((bins_count, len(section__x_distances))),
        "U": np.zeros((bins_count, len(section__x_distances))),
    }

    # Iterating over the X sections
    for idx in range(len(section__x_distances)):
        # Extracting the section of interest
        target_section = __extractSection(poly_data, section__x_distances[idx])

        # Extracting the cells and the values of the flow quantities associated
        cells = __cellsValues(target_section)

        # Extracting the bins and computing the values of the associated flow quantities
        bins = __surfacesBinning(cells)

        # Appending the signal of the current section to the nd signal
        for key in flow_signals.keys():
            flow_signals[key][:, idx] = bins[key]
            flow_signals[key][:, idx] = bins[key]

        #Utils.displayData(flow_signals["p"][:, idx])

    return flow_signals