import vtk
import utils
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


# Global variables and constants
bins_count = 1024 # Numer of bins
section__x_distance = 5 # X coordinate in which the signal is generated
sections_length = 1000.0 # Y length of the cutting sections
free_stream__gradient = 0.17453 # Gradient of the free stream
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream


"""
Given a VTK poly_data object cuts the mesh in the selected sections
"""
def __extractSection(poly_data):
    # Creating the cutting planes
    plane1 = vtk.vtkPlane()
    plane1.SetOrigin(section__x_distance, 0, 0.5)
    plane1.SetNormal(1, 0, 0) #Orthogonal to the x axis

    plane2 = vtk.vtkPlane()
    plane2.SetOrigin(section__x_distance, 0, 0.5)
    plane2.SetNormal(0, 0, 1) #Orthogonal to the z axis

    # Cutting the space in the first direction
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane1)
    cutter.SetInputData(poly_data)
    cutter.Update()

    # Extracting the first target section
    target_section = cutter.GetOutput()

    # Cutting the space in the second direction
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane2)
    cutter.SetInputData(target_section)
    cutter.Update()

    # Extracting the final target section
    target_section = cutter.GetOutput()

    return target_section


"""
Given a grid, extracts the flow quantities associated to each cell.
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
        # Extracting the i-th cell and its points
        cell = target_section.GetCell(idx)
        pts = vtk_to_numpy(cell.GetPoints().GetData())

        # Computing the center of the cell
        center = np.mean(pts[:,1])

        # Extracting the flow quantites of the i-th cell
        cell_p = p[idx]
        cell_U = U[idx]

        cells.append({"center": center, "p": cell_p, "U": cell_U})

    return cells


"""
Given the cells of the mesh with their corresponding flow quantities and the number of bins, 
performs the binning operation to generate a 1D signal.
"""
def __extractBins(points):
    # Extracting the boundaries of the bins
    bins_bounds = np.linspace(-sections_length/2, +sections_length/2, num=bins_count)

    bins = []
    # Iterating over the total number of bins
    for idx in range(len(bins_bounds) - 1):        
        # Extracting the points of the mesh belonging to the i-th bin
        bin_points = [point for point in points if point["center"] >= bins_bounds[idx] and point["center"] < bins_bounds[idx+1]]

        # Computing the pressure and velocity field associated to the i-th bin
        bin_p = float(np.mean([point["p"] for point in bin_points])) if len(bin_points) > 0 else None
        bin_U = float(np.mean([point["U"] for point in bin_points])) if len(bin_points) > 0 else None

        bins.append({"p": bin_p, "U": bin_U})

    # Obtaining the values of the empty bins by interpolating the values of the adjacent ones
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

    for i in range(1, len(bins)-1):
        if(bins[i]["p"] is None or bins[i]["U"] is None):
            j, lower_bin = i, None
            while(lower_bin is None and j >= 0):
                if(bins[j]["p"] is not None and bins[j]["U"] is not None):
                    lower_bin = bins[j]
                j -= 1

            j, upper_bin = i, None
            while(upper_bin is None and j < len(bins)):
                if(bins[j]["p"] is not None and bins[j]["U"] is not None):
                    upper_bin = bins[j]
                j += 1

            bins[i]["p"] = float(np.mean([lower_bin["p"], upper_bin["p"]]))
            bins[i]["U"] = float(np.mean([lower_bin["U"], upper_bin["U"]]))

    bins = {"p": [bin["p"] for bin in bins], "U": [bin["U"] for bin in bins]}

    return bins


"""
Given a vtk reader and the number of bins, extracts the 1D signal
associated to the flow quantities.
"""
def flowSignals(reader):
    # Extracting the data of the grid
    poly_data = reader.GetOutput()

    # Extracting the section of interest
    target_section = __extractSection(poly_data)

    # Extracting the cells and the values of the flow quantities associated
    cells = __cellsValues(target_section)

    # Extracting the bins and computing the values of the associated flow quantities
    bins = __extractBins(cells)

    return bins