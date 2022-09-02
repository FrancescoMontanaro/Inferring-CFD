import vtk
import Utilities
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


# Global variables and constants
points_per_section = 5 # Number of points for each cutting section
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream
U_gradient_threshold = 5e-5 # Gradient threshold of the velocity vector under which to take the inflection points
p_gradient_threshold = 3e-2 # Gradient threshold of the pressure vector under which to take the inflection points
section__x_distances = np.array([-1, 2, 11]) # X coordinates in which the signals are generated


"""
Given a VTK poly_data object cuts the mesh in the selected sections
"""
def __extractSection(poly_data, section_distance):
    # Creating the cutting planes
    plane1 = vtk.vtkPlane()
    plane1.SetOrigin(section_distance, 0, 0.5)
    plane1.SetNormal(1, 0, 0) #Orthogonal to the x axis

    plane2 = vtk.vtkPlane()
    plane2.SetOrigin(section_distance, 0, 0.5)
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
Given an array containing the values of the flow fields, extracts the 
indices of the 5 most informational points 
"""
def __targetPointsIndices(flow_field, gradient_threshold):
    # Computing the gradient of the pressure and velocity vecotr
    gradient = np.gradient(flow_field)
    gradient = np.abs(gradient)

    # Extracting the index the free stream
    free_stream_idx = 0

    # Extracting the indices of the minimum and maximum values of the selected flow field
    min_idx = np.argmin(flow_field)
    max_idx = np.argmax(flow_field)

    # Extracting the indices of the inflection points of the selected flow field
    inflection_1__idx = 0
    while gradient[inflection_1__idx] <= gradient_threshold and inflection_1__idx < len(gradient) - 1:
        inflection_1__idx += 1

    inflection_2__idx = len(gradient) - 1
    while gradient[inflection_2__idx] <= gradient_threshold and inflection_2__idx >= 0:
        inflection_2__idx -= 1

    return free_stream_idx, min_idx, max_idx, inflection_1__idx, inflection_2__idx


"""
Given a grid, extracts the flow quantities associated to each cell.
"""
def __targetPoints(target_section):
    # Extracting the flow quantities
    p = vtk_to_numpy(target_section.GetCellData().GetArray('p'))
    U = vtk_to_numpy(target_section.GetCellData().GetArray('U'))
    U[:,2] = 0.0 # Removing the z component of the velocity

    # Computing the magnitude of the velocity vector
    U = np.array([np.linalg.norm(u) for u in U])

    # Normalizing the velocity w.r.t. the free stream velocity magnitude
    U /= free_stream__velocity_magnitude

    # Extracting the number of cells belonging to the target section
    num_cells = target_section.GetNumberOfCells()

    cells = np.zeros((num_cells, 3))
    # Iterating over the cells of the section
    for idx in range(num_cells):
        # Extracting the i-th cell
        cell = target_section.GetCell(idx)

        # Extracting the points of the i-th cell
        pts = vtk_to_numpy(cell.GetPoints().GetData())

        # Computing the center of the cell
        y_coord = np.mean(pts[:,1])

        # Extracting the flow quantites of the i-th cell
        cells[idx][0] = y_coord
        cells[idx][1] = p[idx]
        cells[idx][2] = U[idx]

    # Sorting the cells according to their position on the target section
    cells = cells[cells[:,0].argsort()]

    # Extracting the indices of the 5 most informational points of the flow fields
    p__target_points__indices = __targetPointsIndices(cells[:,1], p_gradient_threshold)
    U__target_points__indices = __targetPointsIndices(cells[:,2], U_gradient_threshold)

    p_target_points = np.array([[cells[idx,0], cells[idx,1]] for idx in p__target_points__indices])
    U_target_points = np.array([[cells[idx,0], cells[idx,2]] for idx in U__target_points__indices])

    target_points = {
        "p": p_target_points,
        "U": U_target_points
    }

    return target_points


"""
Given a vtk reader, extracts the 5 most informational points
associated to the flow quantities.
"""
def informativePoints(reader):
    # Extracting the data of the grid
    poly_data = reader.GetOutput()

    # Iterating over the X sections
    target_points = {
        "p": np.zeros((points_per_section, 2, len(section__x_distances))), 
        "U": np.zeros((points_per_section, 2, len(section__x_distances)))
    }

    # Iterating over the cutting sections
    for idx in range(len(section__x_distances)):
        # Extracting the section of interest
        target_section = __extractSection(poly_data, section__x_distances[idx])

        # Extracting the 5 most informational points of the flow fields
        section_target_points = __targetPoints(target_section)

        for key in target_points.keys():
            target_points[key][:, 0, idx] = section_target_points[key][:, 0]
            target_points[key][:, 1, idx] = section_target_points[key][:, 1]

    return target_points