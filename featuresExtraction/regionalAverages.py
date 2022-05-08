import vtk
import Utils
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


# Global variables and constants
free_stream__gradient = 0.17453 # Gradient of the free stream
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream
sections__x_distances = np.array([-1, 1, 10]) # X coordinates of the cutting sections
regions__y_bounds = np.array([-500, -10, -1, -0.1, 0, 0.1, 1, 10, 500]) # Y boundaries of each region


"""
Given the data of a section, computes the centroids and the areas of its cells
"""
def __cellValues(poly_data):
    # Extracting the number of cells of the section
    number_of_cells = poly_data.GetNumberOfCells()

    areas = np.zeros(number_of_cells)
    centroids = np.zeros((number_of_cells, 2))

    # Iterating over the cells
    for idx in range(number_of_cells):
        # Extracting the i-th cell and its points
        cell = poly_data.GetCell(idx)
        cell_points = vtk_to_numpy(cell.GetPoints().GetData())

        # Getting coordinates of the vertices of the triangles
        v0 = [cell_points[0,0], cell_points[0,1]]
        v1 = [cell_points[1,0], cell_points[1,1]]
        v2 = [cell_points[2,0], cell_points[2,1]]

        # Computing centroids
        centroids[idx,:] = [(v0[0]+v1[0]+v2[0])/3,(v0[1]+v1[1]+v2[1])/3]

        # Extracting the area of the i-th cell
        areas[idx] = cell.ComputeArea()

    return centroids, areas


"""
Given a vtk reader, the sections coordinates and the bounds of each region, 
computes the Regional Averages of the flow quantitites.
"""
def __extractRegionalAverages(poly_data, chord):
    # Rearranging the distances of the sections from the origin according to the chord value
    distances = sections__x_distances * chord

    regional_averages = []
    # Iterating over the sections
    for section_distance in distances:
        # Creating the cut function
        plane = vtk.vtkPlane()
        plane.SetOrigin(section_distance, 0, 0.5)
        plane.SetNormal(1, 0, 0)

        # Cutting the space 
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(poly_data)
        cutter.Update()

        # Extracting the data of the section
        cutter_output = cutter.GetOutput()

        # Computing the centroids and the areas of the polygons
        centroids, areas = __cellValues(cutter_output)

        # Extracting the flow quantities
        p = vtk_to_numpy(cutter_output.GetCellData().GetArray('p'))
        U = vtk_to_numpy(cutter_output.GetCellData().GetArray('U'))
        U[:,2] = 0.0 # Removing the z component of the velocity

        # Computing the magnitude of the velocity vector
        U = np.array([np.linalg.norm(u) for u in U])

        # Normalizing the velocity vector w.r.t. the free stream velocity magnitude
        U /= free_stream__velocity_magnitude

        # Iterating over the regions of a section
        for r in range(len(regions__y_bounds) - 1):
            # Extracting the polygons whose centroid belongs to the i-th region
            polygons_indexes = np.array([index for index in range(len(centroids)) if centroids[index][1] >= regions__y_bounds[r] and centroids[index][1] < regions__y_bounds[r+1]])

            # Extracting the areas and the flow quantities of the triangles belonging to i-th the region
            region_areas = np.array([areas[index] for index in polygons_indexes])
            region_p =  np.array([p[index] for index in polygons_indexes])
            region_U =  np.array([U[index] for index in polygons_indexes])

            # Computing the sum of the areas of the polygons belonging to the i-th region
            sum_areas = np.sum(region_areas)

            # Regional averages of pressure and velocity
            average_p = np.dot(region_p, region_areas) / sum_areas if sum_areas > 0 else 0.0
            average_U = np.dot(region_U, region_areas) / sum_areas if sum_areas > 0 else 0.0

            # Saving the results into an array
            regional_averages.append({"p": average_p, "U": average_U})

    return regional_averages


def regionalAverages(reader, chord):
    # Extracting the data of the grid
    poly_data = reader.GetOutput()

    # Computing the regional averages of the flow quantities
    regional_averages = __extractRegionalAverages(poly_data, chord)

    regional_averages = {
        "p": [regional_average["p"] for regional_average in regional_averages],
        "U": [regional_average["U"] for regional_average in regional_averages],
    }

    return regional_averages