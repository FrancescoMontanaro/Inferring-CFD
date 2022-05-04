import vtk
import utils
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


# Global variables and constants
bins_count = 64 # Numer of bins
sections_length = 10.0 # Y length of the cutting sections
sections_distance = 3.0 # X distance of the cutting sections from the origin
maximum_propagation = 30.0 # Maximum streamlines length
streamlines_resolution = 350 # Number of streamlines
free_stream__gradient = 0.17453 # Gradient of the free stream
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream


"""
Given a VTK objects, extracts the points belonging to a specific cell
"""
def __getPointsIds(streamer_output, cell):
    # Loading the cells' ids into a vtkIdList
    vtk_id_list = vtk.vtkIdList()
    streamer_output.GetCellPoints(cell, vtk_id_list)

    # Extracting the ids of the points of the cell
    ids = []
    for idx in range(vtk_id_list.GetNumberOfIds()):
        ids.append(vtk_id_list.GetId(idx))

    return ids


"""
Given a set of points, rotates the points of an angle equal to the
gradient of the free stream.
"""
def __rotatePoints(points):
    points = np.array(points)

    # Rotation coefficients
    l_cos = 1 / np.sqrt(1 + free_stream__gradient**2)
    l_sin = free_stream__gradient * l_cos

    # Rotation matrix
    l_rot = np.array([[l_cos , -l_sin], [-l_sin, l_cos]])
    
    # Rotating the points of the cutting plane according to the gradient of the free stream
    points = (l_rot @ points.T).T

    return points


"""
Given a vtk reader, the distance of the cutting sections (S and E), the  length of the 
cutting sections (S and E) the gradiet of the free stream and the chord of the i-th airfoil,
extracts the streamlines and computes their arrival time from section S to section E.
"""
def __extractArrivalTimes(reader):
    # Extracting the data of the grid
    poly_data = reader.GetOutput()

    # Computing the boundaries of the cutting plane
    section__y_bounds = [-sections_length/2, +sections_length/2]
    section__x_bounds = [-sections_distance, +sections_distance]

    # Setting the z component of the velocity field to 0
    U = vtk_to_numpy(poly_data.GetCellData().GetArray("U"))
    U[:,2] = 0.0

    # Adding a new array with the transformed velocity to the grid
    U_vtk = numpy_to_vtk(U)
    U_vtk.SetName('Velocity')
    poly_data.GetCellData().AddArray(U_vtk)

    # Creating the seed
    center_y = 0.75 * sections_length
    center_x = 2 * sections_distance

    seed = vtk.vtkLineSource()
    seed.SetPoint1((-center_x, -center_y, 0.5))
    seed.SetPoint2((-center_x, center_y, 0.5))
    seed.SetResolution(streamlines_resolution)

    # Creating the Streamtracer object
    streamer = vtk.vtkStreamTracer()
    streamer.SetInputConnection(reader.GetOutputPort())
    streamer.SetSourceConnection(seed.GetOutputPort())
    streamer.SetMaximumPropagation(maximum_propagation)
    streamer.SetInitialIntegrationStep(0.01)
    streamer.SetMaximumError(1e-5)
    streamer.SetIntegrationDirectionToForward()
    streamer.SetIntegratorTypeToRungeKutta45()
    streamer.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_CELLS", "Velocity")
    streamer.Update()

    # Extracting the Streamer's output data
    streamer_output = streamer.GetOutput()

    # Extracting the points of the Streamtracer object and removing the last column (z coordinate)
    points = vtk_to_numpy(streamer_output.GetPoints().GetData())
    points = np.delete(points, -1, axis=1)

    # Rotating the points of an angle equal to the free stream gradient
    points = __rotatePoints(points)

    # Extracting the velocity of the points and removing the last column (z coordinate)
    U = vtk_to_numpy(streamer_output.GetPointData().GetArray("Velocity"))
    U = np.delete(U, -1, axis=1)

    # Computing the magnitude of the velocity of the points
    U = np.array([np.linalg.norm(u) for u in U])

    # Normalizing the velocity w.r.t. the free stream velocity magnitude
    U /= free_stream__velocity_magnitude

    # Extracting the number of streamlines
    num_streamlines = streamer_output.GetNumberOfCells()

    arrival_times = []
    for cell in range(num_streamlines):
        # Extracting the ids of the points of the streamline
        ids = __getPointsIds(streamer_output, cell)

        # Extracting the points of the streamline
        streamline_points = np.array([points[id] for id in ids])

        # Extracting the velocity of the points of the streamline
        streamline_U = np.array([U[id] for id in ids])

        # Checking if the streamline belongs to the vertical section of the cutting plane
        if all(y >= section__y_bounds[0] and y <= section__y_bounds[1] for (_,y) in streamline_points):
            # Computing the distance of the consecutive points of the streamline
            dx = streamline_points[1:,0] - streamline_points[:-1,0]
            dy = streamline_points[1:,1] - streamline_points[:-1,1]

            # Extracting the length of the segments connecting points
            segments_length = np.sqrt(dx**2 + dy**2)

            # Computing the velocity of the edges as the mean of the velocity of the points
            U_edges = (streamline_U[1:] + streamline_U[:-1]) / 2

            # Computing the time distance of the consecutive points
            time_distances = segments_length / U_edges
            time_distances = np.insert(time_distances, 0, 0.0)

            # Filtering the indices of the points belonging to the horiziontal section of the cutting plane
            indices = [idx for idx in range(len(streamline_points)) if streamline_points[idx, 0] >= section__x_bounds[0] and streamline_points[idx, 0] <= section__x_bounds[1]]

            # Extracting the coordinates and the time distances of the points belonging to the cutting section
            region_points = np.array([streamline_points[idx] for idx in indices])
            region_time_distances = np.array([time_distances[idx] for idx in indices])

            # Extracting the closest point not belonging to the region of interest
            lower_idx = indices[0] -1
            upper_idx = indices[-1] + 1

            if lower_idx >= 0 and upper_idx < len(streamline_points):
                # Extracting the coordinates of the points lying on the cutting sections (linear interpolation)
                lower__point_on_section = [
                    section__x_bounds[0],
                    region_points[0,1] + (section__x_bounds[0] - region_points[0,0]) * (streamline_points[lower_idx,1] - region_points[0,1]) / (streamline_points[lower_idx,0] - region_points[0,0])
                ]

                upper__point_on_section = [
                    section__x_bounds[1],
                    region_points[-1,1] + (section__x_bounds[1] - region_points[-1,0]) * (streamline_points[upper_idx,1] - region_points[-1,1]) / (streamline_points[upper_idx,0] - region_points[-1,0])
                ]

                # Computing the distance between the cutting sections and the closest points belonging to it
                lower__section_distance = np.linalg.norm(lower__point_on_section - region_points[0])
                upper__section_distance = np.linalg.norm(upper__point_on_section - region_points[-1])

                # Computing the distance between the closest points belonging to the cutting sections and the
                # closest points not belonging to it
                lower_point_distance = np.linalg.norm(streamline_points[lower_idx] - region_points[0])
                upper_point_distance = np.linalg.norm(streamline_points[upper_idx] - region_points[-1])

                # Computing the ratio of the computed values 
                lower_ratio = lower__section_distance / lower_point_distance
                upper_ratio = upper__section_distance / upper_point_distance

                # Computing the time distance between the closest points to the cutting sections
                # and the cutting section itself
                lower_time_distance = region_time_distances[0] * lower_ratio
                upper_time_distance = time_distances[upper_idx] * upper_ratio

                # Computing the arrival time of the current streamline
                arrival_time = lower_time_distance + np.sum(region_time_distances[1:]) + upper_time_distance

                arrival_times.append({"arrival_time": arrival_time, "y_min": region_points[0,1]})

    return arrival_times


"""
Given the streamlines with their velocity value and the number of bins, 
performs the binning operation to generate a 1D signal.
"""
def __extractBins(arrival_times, bins_count):
    # Extracting the boundaries of the bins
    bins_bounds = np.linspace(-sections_length/2, +sections_length/2, num=bins_count)

    # Creating an array of empty bins
    bins = np.full(bins_count, None)

    # Iterating over the total number of bins
    for idx in range(len(bins_bounds) - 1):
        # Extracting the arrival times of the streamlines belonging to the i-th bin
        bin__arrival_times = [arrival_time["arrival_time"] for arrival_time in arrival_times if arrival_time["y_min"] >= bins_bounds[idx] and arrival_time["y_min"] < bins_bounds[idx+1]]

        # Computing the average arrival time of the streamlines belonging to the i-th bin
        bin__arrival_time = float(np.mean([bin__arrival_time for bin__arrival_time in bin__arrival_times])) if len(bin__arrival_times) > 0 else None

        # Assigning the arrival time to the i-th bin
        bins[idx] = bin__arrival_time

    # Obtaining the values of the empty bins by interpolating the values of the adjacent ones
    if(bins[0] is None):
        i, upper_bin = 0, None
        while(upper_bin is None and i < len(bins)):
            if(bins[i] is not None):
                upper_bin = bins[i]
            i += 1

        bins[0] = upper_bin

    if(bins[-1] is None):
        i, lower_bin = len(bins) - 1, None
        while(lower_bin is None and i > 0):
            if(bins[i] is not None):
                lower_bin = bins[i]
            i -= 1

        bins[-1] = lower_bin

    for i in range(1, len(bins)-1):
        if(bins[i] is None):
            j, lower_bin = i, None
            while(lower_bin is None and j >= 0):
                if(bins[j] is not None):
                    lower_bin = bins[j]
                    lower_weight = 1 / i - j
                j -= 1

            j, upper_bin = i, None
            while(upper_bin is None and j < len(bins)):
                if(bins[j] is not None ):
                    upper_bin = bins[j]
                    upper_weight = 1 / j - i
                j += 1

            bins[i] = np.average([lower_bin, upper_bin], weights=[lower_weight, upper_weight])

    bins = {"arrival_times": list(bins)}

    return bins


"""
Given a vtk reader extracts the arrival times of the streamlines to the arrival section.
"""
def arrivalTimes(reader):
    # Computing the streamlines' arrival time to the arrival section
    arrival_times = __extractArrivalTimes(reader)

    # Extracting the bins
    bins = __extractBins(arrival_times, bins_count)

    return bins