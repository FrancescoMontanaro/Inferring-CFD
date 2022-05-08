import vtk
import Utils
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


# Global variables and constants
maximum_propagation = 30 # Maximum streamlines length
streamlines_resolution = 400 # Number of streamlines
free_stream__gradient = 0.17453 # Gradient of the free stream
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream
sections__x_distances = np.array([[-5.0, -1.0], [-1.0, 1.0], [1.0, 5.0]]) # Boundaries of the X cutting sections
regions__y_bounds = np.array([-10, -3.5, -1.75, -0.75, 0, 0.75, 1.75, 3.5, 10]) # Boundaries of the Y regions


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
Given a VTK reader, extracts the streamlines and computes the time distances
of the consecutive points belonging to them.
"""
def __extractStreamlines(reader):
    # Extracting the data of the grid
    poly_data = reader.GetOutput()

    # Setting the z component of the velocity field to 0
    U = vtk_to_numpy(poly_data.GetCellData().GetArray("U"))
    U[:,2] = 0.0

    # Adding a new array with the transformed velocity to the grid
    U_vtk = numpy_to_vtk(U)
    U_vtk.SetName('Velocity')
    poly_data.GetCellData().AddArray(U_vtk)

    # Creating the seed
    center_y = np.abs(regions__y_bounds).max()
    center_x = 2 * np.abs(sections__x_distances.flatten()).max()

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

    # Extracting the velocity of the points and removing the last column (z coordinate)
    U = vtk_to_numpy(streamer_output.GetPointData().GetArray("Velocity"))
    U = np.delete(U, -1, axis=1)

    # Computing the magnitude of the velocity of the points
    U = np.array([np.linalg.norm(u) for u in U])

    # Normalizing the velocity w.r.t. the free stream velocity magnitude
    U /= free_stream__velocity_magnitude

    # Extracting the number of streamlines
    num_streamlines = streamer_output.GetNumberOfCells()

    # Iterating over the streamlines
    streamlines = []
    for cell in range(num_streamlines):  
        ids = __getPointsIds(streamer_output, cell)

        # Extracting the points of the streamline
        streamline_points = np.array([points[id] for id in ids])
        streamline_points = __rotatePoints(streamline_points)

        # Extracting the velocity of the points of the streamline
        streamline_U = np.array([U[id] for id in ids])

        # Computing the distance of the points 
        dx = streamline_points[1:,0] - streamline_points[:-1,0]
        dy = streamline_points[1:,1] - streamline_points[:-1,1]

        # Extracting the length of the segments connecting points
        segments_length = np.sqrt(dx**2 + dy**2)

        # Computing the velocity of the edges
        U_edges = (streamline_U[1:] - streamline_U[:-1]) / 2 + streamline_U[:-1]

        # Computing the time distance of the consecutive points
        time_distances = segments_length / U_edges
        time_distances = np.insert(time_distances, 0, 0.0)

        streamlines.append({"points": streamline_points, "time_distances": time_distances})

    return streamlines


"""
Given a VTK reader, extracts the streamlines and computes the time distances
of the consecutive points belonging to them.
"""
def __extractRegionalArrivalTimes(streamlines):
    regional__arrival_times = []
    # Iterating over the X sections
    for section_distance in sections__x_distances:
        # Iterating over the Y regions of the current section
        for r in range(len(regions__y_bounds) - 1):
            # Filtering the streamlines belonging to the current Y region
            region_streamlines = [streamline for streamline in streamlines if all(y >= regions__y_bounds[r] and y < regions__y_bounds[r+1] for (_, y) in streamline["points"])]

            # Iterating over the streamlines belonging to the current region
            arrival_times = []
            for streamline in region_streamlines:
                # Filtering the indices of points belonging to the current X section
                indices = [idx for idx in range(len(streamline["points"])) if streamline["points"][idx, 0] >= np.min(section_distance) and streamline["points"][idx, 0] < np.max(section_distance)]

                if len(indices) > 0:
                    # Filtering the coordinates and the velocity magnitude of the points of the streamlines belonging to the current X section
                    region_points = np.array([streamline["points"][idx] for idx in indices])
                    region__time_distances = np.array([streamline["time_distances"][idx] for idx in indices])

                    # Extracting the closest point not belonging to the region of interest
                    lower_idx = indices[0] -1
                    upper_idx = indices[-1] + 1

                    if lower_idx >= 0 and upper_idx < len(streamline["points"]):
                        # Extracting the coordinates of the points lying on the cutting sections (linear interpolation)
                        lower__point_on_section = [
                            section_distance[0],
                            region_points[0,1] + (section_distance[0] - region_points[0,0]) * (streamline["points"][lower_idx,1] - region_points[0,1]) / (streamline["points"][lower_idx,0] - region_points[0,0])
                        ]

                        upper__point_on_section = [
                            section_distance[1],
                            region_points[-1,1] + (section_distance[1] - region_points[-1,0]) * (streamline["points"][upper_idx,1] - region_points[-1,1]) / (streamline["points"][upper_idx,0] - region_points[-1,0])
                        ]

                        # Computing the distance between the cutting sections and the closest points belonging to it
                        lower__section_distance = np.linalg.norm(lower__point_on_section - region_points[0])
                        upper__section_distance = np.linalg.norm(upper__point_on_section - region_points[-1])

                        # Computing the distance between the closest points belonging to the cutting sections and the
                        # closest points not belonging to it
                        lower_point_distance = np.linalg.norm(streamline["points"][lower_idx] - region_points[0])
                        upper_point_distance = np.linalg.norm(streamline["points"][upper_idx] - region_points[-1])

                        # Computing the ratio of the computed values 
                        lower_ratio = lower__section_distance / lower_point_distance
                        upper_ratio = upper__section_distance / upper_point_distance

                        # Computing the time distance between the closest points to the cutting sections
                        # and the cutting section itself
                        lower_time_distance = region__time_distances[0] * lower_ratio
                        upper_time_distance = streamline["time_distances"][upper_idx] * upper_ratio

                        # Computing the arrival time of the current streamline
                        arrival_time = lower_time_distance + sum(region__time_distances[1:]) + upper_time_distance

                        arrival_times.append(arrival_time)

            # Raising an exception if the current region is empty
            if len(arrival_times) == 0:
                raise Exception(f'Empty region: X=[{section_distance[0]},{section_distance[1]}] , Y=[{regions__y_bounds[r]},{regions__y_bounds[r+1]}]')

            # Computing the mean of the arrival time of the streamlines belonging to the current region
            regional_arrival_time = np.mean(arrival_times)    

            regional__arrival_times.append(regional_arrival_time)

    regional__arrival_times = {"arrival_times": regional__arrival_times}

    return regional__arrival_times


"""
Given a VTK reader extracts the regional arrival times of the streamlines to the arrival sections.
"""
def regionalArrivalTimes(reader):
    # Computing the streamlines
    streamlines = __extractStreamlines(reader)

    # Extracting the regional arrival times
    regional__arrival_times = __extractRegionalArrivalTimes(streamlines)

    return regional__arrival_times