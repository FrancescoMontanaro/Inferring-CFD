import vtk
import numpy as np
import matplotlib.pyplot as plt
from Geometry import Point, Line, Segment, Rectangle
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

plt.style.use('seaborn')


'### GLOBAL VARIABLES AND CONSTANTS ###'

sections_length = 10.0 # Y length of the cutting sections
sections_distance = 3.0 # X distance of the cutting sections from the origin
section_origin = Point(0.5, 0.0) # X and Y origin of the cutting plane
maximum_propagation = 30.0 # Maximum streamlines length
streamlines_resolution = 120 # Number of streamlines
free_stream__gradient = 0.17453 # Gradient of the free stream
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream


'### FUNCTIONS ###'

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
Given a vtk reader, the distance of the cutting sections (S and E), the  length of the 
cutting sections (S and E) the gradiet of the free stream and the chord of the i-th airfoil,
extracts the streamlines and computes their arrival time from section S to section E.
"""
def __extractArrivalTimes(reader):
    # Extracting the data of the grid
    poly_data = reader.GetOutput()

    # Cutting plane
    cutting_plane = Rectangle(section_origin, 2*sections_distance, sections_length)

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
    #points = __rotatePoints(points)

    # Extracting the velocity of the points and removing the last column (z coordinate)
    U = vtk_to_numpy(streamer_output.GetPointData().GetArray("Velocity"))
    U = np.delete(U, -1, axis=1)

    # Computing the magnitude of the velocity of the points
    U = np.array([np.linalg.norm(u) for u in U])

    # Normalizing the velocity w.r.t. the free stream velocity magnitude
    U /= free_stream__velocity_magnitude

    # Extracting the number of streamlines
    num_streamlines = streamer_output.GetNumberOfCells()

    arrival_times = np.array([])
    for cell in range(num_streamlines):
        # Extracting the ids of the points of the streamline
        ids = __getPointsIds(streamer_output, cell)

        # Extracting the coordinates and the velocity of the points of the streamline
        streamline_points = [Point(points[id, 0], points[id, 1]) for id in ids]
        streamline_U = [U[id] for id in ids]

        # Extracting the indices of the points belonging to the cutting region
        region_indices = [idx for idx in range(len(streamline_points)) if cutting_plane.containsPoint(streamline_points[idx])]

        region_points = np.array([streamline_points[idx] for idx in region_indices])
        region_U = np.array([streamline_U[idx] for idx in region_indices])

        if len(region_points) > 2:
            # Computing the points of the streamline lying on the cutting sections
            streamline = Line(region_points[0], region_points[-1])

            S = Segment(Point(cutting_plane.west, cutting_plane.south), Point(cutting_plane.west, cutting_plane.north))
            E = Segment(Point(cutting_plane.east, cutting_plane.south), Point(cutting_plane.east, cutting_plane.north))
            
            lower_intersection = streamline.SegmentIntersection(S)
            upper_intersection = streamline.SegmentIntersection(E)

            # Filtering streamlines not intersecting the starting and arrival section E 
            if lower_intersection is not None and upper_intersection is not None:
                # Adding the points of the streamline lying on the section S and E
                region_points = np.insert(region_points, 0, lower_intersection)
                region_U = np.insert(region_U, 0, region_U[0])
                
                region_points = np.append(region_points, upper_intersection)
                region_U = np.append(region_U, region_U[-1])

                # Computing the time distance of the consecutive points
                time_distances = np.zeros(len(region_points)-1)
                for idx in range(len(region_points) - 1):
                    segment_length = region_points[idx].distance(region_points[idx+1])
                    U_edge = (region_U[idx] + region_U[idx+1]) / 2
                    
                    time_distances[idx] = segment_length / U_edge

                # Computing the arrival time of the streamline
                arrival_time = np.sum(time_distances)

                # Adding the obtained value to the main list
                arrival_times = np.append(arrival_times, arrival_time)

    return arrival_times


"""
Function to compute the n-th centered moment of an array
"""
def __nthMoment(data, n):
    n_moment = np.sum((data - np.mean(data)) ** n) / len(data)
    return n_moment


"""
Given the arrival times, computes their distribution statistics.
"""
def __extractStatistics(arrival_times):
    mean = np.mean(arrival_times)
    second_order_moment = __nthMoment(arrival_times, 2)
    third_order_moment = __nthMoment(arrival_times, 3)
    fourth_order_moment = __nthMoment(arrival_times, 4)
    fifth_order_moment = __nthMoment(arrival_times, 5)

    return mean, second_order_moment, third_order_moment, fourth_order_moment, fifth_order_moment


"""
Given a vtk reader extracts the arrival times of the streamlines to the arrival section.
"""
def arrivalTimes(reader):
    # Computing the streamlines' arrival time to the arrival section
    arrival_times = __extractArrivalTimes(reader)

    # Extracting the distribution statistics of the arrival_times
    distribution_statistics = __extractStatistics(arrival_times)

    return [{"distribution_statistics": list(distribution_statistics)}]