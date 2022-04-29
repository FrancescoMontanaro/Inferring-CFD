import vtk
import utils
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


# Global variables and constants
bins_count = 32 # Numer of bins
sections_length = 10 # Y length of the cutting sections
sections_distance = 3 # X distance of the cutting sections from the origin
maximum_propagation = 20 # Maximum streamlines length
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
Given the distance of the sections S and E from the origin, their length, the 
gradient of the free stream and the chord of the airfoil, computes the coordinates 
of the cutting plane and the equation (gradient - m , intercept - q) of the 
sections orthogonal to the free stream.
"""
def __cuttingPlane(chord):
    # Rearranging the distances according to the chord length
    distance = sections_distance * chord
    length = sections_length * chord

    # Extracting the coordinates of the cutting plane according to the distances
    boundaries = np.array([
        [-distance/2, +length/2],
        [-distance/2, -length/2],
        [+distance/2, -length/2],
        [+distance/2, +length/2]
    ])

    # Rotation coefficients
    l_cos = 1 / np.sqrt(1 + free_stream__gradient**2)
    l_sin = free_stream__gradient * l_cos

    # Rotation matrix
    l_rot = np.array([[l_cos , -l_sin], [l_sin, l_cos]])
    
    # Rotating the points of the cutting plane according to the gradient of the free stream
    plane_boundaries = (l_rot @ boundaries.T).T

    return plane_boundaries


"""
Given the coordinates of a point and a polygon, checks if the point belongs
to the polygon's boundaries.
"""
def __belongsToPolygon(point, polygon):
    belongs_to_polygon = False
    num_points = len(polygon)
    j = num_points - 1

    # Iterating over the points of the polygon
    for i in range(num_points):
        # Checks if the point is on the corner of the polygon
        if (point[0] == polygon[i][0]) and (point[1] == polygon[i][1]):
            return True 
        # Checks if the point is inside the boundaries
        if ((polygon[i][1] > point[1]) != (polygon[j][1] > point[1])):
            slope = (point[0]-polygon[i][0])*(polygon[j][1]-polygon[i][1])-(polygon[j][0]-polygon[i][0])*(point[1]-polygon[i][1])
            if slope == 0:
                return True
            if (slope < 0) != (polygon[j][1] < polygon[i][1]):
                belongs_to_polygon = not belongs_to_polygon
        j = i

    return belongs_to_polygon


"""
Given a vtk reader, the distance of the cutting sections (S and E), the  length of the 
cutting sections (S and E) the gradiet of the free stream and the chord of the i-th airfoil,
extracts the streamlines and computes their arrival time from section S to section E.
"""
def __extractArrivalTimes(reader, chord):
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
    center = - maximum_propagation / 2

    seed = vtk.vtkLineSource()
    seed.SetPoint1((center, center, 0))
    seed.SetPoint2((center, -center, 0))
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

    # Creating the cutting plane
    cutting_plane = __cuttingPlane(chord)

    # Extracting the points of the Streamtracer object and removing the last column (z coordinate)
    points = vtk_to_numpy(streamer_output.GetPoints().GetData())
    points = np.delete(points, -1, axis=1)

    # Extracting the velocity of the points and removing the last column (z coordinate)
    U = vtk_to_numpy(streamer_output.GetPointData().GetArray("Velocity"))
    U = np.delete(U, -1, axis=1)

    # Normalizing the velocity components w.r.t. the free stream velocity magnitude
    U[:,0] -= free_stream__velocity_magnitude * np.cos(free_stream__gradient)
    U[:,1] -= free_stream__velocity_magnitude * np.sin(free_stream__gradient)

    # Computing the magnitude of the velocity of the points
    U = np.array([np.linalg.norm(u) for u in U])

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

        # Extracting the indices of the points belonging to the cutting plane
        indices = [idx for idx in range(len(streamline_points)) if __belongsToPolygon(streamline_points[idx], cutting_plane)]

        # Filtering the points belonging to the cutting plane
        streamline_points = np.array([streamline_points[idx] for idx in indices])
        streamline_U = np.array([streamline_U[idx] for idx in indices])

        if len(streamline_points) > 0:
            # Computing the mean velocity of the points of the streamline
            U_mean = np.mean(streamline_U)

            # Computing the distance of the cutting sections 
            distance = 2 * sections_distance * np.cos(free_stream__gradient)

            # Computing the arrival time of the current streamline to the arrival section
            arrival_time = distance / U_mean

            arrival_times.append({"arrival_time": arrival_time, "y_coordinate": np.min(streamline_points[:,1])})

    return arrival_times


"""
Given the streamlines with their velocity value and the number of bins, 
performs the binning operation to generate a 1D signal.
"""
def __extractBins(arrival_times, bins_count):
    # Extracting the lower bound coordinate of the section
    lower_bound = - sections_length / (2 * np.cos(free_stream__gradient))

    bins = []
    # Iterating over the total number of bins
    for idx in range(bins_count):
        # Computing the bounds of the i-th bin
        bin_bounds = (lower_bound + idx * sections_length / bins_count), (lower_bound + (idx+1) * sections_length / bins_count)

        # Extracting the arrival times of the streamlines belonging to the i-th bin
        bin__arrival_times = [arrival_time for arrival_time in arrival_times if arrival_time["y_coordinate"] >= bin_bounds[0] and arrival_time["y_coordinate"] < bin_bounds[1]]
        # Computing the center coordinate of the i-th bin
        center = np.mean(bin_bounds)

        # Computing the average arrival time of the streamlines belonging to the i-th bin
        bin__arrival_time = float(np.mean([bin__arrival_time["arrival_time"] for bin__arrival_time in bin__arrival_times])) if len(bin__arrival_times) > 0 else None

        bins.append({"center": center, "arrival_time": bin__arrival_time})

    # Obtaining the values of the empty bins by interpolating the values of the adjacent ones
    if(bins[0]["arrival_time"] is None):
        upper_bin = None
        i = 0
        while(upper_bin is None and i < len(bins)):
            if(bins[i]["arrival_time"] is not None):
                upper_bin = bins[i]
            i += 1

        bins[0]["arrival_time"] = upper_bin["arrival_time"]

    if(bins[-1]["arrival_time"] is None):
        lower_bin = None
        i = len(bins) - 1
        while(lower_bin is None and i > 0):
            if(bins[i]["arrival_time"] is not None):
                lower_bin = bins[i]
            i -= 1

        bins[-1]["arrival_time"] = lower_bin["arrival_time"]

    for i in range(1, len(bins)-1):
        if(bins[i]["arrival_time"] is None):
            lower_bin = None
            j = i
            while(lower_bin is None and j >= 0):
                if(bins[j]["arrival_time"] is not None):
                    lower_bin = bins[j]
                j -= 1

            upper_bin = None
            j = i
            while(upper_bin is None and j < len(bins)):
                if(bins[j]["arrival_time"] is not None ):
                    upper_bin = bins[j]
                j += 1

            bins[i]["arrival_time"] = np.mean([lower_bin["arrival_time"], upper_bin["arrival_time"]])

    bins = {"arrival_times": [bin["arrival_time"] for bin in bins]}

    return bins


"""
Given a vtk reader extracts the arrival times of the streamlines to the arrival section.
"""
def arrivalTimes(reader, chord):
    # Computing the streamlines' arrival time to the arrival section
    arrival_times = __extractArrivalTimes(reader, chord)

    # Extracting the bins
    bins = __extractBins(arrival_times, bins_count)

    print(bins)

    return bins