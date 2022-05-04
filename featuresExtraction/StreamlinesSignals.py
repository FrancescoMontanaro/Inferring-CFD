import vtk
import utils
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


# Global variables and constants
bins_count = 1024 # Numer of bins
sections_length = 10.0 # Y length of the cutting sections
maximum_propagation = 30.0 # maximum streamlines length
streamlines_resolution = 350 # Number of streamlines
free_stream__gradient = 0.17453 # Gradient of the free stream
free_stream__velocity_magnitude = 30.0 # Magnitude of the velocity of the free stream


"""
Given the streamlines data, plots them in a 2D space.
"""
def __plotStreamlines(streamlines):
    for streamline in streamlines:
        plt.plot(streamline["points"][:,0], streamline["points"][:,1], linewidth=1, color="black")
    plt.show()


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
Given a VTK reader object, extracts the streamlines's velocity field
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
    center = maximum_propagation / 2

    seed = vtk.vtkLineSource()
    seed.SetPoint1((-center, -center, 0))
    seed.SetPoint2((-center, center, 0))
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

    # Extracting the magnitude of the velocity of the points
    U = vtk_to_numpy(streamer_output.GetPointData().GetArray("Velocity"))
    U = np.delete(U, -1, axis=1)

    # Computing the magnitude of the velocity vector
    U = np.array([np.linalg.norm(u) for u in U])

    # Normalizing the velocity w.r.t. the free stream velocity magnitude
    U /= free_stream__velocity_magnitude

    # Extracting the points of the Streamtracer object and removing the last column (z coordinate)
    points = vtk_to_numpy(streamer_output.GetPoints().GetData())
    points = np.delete(points, -1, axis=1)

    # Extracting the number of streamlines
    num_streamlines = streamer_output.GetNumberOfCells()

    # Iterating over the streamlines
    streamlines = []
    for cell in range(num_streamlines):    
        # Extracting the ids of the points of the streamline
        ids = __getPointsIds(streamer_output, cell)

        # Extracting the points of the streamline
        streamline_points = np.array([points[id] for id in ids])

        # Extracting the velocity of the points the streamline
        streamline_U = np.array([U[id] for id in ids])

        # Computing the distance of the points 
        dx = streamline_points[1:,0] - streamline_points[:-1,0]
        dy = streamline_points[1:,1] - streamline_points[:-1,1]

        # Extracting the length of the segments connecting points
        segments_length = np.sqrt(dx**2 + dy**2)

        # Extracting the length of the streamline
        streamline_length = np.sum(segments_length)

        # Computing the velocity of the edges
        U_edges = (streamline_U[1:] - streamline_U[:-1]) / 2 + streamline_U[:-1]

        # Computing the mean velocity of the streamlines
        U_mean = np.dot(U_edges, segments_length) / streamline_length

        streamlines.append({"U_mean": U_mean, "y_min": streamline_points[0,1]})

    return streamlines


"""
Given the streamlines with their velocity value and the number of bins, 
performs the binning operation to generate a 1D signal.
"""
def __extractBins(streamlines):
    # Extracting the boundaries of the bins
    bins_bounds = np.linspace(-sections_length/2, +sections_length/2, num=bins_count)

    # Creating an array of empty bins
    bins = np.full(bins_count, None)

    # Iterating over the total number of bins
    for idx in range(len(bins_bounds) - 1):
        # Extracting the arrival times of the streamlines belonging to the i-th bin
        bin_streamlines = [streamline for streamline in streamlines if streamline["y_min"] >= bins_bounds[idx] and streamline["y_min"] < bins_bounds[idx+1]]

        # Computing the average arrival time of the streamlines belonging to the i-th bin
        bin__U_mean = float(np.mean([bin_streamline["U_mean"] for bin_streamline in bin_streamlines])) if len(bin_streamlines) > 0 else None

        bins[idx] = bin__U_mean

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
                    lower_weigth = 1 / i - j
                j -= 1

            j, upper_bin = i, None
            while(upper_bin is None and j < len(bins)):
                if(bins[j] is not None ):
                    upper_bin = bins[j]
                    upper_weight = 1 / j - i
                j += 1

            bins[i] = np.average([lower_bin, upper_bin], weights=[lower_weigth, upper_weight])

    bins = {"U": list(bins)}

    return bins


"""
Given a vtk reader and the number of bins, extracts the 1D signal
associated to the streamlines.
"""
def streamlinesSignals(reader):
    # Computing streamlines
    streamlines = __extractStreamlines(reader)

    # Extracting the bins
    bins = __extractBins(streamlines)

    return bins