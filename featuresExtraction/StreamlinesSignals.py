import vtk
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


# Global variables and constants
bins_count = 512 # Numer of bins
section__y_bounds = [-5, 5] # Y Boundaries of the signal
maximum_propagation = 20 # maximum streamlines length
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

    # Extracting the magnitude of the velocity of the points
    U = vtk_to_numpy(streamer_output.GetPointData().GetArray("Velocity"))
    U = np.delete(U, -1, axis=1)

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
        streamline_length = sum(segments_length)

        # Computing the velocity of the edges
        U_edges = (streamline_U[1:] - streamline_U[:-1]) / 2 + streamline_U[:-1]

        # Normalizing the velocity components
        U_edges[:,0] -= free_stream__velocity_magnitude * np.cos(free_stream__gradient)
        U_edges[:,1] -= free_stream__velocity_magnitude * np.sin(free_stream__gradient)
        
        # Computing the magnitude of the velocity vector
        U_edges = np.array([np.linalg.norm(u) for u in U_edges])

        # Computing the mean velocity of the streamlines
        U_mean = np.dot(U_edges, segments_length) / streamline_length

        streamlines.append({"U_mean": U_mean, "y_coordinate": np.min(streamline_points[:,1])})

    return streamlines


"""
Given the streamlines with their velocity value and the number of bins, 
performs the binning operation to generate a 1D signal.
"""
def __extractBins(streamlines):
    # Computing the length of the section
    section_length = np.sqrt((np.min(section__y_bounds) - np.max(section__y_bounds))**2)

    # Extracting the lower bound coordinate of the section
    lower_bound = np.min(section__y_bounds)

    bins = []
    # Iterating over the total number of bins
    for idx in range(bins_count):
        # Computing the bounds of the i-th bin
        bin_bounds = (lower_bound + idx * section_length / bins_count), (lower_bound + (idx+1) * section_length / bins_count)

        # Extracting the arrival times of the streamlines belonging to the i-th bin
        bin_streamlines = [streamline for streamline in streamlines if streamline["y_coordinate"] >= bin_bounds[0] and streamline["y_coordinate"] < bin_bounds[1]]
        # Computing the center coordinate of the i-th bin
        center = np.mean(bin_bounds)

        # Computing the average arrival time of the streamlines belonging to the i-th bin
        bin__U_mean = float(np.mean([bin_streamline["U_mean"] for bin_streamline in bin_streamlines])) if len(bin_streamlines) > 0 else None

        bins.append({"center": center, "U": bin__U_mean})

    # Obtaining the values of the empty bins by interpolating the values of the adjacent ones
    if(bins[0]["U"] is None):
        upper_bin = None
        i = 0
        while(upper_bin is None and i < len(bins)):
            if(bins[i]["U"] is not None):
                upper_bin = bins[i]
            i += 1

        bins[0]["U"] = upper_bin["U"]

    if(bins[-1]["U"] is None):
        lower_bin = None
        i = len(bins) - 1
        while(lower_bin is None and i > 0):
            if(bins[i]["U"] is not None):
                lower_bin = bins[i]
            i -= 1

        bins[-1]["U"] = lower_bin["U"]

    for i in range(1, len(bins)-1):
        if(bins[i]["U"] is None):
            lower_bin = None
            j = i
            while(lower_bin is None and j >= 0):
                if(bins[j]["U"] is not None):
                    lower_bin = bins[j]
                j -= 1

            upper_bin = None
            j = i
            while(upper_bin is None and j < len(bins)):
                if(bins[j]["U"] is not None ):
                    upper_bin = bins[j]
                j += 1

            bins[i]["U"] = np.mean([lower_bin["U"], upper_bin["U"]])

    bins = {"U": [bin["U"] for bin in bins]}

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