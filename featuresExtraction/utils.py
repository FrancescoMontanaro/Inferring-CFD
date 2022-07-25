import os
import vtk
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

results_file = ""

"""
Given a path, extracts a list of tuples containig the name of the vtk files, the path
of the simulation data file and the path of target data file
"""
def getDataFiles(data_path):
    data_files = []
    # Iterating over the folders and files of the path
    for path, _, files in os.walk(data_path):
        for file in files:
            file_components = file.split(".")
            if(file_components[-1] == "vtk" and "_walls" not in file):
                data_files.append({"name": file_components[0], "path": os.path.join(path, file)})

    return data_files
    

"""
Given a VTK file, extracts the corresponding vtk reader
"""
def readVtk(file_path):
    # Creating the UnstructuredGridReader object
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    return reader


"""
Given a VTK file, extracts the target values to be predicted
"""
def targetValues(file_name):
    # Extracting the chord of the airfoil from its boundaries
    chord = 1.0

    # Saving the values of the naca numbers into a dictionary
    naca_numbers = [
        int(file_name[0]) * chord, # maximum_camber
        int(file_name[1]) * chord, # maximum_camber_position
        int(file_name[2:]) * chord # maximum_thickness
    ]

    return naca_numbers


"""
Given an array of bins and the flow quantity, displays the signal in a 
2D space.
"""
def displayData(data):
    # Plotting the data
    plt.plot(data)
    plt.show()


"""
Given a vtk output data, plots the mesh
"""
def PlotVTKData(poly_data):
    if type(poly_data) != "vtkmodules.vtkCommonDataModel.vtkPolyData":
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputData(poly_data)
        geometry_filter.Update()
        poly_data = geometry_filter.GetOutput()

    # Creating the mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    # Creating the actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.VisibilityOn()
    actor.GetProperty().SetColor(255, 255, 255)
    actor.VisibilityOn()

    # Renderer
    renderer = vtk.vtkRenderer()

    # Renderer window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName('StreamLines')

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    render_window.SetInteractor(interactor)
    render_window.SetSize(1000, 800)

    renderer.AddActor(actor)

    interactor.Initialize()
    render_window.Render()
    interactor.Start()



"""
Saves the results to a compressed .npz file
"""
def saveData(data, file_path=None):
    if file_path is None:
        current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        file_name = f'dataset_{current_datetime}.npz'

        file_path = os.path.join('Dataset', file_name)

    np.savez_compressed(file_path, **data)