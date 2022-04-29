import os
import vtk
import json
from pathlib import Path
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
            # Extracting the vtk files
            file_components = file.split(".")
            if(file_components[-1] == "vtk" and "walls" not in file):
                target_file_path = os.path.join(path, ("%s_walls.vtk" % file_components[0]))
                if(os.path.exists(target_file_path)):
                    data_files.append({"file_name": file_components[0], "data_path": os.path.join(path, file), "target_path": target_file_path})

    return data_files


"""
Saves the results into a json file
"""
def saveResults(result):
    data = []
    with open(results_file, 'r') as destination_file:
        data = json.load(destination_file)
        destination_file.close()

    data.append(result)

    with open(results_file, 'w') as destination_file:
        json.dump(data, destination_file, indent=2)
        destination_file.close()


"""
Creates a json file in the destination directory to save the obtained results
"""
def CreateResultsFile(destination_path=None, file_name=None):
    current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    destination_path = destination_path if destination_path else "./Dataset"
    file_name = file_name if file_name else f'dataset_{current_datetime}.json'

    file_path = os.path.join(destination_path, file_name)

    if(not os.path.exists(destination_path)):
        Path(destination_path).mkdir(parents=True)

    with open(file_path, 'w') as destination_file:
        json.dump([], destination_file, indent=2)
        destination_file.close()

    global results_file
    results_file = file_path
    

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
def targetValues(file_name, file_path):
    # Creating the PolyDataReader object
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Extracting the data of the mesh
    poly_data = reader.GetOutput()

    # Extracting the chord of the airfoil from its boundaries
    #bounds = poly_data.GetBounds()
    #chord = np.abs(bounds[1] - bounds[0])
    chord = 1.0

    # Saving the values of the naca numbers into a dictionary
    naca_numbers = {
        "maximum_camber": (int(file_name[0]) / 100) * chord,
        "maximum_camber_position": (int(file_name[1]) / 10) * chord,
        "maximum_thickness": (int(file_name[2:]) / 100) * chord
    }

    return chord, naca_numbers


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