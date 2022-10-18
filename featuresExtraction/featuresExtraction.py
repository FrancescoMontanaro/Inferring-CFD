import Utilities
from Sensor import sensorSignal
from FlowSignals import flowSignals
from RegionalAverages import regionalAverages


# Gloabl Variables
num_vtk_files = 1 # Number of vtk files to process (-1 for all of them)
save_data = False # Flag to save the data extracted to a local file
file_name = None # Name of the file in which to save the data
#data_path = "/Volumes/T5/files/" # Dataset path
data_path = "./VTK_files/0005" # Dataset path
featresExtractor = sensorSignal # Features to extract

# Extracting the vtk files
data_files = Utilities.getDataFiles(data_path)
data_files = data_files[:num_vtk_files]

# Iterating over the vtk files
data = {"naca_numbers": []}
for data_file in data_files:
    try:
        # Reading the vtk file
        reader = Utilities.readVtk(data_file["path"])

        # Loading the vtk file of the target shape and extracting its labels
        naca_numbers = Utilities.targetValues(data_file["name"])

        # Extracting the informative points of the flow fields
        samples = featresExtractor(reader)

        # Adding the features extracted to the main list
        for sample in samples:
            data["naca_numbers"].append(naca_numbers)
            for key in sample.keys():
                if key not in data:
                    data[key] = []
                data[key].append(sample[key])

    except Exception as e:
        # Displaying errors
        print(f'ERROR OCCURREND FOR FILE: {data_file["name"]} --> {str(e)}')

    else:
        # Displaying progress
        print(f'NACA {data_file["name"]} --> {(data_files.index(data_file) + 1)}/{len(data_files)} files processed')


# Saving the result into the destination file
if save_data:
    Utilities.saveData(data, file_name=file_name)