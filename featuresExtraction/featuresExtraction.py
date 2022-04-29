import utils
from FlowSignals import flowSignals
from ArrivalTimes import arrivalTimes
from RegionalAverages import regionalAverages
from StreamlinesSignals import streamlinesSignals


# Input data path
data_path = "/Volumes/T5/files" #CHANGE ME

flow_signals = False
arrival_times = False
regional_averages = False
streamlines_signals = True


# Extracting the vtk files
data_files = utils.getDataFiles(data_path)

# Creating the file to store the results
utils.CreateResultsFile()

# Iterating over the vtk files
for data_file in data_files:
    try:
        # Reading the vtk file
        reader = utils.readVtk(data_file["data_path"])

        # Loading the vtk file of the target shape and extracting its labels
        chord, naca_numbers = utils.targetValues(data_file["file_name"], data_file["target_path"])

        if regional_averages:
            # Extracting the regional averages of the flow quantities
            features = regionalAverages(reader, chord)

        if arrival_times:
            # Extracting the streamlines arrival times
            features = arrivalTimes(reader, chord)

        if flow_signals:
            # Extracting the signals associated to the flow quantities
            features = flowSignals(reader)

        if streamlines_signals:
            # Extracting the signals associated to the streamlines
            features = streamlinesSignals(reader)

    except Exception as e:
        # Displaying errors
        print(f'ERROR OCCURREND FOR FILE: {data_file["file_name"]} --> {str(e)}')

    else:
        # Saving the result into the destination file
        utils.saveResults({"features": features, "naca_numbers": naca_numbers})

        # Displaying progress
        print(f'{data_file["file_name"]} --> {(data_files.index(data_file) + 1)}/{len(data_files)} files processed')