import Utils
from FlowSignals import flowSignals
from ArrivalTimes import arrivalTimes
from RegionalAverages import regionalAverages
from InformativePoints import informativePoints
from StreamlinesSignals import streamlinesSignals
from RegionalArrivalTimes import regionalArrivalTimes


# Input data path
data_path = "/Volumes/T5/files/" #CHANGE ME

flow_signals = False
arrival_times = False
regional_averages = False
informative_points = False
streamlines_signals = False
regional_arrival_times = False

save_data = False

# Extracting the vtk files
data_files = Utils.getDataFiles(data_path)

# Creating the file to store the results
if save_data:
    Utils.CreateResultsFile()

# Iterating over the vtk files
for data_file in data_files:
    try:
        # Reading the vtk file
        reader = Utils.readVtk(data_file["data_path"])

        # Loading the vtk file of the target shape and extracting its labels
        naca_numbers = Utils.targetValues(data_file["file_name"])

        if informative_points:
            # Extracting the informative points of the flow fields
            features = informativePoints(reader)

        if regional_averages:
            # Extracting the regional averages of the flow quantities
            features = regionalAverages(reader)

        if arrival_times:
            # Extracting the streamlines arrival times
            features = arrivalTimes(reader)

        if flow_signals:
            # Extracting the signals associated to the flow quantities
            features = flowSignals(reader)

        if streamlines_signals:
            # Extracting the signals associated to the streamlines
            features = streamlinesSignals(reader)

        if regional_arrival_times:
            # Extracting the signals associated to the streamlines
            features = regionalArrivalTimes(reader)

    except Exception as e:
        # Displaying errors
        print(f'ERROR OCCURREND FOR FILE: {data_file["file_name"]} --> {str(e)}')

    else:
        # Saving the result into the destination file
        if save_data:
            Utils.saveResults({"features": features, "naca_numbers": naca_numbers})

        # Displaying progress
        print(f'{data_file["file_name"]} --> {(data_files.index(data_file) + 1)}/{len(data_files)} files processed')