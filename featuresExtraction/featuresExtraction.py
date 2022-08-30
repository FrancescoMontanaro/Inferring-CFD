import Utils
from Sensor import sensorSignal
from FlowSignals import flowSignals
from ArrivalTimes import arrivalTimes
from RegionalAverages import regionalAverages
from InformativePoints import informativePoints
from StreamlinesSignals import streamlinesSignals
from RegionalArrivalTimes import regionalArrivalTimes


# Input data path
#data_path = "/Volumes/T5/files/" #CHANGE ME
data_path = "./0005"

# Features to extract
flow_signals = False
sensor_signal = True
arrival_times = False
regional_averages = False
informative_points = False
streamlines_signals = False
regional_arrival_times = False

# Flag to save the data extracted to a a local file
save_data = False

# Extracting the vtk files
data_files = Utils.getDataFiles(data_path)

# Iterating over the vtk files
data = {"naca_numbers": []}
for data_file in data_files:
    try:
        # Reading the vtk file
        reader = Utils.readVtk(data_file["path"])

        # Loading the vtk file of the target shape and extracting its labels
        naca_numbers = Utils.targetValues(data_file["name"])
        data["naca_numbers"].append(naca_numbers)

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

        if sensor_signal:
            # Extracting the signals associated to the sensor
            features = sensorSignal(reader)

        # Adding the features extracted to the main list
        for key in features.keys():
            if key not in data:
                data[key] = []
            data[key].append(features[key])

    except Exception as e:
        # Displaying errors
        print(f'ERROR OCCURREND FOR FILE: {data_file["name"]} --> {str(e)}')

    else:
        # Displaying progress
        print(f'NACA {data_file["name"]} --> {(data_files.index(data_file) + 1)}/{len(data_files)} files processed')

# Saving the result into the destination file
if save_data:
    Utils.saveData(data)