import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import sys

def parse_log_file(path):
    f = open(path, "r")
    text = f.read()
    f.close()
    result = {}

    # MSE
    matches = re.findall("mse:[0-9]+.[0-9]*", text)
    if len(matches) != 0:
        numbers = re.findall("[0-9]+.[0-9]*", matches[0])
        result["mse"] = float(numbers[0])
    
    # MAE
    matches = re.findall("mae:[0-9]+.[0-9]*", text)
    if len(matches) != 0:
        numbers = re.findall("[0-9]+.[0-9]*", matches[0])
        result["mae"] = float(numbers[0])

    # Train dataset length
    matches = re.findall("train [0-9]+", text)
    if len(matches) != 0:
        numbers = re.findall("[0-9]+", matches[0])
        result["train_len"] = int(numbers[0])

    # Validation dataset length
    matches = re.findall("val [0-9]+", text)
    if len(matches) != 0:
        numbers = re.findall("[0-9]+", matches[0])
        result["val_len"] = int(numbers[0])

    # Test dataset length
    matches = re.findall("test [0-9]+", text)
    if len(matches) != 0:
        numbers = re.findall("[0-9]+", matches[0])
        result["test_len"] = int(numbers[0])

    # Cost time
    matches = re.findall("cost time: [0-9]+.[0-9]*", text)
    if len(matches) != 0:
        total_time = 0
        for match in matches:
            total_time += float(re.findall("[0-9]+.[0-9]*", match)[0])
        result["cost_time"] = total_time

    # Epochs count
    matches = re.findall("Epoch: [0-9]+", text)
    ## Extract epochs count using last match from file
    if len(matches) != 0:
        numbers = re.findall("[0-9]+", matches[-1])
        result["epochs_count"] = int(numbers[0])

    # Compression parameters
    ## divide_dataset_size
    matches = re.findall("divide_dataset_size=[0-9]+", text)
    if len(matches) != 0:
        value = re.findall("[0-9]+", matches[0])
        result["divide_dataset_size"] = int(value[0])

    return result

def parse_directory(dir_path, print=True):
    results = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        results.append( parse_log_file(file_path) )
    df = pd.DataFrame(results)
    df = df.sort_values('divide_dataset_size')
    df = df.drop('divide_dataset_size', axis=1)

    if print == True:
        print(df.to_string(index=False))

    return df

def compare_benchmarks(dir1, dir2):
    results1 = parse_directory(dir1, False)
    results2 = parse_directory(dir2, False)

    # Remove trailing "/" from dir names if needed
    if dir1[-1] == "/":
        dir1 = dir1[:-1]
    if dir2[-1] == "/":
        dir2 = dir2[:-1]

    # Draw chart
    fig = plt.figure()

    x = results1["train_len"] + results1["val_len"]
    plt.plot(x, results1["mse"], label=os.path.basename(dir1))
    plt.plot(x, results2["mse"], label=os.path.basename(dir2))

    ## Display points counts in decreasing order
    plt.gca().invert_xaxis()

    plt.title('MSE comparison between two benchmarks')
    plt.ylabel('MSE')
    plt.xlabel('Points count (train+val dataset length)')
    plt.legend()
    plt.show()

# Main
l = len(sys.argv)
if l != 2 and l != 3:
    raise Exception("Wrong format:\n\tpython exploration/results.py path/to/log\n\tpython exploration/results path/to/log/dir1 path/to/log/dir2")
path = sys.argv[1]

if os.path.isfile(path):
    result = parse_log_file(path)
    print(result)
elif l == 2 and os.path.isdir(path):
    parse_directory(path)
elif l == 3:
    compare_benchmarks(sys.argv[1], sys.argv[2])
else:
    raise Exception("Input path is not a file neither a directory (?).")
