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

    # Epochs count
    matches = re.findall("Epoch: [0-9]+", text)
    ## Extract epochs count using last match from file
    if len(matches) != 0:
        numbers = re.findall("[0-9]+", matches[-1])
        result["epochs_count"] = int(numbers[0])

    return result


if len(sys.argv) != 2:
    raise Exception("Wrong format:\n\tpython exploration/results.py path/to/log")
path_to_log = sys.argv[1]
result = parse_log_file(path_to_log)
print(result)