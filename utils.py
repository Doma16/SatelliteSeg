import json

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_json_variable(path, var):
    data = read_json(path)
    return data[var]
    