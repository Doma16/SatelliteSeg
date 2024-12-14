import json

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_json_variable(path, var):
    data = read_json(path)
    return data[var]
    
def count_parameters(model):
    return sum((x.numel() for x in model.parameters()))

def get_save_name(model, *configs):
    name = model.name
    for el in configs[0]:
        name += '_' + str(el)
    return name