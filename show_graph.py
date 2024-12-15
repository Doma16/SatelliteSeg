import os
import json
import matplotlib.pyplot as plt

file = 'model/trained_models/'
model = 'SMPUNET_32_0.0001_300_True_torch.float32_score.json'
path = os.path.join(file, model)

with open(path, 'r') as f:
    data = json.load(f)

loss = data['loss']
f1 = data['f1']
plt.plot(range(len(loss)), loss, label='loss')
plt.plot(range(len(f1)), f1, label='f1')
plt.legend()
plt.show()