import torch

lst = []
data = [0, 1, 2, 3]

for i in range(100):
    lst.append(data)

# Generate batches
batch_size = 25
batches = []
for bstart in range(0, 100, batch_size):
    batches.append(lst[bstart:bstart+batch_size])

t = torch.as_tensor(batches)
