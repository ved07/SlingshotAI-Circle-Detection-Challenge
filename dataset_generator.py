"""
Needed to generate and store data for training as lazy generators result in new training set each time,
though this performed well on preliminary training, it is not the "right" approach, and may result in unintended
functionality.
"""
# importing dependencies
import torch
import numpy as np

# importing files from package
import circle_detection as cd
# Trained with 100,000 examples
TOTAL_SIZE = 1000
SPLIT = {"train": 0.8, "validation": 0.19, "test": 0.01}

print(f"Dataset of size: {TOTAL_SIZE} and split: {SPLIT}")

# examples
examples = cd.generate_examples()


# reformatting helper function to make data PyTorch friendly
def reformat_example(sample_x, sample_y):
    sample_x = sample_x.astype(dtype=np.float32)
    sample_x.resize((1, 100, 100))
    sample_x = torch.from_numpy(sample_x)
    sample_x.requires_grad = True
    sample_y = torch.tensor([sample_y.row, sample_y.col, sample_y.radius], dtype=torch.float32, requires_grad=True)

    return sample_x, sample_y


dataset = []
# use a counter to check when generating validation / test data
counter = 0

for example in examples:
    if counter > TOTAL_SIZE: break
    x, y = reformat_example(example[0], example[1])

    dataset.append((x, y))
    counter += 1



