# import dependencies
import torch
from torch import optim
import matplotlib.pyplot as plt

# import files from within package
import circle_detection as cd
import networks
import dataset_generator as dg
# CONSTANTS
SPLIT = dg.SPLIT
DATASET_SIZE = dg.TOTAL_SIZE

TRAIN_SIZE = int(SPLIT["train"] * DATASET_SIZE)
VAL_SIZE = int(SPLIT["validation"] * DATASET_SIZE)
TEST_SIZE = int(SPLIT["test"] * DATASET_SIZE)
STEPS = 100

print(f"Train size: {TRAIN_SIZE}; Validation size: {VAL_SIZE}; Test size: {TEST_SIZE}, Steps: {STEPS}")
# Datasets
dataset = dg.dataset
train_set = dataset[:TRAIN_SIZE]
val_set = dataset[TRAIN_SIZE:VAL_SIZE+TRAIN_SIZE]
test_set = dataset[VAL_SIZE+TRAIN_SIZE:]
print(len(val_set))
# losses
train_losses = []
val_losses = []

print("GPU in use" if torch.cuda.is_available() else "CPU in use")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model and SGD Optimizer
net = networks.NotSimpleNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

for i in range(STEPS):

    # tracking loss per epoch
    loss_epoch = torch.tensor(0.).to(device)

    for sample in train_set:

        # obtain x, y, and y'
        x, y = sample
        x, y = x.to(device), y.to(device)
        y_hat = net(x)

        loss = 1-cd.iou(y, y_hat)
        loss = loss.to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track losses
        loss_epoch += loss.detach()

    val_loss_epoch = torch.tensor(0.)
    val_loss_epoch = val_loss_epoch.to(device)

    for x, y in val_set:
        # ensure that all gradients are detached
        x, y = x.detach(), y.detach()
        x, y = x.to(device), y.to(device)
        y_hat = net(x).detach()
        # Jaccard index is an accuracy measure; use negative for loss function
        loss = 1-cd.iou(y, y_hat).detach()
        loss = loss.to(device)
        val_loss_epoch += loss

    train_losses.append(float(loss_epoch / TRAIN_SIZE))
    val_losses.append(float(val_loss_epoch / VAL_SIZE))

    print(f"Epoch: {i}, Mean Training Loss:{loss_epoch / TRAIN_SIZE}; Val. Loss:{val_loss_epoch / VAL_SIZE}")
    torch.save(net.state_dict(), f"models/circleDetectEpoch{i}.pth")


# Plot model performance
plt.title("Model Performance")
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.show()


# Evaluate against test data
print("TRAINING COMPLETE")
print("Test Accuracy")
net.eval()
test_loss = torch.tensor(0.).to(device)

for x, y in test_set:
    x, y = x.to(device), y.to(device)
    y_hat = net(x).detach()
    test_loss += cd.iou(y, y_hat)
    print(f"y:{y}; y': {y_hat}")

print(f"Loss on Test set: {float(test_loss/TEST_SIZE)}")
