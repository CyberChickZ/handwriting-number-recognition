import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)  # First fully connected layer: input size 784 (28*28), output size 64
        self.fc2 = torch.nn.Linear(64, 64)     # Second hidden layer: 64 -> 64
        self.fc3 = torch.nn.Linear(64, 64)     # Third hidden layer: 64 -> 64
        self.fc4 = torch.nn.Linear(64, 10)     # Output layer: 64 -> 10 (for 10 digit classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))         # Apply ReLU activation after first layer
        x = torch.nn.functional.relu(self.fc2(x))         # Apply ReLU after second layer
        x = torch.nn.functional.relu(self.fc3(x))         # Apply ReLU after third layer
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)  # Apply log softmax to output layer
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])  # Convert image to tensor
    data_set = MNIST("", is_train, transform=to_tensor, download=True)  # Download MNIST dataset
    return DataLoader(data_set, batch_size=15, shuffle=True)  # Load data in batches of 15, randomly shuffled


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for (x, y) in test_data:  # Loop over test batches
            outputs = net.forward(x.view(-1, 28*28))  # Flatten images and compute predictions
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:  # Compare predicted label with true label
                    n_correct += 1
                n_total += 1
    return n_correct / n_total  # Return accuracy


def main():
    train_data = get_data_loader(is_train=True)   # Load training data
    test_data = get_data_loader(is_train=False)   # Load test data
    net = Net()                                    # Initialize the neural network
    
    print("initial accuracy:", evaluate(test_data, net))  # Check initial (untrained) accuracy
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Use Adam optimizer for training
    
    for epoch in range(2):  # Train for 2 epochs
        for (x, y) in train_data:
            net.zero_grad()  # Reset gradients
            output = net.forward(x.view(-1, 28*28))  # Flatten input and do forward pass
            loss = torch.nn.functional.nll_loss(output, y)  # Compute loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the network weights
        
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))  # Evaluate after each epoch


    for (n, (x, _)) in enumerate(test_data):  # Take a few samples from test data
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))  # Predict label
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))  # Show image
        plt.title("prediction: " + str(int(predict)))  # Show predicted label
        plt.savefig("test_" + str(n) + ".png") # Save prediction plots
    plt.show()  # Display plots
    
    torch.save(net.state_dict(), "model.pt")  # Save the trained model

if __name__ == "__main__":
    main()