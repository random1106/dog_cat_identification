import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.conv5 = nn.Conv2d(256, 512, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 2)
        # self.fc3 = nn.Linear(100, 10)
        # self.fc4 = nn.Linear(10, 2)

    def forward(self, X):
        X = self.pool(self.conv1(X))
        X = self.pool(self.conv2(X))
        X = self.pool(self.conv3(X))
        X = self.pool(self.conv4(X))
        X = self.conv5(X)
        X = X.reshape(X.shape[0], -1)
        X = torch.relu(self.fc1(X))
        X = self.fc2(X)
        # X = torch.relu(self.fc3(X))
        # X = self.fc4(X)
        return X

    def fit(self, X_train, X_valid, y_train, y_valid, epochs = 50, batch_size = 50, lr = 0.001):
        self.train()
        optimizer =  torch.optim.Adam(self.parameters(), lr = lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for start in range(0, len(y_train), batch_size):
                end = min(start + batch_size, len(y_train))
                loss = criterion(self.forward(X_train[start:end]), y_train[start:end])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(f"Epoch {epoch + 1}, training loss: {loss.item()}")
            print(f"training correct rate {self.correct_rate(X_train, y_train)/len(y_train)}")
            print(f"validation correct rate {self.correct_rate(X_valid, y_valid)/len(y_valid)}")

    def correct_rate(self, X, y):
        correct = 0
        for x, target in zip(X, y):
            correct += self.predict_each_row(x.unsqueeze(0)) == target
        return correct
                
    def predict_each_row(self, x):
        with torch.no_grad():
            return torch.argmax(self.forward(x))
        
    def validate(self, X, y):
        self.eval()
        correct = 0
        for i in range(len(y)):
            correct += self.predict(X[i]) == y[i]
        print(f"Correctedness Rate for Validation: {correct}/{len(y)} = {correct/len(y)}")