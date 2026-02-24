# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Predict future stock prices using an RNN model based on historical closing prices from trainset.csv and testset.csv, with data normalized using MinMaxScaler.
## Design Steps

### Step 1:
Write your own steps

### Step 2:

### Step 3:



## Program
#### Name:SELVARANI S
#### Register Number:212224040301 
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Take last time step output
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)

        return out

model =model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the Model

num_epochs = 20
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')






```

## Output

### True Stock Price, Predicted Stock Price vs time


<img width="811" height="683" alt="image" src="https://github.com/user-attachments/assets/94a09eeb-84f0-467b-8e1e-cb3da343f27e" />
 

Include the predictions on test data
<img width="1047" height="700" alt="image" src="https://github.com/user-attachments/assets/8a6d8cc8-8f70-4a4e-8785-b19bc2acf569" />


## Result
The RNN model was successfully implemented for stock price prediction.

