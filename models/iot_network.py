import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# 1. Neural Network Architecture
# ====================================================================
class MultiLayerLSTM(nn.Module):
    """
    Multi-layer LSTM network
    """
    def __init__(self, args):
        super(MultiLayerLSTM, self).__init__()
        self.input_size = 45
        self.hidden_size = 128
        self.num_layers = 5
        
        # batch_first=True -> input tensor shape is (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        
        # Final fully connected layer mapping hidden units to categorical outputs
        self.fc = nn.Linear(self.hidden_size, args.num_classes)

    def forward(self, x):
        # Initialize hidden and cell states dynamically based on batch size and device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the LAST time step
        out = self.fc(out[:, -1, :])
        return out


# ====================================================================
# 2. Local Client Training Operations
# ====================================================================
class LocalUpdate(object):
    def __init__(self, args, train_dataset, idxs, logger, cid, my_aoi, include_time, class_names):
        self.args = args
        self.logger = logger
        self.cid = cid
        self.device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.aoi = my_aoi
        
        self.trainloader = self.get_trainloader(train_dataset, list(idxs), include_time, class_names)

    def get_trainloader(self, train_dataset, idxs, include_time, class_names):
        """Filters client data based on AoI and prepares the DataLoader."""
        client_dataset = train_dataset.loc[idxs]
        
        # Filter available classes based on Age of Information (AoI)
        target_label = [
            name for i, name in enumerate(class_names) 
            if include_time[i] <= 19 - int(self.aoi)
        ]
        
        filtered_df = client_dataset[client_dataset['label'].isin(target_label)]

        # Stratified subsampling if the client has too much data
        if len(filtered_df) >= 500:
            filtered_df, _ = train_test_split(
                filtered_df, 
                train_size=500, 
                stratify=filtered_df['label'], 
                random_state=42
            )

        # Separate features and labels
        feature_cols = filtered_df.columns.drop(['sixclass', 'label', 'six_encoded'])
        label_col = ['six_encoded']
        
        X_train_tensor = torch.tensor(filtered_df[feature_cols].values, dtype=torch.float32)
        y_train_tensor = torch.tensor(filtered_df[label_col].values, dtype=torch.long)
        
        # Reshape X from 2D (samples, features) to 3D (samples, seq_len=1, features) for LSTM
        if X_train_tensor.dim() == 2:
            X_train_tensor = X_train_tensor.unsqueeze(1)

        # Create DataLoader
        train_tensordataset = TensorDataset(X_train_tensor, y_train_tensor)
        trainloader = DataLoader(train_tensordataset, batch_size=self.args.local_bs, shuffle=True)

        return trainloader
    
    def compute_grad_mean(self, model):
        """Computes the average gradient norm over the local dataset."""
        model.train()
        norms = []
        
        for batch_idx, (features, labels) in enumerate(self.trainloader):
            features, labels = features.to(self.device), labels.to(self.device)
            features.requires_grad = True
    
            model.zero_grad()
            target_pred = model(features)
            labels = labels.squeeze()
            
            y = self.criterion(target_pred, labels)
            dy = torch.autograd.grad(y, model.parameters(), retain_graph=False)
            
            # Flatten gradients into a single vector
            target_grad = torch.cat([g.detach().clone().reshape(1, -1) for g in dy], 1).reshape(1, -1)
            target_grad = target_grad.to(self.device)
            
            grad_norm = torch.norm(target_grad).detach().cpu().numpy()
            norms.append(grad_norm)
            
        return np.mean(norms)
            
    def update_weights(self, model, global_round):
        """Performs local training on the client."""
        model.train()
        epoch_loss = []

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (features, labels) in enumerate(self.trainloader):
                features, labels = features.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                log_probs = model(features)
                labels = labels.squeeze()
                
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(f'| Global Round : {global_round} | Local Epoch : {iter} | '
                          f'[{batch_idx * len(features)}/{len(self.trainloader.dataset)} '
                          f'({100. * batch_idx / len(self.trainloader):.0f}%)]\tLoss: {loss.item():.6f}')
                          
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


# ====================================================================
# 3. Global Testing / Inference
# ====================================================================
def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and average loss. """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Clean preparation of testing data
    feature_cols = test_dataset.columns.drop(['sixclass', 'label', 'six_encoded'])
    label_col = ['six_encoded']

    X_test_tensor = torch.tensor(test_dataset[feature_cols].values, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_dataset[label_col].values, dtype=torch.long)

    # Reshape X from 2D (samples, 45) to 3D (samples, 1, 45) for the LSTM
    if X_test_tensor.dim() == 2:
        X_test_tensor = X_test_tensor.unsqueeze(1)

    test_tensordataset = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = DataLoader(test_tensordataset, batch_size=args.local_bs, shuffle=False) 

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(testloader):
            features, labels = features.to(device), labels.to(device)

            # Inference
            outputs = model(features)
            labels = labels.squeeze()
            
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct / total
    avg_loss = loss / len(testloader)
    
    return accuracy, avg_loss