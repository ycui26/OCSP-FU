import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import warnings

warnings.filterwarnings('ignore')

# ====================================================================
# 1. Neural Network Architectures
# ====================================================================
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCIFAR(nn.Module):
    def __init__(self, args):
        super(CNNCIFAR, self).__init__()
        # Convolutional layers
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Fully Connected layers
        self.hidden_layer = nn.Linear(128 * 2 * 2, 206)
        self.output_layer = nn.Linear(206, args.num_classes) # Use args.num_classes for flexibility
        
    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

# ====================================================================
# 2. Perturbation & Data Wrapper
# ====================================================================
def fixed_flip(label):
    """Deterministic label flipping map."""
    flip_map = {0: 9, 1: 7, 2: 5, 3: 8, 4: 6, 5: 2, 6: 4, 7: 1, 8: 3, 9: 0}
    # Support tensor or integer inputs
    lbl_int = int(label.item() if isinstance(label, torch.Tensor) else label)
    return flip_map.get(lbl_int, lbl_int)

def random_flip(label, num_classes=100):
    """Randomly flips to another class."""
    return np.random.randint(0, num_classes)


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around PyTorch Dataset class."""
    def __init__(self, args, dataset, idxs, my_perturb, my_aoi):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.my_perturb = my_perturb
        self.my_aoi = my_aoi
        self.args = args

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        
        
        if isinstance(image, torch.Tensor):
            image = image.clone().detach()
        else:
            image = torch.tensor(image, dtype=torch.float32)

        if self.my_perturb is not None:
            if self.my_perturb == 'mislabel':
                flip_rate = self.args.perturb_rate * np.log(self.my_aoi)
                if np.random.rand() <= flip_rate:
                    if self.args.dataset == 'mnist':
                        label = fixed_flip(label)
                    elif self.args.dataset == 'cifar':
                        label = random_flip(label, num_classes=self.args.num_classes)
                        
            elif self.my_perturb == 'noise':
                std = self.args.perturb_rate / (1 + np.exp(-self.my_aoi / 100))
                # Add Gaussian noise
                image = image + torch.randn(image.size()) * std
            else:
                print(f"Warning: Unrecognized perturbation type '{self.my_perturb}'")

        # Convert label to Long Tensor (required by CrossEntropy/NLLLoss)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# ====================================================================
# 3. Local Client Training Operations
# ====================================================================
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, cid, my_aoi):
        self.args = args
        self.logger = logger
        self.cid = cid
        self.device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
        self.aoi = my_aoi
        
        if args.dataset == 'mnist':
            self.criterion = nn.NLLLoss().to(self.device)
        elif args.dataset == 'cifar':
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            
        self.trainloader, self.validloader, self.testloader = self.train_val_test(args, dataset, list(idxs))

    def train_val_test(self, args, dataset, idxs):
        """Splits data into train, val, test (80%, 10%, 10%)."""
        num_items = len(idxs)
        idxs_train = idxs[:int(0.8 * num_items)]
        idxs_val = idxs[int(0.8 * num_items):int(0.9 * num_items)]
        idxs_test = idxs[int(0.9 * num_items):]

        trainloader = DataLoader(DatasetSplit(args, dataset, idxs_train, self.args.perturb, self.aoi),
                                 batch_size=self.args.local_bs, shuffle=True)
        # No perturbation on validation and test sets
        validloader = DataLoader(DatasetSplit(args, dataset, idxs_val, None, None),
                                 batch_size=max(1, int(len(idxs_val)/5)), shuffle=False)
        testloader = DataLoader(DatasetSplit(args, dataset, idxs_test, None, None),
                                batch_size=max(1, int(len(idxs_test)/5)), shuffle=False)
                                
        return trainloader, validloader, testloader

    def compute_grad_mean(self, model):
        """Computes the average gradient norm over the test set."""
        norms = []
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            images.requires_grad = True

            model.zero_grad()
            target_pred = model(images)
            y = self.criterion(target_pred, labels)
            
            # Compute gradients of the loss w.r.t. model parameters
            dy = torch.autograd.grad(y, model.parameters(), retain_graph=False)
            target_grad = torch.cat([g.detach().clone().reshape(1, -1) for g in dy], 1).reshape(1, -1)
            
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
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(f'| Global Round : {global_round} | Local Epoch : {iter} | '
                          f'[{batch_idx * len(images)}/{len(self.trainloader.dataset)} '
                          f'({100. * batch_idx / len(self.trainloader):.0f}%)]\tLoss: {loss.item():.6f}')
                    
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

# ====================================================================
# 4. Global Testing / Inference
# ====================================================================
def test_inference(args, model, test_dataset):
    """Returns the test accuracy, top-5 accuracy, and loss."""
    model.eval()
    loss, total, correct, correct_5 = 0.0, 0.0, 0.0, 0.0
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    if args.dataset == 'mnist':
        criterion = nn.NLLLoss().to(device)
    elif args.dataset == 'cifar':
        criterion = nn.CrossEntropyLoss().to(device)
        
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Top-1 Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            
            # Top-5 Prediction
            max_k = min(5, outputs.size(1))
            _, pred = outputs.topk(max_k, 1, largest=True, sorted=True)
            labels_expand = labels.view(labels.size(0), -1).expand_as(pred)
            correct_k = pred.eq(labels_expand).float()
            correct_5 += correct_k.sum().item()

    accuracy = correct / total
    top5acc = correct_5 / total
    avg_loss = loss / len(testloader) 
    
    return accuracy, top5acc, avg_loss