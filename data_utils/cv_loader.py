import numpy as np
import torch
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings('ignore')

def allocate_dataset(args):
    """
    Load datasets and allocate them to clients in a Non-IID (shard-based) manner.
    
    Args:
        args: Argument parser object containing dataset name and num_users.
        
    Returns:
        train_dataset: Torchvision dataset object for training.
        test_dataset: Torchvision dataset object for testing.
        dict_users: Dictionary mapping client_idx to their allocated data indices.
    """
    # ==========================================
    # 1. Dataset Loading & Transformations
    # ==========================================
    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        data_dir = './data/mnist/'
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
        
        # 60,000 training images = 1200 shards * 50 images/shard
        num_shards, num_imgs = 1200, 50
        
    elif args.dataset == 'cifar':
        # CIFAR-100 Normalization values
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4911, 0.4820, 0.4467), (0.2022, 0.1993, 0.2009))
        ])

        # Fixed the path consistency (used './data/cifar-100/' for both)
        data_dir = './data/cifar-100/'
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=apply_transform)
        
        # 50,000 training images = 1000 shards * 50 images/shard
        num_shards, num_imgs = 1000, 50
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")

    # ==========================================
    # 2. Prepare for Data Partitioning (Non-IID)
    # ==========================================
    idx_shard = list(range(num_shards))
    dict_users_lists = {i: [] for i in range(args.num_users)}
    
    idxs = np.arange(num_shards * num_imgs)
    
    
    # Check if targets is a tensor, if not, convert it (CIFAR uses lists, MNIST uses tensors)
    targets = train_dataset.targets
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    labels = targets.numpy()

    # Sort labels to group identical classes together (creates the Non-IID nature)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client
    min_shard = 1
    max_shard = 30

    # ==========================================
    # 3. Allocate Shards to Clients
    # ==========================================
    # Divide the shards into random chunks for every client
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=args.num_users)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards).astype(int)

    # Strategy: If we generated more shards than available, ensure everyone gets at least 1 first.
    if sum(random_shard_size) > num_shards:
        # Pass 1: Ensure at least one shard per client
        for i in range(args.num_users):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            
            for rand in rand_set:
                dict_users_lists[i].extend(idxs[rand*num_imgs : (rand+1)*num_imgs])
                
        random_shard_size = random_shard_size - 1

        # Pass 2: Distribute remaining shards
        for i in range(args.num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = min(random_shard_size[i], len(idx_shard))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            
            for rand in rand_set:
                dict_users_lists[i].extend(idxs[rand*num_imgs : (rand+1)*num_imgs])
                
    else:
        # Standard allocation if total assigned shards <= available shards
        for i in range(args.num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            
            for rand in rand_set:
                dict_users_lists[i].extend(idxs[rand*num_imgs : (rand+1)*num_imgs])

        # Handling leftovers: Give them to the client with the least data
        if len(idx_shard) > 0:
            shard_size = len(idx_shard)
            k = min(dict_users_lists, key=lambda x: len(dict_users_lists[x]))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            
            for rand in rand_set:
                dict_users_lists[k].extend(idxs[rand*num_imgs : (rand+1)*num_imgs])

    # Convert the Python lists back to integer numpy arrays for output
    dict_users = {i: np.array(dict_users_lists[i], dtype=int) for i in range(args.num_users)}

    return train_dataset, test_dataset, dict_users