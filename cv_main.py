import os
import copy
import numpy as np
import csv
import random
import torch
import argparse
import warnings

# Suppress PyTorch/NumPy warnings for cleaner output
warnings.filterwarnings('ignore')

from data_utils.cv_loader import allocate_dataset
from models.cv_network import CNNMnist, CNNCIFAR, LocalUpdate, test_inference
from utils.helpers import average_weights, select_users_by_value


def args_parser():
    parser = argparse.ArgumentParser(description="Federated Learning with Selection and Unlearning")
    parser.add_argument('--exp_name', type=str, default='test', help="Name of the experiment/log folder")
    
    # System parameters 
    parser.add_argument('--num_users', type=int, default=100, help="Total number of clients")
    parser.add_argument('--K', type=int, default=50, help="Number of sampled clients in each aggregation")
    parser.add_argument('--num_selections', type=int, default=200, help="Rounds of client selection, T")
    parser.add_argument('--L1', type=float, default=0.1, help='The first parameter in theta')
    parser.add_argument('--alpha', type=float, default=0.01, help='Loss and payment scaling parameter')
    parser.add_argument('--kc', type=float, default=10, help='Data collection time')
    parser.add_argument('--kf', type=float, default=20, help='FU execution time')
    parser.add_argument('--V', type=float, default=1, help='Fairness scaling parameter')
    parser.add_argument('--gamma', type=float, default=0.1, help='UCB parameter')
    parser.add_argument('--bw_ratio', type=float, default=0.1, help='Ratio of updated data (transmission capacity)')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar'], help="Dataset name")
    parser.add_argument('--num_channels', type=int, default=1, help="Number of channels (mnist:1, cifar:3)")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes (mnist:10, cifar:100)")
    
    # Data drift setting
    parser.add_argument('--perturb', type=str, default='mislabel', choices=['mislabel', 'noise'], help="Drift type")
    parser.add_argument('--perturb_rate', type=float, default=0.1, help="Flip rate or noise standard deviation")
    
    # Model architecture and training settings
    parser.add_argument('--epochs', type=int, default=5, help="Number of global aggregations after each selection, phi")
    parser.add_argument('--local_ep', type=int, default=1, help="Number of local epochs (E)")
    parser.add_argument('--local_bs', type=int, default=10, help="Local batch size (B)")
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help="Type of optimizer")
    parser.add_argument('--verbose', type=int, default=0, help='Print detailed training logs')
    parser.add_argument('--gpu', type=int, default=1, help="1 to use GPU, 0 to use CPU")

    args = parser.parse_known_args()[0]
    return args

if __name__ == '__main__':
    args = args_parser()
    
    log_dir = os.path.join('./log', args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    csv_file_path = os.path.join(log_dir, 'selection_acc_5acc_loss_payment.csv')
    # Write CSV header if the file is new
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Selection_T", "Test_Acc", "Test_Top5_Acc", "Test_Loss", "Total_Payment"])

    # Dynamic Device Selection
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"====== Using Device: {device} ======")
    logger = None 

    # 1. Allocate Dataset
    train_dataset, test_dataset, user_groups = allocate_dataset(args) 

    # 2. Initialize Global Model
    if args.dataset == 'mnist':
        global_model = CNNMnist(args=args).to(device)
    elif args.dataset == 'cifar':
        global_model = CNNCIFAR(args=args).to(device)
    else:
        raise ValueError("Unsupported dataset.")
        
    global_model.train()
    
    # 3. System States Initialization
    aoi = np.zeros(args.num_users)
    Q = np.zeros(args.num_users)
    sample_mean_delta = np.zeros(args.num_users)
    ucb_delta = np.zeros(args.num_users)
    obs_delta = np.zeros(args.num_users)
    n = np.zeros(args.num_users) # Selection counter
    d = np.array([len(user_groups[i]) for i in range(args.num_users)]) # User data sizes
    
    # 4. Compute Initial Gradient Norms (Gi)
    print("Computing initial gradient norms (Gi) for all users.")
    Gi = np.zeros(args.num_users)
    for idx in range(args.num_users):
        local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, cid=idx, my_aoi=0)
        Gi[idx] = local_model.compute_grad_mean(model=copy.deepcopy(global_model))
    print("Initial Gi computation completed.")
        
    test_accs, test_5accs, test_losses = [], [], []
    hist_part = [None] * args.epochs
    hist_model_states = [None] * args.epochs 
    
    # =======================================================================
    # Main Task Loop (Selections)
    # =======================================================================
    for t in range(args.num_selections):
        print(f"\n========== Starting Selection Task t={t} ==========")
        
        # --- A. Client Selection and Payment ---
        cost = np.random.uniform(low=0, high=1, size=args.num_users) 
        bandwidth = args.bw_ratio * np.sum(d)
        
        est_theta = np.zeros(args.num_users)
        for i in range(args.num_users):
            est_theta[i] = (args.L1 * d[i] * (ucb_delta[i]**2) * (aoi[i] + args.kc)) / (args.alpha * np.sum(d)) - Q[i] / (args.V * args.alpha)
            
        bid = cost
        g = (est_theta - bid) / d 
        
        selection, gN1 = select_users_by_value(bandwidth, g, d, args) 
        
        price = [(est_theta[i] - d[i]*gN1) * selection[i] for i in range(args.num_users)]
        
        # --- B. Update Age of Information (AoI) and Q Queue ---
        aoi = (aoi + args.kc) * (1 - selection)
        Q += selection - bandwidth / np.sum(d)
        Q = np.clip(Q, a_min=0, a_max=None)
        
        # --- C. Observe Data Criticality ---
        for idx in range(args.num_users):
            if selection[idx] == 1:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, cid=idx, my_aoi=aoi[idx])
                mean_norms = local_model.compute_grad_mean(model=copy.deepcopy(global_model))
                obs_delta[idx] = (mean_norms - Gi[idx]) / max(aoi[idx], 1e-8) 
            else:
                obs_delta[idx] = 0
                
        # Min-Max Normalization of obs_delta
        delta_min, delta_max = np.min(obs_delta), np.max(obs_delta)
        if delta_max > delta_min:
            obs_delta = (obs_delta - delta_min) / (delta_max - delta_min + 1e-8)
        else:
            obs_delta = np.zeros_like(obs_delta)
            
        # --- D. Selective Retraining (FU Algorithm) ---
        need_retrain = False
        
        for epoch in range(args.epochs):
            if hist_part[epoch] is None:
                need_retrain = True
            elif not need_retrain:
                has_intersection = np.intersect1d(selection, hist_part[epoch]).size > 0
                if not has_intersection:
                    print(f'  -> [Epoch {epoch}] No intersection. Skipping training and reusing history.')
                    global_model.load_state_dict(hist_model_states[epoch])
                    continue
                else:
                    print(f'  -> [Epoch {epoch}] Intersection detected! Triggering retraining for current and all subsequent epochs.')
                    need_retrain = True
                    
            # Local Training Execution
            local_weights, local_losses = [], []
            global_model.train()
            sampled_clients = np.random.choice(args.num_users, size=args.K, replace=False)
            
            for idx in sampled_clients:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger, cid=idx, my_aoi=aoi[idx])
                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(loss)
                
            # Aggregation & Model Update
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            
            # Save History
            hist_part[epoch] = sampled_clients
            hist_model_states[epoch] = copy.deepcopy(global_model.state_dict())
            
        print(f'Selection {t}: FU algorithm complete.')
            
        # --- E. Evaluate Global Model ---
        test_acc, test_5acc, test_loss = test_inference(args, global_model, test_dataset)
        print(f'Results  -> Test Acc: {test_acc:.4f} | Top-5 Acc: {test_5acc:.4f} | Loss: {test_loss:.4f}')
        
        test_accs.append(test_acc)
        test_5accs.append(test_5acc)
        test_losses.append(test_loss)
        
        
        with open(csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([t, test_acc, test_5acc, test_loss, sum(price)])
                
        # --- F. Update AoI and UCB for Next Round ---
        aoi += args.kf
        
        for i in range(args.num_users):
            if selection[i] == 1:
                sample_mean_delta[i] = (sample_mean_delta[i] * n[i] + obs_delta[i]) / (n[i] + 1)
                
        n += selection
        sumn = np.sum(n)
        
        for i in range(args.num_users):
            if n[i] > 0:
                temp_dev = np.sqrt(args.gamma * np.log(sumn) / n[i])
            else:
                temp_dev = 0.0 
            ucb_delta[i] = sample_mean_delta[i] + temp_dev