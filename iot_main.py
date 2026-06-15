import os
import copy
import numpy as np
import csv
import random
import torch
import argparse
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from data_utils.iot_loader import allocate_dataset
from models.iot_network import MultiLayerLSTM, LocalUpdate, test_inference
from utils.helpers import average_weights, select_users_by_value


def args_parser():
    parser = argparse.ArgumentParser(description="Federated Unlearning on CICIoT Dataset")
    parser.add_argument('--exp_name', type=str, default='iomt_fu', help="name of folder")
    
    # System parameters
    parser.add_argument('--num_users', type=int, default=100, help="number of clients")
    parser.add_argument('--num_selections', type=int, default=18, help="rounds of client selection, T")
    parser.add_argument('--K', type=int, default=50, help="number of sampled clients in each aggregation")
    
    parser.add_argument('--L1', type=float, default=0.1, help='the first parameter in theta')
    parser.add_argument('--alpha', type=float, default=0.01, help='loss and payment scaling parameter')
    parser.add_argument('--kc', type=float, default=1, help='data collection time')
    parser.add_argument('--kf', type=float, default=1, help='fu execution time')
    parser.add_argument('--V', type=float, default=1, help='fairness scaling parameter')
    parser.add_argument('--gamma', type=float, default=0.1, help='ucb parameter')
    parser.add_argument('--bw_ratio', type=float, default=0.1, help='the ratio of updated data')
    
    # Model architecture and training setting
    parser.add_argument('--epochs', type=int, default=5, help="number of aggregations after each selection, phi")
    parser.add_argument('--num_classes', type=int, default=6, help="number of classes")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help="type of optimizer")
    
    parser.add_argument('--gpu', type=int, default=1, help="1 to use GPU, 0 to use CPU")
    parser.add_argument('--verbose', type=int, default=0, help='verbose logging')
    
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = args_parser()
    
    log_dir = os.path.join('./log', args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    csv_file_path = os.path.join(log_dir, 'selection_acc_loss_payment.csv')
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Selection_T", "Test_Acc", "Test_Loss", "Total_Payment"])

    # Dynamic Device Selection
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"====== Using Device: {device} ======")
    logger = None

    # 2. Allocate Dataset and Encode Labels
    train_dataset, test_dataset, user_groups = allocate_dataset(args)
    
    encoder = LabelEncoder()
    train_dataset['six_encoded'] = encoder.fit_transform(train_dataset['sixclass'])
    test_dataset['six_encoded'] = encoder.transform(test_dataset['sixclass'])

    # 3. System States Initialization
    aoi = np.zeros(args.num_users)
    Q = np.zeros(args.num_users)
    sample_mean_delta = np.zeros(args.num_users)
    ucb_delta = np.zeros(args.num_users)
    obs_delta = np.zeros(args.num_users)
    n = np.zeros(args.num_users) 
    d = np.ones(args.num_users) * 500  # Hardcoded user data size as per original logic
    
    class_names = [
        'benign', 'Malformed Data', 'DoS Connect Flood', 'DDoS Publish Flood', 
        'DDoS Connect Flood', 'DoS Publish Flood', 'DoS SYN', 'DoS ICMP', 
        'DoS UDP', 'DoS TCP', 'DDoS UDP', 'DDoS ICMP', 'DDoS SYN', 'DDoS TCP',
        'Ping Sweep', 'Recon VulScan', 'OS Scan', 'Port Scan', 'ARP Spoofing'
    ]
    include_time = list(range(len(class_names)))
    
    # 4. Initialize Global Model
    global_model = MultiLayerLSTM(args=args).to(device)
    global_model.train()
    
    # 5. Compute Initial Gradient Norms (Gi)
    print("Computing initial gradient norms (Gi) for all IoT users...")
    Gi = np.zeros(args.num_users)
    for idx in range(args.num_users):
        local_model = LocalUpdate(
            args=args, train_dataset=train_dataset, idxs=user_groups[idx], 
            logger=logger, cid=idx, my_aoi=0, include_time=include_time, class_names=class_names
        )
        Gi[idx] = local_model.compute_grad_mean(model=copy.deepcopy(global_model))
    print("Initial Gi computation completed.")
        
    test_accs, test_losses = [], []
    
    # CRITICAL: Store state_dicts, not model references
    hist_part = [None] * args.epochs
    hist_model_states = [None] * args.epochs
    
    # =======================================================================
    # Main Task Loop (Selections)
    # =======================================================================
    for t in range(args.num_selections):
        print(f"\n========== Starting Selection Task t={t} ==========")
        
        # --- A. Client Selection and Payment ---
        cost = np.random.uniform(low=0, high=1, size=(args.num_users,))
        bandwidth = args.bw_ratio * np.sum(d)
        
        est_theta = np.zeros(args.num_users)
        for i in range(args.num_users):
            est_theta[i] = (args.L1 * d[i] * (ucb_delta[i]**2) * (aoi[i] + args.kc)) / (args.alpha * np.sum(d)) - Q[i] / (args.V * args.alpha)
            
        bid = cost
        g = (est_theta - bid) / d
        selection, gN1 = select_users_by_value(args, bandwidth, g, d)
        
        price = [(est_theta[i] - d[i]*gN1) * selection[i] for i in range(args.num_users)]
        
        # --- B. Update Age of Information (AoI) and Q Queue ---
        aoi = (aoi + args.kc) * (1 - selection)
        Q += selection - bandwidth / np.sum(d)
        Q = np.clip(Q, a_min=0, a_max=None)
        
        # --- C. Observe Data Criticality ---
        for idx in range(args.num_users):
            if selection[idx] == 1:
                local_model = LocalUpdate(
                    args=args, train_dataset=train_dataset, idxs=user_groups[idx], 
                    logger=logger, cid=idx, my_aoi=aoi[idx], include_time=include_time, class_names=class_names
                )
                mean_norms = local_model.compute_grad_mean(model=copy.deepcopy(global_model))
                
                obs_delta[idx] = (mean_norms - Gi[idx]) / max(aoi[idx], 1e-8)
            else:
                obs_delta[idx] = 0
                
        # Normalization
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
                    print(f'  -> [Epoch {epoch}] Skip training: Reusing historical model state.')
                    global_model.load_state_dict(hist_model_states[epoch])
                    continue
                else:
                    print(f'  -> [Epoch {epoch}] Intersection detected: Retraining triggered.')
                    need_retrain = True
                    
            local_weights, local_losses = [], []
            global_model.train()
            
            sampled_clients = np.random.choice(args.num_users, size=args.K, replace=False)
            for idx in sampled_clients:
                local_model = LocalUpdate(
                    args=args, train_dataset=train_dataset, idxs=user_groups[idx], 
                    logger=logger, cid=idx, my_aoi=aoi[idx], include_time=include_time, class_names=class_names
                )
                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(loss)
    
            # Aggregation
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            
            # Save Deepcopy State
            hist_part[epoch] = sampled_clients
            hist_model_states[epoch] = copy.deepcopy(global_model.state_dict())
            
        print(f'Selection {t}: FU algorithm finished')
            
        # --- E. Evaluate Global Model ---
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        print(f'Selection: {t} | Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}')
        
        test_accs.append(test_acc)
        test_losses.append(test_loss)
    
        
        with open(csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([t, test_acc, test_loss, sum(price)])
                
        # --- F. Update AoI and UCB Variables ---
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