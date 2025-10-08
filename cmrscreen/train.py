import os
import argparse
import random
import logging
import math
import csv

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
import numpy as np
from chemprop import data
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from misc import AverageMeter, setup_device
from ema import ModelEMA
from dataset import get_sl_data, get_ssl_data, collate_batch
from model import MPNN

RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Deep Learning Model Training')

    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay (L2 regularization) coefficient (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--save-path', type=str, default='./checkpoints/',
                        help='path to save checkpoint models (default: ./checkpoints/)')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use.')
    parser.add_argument('--data-path', type=str, default='./dataset/C.csv',
                        help='path to data (default: ./dataset/C.csv)')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers for data loading (default: 0)')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience (default: 30)')

    parser.add_argument('--use-ema', action='store_true', default=False,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--label-smoothing', type=float, default=0.2,
                        help='label smoothing coefficient (default: 0.2)')

    parser.add_argument('--total-steps', default=600 * 300, type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=10, type=int, help='number of eval steps to run')
    parser.add_argument('--lambda-u', default=0.6, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--mu', default=8, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')

    args = parser.parse_args()
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    return args


def cross_entropy_with_label_smoothing(pred, target, smoothing=0.1):
    n_classes = pred.size(1)
    target = target.long()
    one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
    smooth_one_hot = one_hot * (1 - smoothing) + (smoothing / n_classes)
    loss = (-smooth_one_hot * torch.log_softmax(pred, dim=1)).sum(dim=1).mean()
    return loss


def evaluate(model, data_loader, device, smoothing=0.1):
    model.eval()
    preds, pred_probs, targets = [], [], []
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            bmg, V_d, X_d, target, _, _, _ = batch

            target = target.to(device)
            bmg.to(device)

            pred = model(bmg, V_d, X_d)[:, 0]
            
            if smoothing > 0:
                loss = cross_entropy_with_label_smoothing(pred, target, smoothing)
            else:
                loss = F.cross_entropy(pred, target.long(), reduction='mean')
                
            total_loss += loss.item()

            pred = torch.nn.functional.softmax(pred, dim=1)
            pred_probs.append(pred.cpu().numpy())
            pred = torch.argmax(pred, dim=1).cpu().numpy()

            target = target.cpu().numpy()
            preds.append(pred)
            targets.append(target)

    preds = np.concatenate(preds, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()
    pred_probs = np.concatenate(pred_probs, axis=0)

    accuracy = np.sum(preds == targets) / len(targets)

    TP = np.sum((preds == 1) & (targets == 1))
    FN = np.sum((preds == 0) & (targets == 1))
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0

    roc_auc = roc_auc_score(targets, pred_probs[:, 1].flatten())
    avg_loss = total_loss / len(data_loader)
    return accuracy, FNR, roc_auc, avg_loss


if __name__ == "__main__":
    args = parse_args()
    device = setup_device(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_workers = 0

    for fold in range(1, 11):
        fold_save_path = os.path.join(args.save_path, f'fold-{fold}')
        if not os.path.exists(fold_save_path):
            os.makedirs(fold_save_path)

        logging.info(f'Starting training for fold-{fold}')

        train_dset = get_sl_data(
            args.data_path,
            data_tag='smiles',
            label_tag='class',
            split='train',
            split_label=f'fold-{fold}'
        )
        test_dset = get_sl_data(
            args.data_path,
            data_tag='smiles',
            label_tag='class',
            split='val',
            split_label=f'fold-{fold}'
        )

        unlabeled_dataset = get_ssl_data(
            csv_file='./dataset/unlabeled_smiles.csv',
            data_tag='smiles',
        )

        labeled_trainloader = data.build_dataloader(train_dset, num_workers=num_workers, batch_size=args.batch_size)
        test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False, batch_size=args.batch_size)
        unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_size=args.batch_size * args.mu,
                                           collate_fn=collate_batch,
                                           num_workers=args.num_workers, drop_last=True)

        model = MPNN(n_classes=2)
        model = model.to(device)
        if args.use_ema:
            ema_model = ModelEMA(model, args.ema_decay, device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        csv_file = os.path.join(fold_save_path, 'metrics.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Iters', 'Train_Loss', 'Train_Accuracy', 'Train_FNR', 'Train_ROC_AUC',
                             'Test_Loss', 'Test_Accuracy', 'Test_FNR', 'Test_ROC_AUC'])

        labeled_iter = iter(labeled_trainloader)
        unlabeled_iter = iter(unlabeled_trainloader)

        total_loss = 0
        best_accuracy = 0
        best_fnr = math.inf
        best_roc_auc = 0
        best_loss = float('inf')
        patience_counter = 0
        iters = 0

        for epoch in range(args.epochs):
            losses_x = AverageMeter()
            losses_u = AverageMeter()
            mask_probs = AverageMeter()
            model.train()

            with tqdm(total=args.eval_step, unit='batch') as pbar:
                for batch_idx in range(args.eval_step):
                    try:
                        batch = next(labeled_iter)
                    except:
                        labeled_iter = iter(labeled_trainloader)
                        batch = next(labeled_iter)

                    try:
                        batch_u_w, batch_u_s = next(unlabeled_iter)
                    except:
                        unlabeled_iter = iter(unlabeled_trainloader)
                        batch_u_w, batch_u_s = next(unlabeled_iter)

                    bmg, V_d, X_d, target, _, _, _ = batch
                    target = target.to(device)
                    bmg.to(device)

                    pred_x = model(bmg, V_d, X_d)[:, 0]

                    Lx = cross_entropy_with_label_smoothing(pred_x, target, args.label_smoothing)

                    bmg_w, V_d_w, X_d_w, _, _, _, _ = batch_u_w
                    bmg_s, V_d_s, X_d_s, _, _, _, _ = batch_u_s
                    bmg_w.to(device)
                    bmg_s.to(device)

                    pred_u_w = model(bmg_w, V_d_w, X_d_w)[:, 0]
                    pred_u_s = model(bmg_s, V_d_s, X_d_s)[:, 0]

                    pseudo_label = torch.softmax(pred_u_w.detach() / args.T, dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(args.threshold).float()

                    if args.label_smoothing > 0:
                        n_classes = pred_u_s.size(1)
                        one_hot = torch.zeros_like(pred_u_s).scatter(1, targets_u.unsqueeze(1), 1)
                        smooth_one_hot = one_hot * (1 - args.label_smoothing) + (args.label_smoothing / n_classes)
                        Lu = (-smooth_one_hot * torch.log_softmax(pred_u_s, dim=1)).sum(dim=1) * mask
                        Lu = Lu.mean()
                    else:
                        Lu = (F.cross_entropy(pred_u_s, targets_u, reduction='none') * mask).mean()

                    loss = Lx + args.lambda_u * Lu

                    loss.backward()

                    losses_x.update(Lx.item())
                    losses_u.update(Lu.item())
                    mask_probs.update(mask.mean().item())

                    optimizer.step()
                    if args.use_ema:
                        ema_model.update(model)

                    model.zero_grad()

                    lr = 0.5 * args.lr * (1 + math.cos(math.pi * iters / args.total_steps))

                    iters += 1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    pbar.set_postfix(l=loss.item(), lx=Lx.item(), lu=Lu.item(), mask_prob=mask.mean().item(), lr=lr)
                    pbar.update(1)

            train_accuracy, train_FNR, train_roc_auc, train_loss = evaluate(model, labeled_trainloader, device, args.label_smoothing)
        
            if args.use_ema:
                test_model = ema_model.ema
            else:
                test_model = model
            test_accuracy, test_FNR, test_roc_auc, test_loss = evaluate(test_model, test_loader, device, args.label_smoothing)

            logging.info(f'Epoch: {epoch}, Iters: {iters} / {args.total_steps}')
            logging.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                         f'Train FNR: {train_FNR:.4f}, Train ROC_AUC: {train_roc_auc:.4f}')
            logging.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, '
                         f'Test FNR: {test_FNR:.4f}, Test ROC_AUC: {test_roc_auc:.4f}')

            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, iters, train_loss, train_accuracy, train_FNR, train_roc_auc,
                                 test_loss, test_accuracy, test_FNR, test_roc_auc])

            # Early stopping logic
            if test_loss < best_loss:
                best_loss = test_loss
                best_accuracy = test_accuracy
                best_fnr = test_FNR
                best_roc_auc = test_roc_auc
                save_path = os.path.join(fold_save_path, f'model.pth')
                torch.save(model.state_dict(), save_path)
                logging.info(f'Model saved to {save_path}')
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                logging.info(f'Early stopping triggered after {epoch} epochs')
                break

            logging.info(f'Iters: {iters} / {args.total_steps},'
                         f' Best Accuracy: {best_accuracy:.4f},'
                         f' Best FNR: {best_fnr:.4f},'
                         f' Best ROC_AUC: {best_roc_auc:.4f},'
                         f' Patience Counter: {patience_counter}')

        logging.info(f'Completed training for fold-{fold}')
        logging.info(f'Best Accuracy: {best_accuracy:.4f}, Best FNR: {best_fnr:.4f}, Best ROC_AUC: {best_roc_auc:.4f}')