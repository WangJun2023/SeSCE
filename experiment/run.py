import argparse
import gc
import os
import sys
import numpy as np
import torch
import csv
import time
from tqdm import tqdm

from collections import defaultdict  

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from consRules import generate_extraCLs, generate_balance
from model_SeSCE import Network
from load_data import load_multiview_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "5"
print(f'1.torch version:{torch.__version__} 2.cuda available:{torch.cuda.is_available()}')



if __name__ == '__main__':

    folder_path = "Datasets/"

#     target_datasets = ["HW", "Cifar10_4views",
#             "scene-15_3v",
#             "YoutubeFace10_4Views",
#             "ALOI_100", "EMNIST", "Caltech101-7", "MNIST"
# ]

    target_datasets = [

                       "HW"
                       ]

    for Dataname in tqdm(target_datasets, desc="Datasets"):
        parser = argparse.ArgumentParser(description='train')
        parser.add_argument('--dataset', default=Dataname)
        parser.add_argument('--batch_size', default=256, type=int)
        parser.add_argument("--learning_rate", default=0.0001)
        parser.add_argument("--iter", default=1)
        parser.add_argument("--feature_dim", default=64)
        parser.add_argument('--consRule', type=str, default="balance")
        parser.add_argument('--consIndex', type=int, default=1)
        parser.add_argument('--test_size', type=int, default=0.2)
        parser.add_argument('--lam', type=float, default=0.01)
        parser.add_argument('--beta', type=float, default=0.001)
        parser.add_argument('--epochs', type=int, default=200)
        parser.add_argument('--soft_epochs', type=int, default=100)
        parser.add_argument('--omega', type=float, default=2)
        parser.add_argument('--tail_ratio', type=float, default=0.05)
        parser.add_argument('--tail_sensitivity', type=str, default="False")
        parser.add_argument('--J', type=int, default=10)
        parser.add_argument('--orig_num', type=int, default=None)
        parser.add_argument('--extra_num', type=int, default=None)
        parser.add_argument('--imbCluster', type=int, default=None)

        parser.add_argument('--n_runs', type=int, default=3, help="Number of times to run the experiment")

        parser.add_argument('--use_pseudo', type=str, default="True", help="Whether to use pseudo constraints")
        parser.add_argument('--pseudo_start', type=int, default=50, help="Epoch to start generating pseudo constraints")
        parser.add_argument('--pseudo_interval', type=int, default=50,
                            help="Interval epochs to regenerate pseudo constraints")
        parser.add_argument('--topk_ratio', type=float, default=0.1, help="Threshold for Pseudo Must-Link")
        parser.add_argument('--max_pseudo', type=int, default=1000, help="Max number of pseudo constraints to add")

        args = parser.parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_list, y, train_X_list, train_y, test_X_list, test_y, num_views, n_samples = load_multiview_dataset(
            folder_path, args.dataset, test_size=args.test_size, random_state=1)
        input_dim = [item.shape[1] for item in train_X_list]
        n_clusters = len(np.unique(y))

        # =============================Generate Constraints===========================
        if args.orig_num is None and args.extra_num is None:
            orig_num = int(n_samples * 0.1)
            extra_num = int(n_samples * 0.9)
        else:
            orig_num = args.orig_num
            extra_num = args.extra_num

        J = args.J
        imbCluster = args.imbCluster
        this_y = train_y
        consPath = f'./Constraints/{args.dataset}/{args.consRule}'

        if not os.path.exists(consPath):
            if args.consRule == "extraCLs":
                generate_extraCLs(consPath, this_y, orig_num, extra_num, J, imbCluster)
            elif args.consRule == "balance":
                generate_balance(consPath, this_y, orig_num, extra_num, J)
            else:
                raise ValueError(f"Constraint rule '{args.consRule}' not found.")
        else:
            print(f"Constraints already exist at {consPath}, skipping generation.")

        # ==================== Load Constraints ====================
        constraints_file = f'{consPath}/constraints_{args.consIndex}.npz'
        constraints = np.load(constraints_file, allow_pickle=True)
        ml_base = constraints['ml']
        cl_base = constraints['cl']

        metrics_history = defaultdict(list)


        lab_result_path = f"./results/Ours/{args.dataset}/{args.consRule}"
        if not os.path.exists(lab_result_path):
            os.makedirs(lab_result_path)

        ml_ind1, ml_ind2 = zip(*ml_base) if ml_base.size > 0 else ([], [])
        cl_ind1, cl_ind2 = zip(*cl_base) if cl_base.size > 0 else ([], [])
        ml_ind1, ml_ind2 = np.array(ml_ind1), np.array(ml_ind2)
        cl_ind1, cl_ind2 = np.array(cl_ind1), np.array(cl_ind2)

        if len(ml_ind1) > 0:
            ml_indices = np.arange(len(ml_ind1))
            np.random.shuffle(ml_indices)
            ml_ind1, ml_ind2 = ml_ind1[ml_indices], ml_ind2[ml_indices]

        if len(cl_ind1) > 0:
            cl_indices = np.arange(len(cl_ind1))
            np.random.shuffle(cl_indices)
            cl_ind1, cl_ind2 = cl_ind1[cl_indices], cl_ind2[cl_indices]

        print(f"Starting {args.n_runs} runs for dataset: {args.dataset}")

        for run_idx in range(args.n_runs):
            print(f"\n=== Run {run_idx + 1}/{args.n_runs} ===")

            model = Network(view=num_views, input_size=input_dim, fea_dim=args.feature_dim, n_clusters=n_clusters,
                            device=device).to(device)

            record_log_dir = os.path.join(lab_result_path,
                                          f"log_{args.dataset}_{args.consRule}_{args.consIndex}_run{run_idx}.csv")

            curr_train_X = [x.to(device) for x in train_X_list]
            curr_test_X = [x.to(device) for x in test_X_list]

            use_pseudo_bool = True if args.use_pseudo == "True" else False

            epoch = model.fit(
                record_log_dir=record_log_dir,
                ml_ind1=ml_ind1,
                ml_ind2=ml_ind2,
                cl_ind1=cl_ind1,
                cl_ind2=cl_ind2,
                lam=args.lam,
                beta=args.beta,
                X=curr_train_X,
                lr=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.epochs,
                soft_epochs=args.soft_epochs,
                omega=args.omega,
                use_pseudo=use_pseudo_bool,
                pseudo_start_epoch=args.pseudo_start,
                pseudo_interval=args.pseudo_interval,
                topk_ratio = args.topk_ratio,
                max_pseudo_num=args.max_pseudo
            )


            result_dir = os.path.join(lab_result_path, f"result_{args.dataset}_{args.consRule}.csv")

            if not os.path.exists(result_dir):
                with open(result_dir, "w", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    header = ["Run_ID", "lr", "batch_size", "dataset", "consRule", "consIndex", "epochs",
                              "train_acc", "test_acc", "train_nmi", "test_nmi",
                              "train_ari", "test_ari", "train_f1", "test_f1",
                              "D", "lam", "beta", "omega", "cluster_time",
                              "topk_ratio", "max_pseudo"]
                    writer.writerow(header)

            start_time = time.time()
            train_acc, test_acc, train_nmi, test_nmi, train_ari, test_ari, train_f1, test_f1 = model.assign_Kmeans(
                record_log_dir=record_log_dir,
                X=curr_train_X,
                y=train_y,
                test_X=curr_test_X,
                test_y=test_y)
            end_time = time.time()
            cluster_time = end_time - start_time

            metrics_history['train_acc'].append(train_acc)
            metrics_history['test_acc'].append(test_acc)
            metrics_history['train_nmi'].append(train_nmi)
            metrics_history['test_nmi'].append(test_nmi)
            metrics_history['train_ari'].append(train_ari)
            metrics_history['test_ari'].append(test_ari)
            metrics_history['train_f1'].append(train_f1)
            metrics_history['test_f1'].append(test_f1)
            metrics_history['cluster_time'].append(cluster_time)

            with open(result_dir, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f"Run_{run_idx}", args.learning_rate, args.batch_size,
                                    args.dataset, args.consRule, args.consIndex,
                                    epoch,
                                    train_acc, test_acc,
                                    train_nmi, test_nmi,
                                    train_ari, test_ari,
                                    train_f1, test_f1,
                                    args.feature_dim, args.lam, args.beta, args.omega, cluster_time,
                                    args.topk_ratio, args.max_pseudo])

            del model
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\nComputing statistics for {args.dataset} over {args.n_runs} runs...")

        stats = {}
        for key, values in metrics_history.items():
            stats[f"{key}_mean"] = np.mean(values)
            stats[f"{key}_std"] = np.std(values)

        result_dir = os.path.join(lab_result_path, f"result_{args.dataset}_{args.consRule}.csv")

        with open(result_dir, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)

            row_mean = [
                "MEAN", args.learning_rate, args.batch_size,
                args.dataset, args.consRule, args.consIndex,
                args.epochs,  
                stats['train_acc_mean'], stats['test_acc_mean'],
                stats['train_nmi_mean'], stats['test_nmi_mean'],
                stats['train_ari_mean'], stats['test_ari_mean'],
                stats['train_f1_mean'], stats['test_f1_mean'],
                args.feature_dim, args.lam, args.beta, args.omega,
                stats['cluster_time_mean'], args.topk_ratio, args.max_pseudo
            ]
            writer.writerow(row_mean)

            row_std = [
                "STD", "-", "-",
                "-", "-", "-",
                "-",
                stats['train_acc_std'], stats['test_acc_std'],
                stats['train_nmi_std'], stats['test_nmi_std'],
                stats['train_ari_std'], stats['test_ari_std'],
                stats['train_f1_std'], stats['test_f1_std'],
                "-", "-", "-", "-",
                stats['cluster_time_std'], "-", "-"
            ]
            writer.writerow(row_std)

        print(f"Finished {args.dataset}. Results saved to {result_dir}")