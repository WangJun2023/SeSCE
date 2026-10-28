import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.nn.functional import normalize
import torch.nn.functional as F
from Sample_SelfWeight import AttentionMechanism
import numpy as np
import torch.optim as optim
import math
from torch.nn.utils import clip_grad_norm_
import os
import csv
from torch.nn import Parameter
from evaluation_metric import b3_precision_recall_fscore, acc
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout_rate=0):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout_rate=0):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, fea_dim, n_clusters, device):
        super(Network, self).__init__()
        self.feature_dim = fea_dim
        self.view = view
        self.encoders = []
        self.decoders = []
        self.n_clusters = n_clusters
        self.device = device 

        self.input_size_all = 0
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], fea_dim, 0).to(device))
            self.decoders.append(Decoder(input_size[v], fea_dim, 0).to(device))
            self.input_size_all += input_size[v]

        self.encoders.append(Encoder(self.input_size_all, fea_dim, 0).to(device))
        self.decoders.append(Decoder(self.input_size_all, fea_dim, 0).to(device))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_fusion_module = nn.Sequential(
            nn.Linear(fea_dim, 256),
            nn.ReLU(),
            nn.Linear(256, fea_dim)
        )

        self.attention_mechanism = AttentionMechanism(self.feature_dim)
        self.W = Parameter(torch.Tensor(n_clusters, fea_dim))

    def feature_fusion(self, zs, Wz):
        Wz_expanded = Wz.unsqueeze(-1)
        zs = torch.stack(zs)
        zs_transposed = zs.permute(1, 0, 2)
        weighted_sum = torch.sum(zs_transposed * Wz_expanded, dim=1)
        return normalize(self.feature_fusion_module(weighted_sum), dim=1)

    def recon_loss(self, x_list, xrecon_list):
        if len(x_list) != len(xrecon_list):
            raise ValueError(f"视图数量不匹配")
        total_loss = 0.0
        num_views = len(x_list)
        for i, (x_view, xrecon_view) in enumerate(zip(x_list, xrecon_list)):
            view_loss = torch.mean((xrecon_view - x_view) ** 2)
            total_loss += view_loss
        return total_loss / num_views

    def pairwise_loss(self, z1_list, z2_list, type_values, omega=2):
        total_loss = 0.0
        num_views = len(z1_list)
        for v in range(num_views):
            z1 = z1_list[v]
            z2 = z2_list[v]
            sim = (F.cosine_similarity(z1, z2, eps=1e-8) + 1) / 2
            cos_theta = 2 * sim - 1
            cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos_theta)
            phase_cos = torch.cos(omega * theta)
            phase_cos = phase_cos.clamp(-1, 1)
            phase_sim = torch.where(theta <= np.pi / omega, (phase_cos + 1) / 2, torch.zeros_like(phase_cos))
            adjusted_sim = torch.where(type_values == 1, sim, phase_sim)
            adjusted_sim = adjusted_sim.clamp(0, 1)
            view_loss = F.binary_cross_entropy(adjusted_sim, type_values.float(), reduction='mean')
            total_loss += view_loss
        return total_loss / num_views

    def cross_view_loss(self, view_embeddings, fused_embedding, eps=1e-8):
        V = len(view_embeddings)
        device = fused_embedding.device
        cross_loss = torch.tensor(0.0, device=device)
        for v in range(V):
            z_view = view_embeddings[v]
            z_view_norm = F.normalize(z_view, p=2, dim=1, eps=eps)
            fused_norm = F.normalize(fused_embedding, p=2, dim=1, eps=eps)
            cos_theta = torch.sum(z_view_norm * fused_norm, dim=1)
            cos_theta = cos_theta.clamp(-1 + eps, 1 - eps)
            diff = cos_theta - 1.0
            squared_diff = torch.square(diff)
            view_loss = torch.mean(squared_diff)
            cross_loss = cross_loss + (1.0 / V) * view_loss
        return cross_loss

    def merge_constraints(self, ml_ind1, ml_ind2, cl_ind1, cl_ind2):
        ml_labels = np.ones(len(ml_ind1))
        cl_labels = np.zeros(len(cl_ind1))
        # Handle cases where ml or cl might be empty
        if len(ml_ind1) > 0:
            ml_cons = np.column_stack((ml_ind1, ml_ind2, ml_labels))
        else:
            ml_cons = np.empty((0, 3))

        if len(cl_ind1) > 0:
            cl_cons = np.column_stack((cl_ind1, cl_ind2, cl_labels))
        else:
            cl_cons = np.empty((0, 3))

        mergedCons = np.concatenate((ml_cons, cl_cons), axis=0)
        # Ensure integer type for indices, float for label
        if mergedCons.shape[0] > 0:
            indices = np.random.permutation(np.arange(mergedCons.shape[0]))
            shuffledMergedCons = mergedCons[indices]
            return shuffledMergedCons
        else:
            return np.empty((0, 3))

    def hard_assign(self, z, W):
        cosine = F.linear(F.normalize(z), F.normalize(W))
        y_pred = torch.argmax(cosine, dim=1)
        return cosine, y_pred

    def generate_pseudo_constraints(self, X_list,
                                    orig_ml_ind1, orig_ml_ind2,
                                    orig_cl_ind1, orig_cl_ind2,
                                    n_clusters=None,
                                    topk_ratio=0.1):
        """

        Args:
            X_list: 多视图数据
            orig_ml_ind1, orig_ml_ind2: 原始 Must-Link 索引
            orig_cl_ind1, orig_cl_ind2: 原始 Cannot-Link 索引
            n_clusters: 聚类簇数 (默认使用 self.n_clusters)
            topk_ratio: 每个簇中只取距离中心最近的前百分之多少的样本 (0.0 - 1.0)
        """
        if n_clusters is None:
            n_clusters = self.n_clusters

        self.eval()
        with torch.no_grad():
            _, _, z_tilde = self.forward(X_list)
            z_np = z_tilde.cpu().numpy()

            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(z_np)

            dists_all = kmeans.transform(z_np)
            dists_to_center = dists_all[np.arange(len(z_np)), cluster_labels]

            core_mask = np.zeros(len(z_np), dtype=bool)

            for k in range(n_clusters):
                cluster_indices = np.where(cluster_labels == k)[0]
                if len(cluster_indices) == 0: continue

                k_dists = dists_to_center[cluster_indices]

                cutoff_idx = int(len(cluster_indices) * topk_ratio)
                if cutoff_idx < 1: cutoff_idx = 1  

                sorted_dists = np.sort(k_dists)
                threshold = sorted_dists[min(cutoff_idx, len(sorted_dists) - 1)]

                core_indices = cluster_indices[k_dists <= threshold]
                core_mask[core_indices] = True

            orig_cl_set = set(zip(orig_cl_ind1, orig_cl_ind2))
            orig_cl_set.update(zip(orig_cl_ind2, orig_cl_ind1))  

            orig_ml_set = set(zip(orig_ml_ind1, orig_ml_ind2))
            orig_ml_set.update(zip(orig_ml_ind2, orig_ml_ind1))

            pseudo_ml_list = []
            pseudo_cl_list = []


            # --- 生成 Pseudo ML (同簇核心样本) ---
            for k in range(n_clusters):
                k_core_indices = np.where((cluster_labels == k) & core_mask)[0]
                if len(k_core_indices) < 2: continue

                grid_x, grid_y = np.meshgrid(k_core_indices, k_core_indices)
                pairs = np.column_stack((grid_x.ravel(), grid_y.ravel()))

                # 过滤掉自环 (i, i) 和下三角 (只留 i < j)
                pairs = pairs[pairs[:, 0] < pairs[:, 1]]

                # 过滤掉与原始 CL 冲突的对
                valid_pairs = []
                for p in pairs:
                    if (p[0], p[1]) not in orig_cl_set:
                        valid_pairs.append(p)

                if len(valid_pairs) > 0:
                    pseudo_ml_list.append(np.array(valid_pairs))

            # --- 生成 Pseudo CL (不同簇核心样本) ---
            clusters_cores = {}
            for k in range(n_clusters):
                clusters_cores[k] = np.where((cluster_labels == k) & core_mask)[0]

            # 随机生成 CL (防止 N^2 复杂度)
            target_cl_num = sum([len(x) for x in pseudo_ml_list]) if pseudo_ml_list else 1000
            target_cl_num = min(target_cl_num, 10000)  # 上限

            generated_cl_count = 0
            attempts = 0
            max_attempts = target_cl_num * 5

            temp_cl = []
            while generated_cl_count < target_cl_num and attempts < max_attempts:
                attempts += 1
                # 随机选两个不同的簇
                c1, c2 = np.random.choice(n_clusters, 2, replace=False)

                cores1 = clusters_cores[c1]
                cores2 = clusters_cores[c2]

                if len(cores1) == 0 or len(cores2) == 0: continue

                # 随机选两个点
                idx1 = np.random.choice(cores1)
                idx2 = np.random.choice(cores2)

                # 确保 i < j
                if idx1 > idx2: idx1, idx2 = idx2, idx1

                # 检查冲突：不能是原始 ML
                if (idx1, idx2) not in orig_ml_set:
                    temp_cl.append([idx1, idx2])
                    generated_cl_count += 1

            if len(temp_cl) > 0:
                pseudo_cl_list.append(np.array(temp_cl))

        self.train()

        if len(pseudo_ml_list) > 0:
            pseudo_ml_indices = np.concatenate(pseudo_ml_list, axis=0)
        else:
            pseudo_ml_indices = np.empty((0, 2))

        if len(pseudo_cl_list) > 0:
            pseudo_cl_indices = np.concatenate(pseudo_cl_list, axis=0)
        else:
            pseudo_cl_indices = np.empty((0, 2))

        return pseudo_ml_indices, pseudo_cl_indices

    def forward(self, xs):
        xrs = []
        zs = []
        x_cat = torch.cat(xs, dim=1)
        z_all = self.encoders[self.view](x_cat)
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            z_normalized = F.normalize(z, p=2, dim=1, eps=1e-12)
            zs.append(z_normalized)
        Wz = self.attention_mechanism.compute_attention_weights(z_all, zs)
        for v in range(self.view):
            xr = self.decoders[v](zs[v])
            xrs.append(xr)
        zs = [z.to("cuda") for z in zs]
        z_tilde = self.feature_fusion(zs, Wz)
        return zs, xrs, z_tilde

    def fit(self,
            record_log_dir,
            ml_ind1, ml_ind2, cl_ind1, cl_ind2,
            lam,
            beta,
            X,
            lr=0.001,
            batch_size=256,
            epochs=500,
            soft_epochs=100,
            omega=2,
            use_pseudo=False,
            pseudo_start_epoch=50,
            pseudo_interval=50,
            topk_ratio=0.1,
            max_pseudo_num=1000):

        print("===================== Train ours model ===================")
        optimizer_params = optim.Adam([param for param in self.parameters() if param is not self.W], lr=lr,
                                      weight_decay=0)


        orig_ml_ind1, orig_ml_ind2 = ml_ind1, ml_ind2
        orig_cl_ind1, orig_cl_ind2 = cl_ind1, cl_ind2

        curr_ml_ind1, curr_ml_ind2 = orig_ml_ind1, orig_ml_ind2
        curr_cl_ind1, curr_cl_ind2 = orig_cl_ind1, orig_cl_ind2

        mergedCons = torch.from_numpy(self.merge_constraints(curr_ml_ind1, curr_ml_ind2, curr_cl_ind1, curr_cl_ind2))
        mergedCons = mergedCons.long().to("cuda")
        mergedCons_num = mergedCons.shape[0]

        if mergedCons_num > 0:
            mergedCons_num_batch = int(math.ceil(1.0 * mergedCons_num / batch_size))
        else:
            mergedCons_num_batch = 1  # Fallback

        X_num = X[0].shape[0]
        X_batchsize = max(1, int(math.ceil(1.0 * X_num / mergedCons_num_batch)))

        min_loss = 1e10
        self.train()

        for epoch in range(epochs):
            final_epoch = epoch

            if use_pseudo and epoch >= pseudo_start_epoch and epoch % pseudo_interval == 0:
                print(f"-- Epoch {epoch}: Generating Pseudo Constraints --")
                p_ml, p_cl = self.generate_pseudo_constraints(
                    X,
                    orig_ml_ind1, orig_ml_ind2,
                    orig_cl_ind1, orig_cl_ind2,
                    topk_ratio = topk_ratio  
                )

                # 采样防止数量爆炸
                if len(p_ml) > max_pseudo_num:
                    idx = np.random.choice(len(p_ml), max_pseudo_num, replace=False)
                    p_ml = p_ml[idx]
                if len(p_cl) > max_pseudo_num:
                    idx = np.random.choice(len(p_cl), max_pseudo_num, replace=False)
                    p_cl = p_cl[idx]

                print(f"   Generated Pseudo ML: {len(p_ml)}, Pseudo CL: {len(p_cl)}")

                # 合并原始约束 + 伪约束
                if len(p_ml) > 0:
                    curr_ml_ind1 = np.concatenate([orig_ml_ind1, p_ml[:, 0]])
                    curr_ml_ind2 = np.concatenate([orig_ml_ind2, p_ml[:, 1]])
                else:
                    curr_ml_ind1, curr_ml_ind2 = orig_ml_ind1, orig_ml_ind2

                if len(p_cl) > 0:
                    curr_cl_ind1 = np.concatenate([orig_cl_ind1, p_cl[:, 0]])
                    curr_cl_ind2 = np.concatenate([orig_cl_ind2, p_cl[:, 1]])
                else:
                    curr_cl_ind1, curr_cl_ind2 = orig_cl_ind1, orig_cl_ind2


                mergedCons = torch.from_numpy(
                    self.merge_constraints(curr_ml_ind1, curr_ml_ind2, curr_cl_ind1, curr_cl_ind2))
                mergedCons = mergedCons.long().to("cuda")
                mergedCons_num = mergedCons.shape[0]

                if mergedCons_num > 0:
                    mergedCons_num_batch = int(math.ceil(1.0 * mergedCons_num / batch_size))
                else:
                    mergedCons_num_batch = 1
                X_batchsize = max(1, int(math.ceil(1.0 * X_num / mergedCons_num_batch)))
                print(f"   Total Constraints: {mergedCons_num}, Batches: {mergedCons_num_batch}")


            list_log = [0] * 13
            list_log[0] = epoch

            # ============= no update (Evaluation) ============== #
            self.eval()
            with torch.no_grad():
                current_device = X[0].device

                eval_idx = torch.randperm(mergedCons.shape[0])[:5000]
                eval_cons = mergedCons[eval_idx]

                if eval_cons.shape[0] > 0:
                    inputs1 = [view_data[eval_cons[:, 0].long()] for view_data in X]
                    inputs2 = [view_data[eval_cons[:, 1].long()] for view_data in X]
                    type_values = eval_cons[:, 2]
                    zs1, _, _ = self.forward(inputs1)
                    zs2, _, _ = self.forward(inputs2)
                    loss_pairwise = self.pairwise_loss(zs1, zs2, type_values, omega)
                else:
                    loss_pairwise = torch.tensor(0.0).to(current_device)

                zs, X_recon, z_tilde = self.forward(X)
                loss_recon = self.recon_loss(X, X_recon)
                loss_cross = self.cross_view_loss(zs, z_tilde)
                total_loss = loss_pairwise + lam * loss_recon + beta * loss_cross

                list_log[1] = loss_pairwise.item()
                list_log[2] = loss_recon.item()
                list_log[3] = loss_cross.item()
                list_log[4] = total_loss.item()

                try:
                    delta_loss = total_loss.item() - min_loss
                    delta = delta_loss / min_loss
                    if total_loss.item() < min_loss:
                        min_loss = total_loss.item()
                    if epoch > soft_epochs and delta < 0 and not use_pseudo:  
                        break
                except:
                    pass
            self.train()

            # ============================ update network =============================== #
            total_loss_pairwise = 0.0
            total_loss_recon = 0.0
            total_loss_cross = 0.0

            # Shuffle Constraints and Data
            shuffled_indices = np.random.permutation(mergedCons.shape[0])
            mergedCons = mergedCons[shuffled_indices]
            shuffled_indices_X = np.random.permutation(X[0].shape[0])
            X_shuff = [view_data[shuffled_indices_X] for view_data in X]

            for cons_batch_idx in range(mergedCons_num_batch):
                batch_start = cons_batch_idx * batch_size
                batch_end = min(mergedCons_num, (cons_batch_idx + 1) * batch_size)

                if batch_start >= batch_end: continue  # Prevent empty batch

                batch_indices = mergedCons[batch_start:batch_end]
                px1_list = []
                px2_list = []

                current_device = X[0].device
                batch_indices_tensor = batch_indices.long().to(current_device)

                for view_data in X:
                    px1_view = view_data[batch_indices_tensor[:, 0]]
                    px2_view = view_data[batch_indices_tensor[:, 1]]
                    px1_list.append(px1_view)
                    px2_list.append(px2_view)

                type_values = batch_indices_tensor[:, 2].float()

                optimizer_params.zero_grad()

                zs1, _, _ = self.forward(px1_list)
                zs2, _, _ = self.forward(px2_list)

                # pairwise loss
                loss_pairwise = self.pairwise_loss(zs1, zs2, type_values, omega)

                # recon loss
                start_idx = cons_batch_idx * X_batchsize
                end_idx = min(X_num, (cons_batch_idx + 1) * X_batchsize)
                if start_idx >= end_idx:
                    start_idx = 0
                    end_idx = X_batchsize

                inputs = [view_data[start_idx:end_idx] for view_data in X_shuff]
                zs, xr, z_tilde = self.forward(inputs)
                loss_recon = self.recon_loss(inputs, xr)

                # cross loss
                loss_cross = self.cross_view_loss(zs, z_tilde)

                loss = loss_pairwise + lam * loss_recon + beta * loss_cross
                loss.backward()
                clip_grad_norm_(optimizer_params.param_groups[0]['params'], max_norm=1)
                optimizer_params.step()

                total_loss_pairwise += loss_pairwise.data * px1_list[0].shape[0]
                total_loss_recon += loss_recon.data * px1_list[0].shape[0]
                total_loss_cross += loss_cross.data * px1_list[0].shape[0]

            if epoch % 1 == 0:
                print("(Epoch %d) Pair: %.4f, Recon: %.4f, Cross: %.4f, total: %.4f" % (
                    epoch, float(total_loss_pairwise), float(total_loss_recon), float(total_loss_cross), float(total_loss_pairwise+total_loss_recon+total_loss_cross)))

            # ================================ record log ================================ #
            if not os.path.exists(record_log_dir):
                with open(record_log_dir, "w") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["epoch", "loss_pairwise", "loss_recon", "loss_cross", "loss_total"])  
            with open(record_log_dir, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list_log)

        return final_epoch


    def assign_Kmeans(self,
                      record_log_dir,
                      X, y=None,
                      test_X=None, test_y=None,
                      specific_K=None,
                      n_init=50):

        """
        record log list:
            [   epoch,
                loss_pairwise, loss_recon, loss_cross, loss_total
                acc_train, acc_test,
                nmi_train, nmi_test,
                ari_train, ari_test
                f1_train, f1_test ]
        """
        list_log = [0,
                    0, 0, 0, 0,
                    0, 0,
                    0, 0,
                    0, 0,
                    0, 0]

        print("====================== Use Kmeans to find W for assignments ======================")
        if specific_K is not None:
            self.n_clusters = specific_K
            self.W = Parameter(torch.Tensor(self.n_clusters, self.feature_dim))

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        if y is not None:
            if hasattr(y, 'cpu'):  
                y = y.cpu().numpy()
        if test_y is not None:
            if hasattr(test_y, 'cpu'):  
                test_y = test_y.cpu().numpy()
        silhouette_score_avg = None  

        if specific_K is not None:
            print(f"Finding cluster center W with Kmeans for specific_K={specific_K}")
            _, _, z_tilde = self.forward(X)
            data_np = z_tilde.cpu().detach().numpy()

            sample_size = 5000
            W = self.W
            best_kmeans = None
            best_inertia = float('inf')
            silhouette_scores = []
            sc_n_init = 5
            for i in range(sc_n_init):
                kmeans = KMeans(self.n_clusters, n_init=1)
                y_pred = kmeans.fit_predict(data_np)
                score = silhouette_score(data_np, y_pred, sample_size=sample_size)
                silhouette_scores.append(score)
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_kmeans = kmeans
                print(
                    f"Kmeans run {i + 1}/{sc_n_init} for specific_K={specific_K}, inertia: {kmeans.inertia_}, silhouette score: {score:.5f}")
            average_score = np.mean(silhouette_scores)
            silhouette_score_avg = average_score
            silhouette_score_std = np.std(silhouette_scores)
            print(f"Average Silhouette Score for specific_K={specific_K}: {silhouette_score_avg:.5f}")
            W.data.copy_(torch.Tensor(best_kmeans.cluster_centers_))
        else:
            # ============================= ground truth K ============================ #
            print("Finding cluster center W with Kmeans")
            _, _, z_tilde = self.forward(X)
            W = self.W
            kmeans = KMeans(self.n_clusters, n_init=n_init)
            kmeans.fit(z_tilde.detach().cpu().numpy())
            W.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        _, _, latent = self.forward(X)

        W = self.W
        _, y_pred = self.hard_assign(latent, W)
        y_pred = y_pred.data.cpu().numpy()

        if test_X is not None:
            _, _, test_latent = self.forward(test_X)

            W = self.W
            test_y_pred = self.hard_assign(test_latent, W)[1]
            test_y_pred = test_y_pred.data.cpu().numpy()

        if y is not None:
            print("acc_train: %.5f, nmi_train: %.5f, ari_train:%.5f" % (
            acc(y, y_pred), normalized_mutual_info_score(y, y_pred), adjusted_rand_score(y, y_pred)))
            final_acc = acc(y, y_pred)
            final_nmi = normalized_mutual_info_score(y, y_pred)
            final_ari = adjusted_rand_score(y, y_pred)
            final_f1, _, _ = b3_precision_recall_fscore(y, y_pred)
            """record log"""
            list_log[5] = final_acc
            list_log[7] = final_nmi
            list_log[9] = final_ari
            list_log[11] = final_f1

        if test_X is not None and test_y is not None:
            print("acc_test: %.5f, nmi_test: %.5f, ari_test:%.5f" % (
                acc(test_y, test_y_pred), normalized_mutual_info_score(test_y, test_y_pred), adjusted_rand_score(test_y, test_y_pred)))

            test_acc = acc(test_y, test_y_pred)
            test_nmi = normalized_mutual_info_score(test_y, test_y_pred)
            test_ari = adjusted_rand_score(test_y, test_y_pred)
            test_f1, _, _ = b3_precision_recall_fscore(test_y, test_y_pred)
            """record log"""
            list_log[6] = test_acc
            list_log[8] = test_nmi
            list_log[10] = test_ari
            list_log[12] = test_f1


        if not os.path.exists(record_log_dir):
            with open(record_log_dir, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["epoch",
                                 "loss_pairwise", "loss_recon", "loss_cross", "loss_total",
                                 "acc_train", "acc_test",
                                 "nmi_train", "nmi_test",
                                 "ari_train", "ari_test",
                                 "f1_train", "f1_test"])
        with open(record_log_dir, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list_log)

        if specific_K is not None:
            return final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari, final_f1, test_f1, silhouette_score_avg, silhouette_score_std
        else:
            return final_acc, test_acc, final_nmi, test_nmi, final_ari, test_ari, final_f1, test_f1