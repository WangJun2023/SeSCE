import numpy as np
import os
import torch
from collections import defaultdict
import random


def generate_extraCLs(consPath, y, orig_num, extra_num, J, imbCluster):
    if isinstance(y, torch.Tensor):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    labels = np.unique(y)
    n_samples = len(y)
    print(f"开始生成约束，样本数: {n_samples}, 类别数: {len(labels)}")

    from collections import defaultdict
    import random

    class_to_samples = defaultdict(list)
    for i, label in enumerate(y):
        class_to_samples[label].append(i)


    all_constraints = []
    seen_pairs = set()  


    class_sizes = {label: len(samples) for label, samples in class_to_samples.items()}


    orig_ml, orig_cl = [], []

    total_ml_possible = sum(class_sizes[l] * (class_sizes[l] - 1) // 2 for l in class_sizes)
    print(f"理论最大ML约束数: {total_ml_possible}, 目标原始约束数: {orig_num}")


    target_ml = min(int(orig_num * 0.8), total_ml_possible)
    ml_generated = 0

    sorted_classes = sorted(class_sizes.items(), key=lambda x: x[1], reverse=True)

    for class_id, size in sorted_classes:
        if ml_generated >= target_ml:
            break

        if size >= 2:
            samples = class_to_samples[class_id]
            max_ml_from_class = size * (size - 1) // 2
            needed_from_class = min(target_ml - ml_generated, max_ml_from_class)

            # 对于大类别，使用采样而不是枚举所有组合
            if size > 1000:  
                possible_pairs = set()
                attempts = 0
                max_attempts = needed_from_class * 10  

                while len(possible_pairs) < needed_from_class and attempts < max_attempts:
                    idx1, idx2 = random.sample(samples, 2)
                    pair = tuple(sorted((idx1, idx2)))
                    if pair not in seen_pairs:
                        possible_pairs.add(pair)
                    attempts += 1

                for pair in possible_pairs:
                    orig_ml.append(pair)
                    seen_pairs.add(pair)
                    ml_generated += 1

            else:  # 小类别，可以生成所有组合
                all_pairs = []
                for i in range(size):
                    for j in range(i + 1, size):
                        all_pairs.append(tuple(sorted((samples[i], samples[j]))))

                random.shuffle(all_pairs)
                selected = all_pairs[:needed_from_class]

                for pair in selected:
                    orig_ml.append(pair)
                    seen_pairs.add(pair)
                    ml_generated += 1

    print(f"阶段1完成，生成ML约束: {ml_generated}")


    if ml_generated < orig_num:
        needed_cl = orig_num - ml_generated
        cl_generated = 0

        class_list = list(class_sizes.keys())

        while cl_generated < needed_cl:
            class1, class2 = random.sample(class_list, 2)
            sample1 = random.choice(class_to_samples[class1])
            sample2 = random.choice(class_to_samples[class2])
            pair = tuple(sorted((sample1, sample2)))

            if pair not in seen_pairs:
                orig_cl.append(pair)
                seen_pairs.add(pair)
                cl_generated += 1

        print(f"阶段2完成，生成CL约束: {cl_generated}")

    all_constraints.append((orig_ml, orig_cl))

    if imbCluster is not None:
        chosen_class = imbCluster
    else:
        chosen_class = random.choice(labels)

    chosen_class_indices = class_to_samples[chosen_class]
    other_class_indices = []
    for label in labels:
        if label != chosen_class:
            other_class_indices.extend(class_to_samples[label])

    extra_pairs = []


    for j in range(2, J + 1):
        extra_num_j = (extra_num // (J - 1)) * (j - 1)
        print(f"生成额外约束集 {j - 1}/{J - 1}, 目标数量: {extra_num_j}")

        needed_extra = extra_num_j - len(extra_pairs)

        batch_size = min(10000, needed_extra)
        generated_in_batch = 0

        while generated_in_batch < needed_extra:
            current_batch = min(batch_size, needed_extra - generated_in_batch)

            if len(chosen_class_indices) >= 2:
                # 生成一批ML对
                ml_batch = set()
                ml_to_generate = int(current_batch * 0.7)  

                for _ in range(ml_to_generate * 2):  
                    if len(ml_batch) >= ml_to_generate:
                        break
                    idx1, idx2 = random.sample(chosen_class_indices, 2)
                    pair = tuple(sorted((idx1, idx2)))
                    if pair not in seen_pairs:
                        ml_batch.add(pair)

                # 生成一批CL约束
                cl_batch = set()
                cl_to_generate = current_batch - len(ml_batch)

                for _ in range(cl_to_generate * 2):
                    if len(cl_batch) >= cl_to_generate:
                        break
                    sample_1 = random.choice(chosen_class_indices)
                    sample_2 = random.choice(other_class_indices)
                    pair = tuple(sorted((sample_1, sample_2)))
                    if pair not in seen_pairs and pair[0] != pair[1]:
                        cl_batch.add(pair)

                batch_pairs = list(ml_batch) + list(cl_batch)

                for pair in batch_pairs:
                    if pair not in seen_pairs:
                        extra_pairs.append(pair)
                        seen_pairs.add(pair)
                        generated_in_batch += 1

                        if generated_in_batch >= current_batch:
                            break

        this_ml = orig_ml.copy()
        this_cl = orig_cl.copy()
        for p in extra_pairs:
            if y[p[0]] == y[p[1]]:
                this_ml.append(p)
            else:
                this_cl.append(p)

        all_constraints.append((this_ml, this_cl))
        print(f"约束集 {j} 完成, ML: {len(this_ml)}, CL: {len(this_cl)}")

    os.makedirs(consPath, exist_ok=True)

    for idx, constraints in enumerate(all_constraints, start=1):
        ml_array = np.array(constraints[0], dtype=np.int32)
        cl_array = np.array(constraints[1], dtype=np.int32)

        np.savez_compressed(
            f'{consPath}/constraints_{idx}.npz',
            ml=ml_array,
            cl=cl_array
        )

        print(f"保存约束集 {idx}: ML={len(ml_array)}, CL={len(cl_array)}")

    return all_constraints



def generate_balance(consPath, y, orig_num, extra_num, J):
    if isinstance(y, torch.Tensor):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    n_samples = len(y)
    print(f"开始生成平衡约束，样本数: {n_samples}")

    class_to_samples = defaultdict(list)
    for i, label in enumerate(y):
        class_to_samples[label].append(i)


    seen_pairs = set()

    class_sizes = {label: len(samples) for label, samples in class_to_samples.items()}
    class_list = list(class_sizes.keys())

    all_constraints = []
    orig_ml, orig_cl = [], []


    target_ml = min(int(orig_num * 0.7), n_samples * 10)  

    ml_batch = set()
    batch_size = min(10000, target_ml)

    while len(ml_batch) < target_ml:
        class_id = random.choice(class_list)
        samples = class_to_samples[class_id]

        if len(samples) >= 2:
            idx1, idx2 = random.sample(samples, 2)
            pair = tuple(sorted((idx1, idx2)))

            if pair not in seen_pairs:
                ml_batch.add(pair)

        if len(ml_batch) >= batch_size or len(ml_batch) >= target_ml:
            break

    orig_ml = list(ml_batch)
    for pair in orig_ml:
        seen_pairs.add(pair)

    print(f"生成ML约束: {len(orig_ml)}")

    if len(orig_ml) < orig_num:
        needed_cl = orig_num - len(orig_ml)
        cl_batch = set()

        while len(cl_batch) < needed_cl:
            class1, class2 = random.sample(class_list, 2)
            sample1 = random.choice(class_to_samples[class1])
            sample2 = random.choice(class_to_samples[class2])
            pair = tuple(sorted((sample1, sample2)))

            if pair not in seen_pairs:
                cl_batch.add(pair)

        orig_cl = list(cl_batch)
        for pair in orig_cl:
            seen_pairs.add(pair)

        print(f"生成CL约束: {len(orig_cl)}")

    all_constraints.append((orig_ml, orig_cl))

    extra_pairs = []

    for j in range(2, J + 1):
        extra_num_j = (extra_num // (J - 1)) * (j - 1)
        print(f"生成额外平衡约束集 {j - 1}/{J - 1}, 目标: {extra_num_j}")

        needed_extra = extra_num_j - len(extra_pairs)

        current_batch = []

        ml_ratio = 0.6  
        ml_target = int(needed_extra * ml_ratio)
        cl_target = needed_extra - ml_target

        ml_generated = 0
        while ml_generated < ml_target:
            class_id = random.choice(class_list)
            samples = class_to_samples[class_id]

            if len(samples) >= 2:
                idx1, idx2 = random.sample(samples, 2)
                pair = tuple(sorted((idx1, idx2)))

                if pair not in seen_pairs:
                    current_batch.append(pair)
                    seen_pairs.add(pair)
                    ml_generated += 1

        cl_generated = 0
        while cl_generated < cl_target:
            if len(class_list) >= 2:
                class1, class2 = random.sample(class_list, 2)
                sample1 = random.choice(class_to_samples[class1])
                sample2 = random.choice(class_to_samples[class2])
                pair = tuple(sorted((sample1, sample2)))

                if pair not in seen_pairs:
                    current_batch.append(pair)
                    seen_pairs.add(pair)
                    cl_generated += 1

        extra_pairs.extend(current_batch)

        this_ml = orig_ml.copy()
        this_cl = orig_cl.copy()

        for p in extra_pairs:
            if y[p[0]] == y[p[1]]:
                this_ml.append(p)
            else:
                this_cl.append(p)

        all_constraints.append((this_ml, this_cl))
        print(f"平衡约束集 {j} 完成, ML: {len(this_ml)}, CL: {len(this_cl)}")

    os.makedirs(consPath, exist_ok=True)

    for idx, constraints in enumerate(all_constraints, start=1):
        ml_array = np.array(constraints[0], dtype=np.int32) if constraints[0] else np.array([], dtype=np.int32)
        cl_array = np.array(constraints[1], dtype=np.int32) if constraints[1] else np.array([], dtype=np.int32)

        np.savez_compressed(
            f'{consPath}/constraints_{idx}.npz',
            ml=ml_array,
            cl=cl_array
        )

    return all_constraints

