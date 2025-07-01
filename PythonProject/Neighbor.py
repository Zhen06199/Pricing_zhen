import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

def generate_provider_neighbors(provider_file="providers.txt", target_ids=[0], K=4, output_dir="./neighbor"):
    """
    为指定 provider ID 生成包含自身和其最邻近 K 个 provider 的文件。

    参数：
        provider_file (str): provider 数据文件路径（CSV格式）
        target_ids (list or set): 要处理的 provider ID 列表或集合
        K (int): 邻居数量（默认4）
        output_dir (str): 输出目录（默认 "./neighbor"）

    输出：
        每个指定 provider 一个 TXT 文件，包含其自己和 K 个最相似邻居
        如果总 provider 数不足 K+1，则输出全部 provider
    """
    provider_ids = []
    supplies = []
    latencies = []
    co2s = []
    costs = []

    with open(provider_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            provider_ids.append(int(parts[0]))
            supplies.append(float(parts[1]))
            latencies.append(float(parts[2]))
            co2s.append(float(parts[3]))
            costs.append(float(parts[4]))

    provider_ids = np.array(provider_ids)
    supplies = np.array(supplies)
    latencies = np.array(latencies)
    co2s = np.array(co2s)
    costs = np.array(costs)
    total_provider = len(provider_ids)

    features = np.vstack([latencies, co2s, costs]).T
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)

    dist_matrix = cdist(features_norm, features_norm, metric='euclidean')

    os.makedirs(output_dir, exist_ok=True)
    generated_count = 0
    total_providers = len(provider_ids)

    for i, pid in enumerate(provider_ids):
        if pid not in target_ids:
            continue

        if total_providers <= K + 1:
            # 总数不足，直接输出全部
            selected_indices = np.arange(total_providers)
        else:
            dist_i = dist_matrix[i].copy()
            dist_i[i] = np.inf  # 忽略自身
            nearest_indices = np.argsort(dist_i)[:K]
            selected_indices = np.insert(nearest_indices, 0, i)  # 包含自己

        filename = os.path.join(output_dir, f"provider_{pid}_subset.txt")
        with open(filename, "w") as f:
            f.write("#ID,supply,latency,CO2,cost\n")
            for idx in selected_indices:
                f.write(f"{provider_ids[idx]},{int(supplies[idx])},{latencies[idx]},{co2s[idx]},{costs[idx]}\n")
        generated_count += 1

    print(f"✅ 为 {generated_count} 个 provider 生成了子集文件，输出目录：{output_dir}/")
    return  total_provider


if __name__ == '__main__':
    A = generate_provider_neighbors(target_ids=[0,1,2])