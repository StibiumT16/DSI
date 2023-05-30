import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--v_dim', type=int, default=768)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--k', type=int, default= 10)
parser.add_argument('--c', type=int, default= 100)
args = parser.parse_args()

old_id, X = [], []
with open(args.input_path, 'r') as fr:
    for i, line in enumerate(tqdm(fr)):
        did, vec = line.strip().split('\t')
        old_id.append(did)
        vec = vec.split(',')
        assert len(vec) == args.v_dim
        X.append([float(v) for v in vec])
        
X = np.array(X)

kmeans = KMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=args.seed, tol=1e-7)

mini_kmeans = MiniBatchKMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=3,
                              batch_size=1000, reassignment_ratio=0.01, max_no_improvement=20, tol=1e-7)


def classify_recursion(x_data_pos):
    if x_data_pos.shape[0] <= args.c:
        if x_data_pos.shape[0] == 1:
            return
        for idx, pos in enumerate(x_data_pos):
            new_id_list[pos].append(idx)
        return

    temp_data = np.zeros((x_data_pos.shape[0], args.v_dim))
    for idx, pos in enumerate(x_data_pos):
        temp_data[idx, :] = X[pos]

    if x_data_pos.shape[0] >= 1e3:
        pred = mini_kmeans.fit_predict(temp_data)
    else:
        pred = kmeans.fit_predict(temp_data)

    for i in range(args.k):
        ids, = np.where(pred == i)
        pos_lists = [x_data_pos[id_] for id_ in ids]
        for pos in pos_lists:
            new_id_list[pos].append(i)
        classify_recursion(np.array(pos_lists))

    return

print('Start First Clustering')
pred = mini_kmeans.fit_predict(X)

new_id_list = [[class_] for class_ in pred]

print('Start Recursively Clustering...')
for i in tqdm(range(args.k)):
    pos_lists, = np.where(pred == i)
    classify_recursion(pos_lists)


assert len(old_id) == len(new_id_list)

with open(args.output_path, 'w') as fw:
    for did, docid in zip(old_id, new_id_list):
        its = [str(id) for id in docid] 
        semantic_docid = " ".join(its)
        fw.write(did + '\t' + semantic_docid + '\n')

