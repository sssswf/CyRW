import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd
import csv

def calculate_network_new(G: nx.Graph):
    # 获取图中的基本圈
    cycles = nx.cycle_basis(G)
    node_cycle_count = {node: 0 for node in G.nodes}
    for cycle in cycles:
        for node in cycle:
            node_cycle_count[node] += 1
    # 计算每个基本圈的平均圈数
    cycle_average_counts = {}
    for cycle in cycles:
        cycle_node_counts = [node_cycle_count[node] for node in cycle]
        cycle_average_count = sum(cycle_node_counts) / len(cycle)
        cycle_average_counts[tuple(cycle)] = cycle_average_count
    # 创建一个新的图来存储新生成的节点和边
    pairwise_graph = nx.Graph()
    # 添加原始图的节点
    for node_i in G.nodes:
        pairwise_graph.add_node(node_i)
    # 创建一个字典保存圈名和对应的基本圈
    cycle_dict = {}
    temp = len(G.nodes) + 1
    for cycle in cycles:
        cycle_dict[str(temp)] = cycle
        temp += 1
    # 添加新的圈节点
    for k, v in cycle_dict.items():
        pairwise_graph.add_node(k)
    # 创建连边并计算权重
    for k, v in cycle_dict.items():
        cycle_average_count = cycle_average_counts[tuple(v)]  # 获取当前圈的平均圈数
        for node_i in v:
            # 计算当前节点参与的基本圈数
            node_cycle_count_value = node_cycle_count[node_i]
            # 计算权重
            weight = (node_cycle_count_value * cycle_average_count) / (len(v) ** 2)
            # 将边添加到新的图中并设置权重
            pairwise_graph.add_edge(k, node_i, weight=weight)
    # 创建一个临时列表存储新节点的ID
    temp_list = list(range(len(G.nodes) + 1, temp))
    # 存储原始图的节点
    node_list = list(G.nodes)

    return pairwise_graph, temp_list, node_list

def compute_cycle_vectors(G):
    # 获取基本圈，圈的每个元素是节点而不是边
    basic_cycles = nx.cycle_basis(G)
    # 获取图中的所有边
    edges = list(G.edges())
    # 创建一个字典，将边映射到一个索引，便于后续操作
    edge_index = {tuple(sorted(edge)): i for i, edge in enumerate(edges)}  # 使用tuple(sorted(edge))确保顺序一致
    # 创建圈向量矩阵，每个基本圈对应一行
    cycle_vectors = np.zeros((len(basic_cycles), len(edges)), dtype=int)
    # 填充每个基本圈的圈向量
    for i, cycle in enumerate(basic_cycles):
        for j in range(len(cycle)):
            # 获取该圈中相邻节点对的边
            edge = (cycle[j], cycle[(j + 1) % len(cycle)])  # 获取循环中的边
            sorted_edge = tuple(sorted(edge))  # 对边进行排序以统一顺序
            edge_idx = edge_index[sorted_edge]  # 查找边的索引
            cycle_vectors[i, edge_idx] = 1  # 填充对应位置为1

    return cycle_vectors, basic_cycles, edges


def compute_cycle_similarity(cycle_vectors, basic_cycles, temp_list):
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(cycle_vectors)
    # 计算圈长度相似性
    num_cycles = len(basic_cycles)
    length_similarity_matrix = np.zeros((num_cycles, num_cycles))
    for i in range(num_cycles):
        for j in range(i, num_cycles):
            len_i = len(basic_cycles[i])
            len_j = len(basic_cycles[j])
            length_similarity = 1 / (1 + abs(len_i - len_j))
            length_similarity_matrix[i, j] = length_similarity
            length_similarity_matrix[j, i] = length_similarity
    # 结合余弦相似度和长度相似度
    alpha = 0.9  # 权重参数
    combined_similarity = alpha * similarity_matrix + (1 - alpha) * length_similarity_matrix
    combined_similarity_df = pd.DataFrame(combined_similarity, index=temp_list, columns=temp_list)
    return combined_similarity_df

def creat_cycle_graph(combined_similarity_df):
    similarity_matrix = combined_similarity_df.values
    mst = minimum_spanning_tree(-similarity_matrix)
    mst = mst.toarray()  # 转换为密集数组
    # 构建 1阶圈图
    cycle_graph = nx.Graph()
    cycle_names = combined_similarity_df.index
    # 添加1阶圈图的节点
    for cycle in cycle_names:
        cycle_graph.add_node(cycle)
    # 添加1阶圈图的边 (基于最小生成树)
    for i in range(len(cycle_names)):
        for j in range(i + 1, len(cycle_names)):
            if mst[i, j] != 0:
                cycle_graph.add_edge(cycle_names[i], cycle_names[j])
    # 添加弱连接（选择相似度最低但非0的1%连接）
    for i in range(len(cycle_names)):
        similarities = similarity_matrix[i]
        non_zero_similarities = [(j, similarities[j]) for j in range(len(similarities))
                                 if similarities[j] > 0 and mst[i, j] == 0 and i != j]
        non_zero_similarities.sort(key=lambda x: x[1])
        num_to_add = max(1, int(0.01 * len(non_zero_similarities)))
        for j, sim in non_zero_similarities[:num_to_add]:
            cycle_graph.add_edge(cycle_names[i], cycle_names[j])

    return cycle_graph

def random_matrix(G: nx.Graph):
    z_matrix = nx.adjacency_matrix(G)
    node_num = z_matrix.shape[0]
    original_matrix = np.zeros((node_num, node_num))
    new_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            original_matrix[i][j] = z_matrix[i, j]
    for i in range(node_num):
        for j in range(node_num):
            if original_matrix[i][j] != 0.0:
                new_matrix[i][j] = 1.0 / (np.sum(original_matrix[i]))
    return new_matrix.transpose(), original_matrix


def hyper_random_matrix_new(graph: nx.Graph, cycle_graph):
    b_graph, temp_list, node_list = calculate_network_new(graph)
    m = len(temp_list)  # 1阶圈图的节点数
    n = len(node_list)  # 原始网络的节点数
    # 初始化邻接矩阵 a_matrix
    a_matrix = np.zeros((m, n))
    # 定义 a 矩阵，即高阶二部图的加权邻接矩阵
    for i in range(m):
        for j in range(n):
            # 如果边存在，从 b_graph 中提取边的权重
            if b_graph.has_edge(temp_list[i], node_list[j]):
                # 提取权重（计算时已考虑原始节点的度数和圈的平均度数）
                weight = b_graph[temp_list[i], node_list[j]]['weight']
                a_matrix[i][j] = weight
    # 处理全零列的情况
    for j in range(n):
        if np.all(a_matrix[:, j] == 0):  # 检查列是否全为 0
            # 将该列填充为均匀分布（随机跳转到每一个圈节点）
            a_matrix[:, j] = 1 / m  # 均匀地分配权重，跳转到每一个圈节点
    row_sum = np.zeros(m)
    col_sum = np.zeros(n)

    for i in range(m):
        row_sum[i] = 1 / np.sum(a_matrix[i])

    for j in range(n):
        col_sum_j = np.sum(a_matrix[:, j])
        if col_sum_j != 0:
            col_sum[j] = 1 / col_sum_j
        else:
            col_sum[j] = 0
    # 计算 v_matrix 和 u_matrix
    u_matrix = a_matrix @ np.diag(col_sum)
    v_matrix = a_matrix.transpose() @ np.diag(row_sum)
    cycle_r_matrix, origin_matrix = random_matrix(cycle_graph)
    # 计算最终的权重矩阵
    w_matrix = np.dot(np.dot(v_matrix, cycle_r_matrix), u_matrix)
    return w_matrix


def compute_mix_matrix(c_matrix: np.ndarray, w_matrix: np.ndarray):
    c = c_matrix * w_matrix
    return c

def mix_walk(origin_matrix: np.ndarray, c_matrix: np.ndarray, t: float):
    rw = c_matrix @ origin_matrix
    while True:
        print(f'rw= {rw}')
        rw_2 = c_matrix @ rw
        print(f'rw_2= {rw_2}')
        error_matrix = np.array(rw_2 - rw, dtype=np.float32)
        error = np.linalg.norm(error_matrix, ord=2)
        # error = spectral_radius(error_matrix)
        print(f'error= {error}')
        if error < t:
            return rw
        rw = rw_2

def generate_graph(network):
    path = '/Users/swforward/Desktop/Myself/Code/Cycle-RandomWalk/Data/{}/{}.txt'.format(network,network)
    graph = nx.Graph()
    lines = pd.read_csv(path)
    for line in lines.values:
        graph.add_edge(int(line[0].split(' ')[0]), int(line[0].split(' ')[1]))
    return graph

if __name__ == '__main__':
    NetWorks = ['ca-Erdos']
    for network in NetWorks:
        print(network)
        graph = generate_graph(network)
        G = nx.Graph()
        G.add_edges_from(graph.edges())
        # 去掉所有自环
        G.remove_edges_from([(node, node) for node in G.nodes if G.has_edge(node, node)])
        G = nx.relabel_nodes(G, {node: idx for idx, node in enumerate(G.nodes())})
        b_graph, temp_list, node = calculate_network_new(G)

        # 计算圈向量矩阵
        cycle_vectors, basic_cycles, edges = compute_cycle_vectors(G)
        # 计算圈相似性矩阵
        similarity_matrix = compute_cycle_similarity(cycle_vectors, basic_cycles,temp_list)
        # 基于圈相似性矩阵创建圈图
        cycle_graph = creat_cycle_graph(similarity_matrix)

        # 计算原始网络和圈图网络的随机游走矩阵
        G_new,G_old = random_matrix(G)
        cycle_new,cycle_old = random_matrix(cycle_graph)
        w_matrix = hyper_random_matrix_new(G, cycle_graph)
        list_cycle_dict = []
        mix_matrix = compute_mix_matrix(G_new, w_matrix)
        test_array = np.ones(len(G.nodes)).transpose()
        for i in range(len(G.nodes)):
            test_array[i] = test_array[i] / len(G.nodes)
        mix3_walk_matrix = mix_walk(test_array, mix_matrix, 0.001)
        mix3_list = []
        for i in range(mix3_walk_matrix.shape[0]):
            mix3_list.append(np.sum(mix3_walk_matrix[i]))
        mix3_dic = {}
        for i in range(mix3_walk_matrix.shape[0]):
            mix3_dic[list(G.nodes.keys())[i]] = round(mix3_list[i] / np.sum(mix3_list), 6)
        list_cycle_dict.append(mix3_dic)



        f = open('./Code/Cycle-RandomWalk/Data/{}/{}_CyRW.csv'.format(network,network), 'w')
        writer = csv.writer(f)
        header = ['Node', 'CyRW']
        writer.writerow(header)
        print("Write Data----------------------")
        for key, value in list_cycle_dict[0].items():
            value01 = list_cycle_dict[0][key]
            writer.writerow([key, value01])
        f.close()
        print('{}保存成功'.format(network))
