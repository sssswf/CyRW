import pandas as pd
import numpy as np
import networkx as nx
import random
import copy

def generate_graph(network):
    path = '/Users/swforward/Desktop/Myself/Code/Cycle-RandomWalk/Data/{}/{}.txt'.format(network,network)
    graph = nx.Graph()
    lines = pd.read_csv(path)
    for line in lines.values:
        graph.add_edge(int(line[0].split(' ')[0]), int(line[0].split(' ')[1]))
    return graph

def get_topK(df,k):
    # 计算每个中心性的前 1% 节点
    top_1_percent_nodes = {}
    for col in df.columns[1:]:  # 跳过第一列（节点名）
        # 按照中心性列排序，选取前1%的节点
        sorted_df = df.sort_values(by=col, ascending=False)
        threshold_index = int(len(sorted_df) * k)  # 前1%的节点数
        top_nodes = sorted_df.iloc[:threshold_index, 0].values  # 仅获取 Node 列的值
        top_1_percent_nodes[col] = top_nodes.tolist()  # 转换为列表并保存
    return top_1_percent_nodes

def count_node(G):
    s_num, i_num, r_num = 0, 0, 0
    for node in G:
        if G.nodes[node]['status'] == 'S':
            s_num += 1
        elif G.nodes[node]['status'] == 'I':
            i_num += 1
        else:
            r_num += 1
    return s_num, i_num, r_num

def SIR_network(graph_, source, beta, gamma, step):
    graph = copy.deepcopy(graph_)
    nodes_list = graph.nodes()  # 网络节点个数
    for i in nodes_list:
        graph.nodes[i]['status'] = 'S'  # 将所有节点的状态设置为 易感者（S）
    for j in source:
        graph.nodes[j]['status'] = 'I'  # 将感染源序列中的节点设置为感染源，状态设置为 感染者（I）
    # 记录初始状态
    sir_values = []  # 存储每一次的感染节点数
    sir_values.append(len(source) / len(nodes_list))
    for t in range(step):
        # 记录当前时间步被感染的节点
        newly_infected = []
        # 记录当前时间步即将恢复的节点
        newly_recovered = []
        # 针对每个节点进行状态更新
        for node in nodes_list:
            if graph.nodes[node]['status'] == 'I':
                for neighbor in list(graph.neighbors(node)):
                    if graph.nodes[neighbor]['status'] == 'S' and random.random() < beta:
                        newly_infected.append(neighbor)  # 记录被感染的节点
                if random.random() < gamma:
                    newly_recovered.append(node)  # 记录将要恢复的感染者
        # 在下一时间步更新被感染的节点
        for node in newly_infected:
            graph.nodes[node]['status'] = 'I'
        # 在下一时间步更新将要恢复的节点
        for node in newly_recovered:
            graph.nodes[node]['status'] = 'R'
        # 统计当前状态下的S、I、R数量
        s, i, r = count_node(graph)
        sir = (r + i) / len(nodes_list)  # 计算当前时间步的SIR值
        sir_values.append(sir)  # 将当前SIR值加入结果数组
    return sir_values

# 临界感染率
def infected_beta(graph):
    degree = nx.degree(graph)
    degree_list = []
    degree_sq_list = []
    for i in degree:
        degree_list.append(i[1])
        degree_sq_list.append(i[1] * i[1])
    degree_avg = np.mean(degree_list)
    degree_avg_sq = np.mean(degree_sq_list)
    infected = degree_avg / (degree_avg_sq - degree_avg)
    return infected
