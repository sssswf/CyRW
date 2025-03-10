import copy
import random
import networkx as nx
from collections import defaultdict


# 找到网络中的所有基本圈list
def cycle_list(G):
    basic_cycle_list = nx.cycle_basis(G)
    cycle_list_5 = []
    for cycle in basic_cycle_list:
        if len(cycle) <= 5:
            cycle_list_5.append(cycle)
    return cycle_list_5


def updateNetworkState_HO(G, beta, beta_delta1, beta_delta2, gamma, cumulative_infected, now_infected, recovered, cycle_dic):
    copy_infected = set(now_infected)
    new_infected = set()
    remove_infected = set()
    be_simulate = set(G.nodes) - recovered - copy_infected
    for node in be_simulate:  # 遍历图中处于S态的节点，每一个节点状态进行更新
        # 先算圈的增强效应
        cycle_order = 0
        if len(cycle_dic[node]) != 0:
            for cycle in cycle_dic[node]:
                is_activate = 1
                for node_in_it in cycle:
                    if node_in_it != node and node_in_it not in now_infected:
                        is_activate = 0
                cycle_order += is_activate  # 最终有几个圈就是几阶

        # 再算共圈节点的效应
        co_cycle_order = 0
        if len(cycle_dic[node]) != 0:
            for cycle in cycle_dic[node]:
                for node_in_it in cycle:
                    if node_in_it != node and node_in_it not in G.neighbors(node) and node_in_it in now_infected:
                        co_cycle_order += 1

        # 再算邻居的效应
        pair_order = 0
        for neighbor_node in set(G.neighbors(node)):
            if neighbor_node in now_infected:
                pair_order += 1


        beta_sum = 1 - ((1 - beta) ** pair_order) * ((1 - beta_delta1) ** cycle_order)* ((1 - beta_delta2) ** co_cycle_order)

        if random.random() < beta_sum:
            new_infected.add(node)

    for node in copy_infected:
        if random.random() < gamma:
            remove_infected.add(node)
            now_infected.remove(node)

    now_infected.extend(new_infected)
    recovered |= remove_infected  # 两个set不重复拼接
    cumulative_infected.extend(new_infected)


def get_cyc_neighbors_dict(cycles_list):
    cyc_neighbors_dict = defaultdict(list)
    for cycle in cycles_list:
        for i in range(len(cycle)):
            node = cycle[i]
            neighbors = [cycle[j] for j in range(len(cycle)) if cycle[i] != cycle[j]]
            cyc_neighbors_dict[node].append(tuple(neighbors))  # 使用 tuple 保证元素的不可变性和有序性
    return cyc_neighbors_dict


def cycle_SIR(G, step, beta, beta_delta1, beta_delta2, gamma, initial_infected):
    now_I = copy.deepcopy(initial_infected)
    cumulative_I = copy.deepcopy(initial_infected)
    I = []
    Recovered = []
    R = set()
    cyc_neighbors_dict = get_cyc_neighbors_dict(cycle_list(G))
    for t in range(0, step):
        updateNetworkState_HO(G, beta, beta_delta1, beta_delta2, gamma, cumulative_I, now_I, R, cyc_neighbors_dict)  # 对网络状态进行模拟更新
        I.append(len(cumulative_I)/len(G.nodes()))
        Recovered.append(len(R)/len(G.nodes()))
    return I
