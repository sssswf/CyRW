import pandas as pd
import numpy as np
import networkx as nx
from SIR_model import generate_graph,get_topK,SIR_network,infected_beta
if __name__ == '__main__':
    NetWorks = ['ca-Erdos']
    for network in NetWorks:
        print(network)
        df = pd.read_csv('../Code/Cycle-RandomWalk/Data/{}/{}_CyRW.csv'.format(network, network))
        # 创建一个图，假设是从某些边列表中构建的
        graph = generate_graph(network)
        G = nx.Graph()
        G.add_edges_from(graph.edges())
        # 去掉所有自环
        G.remove_edges_from([(node, node) for node in G.nodes if G.has_edge(node, node)])
        topK_dict = get_topK(df, 0.01)
        print(topK_dict)
        y_all = []
        index = ['CyRW']
        gamma = 1  # 免疫率
        step = 50  # 迭代次数
        beta_value = infected_beta(G)  # 感染率
        for i in index:
            print(i)
            y2_all = []
            seed_node = topK_dict[i]
            print(seed_node)
            for beta in [1.5 * beta_value, 2 * beta_value, 2.5* beta_value, 3 * beta_value]:
                ex_num = 1000
                sir_values_list = []
                for j in range(0, ex_num):
                    if j % 1000 == 0:
                        print('已进行{}次实验'.format(j))
                    sir_source = seed_node  # df转为数组
                    sir_values = SIR_network(G, sir_source, beta, gamma, step)  # sir传播
                    sir_values_list.append(sir_values[step - 1])  # 存储每个方法的Sir传播情况
                y2_all.append(np.mean(sir_values_list))
                print(y2_all)
            y_all.append(y2_all)
        df = pd.DataFrame(y_all)
        df2 = pd.DataFrame(df.values.T)
        df2.to_csv("../Code/Cycle-RandomWalk/Results/{}/{}_results_beta.csv".format(network,network), index=None,header=['CyRW'])
        print("保存成功")