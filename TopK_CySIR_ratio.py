import pandas as pd
import numpy as np
import networkx as nx
from cycle_SIR import cycle_SIR
from SIR_model import generate_graph,get_topK,infected_beta
if __name__ == '__main__':
    NetWorks = ['ca-Erdos']
    for network in NetWorks:
        df = pd.read_csv('../Code/Cycle-RandomWalk/Data/{}/{}_CyRW.csv'.format(network, network))
        graph = generate_graph(network)
        G = nx.Graph()
        G.add_edges_from(graph.edges())
        # 去掉所有自环
        G.remove_edges_from([(node, node) for node in G.nodes if G.has_edge(node, node)])
        ratio_list = [0.01, 0.02, 0.03]
        y_all = []
        beta_value = 1.5*infected_beta(G)
        index = ['CyRW']
        for i in index:
            print(i)
            y2_all = []
            for ratio in ratio_list:
                topK_dict = get_topK(df, ratio)
                seed_node = topK_dict[i]
                ex_num = 1000
                sir_results = []
                for j in range(0, ex_num):
                    I_list = cycle_SIR(G,step=50,beta=beta_value,beta_delta1=0.9*beta_value,beta_delta2=0.9*beta_value,gamma=1,initial_infected=seed_node)
                    sir_results.append(I_list[-1])
                y2_all.append(np.mean(sir_results))
                print(y2_all)
            y_all.append(y2_all)
        df = pd.DataFrame(y_all)
        df2 = pd.DataFrame(df.values.T)
        df2.to_csv("../Code/Cycle-RandomWalk/Results/{}/{}_results_CySIR_ratio.csv".format(network,network), index=None,header=['CyRW'])
        print("保存成功")


