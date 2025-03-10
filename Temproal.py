import networkx as nx
from collections import defaultdict


class CascadeReconstructor:
    def __init__(self, edge_file="/Users/swforward/Desktop/Myself/Code/Cycle-RandomWalk/Data/twitter/edges.txt", cascade_file="/Users/swforward/Desktop/Myself/Code/Cycle-RandomWalk/Data/twitter/cascades.txt"):
        self.G = self.load_graph(edge_file)
        self.cascades = self.load_cascades(cascade_file)

    def load_graph(self, filename):
        """加载网络图并计算节点度数"""
        G = nx.Graph()
        with open(filename) as f:
            for line in f:
                node1, node2 = line.strip().split(',')
                G.add_edge(node1.strip(), node2.strip())
        # 预计算度数
        self.degrees = dict(G.degree())
        return G

    def load_cascades(self, filename):
        """加载级联数据并按时序排序"""
        cascades = []
        with open(filename) as f:
            for line in f:
                events = []
                for item in line.strip().split(','):
                    if not item: continue
                    parts = item.strip().split()
                    if len(parts) >= 2:
                        node = parts[0]
                        timestamp = float(parts[1])
                        events.append((node, timestamp))
                # 按时间戳排序
                events.sort(key=lambda x: x[1])
                cascades.append(events)
        return cascades

    def reconstruct_cascade(self, cascade_id=0):
        """重建指定级联的传播路径"""
        cascade = self.cascades[cascade_id]
        source = cascade[0][0]
        propagation_tree = {
            'source': source,
            'paths': defaultdict(list),
            'probabilities': defaultdict(dict)
        }

        # 构建时间顺序节点列表
        nodes = [node for node, _ in cascade]
        timestamps = {node: ts for node, ts in cascade}

        for i in range(1, len(cascade)):
            current_node, current_ts = cascade[i]
            possible_sources = []

            # 获取当前节点的所有邻居
            neighbors = list(self.G.neighbors(current_node)) if current_node in self.G else []

            # 筛选符合时间条件的邻居
            valid_neighbors = []
            for neighbor in neighbors:
                if neighbor in timestamps and timestamps[neighbor] < current_ts:
                    valid_neighbors.append(neighbor)

            if not valid_neighbors:
                # 如果没有合法传播源，标记为未知
                propagation_tree['paths'][current_node] = []
                continue

            # 计算概率分布
            total_degree = sum(self.degrees[n] for n in valid_neighbors)
            probabilities = {}
            for n in valid_neighbors:
                probabilities[n] = self.degrees[n] / total_degree

            # 记录结果
            propagation_tree['paths'][current_node] = valid_neighbors
            propagation_tree['probabilities'][current_node] = probabilities

        return propagation_tree

    def batch_reconstruct(self):
        """批量重建所有级联"""
        return [self.reconstruct_cascade(i) for i in range(len(self.cascades))]


# 使用示例
if __name__ == "__main__":
    reconstructor = CascadeReconstructor()

    # 重建第一个级联
    cascade_id = 3
    result = reconstructor.reconstruct_cascade(cascade_id)

    print(f"级联 {cascade_id} 的传播路径重建结果：")
    print(f"初始节点：{result['source']}")

    for node in result['paths']:
        parents = result['paths'][node]
        if not parents:
            print(f"{node} 无合法传播源")
            continue

        probabilities = result['probabilities'][node]
        print(f"\n{node} 的可能传播源：")
        for parent, prob in probabilities.items():
            print(f"  {parent}: {prob:.2%}")

        # 选择最大概率的父节点
        best_parent = max(probabilities, key=probabilities.get)
        print(f"  最可能父节点：{best_parent} ({probabilities[best_parent]:.2%})")