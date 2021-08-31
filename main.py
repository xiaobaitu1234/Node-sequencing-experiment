
import pandas as pd
import csv
import networkx as nx
from os.path import join
from queue import Queue

folder = './样本数据'
dst = './dst'

file_map = {
    'Youtube-edge.csv': 'Youtube-node1.csv',
    # 'Wiki-Vote-edge.csv': 'wiki-node1.csv',
    # 'Facebook-edge.csv': 'Facebook-node.csv',
}


def load_edges(file_name):
    """加载数据，构造边列表"""
    edges = []
    with open(file_name, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for item in reader:
            edges.append((item['Source'], item['Target']))
    return edges


def load_graph(file_name):
    """构造Graph"""
    file = join(folder, file_name)
    edges = load_edges(file)
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def load_nodes(file_name):
    file = join(folder, file_name)
    with open(file, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        nodes = list(reader)
    return nodes


def init_feature(G, type='degree'):
    """为图的节点设置属性"""

    data = {}
    if type == 'degree':
        for k, v in G.degree():
            data[k] = v
    if type == 'closnesscentrality':
        data = nx.closeness_centrality(G)
    if type == 'eigencentrality':
        data = nx.eigenvector_centrality(G)
    if type == 'betweenesscentrality':
        data = nx.betweenness_centrality(G)
    for k, v in data.items():
        G.nodes[k][type] = v


def remove(G, type='degree'):
    """执行删除操作，并记录值"""
    node_num = len(G.nodes)  # 节点总数
    label = '{}({})'.format(node_num, types[type])
    x = [int(0.01 * x * node_num) for x in range(1, 101)]  # 每次移除总数

    print("all num is %d" % node_num)
    s_q = Queue()
    for item in x:
        s_q.put(item)
    init_feature(G, type)  # 第一次初始化属性
    counter = 0  # 删除计数器
    rt = []  # 存储返回值
    while len(G.nodes()):
        # 根据属性排序，构造队列q
        def sort(item):
            k, v = item
            return float(v[type])
        sequence = sorted(G.nodes(data=True), key=sort, reverse=True)
        # 构建队列
        q = Queue()
        for k, v in sequence:
            q.put(k)
        # 使用队列 q，每次取最前边的几个
        s = s_q.get()  # 获取本次要删除到的总数
        while True:  # 按顺序删除节点，直到删除数达到s
            # 每次删除一批节点
            if not q.empty():
                node = q.get()
                G.remove_node(node)
                counter += 1
                if counter >= s:
                    break
            else:
                break
        # 计算删除后状态
        temp = {label: round(s / node_num, 2), 'S': s}
        temp.update(caculate(G))
        init_feature(G, type)  # 重新计算属性
        rt.append(temp)
    return rt


def caculate(G):
    """计算图的剩余属性"""
    asf = list(nx.connected_components(G))
    wgs = len(asf)
    if asf:
        largest = max(asf, key=len)
        mgs = len(largest)
        return {'M(G-S)': mgs, 'W(G-S)': wgs}
    else:
        return {'M(G-S)': 0, 'W(G-S)': 0}


types = {
    'degree': 'D',
    # 'eigencentrality': 'E',
    # 'closnesscentrality': 'C',
    # 'betweenesscentrality': 'B'
}


def main():
    for file in file_map:
        g = load_graph(file)
        for type in types:
            temp_g = nx.Graph(g)
            rt = remove(temp_g, type)
            df = pd.DataFrame(rt)
            df.reset_index(drop=True)
            print(df)
            path = join(dst, '{}-{}'.format(type, file))
            df.to_csv(path)


if __name__ == '__main__':
    main()
