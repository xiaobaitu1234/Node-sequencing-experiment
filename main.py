
import pandas as pd
import csv
import networkx as nx
from os.path import join
from queue import Queue

folder = './样本数据'
dst = './dst'

file_map = {
    'Youtube-edge.csv': 'Youtube-node1.csv',
    'Wiki-Vote-edge.csv': 'wiki-node1.csv',
    'Facebook-edge.csv': 'Facebook-node.csv',
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


def init_feature(G, type='degree'):
    """为图的节点设置属性"""

    data = {}
    try:
        # 计算属性字典
        if type == 'degree':
            for k, v in G.degree():
                data[k] = v
        if type == 'closnesscentrality':
            data = nx.closeness_centrality(G)
        if type == 'eigencentrality':
            data = nx.eigenvector_centrality(G)
        if type == 'betweenesscentrality':
            data = nx.betweenness_centrality(G)
        # 为节点赋值
        for k, v in data.items():
            G.nodes[k][type] = v
    except Exception as e:
        pass


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
        print("%d nodes has delete" % s)
        init_feature(G, type)  # 重新计算属性
        rt.append(temp)
    return rt


def caculate(G):
    """计算图的剩余属性"""
    conn_components = list(nx.connected_components(G))  # 获取所有连通子图
    wgs = len(conn_components)  # 获取连通子图数量
    if conn_components:
        largest = max(conn_components, key=len)  # 计算最大连通子图
        mgs = len(largest)  # 求图中节点数量
        return {'M(G-S)': mgs, 'W(G-S)': wgs}
    else:
        return {'M(G-S)': 0, 'W(G-S)': 0}


types = {
    'degree': 'D',
    'betweenesscentrality': 'B',
    'eigencentrality': 'E',
    'closnesscentrality': 'C',
}


def main():
    """控制读取文件、节点删除"""
    for delete_type in types:
        for file in file_map:
            print("%s ---- %s" % (file, delete_type))  # 打印当前文件及类型
            g = load_graph(file)  # 载入图数据
            temp_g = nx.Graph(g)  # 构建图
            rt = remove(temp_g, delete_type)  # 删除节点并获取删除序列
            df = pd.DataFrame(rt)  # 构建dataframe，并输出csv文件
            df.reset_index(drop=True)
            print(df)
            path = join(dst, '{}-{}'.format(delete_type, file))
            df.to_csv(path)


if __name__ == '__main__':
    main()
