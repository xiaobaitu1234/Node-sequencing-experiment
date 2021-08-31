
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
    init_feature(G, file_name)
    return G


def load_nodes(file_name):
    file = join(folder, file_name)
    with open(file, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        nodes = list(reader)
    return nodes


def init_feature(G, file_name):
    """为图的节点设置属性"""
    node_file = file_map[file_name]
    nodes = load_nodes(node_file)
    for info in nodes:
        id = info.pop('Id')
        G.nodes[id].update(info)


def remove(G, type='degree'):
    """执行删除操作，并记录值"""
    node_num = len(G.nodes)  # 节点总数
    label = '{}({})'.format(node_num,types[type])
    x = [int(0.01 * x * node_num) for x in range(1, 101)]  # 每次移除总数
    s_q = Queue()
    for item in x:
        s_q.put(item)

    def sort(item):
        k, v = item
        return float(v[type])
    sequence = sorted(G.nodes(data=True), key=sort, reverse=True)
    # 构建队列
    q = Queue()
    for k, v in sequence:
        q.put(k)
    flag = True
    # 使用队列
    rt = []
    counter = 0
    while flag:
        s = s_q.get()
        if not s_q.qsize():
            break
        while True:
            # 每次删除一批节点
            if not q.empty():
                node = q.get()
                G.remove_node(node)
                counter += 1
                if counter == s:
                    break
            else:
                flag = False
                break
        # 计算删除后状态
        temp = {label: round(s / node_num, 2),'S': s}
        temp.update(caculate(G))
        rt.append(temp)
    return rt


def caculate(G):
    asf = list(nx.connected_components(G))
    last_num = len(G.nodes())
    wgs = len(asf)
    if asf:
        largest = max(asf, key=len)
        mgs = len(largest)
        return {'M(G-S)': mgs, 'W(G-S)': wgs}
    else:
        return {'M(G-S)': 0, 'W(G-S)': 0}


types = {
    'degree':'D',
    'eigencentrality':'E',
    'closnesscentrality':'C',
    'betweenesscentrality':'B'
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
            path = join(dst,'{}-{}'.format(type,file))
            df.to_csv(path)


if __name__ == '__main__':
    main()
