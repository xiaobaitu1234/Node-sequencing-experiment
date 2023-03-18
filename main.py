from __future__ import annotations
from itertools import product

from typing import List, Tuple
import pandas as pd
import csv
import networkx as nx
from os.path import join
from queue import Queue

src_folder = './样本数据'
dst_folder = './dst1'

file_map = {
    'Youtube-edge.csv': 'Youtube-node1.csv',
    'Wiki-Vote-edge.csv': 'wiki-node1.csv',
    'Facebook-edge.csv': 'Facebook-node.csv',
}


def load_edges(
        file_name, direction=True) -> List[Tuple[str | int, str | int]]:
    """
    加载数据，构造边列表

    :param str file_name: 文件名称，要求以Source，Target为列名的csv文件
    :param bool direction: 是否有方向性，若无，则删除重复边。需要进一步打磨
    :return: edge_list
    """
    edges = []
    with open(file_name, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for item in reader:
            edges.append((item['Source'], item['Target']))
    if not direction:
        edges = list(set(edges))
    return edges


def load_graph(file_name) -> nx.Graph:
    """
    构造Graph

    :param str file_name: 文件名称，要求以Source，Target为列名的csv文件
    :return:
    """
    file = join(src_folder, file_name)  # 构造路径
    edges = load_edges(file)  # 加载edge列表
    G = nx.Graph()  # 初始化图
    G.add_edges_from(edges)  # 将边加入图
    return G


def init_feature_value(graph, init_type=None):
    """
    为图的节点设置属性

    :param nx.Graph graph: 待赋值的图
    :param str init_type: [degree, closnesscentrality, eigencentrality, betweenesscentrality]
    :return:
    """
    if init_type is None:
        init_type = "degree"

    data = {}
    try:
        # 计算属性字典
        if init_type == 'degree':
            for k, v in graph.degree():
                data[k] = v
        elif init_type == 'closnesscentrality':
            data = nx.closeness_centrality(graph)
        elif init_type == 'eigencentrality':
            data = nx.eigenvector_centrality(graph)
        elif init_type == 'betweenesscentrality':
            data = nx.betweenness_centrality(graph)
        # 为节点赋值
        for k, v in data.items():
            graph.nodes[k][init_type] = v
    except Exception:
        pass


def remove(graph, calculate_type=None):
    """
    执行删除操作，并记录值

    :param nx.Graph graph: 待迭代删除的图
    :param str calculate_type: 删除依据
    :return:
    """

    if calculate_type is None:
        calculate_type = "degree"

    node_num = len(graph.nodes)  # 节点总数
    col_label = '{}({})'.format(node_num, types[calculate_type])  # 列标记
    remove_num_each_sheaves = [int(0.01 * sheaves * node_num)
                               for sheaves in range(1, 101)]  # 每次移除总数

    print("all num is %d" % node_num)
    queue_delete_num = Queue()  # 记录每一轮删除，需要剩余的节点数量
    for item in remove_num_each_sheaves:
        queue_delete_num.put(item)
    init_feature_value(graph, calculate_type)  # 第一次初始化属性

    cnt = 0  # 删除计数器
    ret_list = []  # 存储返回值
    while len(graph.nodes()):
        # 根据属性排序，构造队列q
        def delete_cmp_role(node_sample):
            """
            指定删除规则

            :param node_sample:
            :return:
            """
            _, value = node_sample
            return float(value[calculate_type])

        """根据属性排序并将node写入队列"""
        sequence = sorted(
            graph.nodes(
                data=True),
            key=delete_cmp_role,
            reverse=True)
        queue_node_sorted = Queue()
        for node_nums, _ in sequence:
            queue_node_sorted.put(node_nums)

        """开始一轮的删除"""
        need_delete = queue_delete_num.get()  # 获取本次要删除到的总数
        while not queue_node_sorted.empty():  # 按顺序删除节点，直到删除数达到 need_delete
            node = queue_node_sorted.get()
            graph.remove_node(node)
            cnt += 1
            if cnt >= need_delete:
                break

        # 计算删除后状态
        row = {col_label: round(need_delete / node_num, 2), 'S': need_delete}
        row.update(calculate_remain_graph(graph))
        print("%d nodes has delete" % need_delete)
        init_feature_value(graph, calculate_type)  # 重新计算属性
        ret_list.append(row)
    return ret_list


def delete_nodes(): ...


def calculate_remain_graph(graph) -> dict:
    """
    计算图的剩余属性
    M(G-S): 最大连通子图的节点数
    W(G-S): 连通子图数量

    :param nx.Graph graph: 待计算的图
    :return:
    """
    conn_components = list(nx.connected_components(graph))  # 获取所有连通子图
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


def batch_remove():
    """
    控制读取文件、节点删除

    :return:
    """
    for file, delete_type in product(file_map.keys(), types):
        print("%s ---- %s" % (file, delete_type))  # 打印当前文件及类型
        graph = load_graph(file)  # 载入图数据
        print("{} 图数据加载完成")
        rt = remove(graph, delete_type)  # 删除节点并获取删除序列
        """将删除序列构建dataframe，并输出csv文件"""
        df = pd.DataFrame(rt)
        df.reset_index(drop=True)
        print(df)
        """输出删除结果到文件"""
        path = join(dst_folder, '{}-{}'.format(delete_type, file))
        df.to_csv(path)


if __name__ == '__main__':
    batch_remove()
