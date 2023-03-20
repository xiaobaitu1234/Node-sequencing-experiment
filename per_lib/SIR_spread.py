# -*- coding: utf-8 -*-

"""

这个文件很NB，读取图，对每个点跑SIR模型100次计算平均传播范围

然后，使用之前计算的每个点的多个值（AIT，Kshell，degree， between....），都进来，把SIR的平均传播加进去


"""

import csv
import os
import sys
import time
from collections import defaultdict
from itertools import count

import networkx as nx
import pandas as pd

from lib.sir.sir import SIRController

sys.path.append('../')

verbose = True  # 是否显示详情


def read_graph(edge_list_path: str):
    """
    从文件中读取图 grid.edgelist

    :param edge_list_path:
    :return:
    """
    graph = nx.read_edgelist(path=edge_list_path, data=True)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


class FeatureCalculate(object):
    """
    该类用于对图进行处理，得出每个节点的AIT值
    """

    def __init__(self, graph):
        """
        初始化图

        :param graph:
        """
        self.graph = graph
        self.DOUBLE_DEGREE = "DoubleDegree"  # 定义常量，为结点设置属性名用
        self.DEGREE = "Degree"  # 定义常量，为结点设置属性名用
        self.DOUBLE_DEGREE_DYNAMIC = "DoubleDegreeDynamic"  # 定义常量，为结点设置属性名用
        self.DEGREE_DYNAMIC = "DegreeDynamic"  # 定义常量，为结点设置属性名用
        self.A_IT_M = "A_it^m"
        self.init_Degree_Double_degree()  # 初始化每个节点度值和重读值，只调用一次，用于之后计算

    def init_Degree_Double_degree(self):
        """
        为每个结点设置度值和二重度

        :return:
        """
        graph = self.graph  # 存储引用，便于调用
        doubleDegree = self.DOUBLE_DEGREE
        degree = self.DEGREE
        # 以上三个变量是为了访问方便

        for node in graph.nodes:
            # 设置每个结点的度值
            graph.nodes[node][degree] = graph.degree(node)
        for node in graph.nodes:
            # 设置每个结点的重度值
            count = graph.nodes[node][degree]
            for nbr in graph.neighbors(node):
                # 遍历结点的邻居节点，累加度值
                count += graph.nodes[nbr][degree]
            graph.nodes[node][doubleDegree] = count

    def initDoubleDegreeDynamic(self):
        """
        计算当前节点的剩余影响力，也即，当前的重度值

        :return:
        """
        label = self.DOUBLE_DEGREE_DYNAMIC
        degree = self.DEGREE_DYNAMIC
        graph = self.graph  # 存储引用，便于调用

        for node in graph.nodes:
            # 设置每个结点的度值
            graph.nodes[node][degree] = graph.degree(node)
        for node in graph.nodes:
            # 设置每个结点的重度值
            count = graph.nodes[node][degree]
            for nbr in graph.neighbors(node):
                # 遍历结点的邻居节点，累加度值
                count += graph.nodes[nbr][degree]
            graph.nodes[node][label] = count

    def init_label(self, label_name):
        """
            为每个节点重设label的值
        """
        label = label_name
        graph = self.graph  # 存储引用，便于调用
        Ait = self.DOUBLE_DEGREE_DYNAMIC
        Ai0 = self.DOUBLE_DEGREE
        Kit = self.DEGREE_DYNAMIC
        Ki0 = self.DEGREE

        # 为计算，需要先重设结点的动态重度值，也即剩余影响力指数
        self.initDoubleDegreeDynamic()
        for node in graph.nodes:
            if graph.nodes[node][Ki0] == 0:
                print("zer0")
            graph.nodes[node][label] = graph.nodes[node][Ait] + (
                1 - graph.nodes[node][Kit] / graph.nodes[node][Ki0]) * (
                    graph.nodes[node][Ai0] - graph.nodes[node][Ait])
            # graph.nodes[node][label] = graph.nodes[node][
            #     Ai0] + graph.nodes[node][Kit] / graph.nodes[node][Ki0] * (
            #           graph.nodes[node][Ait]-graph.nodes[node][Ai0])

    def k_shell_degree(self):
        """
        返回当前对象持有图对应的K-shell分解字典

        :return:
        """
        graph = self.graph  # 存储引用，便于调用
        importance_dict = defaultdict(list)
        dic = nx.core_number(graph)  # 获取k-shell值
        for key, value in dic.items():
            importance_dict[value].append(key)
        return dict(importance_dict)  # 退出外层循环说明全部计算完毕，返回字典

    def k_shell_self(self):
        """
        返回当前对象持有图对应的自定义Label进行K-shell分解字典

        :return:
        """
        copy = self.graph.copy()
        graph = self.graph  # 存储引用，便于调用
        label = self.A_IT_M
        importance_dict = {}  # 存储返回字典
        '''
            由于存在初始读数为0的节点，在这里归为0曾
        '''
        level = 0
        level_node_list = []  # 存储临时结点集
        degree = list(graph.degree)
        for item in degree:
            if item[1] == 0:
                level_node_list.append(item[0])
        if len(level_node_list) > 0:
            importance_dict[level] = []
            importance_dict[level].extend(level_node_list)  # 将记录的结点加入返回字典
            graph.remove_nodes_from(level_node_list)  # 从图中删去已经记录的结点
        '''
            以下是正常操作
        '''
        level = 1  # 初始化 ks = 1
        self.init_label(label)  # 为每个结点设置重度值，用于接下来的ks分解
        while len(graph.nodes) > 0:  # 即存在结点，便计算
            # 新增加的判定条件，存在节点并且不存在边，则将所有剩余节点加入
            if len(graph.edges) == 0:
                level_node_list = []  # 存储临时结点集
                for node in graph.nodes:
                    level_node_list.append(node)
                importance_dict[level] = []
                importance_dict[level].extend(level_node_list)  # 将记录的结点加入返回字典
                break
            importance_dict[level] = []
            min_double_degree = 0
            while True:
                level_node_list = []  # 存储临时结点集
                for node in graph.nodes:  # 取每个结点进行判定
                    if graph.nodes[node][
                            label] <= level:  # 判定当前结点的重度值是否满足小于等于ks
                        level_node_list.append(node)  # 将满足条件的结点加入临时列表
                graph.remove_nodes_from(level_node_list)  # 从图中删去已经记录的结点
                self.init_label(label)  # 由于上一步删除了结点，每次删除要进行重新计算
                importance_dict[level].extend(level_node_list)  # 将记录的结点加入返回字典
                if not len(graph.nodes):  # 若不满足计算条件，则直接返回
                    self.graph = copy
                    return importance_dict
                '''
                    接下来要判定当前 ks = level 的结点是否全部被记录，
                    即，判定剩余节点的label值是否均大于level
                '''
                min_double_degree = min(
                    [graph.nodes[node][label] for node in graph.nodes])
                if min_double_degree > level:
                    break
            level = min_double_degree  # 设置level为当前最小的重度值
        self.graph = copy
        return importance_dict  # 退出外层循环说明全部计算完毕，返回字典

    def cal_HSI(self):
        """
        计算HSI值

        :return:
        """
        graph = self.graph.copy()
        # 获取k-shell分解后的节点集合
        k_shell = self.k_shell_self()
        # 获取最大shell值对应的节点集
        max_key = max(k_shell.keys())
        max_node_set = k_shell[max_key]
        # 从图中删除节点集
        graph.remove_nodes_from(max_node_set)

        W = len(list(nx.connected_components(graph)))

        S = len(max(nx.connected_components(graph), key=len))
        E = W / S
        HSI = len(max_node_set) / E
        print("HSI值为", HSI)
        print("最大shell值对应", max_node_set)


def setAttr(graph):
    """
    设置图的多种属性值

    :param graph:
    :return:
    """
    DEGREE = "degree"
    PAGERANKS = "pageranks"
    BETWEENNESS = "betweenesscentrality"
    KSHELL = "k_shell_level"
    ASHELL = "ait_shell_level"

    test = FeatureCalculate(graph.copy())
    # 为graph的每个节点设置pagerak
    # pagerank = nx.pagerank(graph)
    # for node in graph.nodes:
    #     graph.nodes[node][PAGERANKS] = pagerank[node]
    # if verbose:
    #     print("pagerank finish")

    # 为graph的每个节点设置介数中心性
    betweenness_dict = nx.betweenness_centrality(graph)
    for node in graph.nodes:
        graph.nodes[node][BETWEENNESS] = betweenness_dict[node]
    if verbose:
        print("betweenesscentrality finish")

    # 为graph的每个节点添加degree值
    degrees = graph.degree
    for node in graph.nodes:
        graph.nodes[node][DEGREE] = degrees[node]
    if verbose:
        print("degree finish")

    # 为graph的每个节点添加k_shell值
    k_shell_set = test.k_shell_degree()
    for key in k_shell_set.keys():
        for item in k_shell_set[key]:
            graph.nodes[item][KSHELL] = key
    if verbose:
        print("k_shell_level finish")

    # 为graph的每个节点添加shell值
    k_shell_set = test.k_shell_self()
    for key in k_shell_set.keys():
        for item in k_shell_set[key]:
            graph.nodes[item][ASHELL] = key
    if verbose:
        print("ait_shell_level finish")


def write_csv(graph, output_path):
    """
    将图以及点的数据写入csv文件

    :param graph:
    :param output_path:
    :return:
    """
    LABEL = "label"
    wt = []
    for node, dic in list(graph.nodes(data=True)):
        row = {LABEL: node}
        row.update(dic)
        wt.append(row)
    pd.DataFrame(wt).to_csv(output_path)


def calculate_graph_features(input, output):
    """
    计算每个结点的多种特征属性

    :param input: 输入文件路径
    :param output: 输出文件路径
    :return:
    """
    print(input, "begin")
    graph = read_graph(input)  # 读取数据
    setAttr(graph)  # 设置图的属性
    write_csv(graph, output)  # 写入文件中
    print(input, "finish")


def get_data(file_path):
    """
    读取CSV数据

    :param file_path:
    :return:
    """
    col_title = None         # 存储列名
    lst = []             # 存储所有读取的数据
    with open(file_path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            if i == 0:
                col_title = rows     # 读取列名
                continue
            lst.append(rows)     # 存储每一行的数据
    ret = {"title": col_title, "data": lst}
    return ret


if __name__ == "__main__":

    root = os.path.abspath(os.path.dirname(__file__))  # 获取文件路径
    root_path = os.path.join(root, "传递样本")
    # 保证这里与传染率字典一致，否则找不到传染率，会报错。并且输入的图要求为edgelist格式，需要手动处理
    names = [
        "actors",
    ]

    irate_dic = {"CA-AstroPh": 0.02,
                 "CA-CondMat": 0.05,
                 "CA-GrQc": 0.06,
                 "CA-HepTh": 0.09,
                 "CA-HepPh": 0.01,
                 "jazz": 0.03,
                 'actors': 0.01,
                 "Wiki-Vote": 0.01,
                 "musae_FR_edges": 0.01,
                 "netscience": 0.15,
                 }
    # 平均传播范围的标签
    average_spread_label = "Mi"
    # 执行次数
    times = 100

    """
        计时开始
    """
    cpu_start = time.clock()

    for name in names:
        infection = irate_dic[name]  # 传染率
        recover = 1  # 恢复率

        path = os.path.join(root_path, name + ".edgelist")  # 图的存储路径
        node_path = os.path.join(root_path, name + ".csv")  # 点的特征值输出路径
        if not os.path.exists(node_path):
            calculate_graph_features(path, node_path)
        # break
        # 读取图
        cal_graph = read_graph(path)

        cnt = count(start=0, step=1)  # 记录当前处理到第几个节点
        node_nums = len(cal_graph)
        for node in cal_graph:
            controller = SIRController(cal_graph)
            controller.configure_sir([node], infection, recover)
            result = controller.run(times)
            # 用于计算平均传播范围
            result["spread"] = result['i'] + result['r']
            sum_of_max = result.groupby('times')['spread'].agg('max').sum()
            cal_graph.nodes[node][average_spread_label] = sum_of_max / times
            print(
                "this is the {} of {} in {}".format(
                    next(cnt), node_nums, name))

        # 添加之前记录的属性
        features = get_data(node_path)
        title, data = features["title"], features["data"]
        for line in data:
            node = line[1]
            for index in range(2, len(line)):
                cal_graph.nodes[node][title[index]] = line[index]
        # 写回文件, 如果不担心出意外，可以直接写回原文件
        write_csv(cal_graph, os.path.join(root_path, name + "_result.csv"))
    """
        计时结束
    """
    cpu_end = time.clock()
    print('cpu:', cpu_end - cpu_start)
