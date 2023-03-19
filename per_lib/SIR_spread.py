# -*- coding: utf-8 -*-

"""

这个文件很NB，读取图，对每个点跑SIR模型100次计算平均传播范围

然后，使用之前计算的每个点的多个值（AIT，Kshell，degree， between....），都进来，把SIR的平均传播加进去


"""
import copy
import csv
import os
import random
import time
from collections import defaultdict
from multiprocessing import Manager, Pool

import networkx as nx
import pandas as pd


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
    pagerank = nx.pagerank(graph)
    for node in graph.nodes:
        graph.nodes[node][PAGERANKS] = pagerank[node]

    # 为graph的每个节点设置介数中心性
    betweenness_dict = nx.betweenness_centrality(graph)
    for node in graph.nodes:
        graph.nodes[node][BETWEENNESS] = betweenness_dict[node]

    # 为graph的每个节点添加degree值
    degrees = graph.degree
    for node in graph.nodes:
        graph.nodes[node][DEGREE] = degrees[node]

    # 为graph的每个节点添加k_shell值
    k_shell_set = test.k_shell_degree()
    for key in k_shell_set.keys():
        for item in k_shell_set[key]:
            graph.nodes[item][KSHELL] = key

    # 为graph的每个节点添加shell值
    k_shell_set = test.k_shell_self()
    for key in k_shell_set.keys():
        for item in k_shell_set[key]:
            graph.nodes[item][ASHELL] = key


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


def run_sir(m: "节点的邻居列表", Infect_index: "infection list",
            A: "nodes_index", recover, infection):
    # 输入邻接矩阵，感染列表，以及所有节点的下标，感染率和恢复率，输出最终感染结点数
    # 先定义三种节点
    # 可被感染的节点
    S = []
    # 已经被感染的节点
    I = Infect_index[:]
    # 已经痊愈的节点
    R = []
    # 初始化S,除了I就全是S,R初始化时为0
    for i in A:
        if i in Infect_index:
            continue
        S.append(i)

    # 计步器(时间步)
    count = 0
    while True:
        # 遍历感染者人群去传染未感染者
        tempI = copy.copy(I)
        for infect in tempI:
            # 遍历当前感染者的邻居
            for nbr in m[infect]:
                # 如果当前感染者的邻居是感染者或者是恢复者,继续循环
                if nbr in S:
                    # 有一定的概率被传染并不一定会被传染
                    p = random.uniform(0, 1)
                    if p < infection:
                        I.append(nbr)
                        S.remove(nbr)
            # 感染者人群也会有一定的概率痊愈
            q = random.uniform(0, 1)
            if q < recover:
                I.remove(infect)
                R.append(infect)
        count += 1

        if len(I) == 0:
            break
    return len(R) + len(I)


def init_data(G):
    # 初始化数据并执行
    # 总节点数N
    N = len(G.nodes())
    # 构造邻接矩阵
    m = [[] for _ in range(N)]

    # A是所有的节点,无重复
    A = list(G.nodes)
    # A_index是所有节点的索引
    A_index = list(range(N))
    # 构造节点与邻接矩阵下标映射
    for i in range(N):
        G.nodes[A[i]]["index"] = A_index[i]

    # 构造所有节点的邻居列表
    for node in A:
        # 获取节点的索引
        index1 = G.nodes[node]["index"]
        # 获取node的邻居节点
        for nbr in G.neighbors(node):
            index2 = G.nodes[nbr]["index"]
            m[index1].append(index2)

    return A_index, A, m

# 处理每个结点的平均传播范围


def task_node(q, node_index, m, A_index, recover, infection, times):
    count_list = []
    for _ in range(times):
        count_list.append(
            run_sir(
                m,
                [node_index],
                A_index,
                recover,
                infection))
    # 计数列表自增
    q.put(node_index)
    # 输出当前执行结点个数
    print("this is the %d in %d" % (q.qsize(), len(A_index)))
    return node_index, sum(count_list) / times


if __name__ == "__main__":

    root = os.path.abspath(os.path.dirname(__file__))  # 获取文件路径
    root_path = os.path.join(root, "传递样本")
    names = ["jazz"]

    irate_dic = {"CA-AstroPh": 0.02,
                 "CA-CondMat": 0.05,
                 "CA-GrQc": 0.06,
                 "CA-HepTh": 0.09,
                 "CA-HepPh": 0.01,
                 "jazz": 0.03,
                 "netscience": 0.15}
    # 平均传播范围的标签
    Label = "Mi"
    # 执行次数
    times = 100

    """
        计时开始
    """
    cpu_start = time.clock()

    for name in names:

        # 传染率
        infection = irate_dic[name]
        # 回复率
        recover = 1
        # 图的存储路径
        path = os.path.join(root_path, name + ".edgelist")
        node_path = os.path.join(root_path, name + ".csv")
        calculate_graph_features(path, node_path)

        # 读取图
        G = read_graph(path)
        # 处理数据，构造指定格式数据
        A_index, A, m = init_data(G)
        # 对每个节点进行测试
        """
        开启多进程,设置计数器counter
        """
        q = Manager().Queue(len(A_index))
        processes = 1
        # 生成参数列表
        data_list = []
        for node_index in A_index:
            data_list.append(
                (q, node_index, m, A_index, recover, infection, times))
        # 开启多进程
        pool = Pool(processes=processes)
        # map分配进程
        res = pool.starmap(task_node, data_list)
        # 收集结果
        pool.close()
        pool.join()
        """
        关闭多进程
        """
        # 统计结果
        for node_index, average_spread in res:
            G.nodes[A[node_index]][Label] = average_spread
            # print(node_index,"节点，平均传播为",str(average_spread))
        """
        收集结果完成
        """
        # 添加之前记录的属性

        rt = get_data(node_path)
        title, data = rt["title"], rt["data"]
        for line in data:
            node = line[1]
            for index in range(2, len(line)):
                G.nodes[node][title[index]] = line[index]
        # 写回文件
        write_csv(G, os.path.join(root_path, name + "_result.csv"))
    """
        计时结束
    """
    cpu_end = time.clock()
    print('cpu:', cpu_end - cpu_start)
