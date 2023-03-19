# 引包

"""
这个文件可以出图3

即计算每次按比例删除后，剩余的HSI值和F(i)值

"""

import copy
import os

import networkx as nx
import numpy as np
import pandas as pd


def declare_folder(path: str) -> None:
    """
    避免出现文件夹不存在的情况

    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


root_path = os.path.abspath(os.path.dirname(__file__))  # 获取文件路径

graph_path = os.path.join(root_path, "传递样本")  # 存储图的文件夹
# 存储节点信息的文件夹，里边有每一个节点的多种特征值，用于排序删除
nodes_path = os.path.join(root_path, "传递样本")
result_path = os.path.join(root_path, "迭代删除结果目录")  # 结果输出目录
declare_folder(result_path)
declare_folder(graph_path)
declare_folder(nodes_path)

file_names = [
    # 'CA-AstroPh',
    # 'CA-CondMat',
    # 'CA-GrQc',
    # 'CA-HepTh',
    # 'CA-HepPh',
    'jazz',
    # 'netscience',
]


def read(file_path: str):
    """
    从文件中读取图 grid.edgelist
    以networkx的方式读取图, 保留最大连通子图

    :param file_path:
    :return:
    """
    if ".gml" in file_path:
        graph = read_gml(file_path)
    else:
        graph = nx.read_edgelist(path=file_path, data=True)
    graph.remove_edges_from(nx.selfloop_edges(graph))  # 删除自循环
    largest_components = max(nx.connected_components(graph), key=len)
    ori = list(graph.nodes())
    des = list(largest_components)
    remove = [x for x in ori if x not in des]
    graph.remove_nodes_from(remove)
    return graph


def read_gml(path):
    H = nx.read_gml(path)
    return H


# 初始化配置信息，包括删除比例i列表
remove_rate_list = [0.01 * x for x in range(1, 101)]


types = ["degree", "k_shell_level", "ait_shell_level", "random"]
type_trans_dic = {
    "degree": 'DC',
    "betweenesscentrality": 'BC',
    "k_shell_level": 'KS',
    "ait_shell_level": 'IKs',
    "random": 'RAM'
}
label = "label"


# 此处开始计算各数据集
for name in file_names:
    # 构造输出
    output = []
    # 构造图路径，节点信息路径
    # if name == 'netscience':
    #     path = os.path.join(graph_path, name + ".gml")
    # else:
    path = os.path.join(graph_path, name + ".edgelist")
    # 构造节点信息路径
    node_path = os.path.join(nodes_path, name + ".csv")

    G = read(path)  # 读入图
    nodes_num = len(G.nodes)  # 存储图的节点数
    attr = pd.read_csv(node_path)  # 获取节点数据

    for sort_type in types:  # 读入排序指标
        if sort_type == "random":
            process_data = attr[[label, "degree"]]

            process_data = process_data.reindex(
                np.random.permutation(process_data.index))  # 随机打乱顺序
        else:
            process_data = attr[[label, sort_type]]  # 读取节点标签和对应类型属性列
            process_data = process_data.sort_values(
                by=sort_type, ascending=False)  # 按降序排列
            max_value = max(process_data[sort_type])  # 记录最大shell值
            max_shell_count = process_data.loc[:, sort_type].value_counts()[
                max_value]  # 记录最大值出现次数

        process_data.reset_index(drop=True, inplace=True)  # index重排序保证后续获取多行正确

        for remove_rate in remove_rate_list:  # 读取待删除数目，以获得待删除数据项
            if remove_rate == 1:
                result = {"type": type_trans_dic[sort_type],
                          "remove_rate": remove_rate,
                          "HSI": '',
                          "F(i)": '',
                          "Cm": Cm,
                          "Wg": 0,
                          "Sg": 0,
                          "remove_num": nodes_num,
                          'k': 0,
                          'k^2': 0,
                          'k0': 0}
                continue

            # 计算待删除结点数目
            remove_num = int(len(process_data) * remove_rate)
            # 获取待删除结点列表
            need_remove = process_data.loc[:remove_num - 1, label]
            # 拷贝原图
            graph_copy = copy.deepcopy(G)
            # 删除图中数据
            graph_copy.remove_nodes_from([str(x) for x in list(need_remove)])
            # 获取所有连通子图,以计算连通分支数
            components_num = len(list(nx.connected_components(graph_copy)))
            # 获取最大连通子图
            largest_components = max(
                nx.connected_components(graph_copy), key=len)

            Wg = components_num  # 连通分支数
            Cm = max_shell_count  # 最大shell值得节点数目
            Sg = len(largest_components)  # 最大连通分支节点数

            HSI = Cm * Sg / Wg

            N = nodes_num  # 原图节点总数
            Fi = Sg / N

            # 计算k-k0
            degrees = graph_copy.degree()
            degrees = [x for _, x in degrees]
            k_bottom = sum(degrees) / len(degrees)
            k_up = sum([x**2 for x in degrees]) / len(degrees)
            k0 = k_up / k_bottom if k_bottom > 0 else 0

            result = {"type": type_trans_dic[sort_type],
                      "remove_rate": remove_rate,
                      "HSI": HSI,
                      "F(i)": Fi,
                      "Cm": Cm,
                      "Wg": Wg,
                      "Sg": Sg,
                      "remove_num": remove_num,
                      'k': k_bottom,
                      'k^2': k_up,
                      'k0': k0}  # 生成结果

            output.append(result)  # 合并结果

    df = pd.DataFrame(output)  # 转为dataframe输出
    df.to_csv(os.path.join(result_path, name + ".csv"))
