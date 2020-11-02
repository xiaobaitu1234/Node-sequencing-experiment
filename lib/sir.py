import networkx as nx
from numba import njit
from test_numba import tick
from random import uniform
from numba.typed import List
from pandas import DataFrame


def prepare_data_from_graph(g):
    """
    从g中提取节点和边

    :param g: networkx，Graph
    :return: (name_index_dict,edge_list),(name_index_dict):节点编码字典，key为节点名称，value为节点标识数字下标，所有value构成联系自然数;(edge_list):边列表，列表内元素为编码后的边元组
    """
    name_index_dict = dict(zip(g.nodes, range(len(g.nodes))))  # 构造字典
    edge_list = [(name_index_dict[x], name_index_dict[y])
                 for x, y in g.edges()]  # 对边进行编码
    return name_index_dict, edge_list


def encoding_node_name(name_index_dict, node_list):
    """
    使用name_index_dict对node_list进行编码

    :param name_index_dict: 节点名称-替换值字典
    :param node_list: 需编码的节点名称
    :return: list
    """
    result = [name_index_dict[x] for x in node_list]
    return result


@njit
def sir_simulate(node_nums, infection_list, edges,
                 infect_rate, recover_rate, times=100):
    """
    批量执行SIR模拟，并返回列表

    :param node_nums: 节点总数
    :param infection_list: 感染节点下标
    :param edges: 边集合
    :param infect_rate: 感染率
    :param recover_rate: 恢复率
    :param times: 模拟次数
    :return: result->list[list],内层list结构为[time, step, s, i, r]，含义为次数，时间步，各状态节点数量
    """
    # 构造常量
    label_s, label_i, label_r = 's', 'i', 'r'

    # 处理输入数据，作为SIR模拟输入,以下均为初始数据，后续拷贝使用
    backup_i = List(infection_list)  # 记录初始感染节点
    num_s, num_i, num_r = node_nums - \
        len(backup_i), len(backup_i), 0  # 记录每个状态的节点的数量
    status_dict = {}  # 状态字典，在判定某个节点的状态时使用
    for i in range(node_nums):  # 初始化字典
        status_dict[i] = label_s
    for i in backup_i:  # 减少判断
        status_dict[i] = label_i
    # 构造邻接矩阵
    list_contain_type_placeholder = -1  # 用于初始化列表的类型,默认输入为int，方便下标调用邻接矩阵
    matrix = [[list_contain_type_placeholder]
              for _ in range(node_nums)]  # 初始化邻接矩阵
    for nbr in matrix:  # 删除初始化数据
        nbr.remove(list_contain_type_placeholder)
    for x, y in edges:  # 构建邻接矩阵
        matrix[x].append(y)

    result = []  # 存储返回结果
    for time in range(1, times + 1):  # 按次数执行
        time_num_s, time_num_i, time_num_r = num_s, num_i, num_r  # 拷贝计数器
        time_backup_i = List(backup_i)  # 拷贝感染列表
        time_status_dict = {}  # 拷贝状态字典
        for key in status_dict:
            time_status_dict[key] = status_dict[key]
        # 每次具体执行部分，初始化计步器
        count = 0  # 计步器
        result.append([time, count, time_num_s,
                       time_num_i, time_num_r])  # 记录状态
        while time_num_i != 0:  # 直到没有感染节点，停止模拟
            for infect in List(time_backup_i):  # 拷贝i列表，使执行时修改i列表不影响执行
                for nbr in matrix[infect]:  # 遍历节点的邻居
                    if time_status_dict[nbr] == label_s:  # 如果该邻居为可感染状态，则进行判定
                        p = uniform(0, 1)
                        if p < infect_rate:  # 判定本次是否感染
                            time_num_i, time_num_s = time_num_i + 1, time_num_s - 1  # 感染后，节点计数器变化
                            time_status_dict[nbr] = label_i  # 感染后，节点状态字典变化
                            time_backup_i.append(nbr)  # 感染后，感染节点列表新增节点
                q = uniform(0, 1)  # 判定是否恢复
                if q < recover_rate:
                    time_num_i, time_num_r = time_num_i - 1, time_num_r + 1  # 恢复后，节点计数器变化
                    time_status_dict[infect] = label_r  # 恢复后，节点状态字典变化
                    time_backup_i.remove(infect)  # 恢复后，感染节点列表移除当前节点
            count += 1  # 计步器自增
            result.append([time, count, time_num_s,
                           time_num_i, time_num_r])  # 记录状态
    column = ['times', 'step', 's', 'i', 'r']
    return result, column


def load_graph():
    """
    载入图

    :return: graph
    """
    path = r'D:\GitHub\SIR\样本数据\CA-HepTh-edge.csv'
    with open(path) as f:
        data = f.readlines()[1:-1]
        data = [x.replace('\n', '') for x in data]
        data = [x.split(',') for x in data]
    G = nx.Graph()
    G.add_edges_from(data)
    return G


def test(graph, infection_list, irate, rrate, times):
    """
    测试用例

    :param graph: networkx.Graph
    :param infection_list: 感染节点名称列表
    :param irate: 感染率
    :param rrate: 恢复率
    :param times: 模拟次数
    :return: dataframe
    """
    name_dict, edge_list = prepare_data_from_graph(graph)  # 获取图的信息
    infection_list = encoding_node_name(name_dict, infection_list)  # 对感染节点编码

    typed_infection_list = List(infection_list)  # 转换为numba可以接受的列表
    typed_edge_list = List(edge_list)  # 转换为numba可以接受的列表
    args = {
        'node_nums': len(name_dict.keys()),
        'infection_list': typed_infection_list,
        'edges': typed_edge_list,
        'infect_rate': irate,
        'recover_rate': rrate,
        'times': times
    }
    with tick('dict'):
        rt, column = sir_simulate(**args)
        df = DataFrame.from_records(rt, columns=column)
    return df


if __name__ == '__main__':
    irate, rrate, times = 0.1, 1, 10
    G = load_graph()
    # 构造传播源
    infection_list = [
        '71788',
        '8168',
        '33111',
        '17284',
        '40942',
        '31512',
        '75297',
        '36663',
        '66142',
        '30344',
    ]
    args = {
        'irate': 0.1,
        'rrate': 1,
        'times': 10,
        'graph': G,
        'infection_list': infection_list
    }

    result = test(**args)
    print(result)
