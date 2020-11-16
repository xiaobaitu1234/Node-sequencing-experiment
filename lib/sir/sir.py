import networkx as nx
from numba import njit
from random import uniform
from numba.typed import List
from pandas import DataFrame


class SIRController(object):
    """
    本类用于完成SIR模拟并返回结果

    Simple example
    --------------

    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> ......
    >>> controller = SIRController(G)
    >>> controller.configure_sir(infection_list,infection_rate,recover_rate)
    >>> result = controller.run(times)
    """

    def __init__(self, graph):
        """
        :param graph: networkx的Graph实例
        """
        self._graph = nx.Graph(graph)
        self._name_index_dict = None  # 存储节点名称编码字典
        self._edge_list = None  # 存储编码后的边列表
        self._init()  # 初始化编码字典，边列表
        self.node_num = len(graph.nodes)  # 节点数

        self._infection_rate = None  # 感染率
        self._recover_rate = None  # 恢复率
        self._infection_list = None  # 传播源

    def _init(self):
        """
        初始化name_index_dict和edge_list
        """
        g = self._graph
        self._name_index_dict = dict(zip(g.nodes, range(len(g.nodes))))  # 构造字典
        encoding_dict = self._name_index_dict
        self._edge_list = (
            (encoding_dict[x],
             encoding_dict[y]) for x,
            y in g.edges())  # 对边进行编码

    def _nodes_encoding(self, infection_list):
        """
        对输入的感染节点进行编码，用于后续调用传播
        :return:
        """
        encoding_dict = self._name_index_dict
        return (encoding_dict[x] for x in infection_list)

    def configure_sir(self, infection_list, infection_rate, recover_rate):
        """
        配置SIR模拟
        :param infection_list: 初始传播源
        :param infection_rate: 感染率
        :param recover_rate: 恢复率
        :return:
        """
        self._infection_rate = infection_rate
        self._recover_rate = recover_rate
        self._infection_list = self._nodes_encoding(infection_list)  # 对传播源编码

    def run(self, simulate_times=1):
        """
        输入模拟次数，根据之前的配置进行模拟
        :param simulate_times:
        :return: pandas.Dataframe对象
        """
        if self._recover_rate is None:
            raise RuntimeError(
                "Please run configure_sir method first or check if correctly configured")
        # 将输入参数构造为numba可以接受的列表形式
        typed_infection_list = List(self._infection_list)
        typed_edge_list = List(self._edge_list)
        # 构造参数字典
        params_dict = {
            'node_nums': self.node_num,
            'infection_list': typed_infection_list,
            'edges': typed_edge_list,
            'infect_rate': self._infection_rate,
            'recover_rate': self._recover_rate,
            'simulate_times': simulate_times
        }
        rt, column = sir_simulate(**params_dict)  # 执行模拟
        return DataFrame.from_records(rt, columns=column)  # 将结果构建为Dataframe并返回


@njit
def sir_simulate(node_nums, infection_list, edges,
                 infect_rate, recover_rate=1, simulate_times=100):
    """
    批量执行SIR模拟，并返回列表

    :param node_nums: 节点总数
    :param infection_list: 感染节点下标
    :param edges: 边集合
    :param infect_rate: 感染率
    :param recover_rate: 恢复率
    :param simulate_times: 模拟次数
    :return: (list,column_name)、list为二级列表，存储结果，column_name指示列表含义
    """
    """
        处理输入数据，作为SIR模拟输入,以下均为初始状态数据，后续拷贝使用,
        backup_i: 感染节点列表
        num_s、i、r: 各状态节点数量
        status_dict: 各节点状态字典
        matrix: 邻接矩阵
    """
    backup_i = List(infection_list)  # 拷贝初始感染节点
    num_s, num_i, num_r = node_nums - \
        len(backup_i), len(backup_i), 0  # 记录每个状态的节点的数量
    # 初始化状态字典，用于快速判断每个节点的状态
    status_dict = {}
    label_s, label_i, label_r = 's', 'i', 'r'
    for i in range(node_nums):  # 初始化字典
        status_dict[i] = label_s
    for i in backup_i:  # 减少判断
        status_dict[i] = label_i
    # 以下均为构造邻接矩阵
    list_contain_type_placeholder = -1  # 用于初始化列表的类型,默认输入为int，方便下标调用邻接矩阵
    matrix = [[list_contain_type_placeholder]
              for _ in range(node_nums)]  # 初始化邻接矩阵
    for nbr in matrix:  # 删除初始化数据
        nbr.remove(list_contain_type_placeholder)
    for x, y in edges:  # 构建邻接矩阵
        matrix[x].append(y)
    # 至此，所有初始步骤构建完毕.

    simulate_result_list = []  # 存储返回结果
    for time in range(1, simulate_times + 1):  # 按次数执行
        """
            模拟时，先拷贝上述三个备份数据，即
            backup_i
            num_s、i、r
            status_dict
            由于邻接矩阵不会变动，因此无需处理
        """
        time_num_s, time_num_i, time_num_r = num_s, num_i, num_r  # 拷贝计数器
        time_backup_i = List(backup_i)  # 拷贝感染列表
        time_status_dict = {}  # 拷贝状态字典
        for key in status_dict:
            time_status_dict[key] = status_dict[key]
        # 每次具体执行部分，初始化计步器
        step_counter = 0  # 计步器
        simulate_result_list.append(
            [time, step_counter, time_num_s, time_num_i, time_num_r])  # 记录初始状态
        while time_num_i != 0:  # 直到没有感染节点，停止模拟
            for infect in List(
                    time_backup_i):  # 对所有感染节点遍历，(拷贝感染列表，使执行时修改感染列表时不影响当前时间步的执行)
                for nbr in matrix[infect]:  # 遍历当前感染节点的邻居
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
            step_counter += 1  # 计步器自增
            simulate_result_list.append(
                [time, step_counter, time_num_s, time_num_i, time_num_r])  # 记录状态
    column = ['times', 'step', 's', 'i', 'r']  # 返回列名，指示返回结果中每一列的含义
    return simulate_result_list, column
