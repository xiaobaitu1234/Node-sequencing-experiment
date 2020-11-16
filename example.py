import networkx as nx
from lib.sir import SIRController


def load_graph():
    """
    载入图

    :return: graph
    """
    path = r'D:\GitHub\Node-sequencing-experiment\样本数据\CA-HepTh.edgelist'
    g = nx.read_edgelist(path)
    return g


def main(infection_rate, recover_rate, simulate_times):
    """
    测试用例

    :param graph: networkx.Graph
    :param infection_list: 感染节点名称列表
    :param infection_rate: 感染率
    :param recover_rate: 恢复率
    :param simulate_times: 模拟次数
    :return: dataframe
    """
    graph = load_graph()
    # 选取度值做传播实验
    degree = nx.degree(graph)
    items = list(degree)
    items.sort(key=lambda x: x[1],reverse=True)
    # 排序选取度值前10个
    infection_list = map(lambda x:x[0],items[0:10])
    # 计算为列表
    infection_list = list(infection_list)

    sir_controller = SIRController(graph)  # 处理图的信息
    sir_controller.configure_sir(infection_list, infection_rate, recover_rate)
    print(sir_controller.run(simulate_times))


if __name__ == '__main__':
    main(0.09, 1, 10)
