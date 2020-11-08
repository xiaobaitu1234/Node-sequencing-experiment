import networkx as nx
from lib.sir import SIRController


def load_graph():
    """
    载入图

    :return: graph
    """
    path = r'D:\GitHub\SIR\样本数据\CA-HepTh-edge.csv'
    with open(path) as f:
        data = f.readlines()[1:-1]
        data = (x.replace('\n', '') for x in data)
        data = (x.split(',') for x in data)
    g = nx.Graph()
    g.add_edges_from(data)
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
    sir_controller = SIRController(graph)  # 处理图的信息
    sir_controller.configure_sir(infection_list, infection_rate, recover_rate)
    print(sir_controller.run(simulate_times))


if __name__ == '__main__':
    main(0.1, 1, 10)
