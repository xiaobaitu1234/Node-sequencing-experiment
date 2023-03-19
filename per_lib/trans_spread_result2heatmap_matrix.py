"""

处理传播结果

"""

import os

import pandas as pd

root = os.path.abspath(os.path.dirname(__file__))  # 获取文件路径


def declare_folder(path: str) -> None:
    """
    避免出现文件夹不存在的情况

    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def multi_value_average(file_names: list,
                        src_folder=None, dst_folder=None, cols=None, split_nums=100):
    """
    由于某列的数据过于分散，使用极差等分的思想进行离散化

    :param file_names: 待处理的文件名
    :param src_folder: 数据源目录
    :param dst_folder: 数据存储路径
    :param cols: 待处理的列
    :param split_nums: 离散化的尺度
    :return:
    """
    if src_folder is None:
        src_folder = "原始数据"
    if dst_folder is None:
        dst_folder = "等分结果均值"

    data_root_path = os.path.join(root, src_folder)  # 原始数据读取地址
    declare_folder(data_root_path)

    result_path = os.path.join(root, dst_folder)  # 结果存放父地址
    declare_folder(result_path)

    result_root_path = os.path.join(result_path, str(split_nums))  # 具体结果存放目录
    declare_folder(result_root_path)

    file_type = ".csv"  # 文件类型/后缀,默认为csv
    if cols is None:  # 默认对这两列进行等分
        cols = ["betweenesscentrality", "ait_shell_level"]  # 进行等分处理的列名

    for name in file_names:  # 对每个文件进行处理
        path = os.path.join(data_root_path, name + file_type)  # 构造路径
        df = pd.read_csv(path)  # 读取数据

        for column in cols:  # 对每个需平均的列进行处理
            max_value, min_value = max(df[column]), min(df[column])  # 记录极值
            diff = max_value - min_value  # 求极差
            split = diff / (split_nums - 1)  # 将极差分段
            # 使用最小值加极差分段，构造极差序列
            steps = [min_value + split * times for times in range(split_nums)]
            # 将该列的值分配到极差序列
            row_index = list(df.index)
            for idx in row_index:
                value = df[column][idx]
                for barrier in steps:
                    if value > barrier:
                        pass
                    else:
                        df.at[idx, column] = barrier  # 向上取整
                        break
        result = os.path.join(result_root_path, name + file_type)  # 构造路径
        df[['label', 'Mi', 'betweenesscentrality', 'degree',
            'k_shell_level', 'ait_shell_level']].to_csv(result)  # 写回文件


def trans_csv_matrix(file_names):
    """
    这个文件负责把等分结果转换为热力图源数据，后续用matlab进行出图

    :return:
    """

    data_root_path = os.path.join(root, '等分结果均值', '100')  # 原始数据读取地址
    declare_folder(data_root_path)
    result_path = os.path.join(root, '矩阵结果')  # 结果存放地址
    declare_folder(result_path)
    file_type = ".csv"  # 文件类型/后缀,暂时只支持csv
    SPREAD_AREA = "Mi"  # 定义列名常量
    columns = ["degree", "betweenesscentrality", "k_shell_level"]
    row = "ait_shell_level"

    for name in file_names:

        path = os.path.join(data_root_path, name + file_type)  # 构造路径

        df = pd.read_csv(path)  # 读取数据
        df[row].map(lambda x: ('%.4f') % x)  # 保留四位小数
        for col in columns:
            df[col].map(lambda x: ('%.4f') % x)  # 保留四位小数

            means = df[SPREAD_AREA].groupby(
                [df[col], df[row]]).mean()  # 按两列聚合分组

            result = means.unstack()  # 填充空值
            result = result.reindex(index=result.index[::-1])

            # 构造目标路径
            target_path = os.path.join(
                result_path,
                name +
                "-row-" +
                row +
                "-col-" +
                col +
                "-result" +
                file_type)
            result.to_csv(target_path)


def av_matrix(filenames):
    """
    平均并转为热力图矩阵

    :param filenames:
    :return:
    """
    multi_value_average(filenames)
    trans_csv_matrix(filenames)


if __name__ == "__main__":
    file_names = ['CA-AstroPh', 'CA-CondMat', 'CA-GrQc', 'CA-HepTh', 'CA-HepPh', 'jazz', 'netscience',
                  'jazz_result']  # 需要处理的文件名
    av_matrix(file_names)
