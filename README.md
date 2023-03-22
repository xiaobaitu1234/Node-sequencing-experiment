# Node sequencing experiment
 节点排序实验

## 文件描述

- per_lib
  - HSI_FI
    - 使用SIR_spread计算的节点特征值，按照多种排序进行迭代删除，进行求值
  - SIR_spread
    - 使用传递样本中的edgelist文件，计算每个节点的多种特征值，并对每个节点求平均传播范围
  - trans_spread_result2heatmap_matrix
    - 将SIR的传播结果，构造为热力图的矩阵数据。后续可以通过matlab等工具绘制