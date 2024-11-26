前提：必须运行long_tail中划分 group degree的代码，因为我们是按照不同group degree中实体的占比均匀采样query的
1. 运行select_query.py文件，从划分好group degree的数据集中均匀采样query

2. 运行judge.py

3. 人工打标签后，运行static.py计算指标值