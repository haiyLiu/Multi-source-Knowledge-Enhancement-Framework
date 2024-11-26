1. 首先对于一个新类型的数据集，例如从FB换到WN上，那么需要计算WN上的实体频率，运行entity_frequency_test.py，即对于test数据集中的实体，从train数据集中统计实体出现的频率

2. 对于类似于tranE和SimKGC的test.json格式的文件，需要先对格式进行处理，处理为和LLM answer类似的数据格式，运行 small_experiments/process_trans_simkgc.py

3. 对test数据根据实体频率进行划分为小文件，运行 small_experiments/long_tail/long_tail.py，注意修改config_fb.json里的eval_type和degree，每次只能运行一类方法

4. 对每个小文件进行评估, 运行 small_experiments/long_tail/evaluate.py，注意修改config_fb.json里的eval_type和degree

5. 运行small_experiments/long_tail/plot_eval.py文件，绘图