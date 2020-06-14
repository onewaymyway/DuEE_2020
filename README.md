# 事件抽取模型（基于paddlehub）
本模型在官方PaddleHub版本上进行修改得到
官方原版地址:https://github.com/PaddlePaddle/Research/tree/master/KG/DuEE_baseline/DuEE-PaddleHub

本方案github地址:https://github.com/onewaymyway/DuEE_2020

本方案在官方baseline的基础上的改动

1.在网络结构上在CRF层前面增加了双向GRU层（代码见sequence_label.py中SequenceLabelTaskSP类）

2.将trigger预测结果拼接到text前面进行第二阶段的role预测(代码见data_process.py的data_process函数中model=role1的情况)，这个改动可以解决同一个句子不同event之间role重叠的问题

3.在训练上，本方案先只用train进行训练，然后再将dev放入train进行最后的训练

4.增加了简单的最终结果剔除机制(代码见datachecker.py)

建议使用AIStudio环境跑这个项目，最好是直接Fork本人分享的项目，

项目地址:
....



### 环境准备

- python适用版本 2.7.x（本代码测试时使用依赖见 ./requirements.txt ）
-  paddlepaddle-gpu >= 1.7.0、paddlehub >= 1.6.1
-  请转至paddlepaddle官网按需求安装对应版本的paddlepaddle

#### 依赖安装
> pip install -r ./requirements.txt


### 模型训练

各个步骤在notebook文件里(project.ipynb)都有详细说明
按照notebook的顺序执行就可以了，这里就不详细说明了

