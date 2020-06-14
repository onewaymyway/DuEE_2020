#!/usr/bin/env python
#coding:utf-8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning on sequence labeling task."""
from paddlehub.common.logger import logger
import argparse
import ast
import json
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

from data_process import data_process
from data_process import schema_process
from data_process import write_by_lines

from nlputils import MyLog
import sys
print(sys.argv)
# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--data_dir", type=str, default=None, help="data save dir")
parser.add_argument("--schema_path", type=str, default=None, help="schema path")
parser.add_argument("--train_data", type=str, default=None, help="train data")
parser.add_argument("--dev_data", type=str, default=None, help="dev data")
parser.add_argument("--test_data", type=str, default=None, help="test data")
parser.add_argument("--predict_data", type=str, default=None, help="predict data")
parser.add_argument("--do_train", type=ast.literal_eval, default=False, help="do train")
parser.add_argument("--do_eval", type=ast.literal_eval, default=False, help="do eval")
parser.add_argument("--mixtrain", type=ast.literal_eval, default=False, help="mixtrain")
parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="do predict")
parser.add_argument("--do_predict2", type=ast.literal_eval, default=False, help="do predict")
parser.add_argument("--do_model", type=str, default="trigger", choices=["trigger", "role","role1"], help="trigger or role")
parser.add_argument("--predictmodel", type=str, default="best_model", help="bestmodel")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--eval_step", type=int, default=200, help="eval step")
parser.add_argument("--model_save_step", type=int, default=3000, help="model save step")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--add_crf", type=ast.literal_eval, default=True, help="add crf")
parser.add_argument("--add_gru", type=ast.literal_eval, default=False, help="add gru")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--outmark", type=str, default="", help="outmark")

parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

schema_labels = schema_process(args.schema_path, args.do_model)

# 先把数据处理好保存下来
train_data = data_process(args.train_data, args.do_model)  # 处理训练数据
dev_data = data_process(args.dev_data, args.do_model,False,schema_labels) # 处理dev数据
test_data = data_process(args.test_data, args.do_model)
print("train",len(train_data))
print("dev",len(dev_data))
if args.mixtrain:
    #将dev数据加到train里面进行训练
    train_data=train_data+dev_data[1:]
    #dev_data=train_data[:]
    print("mix",len(train_data))
    

    


write_by_lines("{}/{}_train.tsv".format(args.data_dir, args.do_model), train_data)
write_by_lines("{}/{}_dev.tsv".format(args.data_dir, args.do_model), dev_data)
write_by_lines("{}/{}_test.tsv".format(args.data_dir, args.do_model), test_data)
if args.predict_data:
    predict_sents, predict_data = data_process(args.predict_data, args.do_model, is_predict=True)
    write_by_lines("{}/{}_predict.tsv".format(args.data_dir, args.do_model), predict_data)


schema_labels = schema_process(args.schema_path, args.do_model)

from paddlehub.finetune.evaluate import chunk_eval, calculate_f1
from paddlehub.common.utils import version_compare
import paddle
import os

class SequenceLabelTaskSP(hub.SequenceLabelTask):
    '''
    扩展序列标注任务
    增加从非best_model目录加载模型的功能
    添加gru层
    '''
    def __init__(self,
                 feature,
                 max_seq_len,
                 num_classes,
                 feed_list,
                 data_reader,
                 startup_program=None,
                 config=None,
                 metrics_choices="default",
                 add_crf=False):

        print("SequenceLabelTaskSP")

        super(SequenceLabelTaskSP, self).__init__(
            feature=feature,
            max_seq_len=max_seq_len,
            num_classes=num_classes,
            feed_list=feed_list,
            data_reader=data_reader,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices,
            add_crf=add_crf)

    def init_if_load_best_model(self):
        '''
        支持从自定义的目录加载bestmodel
        '''
        if not self.is_best_model_loaded:
            best_model_path = os.path.join(self.config.checkpoint_dir,
                                           args.predictmodel)
            logger.info("Load the best model from %s" % best_model_path)
            if os.path.exists(best_model_path):
                self.load_parameters(best_model_path)
                self.is_checkpoint_loaded = False
                self.is_best_model_loaded = True
            else:
                self.init_if_necessary()
        else:
            logger.info("The best model has been loaded")
    def _build_net(self):
        self.seq_len = fluid.layers.data(
            name="seq_len", shape=[1], dtype='int64', lod_level=0)

        if version_compare(paddle.__version__, "1.6"):
            self.seq_len_used = fluid.layers.squeeze(self.seq_len, axes=[1])
        else:
            self.seq_len_used = self.seq_len

        #增加gru层相关的代码
        grnn_hidden_dim = 256  # 768
        crf_lr = 0.2
        bigru_num = 2
        init_bound = 0.1

        def _bigru_layer(input_feature):
            """define the bidirectional gru layer
            """
            pre_gru = fluid.layers.fc(
                input=input_feature,
                size=grnn_hidden_dim * 3,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            gru = fluid.layers.dynamic_gru(
                input=pre_gru,
                size=grnn_hidden_dim,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            pre_gru_r = fluid.layers.fc(
                input=input_feature,
                size=grnn_hidden_dim * 3,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            gru_r = fluid.layers.dynamic_gru(
                input=pre_gru_r,
                size=grnn_hidden_dim,
                is_reverse=True,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
            return bi_merge

        if self.add_crf:
            unpad_feature = fluid.layers.sequence_unpad(
                self.feature, length=self.seq_len_used)
            
            #增加gru层相关的代码
            input_feature = unpad_feature
            for i in range(bigru_num):
                bigru_output = _bigru_layer(input_feature)
                input_feature = bigru_output

            unpad_feature=input_feature  
            self.emission = fluid.layers.fc(
                size=self.num_classes,
                input=unpad_feature,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(low=-0.1, high=0.1),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            size = self.emission.shape[1]
            fluid.layers.create_parameter(
                shape=[size + 2, size], dtype=self.emission.dtype, name='crfw')
            self.ret_infers = fluid.layers.crf_decoding(
                input=self.emission, param_attr=fluid.ParamAttr(name='crfw'))
            ret_infers = fluid.layers.assign(self.ret_infers)
            return [ret_infers]
        else:
            self.logits = fluid.layers.fc(
                input=self.feature,
                size=self.num_classes,
                num_flatten_dims=2,
                param_attr=fluid.ParamAttr(
                    name="cls_seq_label_out_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                bias_attr=fluid.ParamAttr(
                    name="cls_seq_label_out_b",
                    initializer=fluid.initializer.Constant(0.)))

            self.ret_infers = fluid.layers.reshape(
                x=fluid.layers.argmax(self.logits, axis=2), shape=[-1, 1])

            logits = self.logits
            logits = fluid.layers.flatten(logits, axis=2)
            logits = fluid.layers.softmax(logits)
            self.num_labels = logits.shape[1]
            return [logits]

class EEDataset(BaseNLPDataset):
    """EEDataset"""
    def __init__(self, data_dir, labels, model="trigger"):
        pdf="{}_predict.tsv".format(model)
        if not args.predict_data:
            pdf="{}_test.tsv".format(model)
        print("labels:",labels)
        # 数据集存放位置
        super(EEDataset, self).__init__(
            base_path=data_dir,
            train_file="{}_train.tsv".format(model),
            dev_file="{}_dev.tsv".format(model),
            test_file="{}_test.tsv".format(model),
            # 如果还有预测数据（不需要文本类别label），可以放在predict.tsv
            predict_file=pdf,
            train_file_with_header=True,
            dev_file_with_header=True,
            test_file_with_header=True,
            predict_file_with_header=True,
            # 数据集类别集合
            label_list=labels)


def main():
    # Load Paddlehub pretrained model
    # 更多预训练模型 https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel

    #设置使用的预训练模型
    model_name = "ernie_tiny"
    #model_name = "chinese-roberta-wwm-ext-large"
    module = hub.Module(name=model_name)


    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use SequenceLabelReader to read dataset
    dataset = EEDataset(args.data_dir, schema_labels, model=args.do_model)
    reader = hub.reader.SequenceLabelReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        sp_model_path=module.get_spm_path(),
        word_dict_path=module.get_word_dict_path())

    # Construct transfer learning network
    # Use "sequence_output" for token-level output.
    sequence_output = outputs["sequence_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of module need
    feed_list = [
        inputs["input_ids"].name, inputs["position_ids"].name,
        inputs["segment_ids"].name, inputs["input_mask"].name
    ]

    # Select a finetune strategy
    strategy = hub.AdamWeightDecayStrategy(
        warmup_proportion=args.warmup_proportion,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate)
        
    print("use_cuda:",args.use_gpu)

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
            eval_interval=args.eval_step,
            save_ckpt_interval=args.model_save_step,
            use_data_parallel=args.use_data_parallel,
            use_cuda=args.use_gpu,
            num_epoch=args.num_epoch,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            strategy=strategy)

    # Define a sequence labeling finetune task by PaddleHub's API
    # If add crf, the network use crf as decoder

    print("add_gru",args.add_gru)
    print("add_crf",args.add_crf)

    if args.add_gru:

        seq_label_task = SequenceLabelTaskSP(
            data_reader=reader,
            feature=sequence_output,
            feed_list=feed_list,
            max_seq_len=args.max_seq_len,
            num_classes=dataset.num_labels,
            config=config,
            add_crf=args.add_crf)
    else:
        seq_label_task = hub.SequenceLabelTask(
        data_reader=reader,
        feature=sequence_output,
        feed_list=feed_list,
        max_seq_len=args.max_seq_len,
        num_classes=dataset.num_labels,
        config=config,
        add_crf=args.add_crf)
        
    

    # 创建 LogWriter 对象
    log_writer = MyLog(mode="role2")  
    seq_label_task._tb_writer=log_writer
    
    # Finetune and evaluate model by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    if args.do_train:
        print("start finetune and eval process")
        seq_label_task.finetune_and_eval()
        seq_label_task.best_score=-999
        
    if args.do_eval:
        print("start eval process")
        seq_label_task.eval()

    if args.do_predict:
        print("start predict process")
        ret = []
        id2label = {val: key for key, val in reader.label_map.items()}
        input_data = [[d] for d in predict_data]
        run_states = seq_label_task.predict(data=input_data[1:])
        results = []
        for batch_states in run_states:
            batch_results = batch_states.run_results
            batch_infers = batch_results[0].reshape([-1]).astype(np.int32).tolist()
            seq_lens = batch_results[1].reshape([-1]).astype(np.int32).tolist()
            current_id = 0
            for length in seq_lens:
                seq_infers = batch_infers[current_id:current_id + length]
                seq_result = list(map(id2label.get, seq_infers[1: -1]))
                current_id += length if args.add_crf else args.max_seq_len
                results.append(seq_result)

        ret = []
        for sent, r_label in zip(predict_sents, results):
            sent["labels"] = r_label
            ret.append(json.dumps(sent, ensure_ascii=False))
        write_by_lines("{}.{}{}.pred".format(args.predict_data, args.do_model, args.outmark), ret)
    
    if args.do_predict2:
        print("start predict2 process")
        ret = []
        id2label = {val: key for key, val in reader.label_map.items()}
        input_data = [[d] for d in predict_data]
        run_states = seq_label_task.predict(data=input_data[1:])
        results = []
        for batch_states in run_states:
            batch_results = batch_states.run_results
            batch_infers = batch_results[0].reshape([-1]).astype(np.int32).tolist()
            seq_lens = batch_results[1].reshape([-1]).astype(np.int32).tolist()
            current_id = 0
            for length in seq_lens:
                seq_infers = batch_infers[current_id:current_id + length]
                seq_result = list(map(id2label.get, seq_infers[1: -1]))
                current_id += length if args.add_crf else args.max_seq_len
                results.append(seq_result)

        ret = []
        for sent, r_label in zip(predict_sents, results):
            sent["labels"] = r_label
            ret.append(json.dumps(sent, ensure_ascii=False))
        write_by_lines("{}.{}{}.pred".format(args.predict_data, args.do_model, args.outmark), ret)


if __name__ == "__main__":
    main()
