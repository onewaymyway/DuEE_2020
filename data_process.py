#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""hello world"""
import os
import sys
import json
import argparse
from fileutils import wwprint


def read_by_lines(path, encoding="utf-8"):
    """read the data by line"""
    result = list()
    with open(path, "r") as infile:
        for line in infile:
            result.append(line.strip().decode(encoding))
    return result


def write_by_lines(path, data, t_code="utf-8"):
    """write the data"""
    with open(path, "w") as outfile:
        [outfile.write(d.encode(t_code) + "\n") for d in data]


def get_adptText(text):
    text_a = [
                u"，" if t == u" " or t == u"\n" or t == u"\t" else t
                for t in list(text.lower())
            ]
    return text_a
def data_process(path, model="trigger", is_predict=False,schema_labels=None):
    """data_process"""

    def label_data(data, start, l, _type):
        """label_data"""
        for i in range(start, start + l):
            suffix = u"B-" if i == start else u"I-"
            data[i] = u"{}{}".format(suffix, _type)
        return data

    sentences = []
    output = [u"text_a"] if is_predict else [u"text_a\tlabel"]
    with open(path) as f:
        for line in f:
            d_json = json.loads(line.strip().decode("utf-8"))
            _id = d_json["id"]
            tDTxt=get_adptText(d_json["text"])
            text_a=tDTxt

            if is_predict:
                if "event_lists" in d_json:
                    for ccevent in d_json["event_lists"]:
                        #将event_type和text拼接在一起
                        tAdpt=ccevent+":"+d_json["text"]
                        tDTxt=get_adptText(tAdpt)
                        sentences.append({"text": d_json["text"], "id": _id,"e":ccevent})
                        output.append(u'\002'.join(tDTxt))

                     
                else:
                     sentences.append({"text": d_json["text"], "id": _id})
                     output.append(u'\002'.join(tDTxt))

            else:
                if model == u"trigger":
                    labels = [u"O"] * len(text_a)
                    for event in d_json["event_list"]:
                        event_type = event["event_type"]
                        start = event["trigger_start_index"]
                        trigger = event["trigger"]
                        labels = label_data(labels, start,
                                            len(trigger), event_type)
                    output.append(u"{}\t{}".format(u'\002'.join(text_a),
                                                   u'\002'.join(labels)))
                elif model == u"role":
                    for event in d_json["event_list"]:
                        labels = [u"O"] * len(text_a)
                        for arg in event["arguments"]:
                            role_type = arg["role"]
                            argument = arg["argument"]
                            start = arg["argument_start_index"]
                            labels = label_data(labels, start,
                                                len(argument), role_type)
                        output.append(u"{}\t{}".format(u'\002'.join(text_a),
                                                       u'\002'.join(labels)))
                elif model == u"role1":
                    events={}
                    for event in d_json["event_list"]:
                        tEventType=event["event_type"]
                        if not tEventType in events:
                            events[tEventType]=[]
                        for arg in event["arguments"]:
                            events[tEventType].append(arg)
                    for event,arguments in events.items():
                        labels = [u"O"] * len(text_a)
                        arguments.sort(key=lambda x:(x["argument_start_index"],-len(x["argument"])))
                        for arg in arguments:
                            role_type = arg["role"]
                            argument = arg["argument"]
                            start = arg["argument_start_index"]
                            if schema_labels and not u"B-"+role_type in schema_labels:
                                print("Wrong:",role_type,d_json["text"])
                            labels = label_data(labels, start,
                                                len(argument), role_type)
                        appendStr=event+":"
                        #将eventtype拼接到句子前面
                        labels=[u"O"]*len(appendStr)+labels
                        tTxt=list(appendStr)+text_a
                        output.append(u"{}\t{}".format(u'\002'.join(tTxt),
                                                       u'\002'.join(labels)))
    if is_predict:
        return sentences, output
    else:
        return output


def schema_process(path, model="trigger"):
    """schema_process"""

    def label_add(labels, _type):
        """label_add"""
        if u"B-{}".format(_type) not in labels:
            labels.extend([u"B-{}".format(_type), u"I-{}".format(_type)])
        return labels

    labels = []
    with open(path) as f:
        for line in f:
            d_json = json.loads(line.strip().decode("utf-8"))
            if model == u"trigger":
                labels = label_add(labels, d_json["event_type"])
            elif model == u"role":
                for role in d_json["role_list"]:
                    labels = label_add(labels, role["role"])
            elif model == u"role1":
                for role in d_json["role_list"]:
                    labels = label_add(labels, role["role"])
    labels.append(u"O")
    return labels


def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    if len(labels)>len(text):
        wwprint("warning",text,labels,len(labels),len(text))
    for i, label in enumerate(labels):
        if i>=len(text):
            continue
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret

roleCoDic={
    u"召回方,召回内容": 9, 
    u"原所属组织,离职者": 166, 
    u"死者年龄,死者": 70, 
    u"降价方,降价物": 9, 
    u"解雇方,被解雇人员": 5,
    u"原所属组织,退出方": 7,
    u"时间,活动名称": 6, 
    u"地点,袭击对象": 6,
    u"时间,发布产品": 8,
    u"罢工人数,罢工人员": 4,
    u"时间,夺冠赛事": 11, 
    u"发布方,发布产品": 78, 
    u"被下架方,下架产品": 14, 
    u"所属组织,停职人员": 14, 
    u"地点,活动名称": 13, 
    u"出售方,交易物": 13, 
    u"地点,死者": 12, 
    u"时间,赛事名称": 38, 
    u"所属组织,罢工人员": 21}    
def extract_resultEX(text, labels,cDic):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    for i, label in enumerate(labels):
        if i>=len(text):
            continue
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True
                newTxt=[]
                if cur_type!=None:
                    adkey=cur_type+","+_type
                    #print("adkey",adkey,u"召回方,召回内容")
                    #print(cDic[u"召回方,召回内容"])
                    if cur_type+","+_type in cDic:
                        newTxt+=ret[-1]["text"]
                        wwprint("concat by B","".join(newTxt))
                    if cur_type==_type:
                        wwprint("sametype:",cur_type,ret[-1],text,labels)
                        
                cur_type = _type
                newTxt.append(text[i])
                ret.append({"start": i-len(newTxt)+1, "text": newTxt, "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                
                is_start = True
                newTxt=[]
                if cur_type!=None:
                    adkey=cur_type+","+_type
                    #print("adkey",adkey,u"召回方,召回内容")
                    #print(cDic[u"召回方,召回内容"])
                    if cur_type+","+_type in cDic:
                        newTxt+=ret[-1]["text"]
                        wwprint("concat by I","".join(newTxt))
                        
                cur_type = _type
                newTxt.append(text[i])
                ret.append({"start": i-len(newTxt)+1, "text": newTxt, "type": _type})
                
                #cur_type = _type
                
                #ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
                
    return ret

def adptRet(ret):
    ret.sort(key=lambda x:x["start"])
    lenRet=len(ret)
    rst=[]
    for i in range(lenRet):
        isMerged=False
        tRet=ret[i]
        if i<lenRet-1:
            for j in range(i+1,lenRet):
                
                nRet=ret[j]
                if tRet["type"]==nRet["type"] and tRet["start"]+len(tRet["text"])==nRet["start"]:
                    wwprint("canMerge",tRet,nRet,ret)
                    isMerged=True
                    nRet["start"]=tRet["start"]
                    nRet["text"]=tRet["text"]+nRet["text"]
                    break
        if not isMerged:
            rst.append(tRet)
                
    return rst 
    
def predict_data_process(trigger_file, role_file, schema_file, save_path):
    """predict_data_process"""
    pred_ret = []
    trigger_datas = read_by_lines(trigger_file)
    role_datas = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]
    # 将role数据进行处理
    sent_role_mapping = {}
    for d in role_datas:
        d_json = json.loads(d)
        r_ret = extract_result(d_json["text"], d_json["labels"])
        role_ret = {}
        for r in r_ret:
            role_type = r["type"]
            if role_type not in role_ret:
                role_ret[role_type] = []
            role_ret[role_type].append(u"".join(r["text"]))
        sent_role_mapping[d_json["id"]] = role_ret

    for d in trigger_datas:
        d_json = json.loads(d)
        t_ret = extract_result(d_json["text"], d_json["labels"])
        pred_event_types = list(set([t["type"] for t in t_ret]))
        event_list = []
        for event_type in pred_event_types:
            role_list = schema[event_type]
            arguments = []
            for role_type, ags in sent_role_mapping[d_json["id"]].items():
                if role_type not in role_list:
                    continue
                for arg in ags:
                    if len(arg) == 1:
                        # 一点小trick
                        continue
                    arguments.append({"role": role_type, "argument": arg})
            event = {"event_type": event_type, "arguments": arguments}
            event_list.append(event)
        pred_ret.append({
            "id": d_json["id"],
            "text": d_json["text"],
            "event_list": event_list
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    write_by_lines(save_path, pred_ret)

def predict_data_process2(trigger_file, role_file, schema_file, save_path):
    """predict_data_process"""
    '''
    根据trigger预测结果生成role预测需要的数据
    '''
    pred_ret = []
    trigger_datas = read_by_lines(trigger_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]
    # 将role数据进行处理
    sent_role_mapping = {}


    for d in trigger_datas:
        d_json = json.loads(d)
        t_ret = extract_result(d_json["text"], d_json["labels"])
        pred_event_types = list(set([t["type"] for t in t_ret]))
        event_list = []
        pred_ret.append({
            "id": d_json["id"],
            "text": d_json["text"],
            "event_lists": pred_event_types
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    write_by_lines(save_path, pred_ret)

def predict_data_process3(trigger_file, role_file, schema_file, save_path):
    """predict_data_process3"""
    '''
    根据role预测结果生成最终的结果
    '''
    print("predict_data_p3 new")
    pred_ret = []
    #trigger_datas = read_by_lines(trigger_file)
    role_datas = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]
    # 将role数据进行处理
    sent_role_mapping = {}
    
    itemsDic={}
    for d in role_datas:
        d_json = json.loads(d)
        tID=d_json["id"]
        tTxtStr=d_json["text"]
        if not tID in itemsDic:
            tItem={
                "id":tID,
                "text":tTxtStr,
                "event_list":[]
            }
            itemsDic[tID]=tItem
            pred_ret.append(tItem)
        tItem=itemsDic[tID]
        #print("d_json:",d_json)
        tEventType=d_json["e"]
        tEventO={
            "event_type":tEventType,
            "arguments":[]
        }
        tItem["event_list"].append(tEventO)
        #r_ret = extract_result(d_json["e"]+":"+d_json["text"], d_json["labels"])
        #role预测的时候句子是 event_type:text,解码的时候也要这样解码才能对上
        r_ret = extract_resultEX(d_json["e"]+":"+d_json["text"], d_json["labels"],roleCoDic)
        role_ret = {}
        role_list = schema[tEventType]
        for r in r_ret:
            role_type = r["type"]
            if role_type not in role_list:
                #事件类型不包含对于的role类型
                wwprint("wrong role:",d_json["text"],r)
                continue
            tRole={
                "role":r["type"],
                "argument":u"".join(r["text"])
            }
            tEventO["arguments"].append(tRole)
        

    
        
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    write_by_lines(save_path, pred_ret)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Official evaluation script for DuEE version 0.1.")
    parser.add_argument(
        "--trigger_file",
        help="trigger model predict data path",
        required=False)
    parser.add_argument(
        "--role_file", help="role model predict data path", required=False)
    parser.add_argument(
        "--schema_file", help="schema file path", required=True)
    parser.add_argument("--save_path", help="save file path", required=True)
    parser.add_argument("--action", type=str, default="predict_data_process", choices=["predict_data_process", "predict_data_process2", "predict_data_process3"], help="predict_data_process")
    args = parser.parse_args()
    if args.action=="predict_data_process":
        predict_data_process(args.trigger_file, args.role_file, args.schema_file,
                         args.save_path)
    elif args.action=="predict_data_process2":
        predict_data_process2(args.trigger_file, args.role_file, args.schema_file,
                         args.save_path)
    elif args.action=="predict_data_process3":
        predict_data_process3(args.trigger_file, args.role_file, args.schema_file,
                         args.save_path)
    
