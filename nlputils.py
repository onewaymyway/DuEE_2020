#!/usr/bin/env python
#coding:utf-8

import json
import numpy as np
import os

from visualdl import LogWriter

class MyLog():
    '''
    本类用于适配PaddleHub在AIStudio中VisualDL的使用
    使用方式：
    # 创建 LogWriter 对象
    log_writer = MyLog(mode="role2")  
    seq_label_task._tb_writer=log_writer
    '''
    
    def __init__(self,mode="train",logDir="../log"):
        self.mode=mode
        self.varDic={}
        self.log_writer = LogWriter(logDir, sync_cycle=10)   
        
        
        
    def add_scalar(self,tag,scalar_value,global_step):
        if not tag in self.varDic:
            with self.log_writer.mode(self.mode) as writer:
                self.varDic[tag]=writer.scalar(tag)
        self.varDic[tag].add_record(global_step,scalar_value)      
        
def saveFile(filepath,content):
    '''
    保存文件
    '''
    f=open(filepath,"w")
    fc=f.write(content.encode('utf-8'))
    f.close()
    

def saveJsonLines(path, data):
    '''
    保存文件
    '''
    lines=[]
    for line in data:
        lines.append(json.dumps(line, ensure_ascii=False))
        
    content="\n".join(lines)
    saveFile(path,content)

def readFile(filepath):
    '''
    读取文件
    '''
    f=open(filepath,"r")
    fc=f.read().decode('utf-8')
    f.close()
    #print(fc)
    return fc
def readJsonLines(filepath):
    '''
    读取文件
    '''
    print("readJsonLines",filepath)
    lines=readFile(filepath).split("\n")
    linefiles=[]
    for line in lines:

        line=line.strip()
        if not line:
            continue
        dd=json.loads(line)
        linefiles.append(dd)
    return linefiles


import paddlehub as hub        

class LACTager(object):
    '''
    封装的分词工具
    '''
    def __init__(self):
        self.module = hub.Module(name="lac")
        
    def getTagResult(self,text):
        inputs = {"text": [text]}
        results = self.module.lexical_analysis(data=inputs)
        result=results[0]
        return result
        
    def getTag(self,text):
        result=self.getTagResult(text)
        start=0
        rst=[]
        for word,ner in zip(result["word"],result["tag"]):
            rst.append([word,ner])
        return rst
        
    def getLabels(self,text):
        result=self.getTagResult(text)
        labels=[""]*len(text)
        start=0
        for word,ner in zip(result["word"],result["tag"]):
            #print(word,ner)
            label_dataOT(labels,start,len(word),ner)
            start+=len(word)

        return labels