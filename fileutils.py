#!/usr/bin/env python
#coding:utf-8

import json
import numpy as np
import os

def wwprint(*args):
    '''
    封装的print函数
    用于解决python2.7环境下中文输出乱码的问题
    '''
    try:
        print(json.dumps(args,ensure_ascii=False))
    except:
        print(json.dumps(args))
    
def saveFile(filepath,content):
    f=open(filepath,"w")
    fc=f.write(content.encode('utf-8'))
    f.close()
    

def saveJsonLines(path, data):
    lines=[]
    for line in data:
        lines.append(json.dumps(line, ensure_ascii=False))
        
    content="\n".join(lines)
    saveFile(path,content)

def readFile(filepath):
    f=open(filepath,"r")
    fc=f.read().decode('utf-8')
    f.close()
    #print(fc)
    return fc
def readJsonLines(filepath):
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