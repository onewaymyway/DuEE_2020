#coding:utf-8
#!/usr/bin/env python

from nlputils import readJsonLines,saveJsonLines
from fileutils import wwprint
from nlputils import LACTager
import collections
import argparse

class TriggerInfo():
    '''
    事件role统计
    '''

    def __init__(self,trigger):
        self.trigger=trigger
        self.reset()
        
    def reset(self):
        #事件出现次数
        self.count=0
        #事件无role的次数
        self.empty=0
        #事件出现的role
        self.roles=[]
        #事件出现role的个数统计，单个事件同个role类型可多次计数
        self.roleDic={}
        #事件出现role的个数统计，单个事件相同role类型只算一次
        self.roleSeenDic={}
        self.roleRateDic={}
        self.roleSeenRateDic={}
        #事件出现的role的长度,用于后期将明显低于最小长度的role筛除
        self.roleAugDic={}
        self.roleAugDD={}

    def addCount(self):
        self.count+=1
    def addEmpty(self):
        self.empty+=1
    def addRole(self,role,ag):
        if not role in self.roleDic:
            self.roleDic[role]=0
            self.roleAugDic[role]=[]
            self.roles.append(role)
        self.roleDic[role]+=1
        self.roleAugDic[role].append(len(ag))

    def addByEvent(self,event):
        roles=event["arguments"]
        seen={}

        #增加出现次数
        self.addCount()
        if len(roles)==0:
            #无role的情况
            self.addEmpty()

        #统计role的出现
        for role in roles:
            roleTxt=role["role"]
            argument=role["argument"]
            seen[roleTxt]=True
            self.addRole(roleTxt,argument)
        
        #类型去重的role统计
        for key in seen:
            self.roleSeenDic[key]+=1

    def checkEvent(self,event,rateLimit=90,text=None):
        roles=event["arguments"]
        seen={}

        #是否有相同role出现
        hasSameRole=False
        minRole=[]
        rolesNew=[]
        preRole=None
        for role in roles:
            roleTxt=role["role"]
            argument=role["argument"]
            if roleTxt in seen:
                hasSameRole=True
                seen[roleTxt]+=1
            else:
                seen[roleTxt]=1
            if len(argument)<self.roleAugDD[roleTxt]:
                #role长度小于最小长度的情况，过滤role或者把role拼接到前面相同的role里
                minRole.append(role)
                if preRole and preRole["role"]==roleTxt:
                    preRole["argument"]+=argument
            else:
                
                rolesNew.append(role)
                preRole=role

        event["arguments"]=rolesNew

        rolesNew2=[]


        #去除有包含关系的role,留长的还是短的好？
        for i,oRole in enumerate(rolesNew):
            hasOverlap=False
            for j in range(i+1,len(rolesNew)):
                role=rolesNew[j]
                if oRole["role"]==role["role"]:
                    if role["argument"].find(oRole["argument"])>=0:
                        wwprint("Over:",text)
                        wwprint("Overlap:",oRole,role,rolesNew)
                        hasOverlap=True
                        break
            if not hasOverlap:
                rolesNew2.append(oRole)




        event["arguments"]=rolesNew2

        if hasSameRole:
            wwprint("HasSame:",text)
            wwprint( "roles:",roles)

        #通过role出现的情况为事件成立的概率打个分
        sumRate=0
        for role,rc in seen.items():
            if rc>1:
                wwprint( "SameRole",role)
            if self.roleSeenRateDic and self.roleSeenRateDic[role]:
                sumRate+=rc*self.roleSeenRateDic[role]
        
        if sumRate>0 and sumRate<50:
            wwprint( "UnderRate",text,sumRate)
            wwprint( "UnderRate",event)
        if len(minRole)>0:
            wwprint( "minRole",text)
            wwprint( "roles:",roles)
            wwprint( "minRole",minRole)
        lost=[]

        #统计某事件中高概率出现的role却没出现的情况
        for role,rate in self.roleSeenRateDic.items():
            if rate>rateLimit:
                if not role in seen:
                    lost.append([role,rate])
        return lost

    def setBySchema(self,schemaData):
        self.reset()
        roles=schemaData["role_list"]
        for roleData in roles:
            role=roleData["role"]
            self.roleDic[role]=0
            self.roleAugDic[role]=[]
            self.roleAugDD[role]=0
            self.roleSeenDic[role]=0
            self.roles.append(role)

    def doSum(self):
        self.roleRateDic={}
        self.roleSeenRateDic={}
        emCount=self.count-self.empty
        wwprint( self.trigger,self.count,self.empty,emCount)
        
        #print(self.roleDic)
        for key,count in self.roleDic.items():
            seenCount=self.roleSeenDic[key]
            self.roleRateDic[key]=round(count*100/emCount,2)
            self.roleSeenRateDic[key]=round(seenCount*100/emCount,2)
        wwprint( self.roleRateDic)
        wwprint( self.roleSeenRateDic)

        self.roleAugDD={}
        for role,ags in self.roleAugDic.items():
            self.roleAugDD[role]=min(ags)
        wwprint( self.roleAugDD)

    def showSum(self):

        print("++++++++++++++++++++++++++++++++++++")
        self.doSum()
        
class SchemaInfo():
    '''
    事件统计
    '''
    def __init__(self):
        self.eventDic={}
        #原来打算用分词信息对role进行补全或者剔除，但是实际没有做
        self.lac=LACTager()
    def setSchemaFile(self,schemafile):
        schemeData=readJsonLines(schemafile)
        for data in schemeData:
            tf=TriggerInfo(data["event_type"])
            tf.setBySchema(data)
            self.eventDic[tf.trigger]=tf

    def addDataFile(self,dataFile):
        datas=readJsonLines(dataFile)
        for data in datas:
            text=data["text"]
            for event in data["event_list"]:
                etype=event["event_type"]          
                triggerInfo=self.getTriggerInfo(etype)
                triggerInfo.addByEvent(event)
               

    def getTriggerInfo(self,trigger):
        return self.eventDic[trigger]

    def setSchemaData(self,schemaDataFile):
        pass

    def showSum(self):
        for key,tirggerInfo in self.eventDic.items():
            tirggerInfo.showSum()
    def doSum(self):
        for key,tirggerInfo in self.eventDic.items():
            tirggerInfo.doSum()

    def checkFile(self,dataFile,rateLimit=90,showWarning=True,saveFile=False):
        datas=readJsonLines(dataFile)
        warns=[]
        roleCount=0
        roleCharCount=0
        emptyCount=0
        emptyRoleCount=0
        roleCountDic={}
        roleAugDic={}
        for data in datas:
            text=data["text"]
            if len(data["event_list"])==0:
                emptyCount+=1
            preRoleCount=roleCount
            for event in data["event_list"]:
                etype=event["event_type"]
                roles=event["arguments"]
                for role in roles:
                    roleCount+=1
                    role_type=role["role"]
                    role_aug=role["argument"]
                    if not role_type in roleAugDic:
                        roleAugDic[role_type]={}
                    roleAugDic[role_type][role_aug]=True
                    roleCharCount+=len(role["argument"])
                triggerInfo=self.getTriggerInfo(etype)
                lost=triggerInfo.checkEvent(event,rateLimit,text)
                if len(lost)>0:
                    warns.append(event)
                    if showWarning:
                        wwprint( "warning:",text,len(data["event_list"]))
                        wwprint( "tags:",self.lac.getTag(text))
                        wwprint( "warning event:",event)
                        wwprint( "warning lost",lost)
            dRole=roleCount-preRoleCount
            if dRole not in roleCountDic:
                roleCountDic[dRole]=0
            roleCountDic[dRole]+=1
        warnCount=len(warns)
        wwprint( "warnCount:",warnCount)
        wwprint( "role",roleCount,roleCharCount)
        wwprint( "warnrate:",round(warnCount*100/roleCount,2))
        dataCount=len(datas)
        wwprint( "empty:",emptyCount,roleCountDic)
        items=list(roleCountDic.items())
        items.sort()
        for key,count in items:
            tt=key*count
            wwprint( "c",key,count,round(count*100/dataCount,2),round(tt*100/roleCount,2))
        if saveFile:
            saveJsonLines(dataFile.replace(".json","_md.json"),datas)
        #print(roleAugDic)
        for role,roleDic in roleAugDic.items():
            wwprint( role,list(roleDic.keys()))

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--targetFile", type=str, default=None, help="./data1/test1fn_pred.json")
args = parser.parse_args()

def main():

    
    #删选不合适的结果 比如明显低于限制的
    #类型不匹配的
    #去重
    #重组
    #高效触发词
    schemaInfo=SchemaInfo()
    schemaInfo.setSchemaFile("./data1/event_schema.json")
    schemaInfo.addDataFile("./data1/train.json")
    #schemaInfo.showSum()

    schemaInfo.checkFile(args.targetFile,90,False,True)#

if __name__ == "__main__":
    main()