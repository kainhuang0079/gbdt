import random
import time
import os
import sys
import math 
import datetime

StatDict = {}
lines = []
BEGIN_DATE = "2014-11-18"
END_DATE = "2014-12-18"
PRO_LIMIT = 0.02
PER_DAYS = -6
PER_DAYS_UID_IID_OP = -2
EPS = 0.000001

def Add(key , value):
    if key not in StatDict:
        StatDict[key] = 0
    StatDict[key] += value

def Get(key):
    if key not in StatDict:
        return 0
    else:
        return StatDict[key]

def Set(key,value):
    StatDict[key] = value

def GetNextDate(datestr, delta):
    date = datetime.datetime.strptime(datestr,'%Y-%m-%d')
    date = date + datetime.timedelta(days = delta)
    return date.strftime('%Y-%m-%d')

def GetWeekDay(datestr):
    date = datetime.datetime.strptime(datestr,'%Y-%m-%d')
    return date.weekday()

def Stat(line):
    col = line.strip().split(',') 
    nowtime = col[5]
    uid = col[0]
    iid = col[1]
    hour = col[6]
    behavior = col[2]
    geohash = col[3][0:6]
    item_category = col[4]

    key = nowtime + '|' + uid + '|' + behavior
    Add(key , 1)

    key = nowtime + '|' + iid + '|' + behavior
    Add(key, 1)

    key = nowtime + '|' + uid + '|' + iid + '|' + behavior
    Add(key ,1)

    key = nowtime + '|' + uid + '|' + item_category + '|' + behavior
    Add(key,1)

    key = nowtime + '|' + item_category + '|' + behavior
    Add(key,1)
    if geohash != '':
        key = nowtime + '|' + geohash + '|' + behavior
        Add(key,1)

        key = nowtime + '|' + uid + '|' + iid + '|' + geohash + '|' + behavior
        Add(key,1)

        key = nowtime + '|' + uid + '|' + geohash + '|' +behavior
        Add(key,1)

        key = nowtime + '|' + iid + '|' + geohash + '|' + behavior
        Add(key,1)

    key = nowtime + '|' + uid + '|' + iid + '|' + 'firsthour'
    if key not in StatDict:
        Set(key,int(hour))

    key = nowtime + '|' + uid + '|' + iid + '|' + 'lasthour'
    Set(key,int(hour))

def GetInstance(nowtime, uid, iid, behavior, item_category, geohash, tag):
    
    vec = []
    vec.append(nowtime);
    if behavior == '4':
        vec.append(1)
    else:
        vec.append(0)
#    if vec[1] == 1 and tag == 'test':
 #       return 
    if ((vec[1] == 0)) :
        if tag == 'train' :#or tag == 'test':
            r = random.random()
#            print 'r = %f PRO_LIMIT = %f' % (r,PRO_LIMIT)
            if r > PRO_LIMIT:
                return 
    if tag == 'check' and vec[1] == 0:
        return 
    vec.append(uid);
    vec.append(iid);

    days = 0
    for per_days in range(PER_DAYS,1):
#        print GetNextDate(nowtime, per_days),BEGIN_DATE 
        if cmp(GetNextDate(nowtime,per_days), BEGIN_DATE) == -1:
            continue
        else:
            days += 1
    if 0 == days:
        print "days == 0", nowtime, uid, iid, behavior, item_category
        return

    
    uid_iid_op_days = 0
    for per_days in range(PER_DAYS_UID_IID_OP,1):
        if cmp(GetNextDate(nowtime,per_days),BEGIN_DATE) == -1:
            continue
        else:
            uid_iid_op_days += 1
    if 0 == uid_iid_op_days:
        return 

    sum_ys_ui_op = 0

    for i in range(1,5):
        key = GetNextDate(nowtime,0) + '|' + uid + '|' + iid + '|' + str(i)
        tmp = Get(key)
        sum_ys_ui_op += tmp
#        vec.append(tmp)

    #if sum_ys_ui_op == 0 and ( tag == 'train' or tag == 'test' ):
     #   return 

    uid_iid_op = [0,0,0,0,0]

    for per_days in range(PER_DAYS_UID_IID_OP,1):
        for i in range(1,5):
            key = GetNextDate(nowtime,per_days) + '|' + uid + '|' + iid + '|' + str(i)
            uid_iid_op[i] += Get(key)

    sum_uid_iid_op = 0
    for i in range(1,5):
        sum_uid_iid_op += int(uid_iid_op[i])
        vec.append(uid_iid_op[i] * 1.0 /uid_iid_op_days)

    if sum_uid_iid_op == 0 and (tag == 'train' or tag == 'test'):
        #print "days == 0", nowtime, uid, iid, behavior, item_category
        return 

    for i in range(1,5):
        if sum_uid_iid_op != 0:
            vec.append(uid_iid_op[i] * 1.0/sum_uid_iid_op)
        else:
            vec.append(0)

    for i in range(1,4):
        if uid_iid_op[i] != 0:
            vec.append(uid_iid_op[4] * 1.0/uid_iid_op[i])
        else:
            vec.append(0)

    uid_op = [0,0,0,0,0]
    for per_days in range(PER_DAYS, 1):
        for i in range(1,5):
            key = GetNextDate(nowtime, per_days) + '|' + uid + '|' + str(i)
            uid_op[i] += Get(key)
    sum_uid_op = 0
    for i in range(1,5):
        sum_uid_op += uid_op[i]
        vec.append(uid_op[i] * 1.0 / days)
    
    for i in range(1,5):
        if sum_uid_op != 0:
            vec.append(uid_op[i] * 1.0 / sum_uid_op)
        else:
            vec.append(0)
    
    for i in range(1,4):
        if uid_op[i] != 0:
            vec.append(uid_op[4] * 1.0/uid_op[i])
        else:
            vec.append(0)

    

    iid_op = [0,0,0,0,0]
    for per_days in range(PER_DAYS,1):
        for i in range(1,5):
            key = GetNextDate(nowtime,per_days) + '|' + iid + '|' + str(i)
            iid_op[i] += Get(key)

    sum_iid_op = 0
    for i in range(1,5):
        sum_iid_op += iid_op[i]
        vec.append(iid_op[i] * 1.0 / days)

    for i in range(1,5):
        if sum_iid_op != 0:
            vec.append(iid_op[i] * 1.0 / sum_iid_op)
        else:
            vec.append(0)

    for i in range(1,4):
        if iid_op[i] != 0:
            vec.append(iid_op[4] * 1.0 / iid_op[i])
        else:
            vec.append(0)


    uid_icate_op = [0,0,0,0,0]
    for per_days in range(PER_DAYS,1):
        for i in range(1,5):
            key = GetNextDate(nowtime,per_days) + '|' + uid + '|' + item_category +'|' + str(i)
            uid_icate_op[i] += Get(key)
    
    sum_uid_icate_op = 0
    for i in range(1,5):
        sum_uid_icate_op += uid_icate_op[i]
        vec.append(uid_icate_op[i] * 1.0 / days)

    for i in range(1,5):
        if sum_uid_icate_op != 0:
            vec.append(uid_icate_op[i] * 1.0 / sum_uid_icate_op)
        else:
            vec.append(0)

    for i in range(1,4):
        if uid_icate_op[i] != 0:
            vec.append(uid_icate_op[4] * 1.0 / uid_icate_op[i])
        else:
            vec.append(0)



    icate_op = [0,0,0,0,0]

    for per_days in range(PER_DAYS,1):
        for i in range(1,5):
            key  = GetNextDate(nowtime,per_days) + '|' + item_category + '|' + str(i)
            icate_op[i] += Get(key)

    sum_icate_op = 0
    for i in range(1,5):
        sum_icate_op += icate_op[i]
        vec.append(icate_op[i] * 1.0 /days)

    for i in range(1,5):
        if sum_icate_op != 0:
            vec.append(icate_op[i] * 1.0 / sum_icate_op)
        else:
            vec.append(0)

    for i in range(1,4):
        if icate_op[i] != 0:
            vec.append(icate_op[4] * 1.0 / icate_op[i])
        else:
            vec.append(0)

    
    '''
    uid_iid_geohash_op = [0,0,0,0,0]
    
    for per_days in range(PER_DAYS,1):
        for i in range(1,5):
            key = GetNextDate(nowtime,per_days) + '|' + uid + '|' + iid + '|' + geohash + '|' + str(i)
            uid_iid_geohash_op[i] += Get(key)


    sum_uid_iid_geohash_op = sum(uid_iid_geohash_op)

    for i in range(1,5):
        vec.append(uid_iid_geohash_op[i] * 1.0 / days)

    for i in range(1,5):
        if sum_uid_iid_geohash_op != 0:
            vec.append(uid_iid_geohash_op[i] *1.0 / sum_uid_iid_geohash_op)
        else:
            vec.append(0)
    '''
    uid_geohash_op = [0,0,0,0,0]

    for per_days in range(PER_DAYS,1):
        for i in range(1,5):
            key = GetNextDate(nowtime,per_days) + '|' + uid + '|' + geohash + '|' + str(i)
            uid_geohash_op[i] += Get(key)

    sum_uid_geohash_op = sum(uid_geohash_op)

    for i in range(1,5):
        vec.append(uid_geohash_op[i] * 1.0 / days)

    for i in range(1,5):
        if sum_uid_geohash_op != 0:
            vec.append(uid_geohash_op[i] * 1.0 / sum_uid_geohash_op)
        else:
            vec.append(0)
    
    geohash_op = [0,0,0,0,0]

    for per_days in range(PER_DAYS,1):
        for i in range(1,5):
            key = GetNextDate(nowtime,per_days) + '|' + geohash + '|' +str(i)
            geohash_op[i] += Get(key)

    sum_geohash_op = sum(geohash_op)
    
    for i in range(1,5):
        vec.append(geohash_op[i] * 1.0 / days)
    
    for i in range(1,5):
        if sum_geohash_op != 0:
            vec.append(geohash_op[i] * 1.0 / sum_geohash_op)
        else:
            vec.append(0)
     
    
    key = nowtime + '|' + uid + '|' + iid + '|' + 'firsthour'
    firsthour = Get(key)

    key = nowtime + '|' + uid + '|' + iid + '|' + 'lasthour'
    lasthour = Get(key)

    vec.append(float(lasthour - firsthour))

    lasttime_one_hot = [0,0,0,0,0]

    lasttime_one_hot[lasthour / 6 + 1] = 1 

    for val in lasttime_one_hot:
        vec.append(val)

    week_day_one_hot = [0,0,0,0,0,0,0]
    week_day_one_hot[GetWeekDay(nowtime)] = 1

    for val in week_day_one_hot:
        vec.append(val)

    if vec[1] == 1:
        stroutput = vec[0] + '\t' + str( vec[1] ) + ',' + vec[2] + ',' + vec[3]
    else:
        stroutput = vec[0] + '\t' + str( vec[1] ) + ',' + vec[2] + ',' + vec[3]


    for i in range(4,len(vec)):
        stroutput += ','
        stroutput += ('%.4f' % vec[i])
    
    stroutput += ('\t' + tag)

    print stroutput
    
     



if __name__ == "__main__":
#    print GetNextDate('2014-12-12',-1)

    if len(sys.argv) < 2:
        print 'Usage: %s <Pset>' % sys.argv[0]
        exit(-1)

    ispset = open(sys.argv[1], 'r')
    psetlines = ispset.readlines()
    Pset = set()
    for p in psetlines:
        Pset.add(p.strip())



    for line in sys.stdin:
        lines.append(line)
    for line in lines:
        Stat(line)

    flagset = {}

    for line in lines:
        col = line.strip().split(',')
        nowtime = col[5]
        uid = col[0]
        iid = col[1]
        hour = col[6]
        behavior = col[2]
        geohash = col[3][0:6]
        item_category = col[4]
        
        
        if iid not in Pset:
            continue

        if cmp(nowtime,'2014-11-20') == 1 and cmp(nowtime,'2014-12-17') == -1:
            key = GetNextDate(nowtime,1) + '|' + uid +'|' + iid + '|' + '4'
            val = Get(key)
            tmp = '0'
            if val >= 1:
                tmp = '4'
            GetInstance(nowtime, uid, iid, tmp, item_category, geohash, 'train')

        if nowtime == '2014-12-17' or nowtime == '2014-12-16':
            tmpkey = uid + '|' + iid
            if tmpkey not in flagset:
                key = '2014-12-18' + '|' + uid + '|' + iid + '|' +'4'
                val = Get(key)
                tmp = '0'
                if val >= 1 :
                    tmp = '4'
                GetInstance('2014-12-17',uid,iid,tmp,item_category,geohash, 'test')
                flagset[tmpkey] = 1

        if nowtime == '2014-12-18':
            GetInstance('2014-12-18',uid,iid,behavior,item_category,geohash,'check')


