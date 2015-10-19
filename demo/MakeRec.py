import os
import sys


if __name__ == '__main__':
    if len(sys.argv) < 7:
        print 'usage:%s <result> <uiinfo> <pset> <check> <output> <topN>' % sys.argv[0]
        exit(-1)

    isresult = open(sys.argv[1],'r')
    isuiinfo = open(sys.argv[2],'r')
    ispset = open(sys.argv[3],'r')
    ischeck = open(sys.argv[4],'r')
    osoutput = open(sys.argv[5],'w')

    topN = int(sys.argv[6])
    resultlines = isresult.readlines()
    uiinfolines = isuiinfo.readlines()
    psetlines = ispset.readlines()
    checklines = ischeck.readlines()

    CheckSet = set()
    for ui in checklines:
        CheckSet.add(ui.strip())

    Pset = set()
    for p in psetlines:
        Pset.add(p.strip())

    ls = []
    for i in range(len(resultlines)):
        tup = (float(resultlines[i].strip()),uiinfolines[i].strip())
        ls.append(tup)
    ls.sort(reverse = True, key = lambda x:x[0])
    
    osoutput.write('user_id,item_id\n')
    cnt = 0
    hit = 0
    for i in range(topN):
        uid,iid = ls[i][1].strip().split(',')
        if iid in Pset:
            cnt += 1
            if ls[i][1] in CheckSet:
                hit += 1
            print ls[i][0]
            osoutput.write(uid + ',' + iid + '\n')

    print 'cnt = ' + str(cnt) + ', hit = ' + str(hit)
    
    

    isresult.close()
    isuiinfo.close()
    ispset.close()
    osoutput.close()


