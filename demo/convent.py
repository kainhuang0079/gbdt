import os 
import sys
SEED = [37,53,29]
MOD = 107
BEGIN_DATE = '2014-11-18'
END_DATE = '2014-12-18'
def hashing(strline,seed,mod):
    ret = 0
    for c in strline:
        ret += ret * seed + int(c)
        ret %= mod
    return ret 

if __name__ == '__main__':
#    print sys.argv
    if(len(sys.argv) < 6):
        print 'Usage:%s <PsetInput> <trainoutput> <testoutput> <testuiinfo> <checkoutput>' % sys.argv[0] 
        exit(0)
    PsetInput = open(sys.argv[1],'r')
    trainoutput = open(sys.argv[2],'w')
    testoutput = open(sys.argv[3],'w')
    testuiinfo = open(sys.argv[4],'w')
    checkoutput = open(sys.argv[5],'w')
    Plis = PsetInput.readlines()
    Pset = set()
    for p in Plis:
        Pset.add(p.strip())
    for line in sys.stdin:
        time,vec,tag = line.strip().split('\t')
        #col = vec.strip().split(',')
        col = vec.strip().split(',')
        vect = []
        #for i in range(MOD):
         #   vect.append('0')
        #for seed in SEED:
         #   ha = hashing("111111" + col[1],seed,MOD)
          #  vect[ha] = '1'
        #for seed in SEED:
         #   ha = hashing("222222" + col[2],seed,MOD)
          #  vect[ha] = '1'
    #    vect.append(col[1])
     #   vect.append(col[2])

     #   sum_uid_iid_op =0 ;
      #  for i in range(3,7):
       #     sum_uid_iid_op += int(float(col[i]))
        
#        if col[2] not in Pset:
#            continue
        for i in range(3, len(col)):
            vect.append(col[i]) 
        vec = ''
        if col[0] == '0':
            vec = '1,'
        elif col[0] == '1':
            vec = '1,'
        vec += col[0] + ',' + ','.join(vect)
        

        if tag == 'test':
            testuiinfo.write(col[1]+','+col[2]+'\n')
            testoutput.write(vec + '\n')
        if tag == 'train': 
            trainoutput.write(vec + '\n')
        if tag == 'check':
            checkoutput.write(col[1]+','+col[2]+'\n')
    PsetInput.close()
    trainoutput.close()
    testoutput.close()
    testuiinfo.close()
    checkoutput.close()
