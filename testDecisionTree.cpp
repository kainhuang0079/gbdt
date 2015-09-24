#include<iostream>
#include<cstdlib>
#include<cstdio>
#include<algorithm>
#include<assert.h>
#include"unity.h"
#include"DecisionTree.h"
#include"gbdtconf.h"
#include"Log.h"
using namespace std;
using namespace gbdt;
using namespace Comm;

int main()
{
	LogInit("w");
	GbdtConf config;
	config.Init("DT.conf");
	puts("aaaa");
	InstancePool instancepool(&config);
	instancepool.Input();
	Instance ** ppInstances;
	puts("bbbb");
	int instanceNum;
	instancepool.GetSubSamplesPtr(ppInstances,instanceNum);
	int count =0;
	for(int i=0;i<instanceNum;i++)
	{
//			ppInstances[i]->print();
	}
	printf("count = %d\n",count);
	printf("instanceNum = %d\n",instanceNum);
	sort(ppInstances,ppInstances + instanceNum,FeatureCmp(2));
	for(int i=0;i<instanceNum;i++)
	{
		ppInstances[i]->print();
	}
	vector<uint32>retvec;
	instancepool.GetSubFeatureIDs(retvec);
/*	for(int i=0;i<retvec.size();i++)
	{
		printf("IDs = %d\n",retvec[i]);
	}
	*/
	printf("IDs size = %u\n",retvec.size());
//	instancepool.print();
	printf("\n\n\n");
	DecisionTree tree(&config);
	tree.Fit(&instancepool);
	tree.print();
	printf("\n\n\n");
	FloatT p;
	instancepool.GetInstance(7164).print();

	printf("\n\n\n");
	tree.Predict(instancepool.GetInstance(7164).X,p);
	printf("predict = %f\n",p);
	return 0;
}
