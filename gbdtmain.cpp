#include"GradientBoosting.h"
#include<iostream>
#include<cstdlib>
#include<cstdio>

using namespace Comm;
using namespace gbdt;
using namespace std;

int main(int argc,char * argv[])
{
	for(int i=0;i<argc;i++)
	{
		cout<<argv[i]<<endl;
	}
	if(argc < 3)
	{
		printf("Usage: %s -c <config> -tr <newTrainOutput> -te <newTestOutput>",argv[0]);
		return -1;
	}
	map<string,string> mp;
	for(int i=1;i+1<argc;i+=2)
	{
		mp[argv[i]] = argv[i+1];
	}

	string sConfig = mp["-c"];

	string sNewTrainFile = "";
	string sNewTestFile = "";

	if(mp.find("-tr") != mp.end() && mp.find("-te") != mp.end())
	{
		sNewTrainFile = mp["-tr"];
		sNewTestFile = mp["-te"];
	}
	GbdtConf config;
	int ret = config.Init(sConfig.c_str());
	if(ret != 0)
	{
		printf("ERR:config Init fail");	
		return -1;
	}
	
	LogInit("w",config.LogLevel);

	InstancePool trainPool(&config);
	ret = trainPool.Input();
	if(ret !=0 )
	{
		printf("ERR:train data Input fail");
		return -1;
	}
	GradientBoostingForest forest(&config);
	InstancePool testinstancepool(&config);
	if(config.TestDataFliePath != "null")
	{
		ret = testinstancepool.Input(config.TestDataFliePath);
		if(ret != 0)
		{
			printf("ERR:testinstancepool input fail");
			return -1;
		}
		forest.SetTestInstancePool(&testinstancepool);
	}
	TimeStat ts(argv[0]);
	ret = forest.Fit(&trainPool);
	if(ret != 0)
	{
		printf("ERR:forest fit fail!");
		return -1;
	}
	

	if(mp.find("-tr") != mp.end() && mp.find("-te") != mp.end())
	{
		InstancePool newTrainPool(&config);
		InstancePool newTestPool(&config);
		forest.GetNewInstancePool(trainPool,newTrainPool);
		forest.GetNewInstancePool(testinstancepool,newTestPool);
		

		FILE * fpNewTrain = fopen(sNewTrainFile.c_str(), "w");
		for(int i=0;i<newTrainPool.Size();i++)
		{
			string sTmp = newTrainPool[i].ToString() + "\n" ;
			fputs(sTmp.c_str(),fpNewTrain);
		}
		fclose(fpNewTrain);


		FILE * fpNewTest = fopen(sNewTestFile.c_str(),"w");
		for(int i=0;i<newTestPool.Size();i++)
		{
			string sTmp = newTestPool[i].ToString() + "\n" ;
			fputs(sTmp.c_str(),fpNewTest);
		}

		fclose(fpNewTest);

	}



	ret = forest.SaveModel();
	if(ret != 0)
	{
		printf("ERR:forest SaveModel fail!");
		return -1;
	}
	printf("train success! FitError = %f\n",forest.FitError());
	return 0;

}
