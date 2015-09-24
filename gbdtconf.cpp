#include<iostream>
#include<cstring>
#include"gbdtconf.h"
#include"Log.h"
namespace gbdt
{
	GbdtConf::GbdtConf(){}
	GbdtConf::~GbdtConf(){}

	int GbdtConf::Init(const char * configFile)
	{
		int ret = makePool(configFile);
		if(ret !=0) {
			Comm::LogErr("GbdtConf::Init makepool fail ");
			return -1;
		}
		const int ValueNum = 21;
		std::string sValueName[ValueNum]={
			"SpliterThreadNum",
			"SearchSplitPointerThreadNum",
			"ResidualThreadNum",
			"LearningRate",
			"FeatureNum",
			"TreeNum",
			"MaxDepth",
			"MinSampleSplit", 
			"MinSampleLeaf", 
			"SubSampleRate",
			"SubFeatureRate",
			"MaxNodes",
			"OutputModelFilePath",
			"LogLevel",
			"IsLearnNewInstances",
			"IsPushBackOgX",
			"OutputNewInstancesFilePath",
			"InputDataFilePath",
			"InputModelFilePath",
			"TestDataFliePath",
			"OutputResultFilePath"
		};
		for(int i = 0; i < ValueNum; i++)
		{
			if(pool.end() == pool.find(sValueName[i]))
			{
				Comm::LogErr("GbdtConf::Init %s not exit",sValueName[i].c_str());
				return -1;
			}
			else
			{
//				std::cout<<sValueName[i]<<" = "<<pool[sValueName[i]]<<std::endl;
			}
		}
		
		SpliterThreadNum = atoi(pool["SpliterThreadNum"].c_str());
		SearchSplitPointerThreadNum = atoi(pool["SearchSplitPointerThreadNum"].c_str());
		ResidualThreadNum = atoi(pool["ResidualThreadNum"].c_str());
		LearningRate = atof(pool["LearningRate"].c_str());
		if(LearningRate > 1 + EPS)return -1;
		FeatureNum = atoi(pool["FeatureNum"].c_str());
		TreeNum = atoi(pool["TreeNum"].c_str());
		MaxDepth = atoi(pool["MaxDepth"].c_str());
		MinSampleSplit = atoi(pool["MinSampleSplit"].c_str());
		MinSampleLeaf = atoi(pool["MinSampleLeaf"].c_str());
		SubSampleRate = atof(pool["SubSampleRate"].c_str());
		if(SubSampleRate < EPS || SubSampleRate > 1 + EPS)return -1;
		SubFeatureRate = atof(pool["SubFeatureRate"].c_str());
		if(SubFeatureRate < EPS ||SubFeatureRate > 1 + EPS)return -1;
		MaxNodes = atoi(pool["MaxNodes"].c_str());
		OutputModelFilePath = pool["OutputModelFilePath"];
		LogLevel = atoi(pool["LogLevel"].c_str());
		IsLearnNewInstances = atoi(pool["IsLearnNewInstances"].c_str());
		IsPushBackOgX = atoi(pool["IsPushBackOgX"].c_str());
		OutputNewInstancesFilePath = pool["OutputNewInstancesFilePath"];
		InputDataFilePath = pool["InputDataFilePath"];
		InputModelFilePath= pool["InputModelFilePath"];
		TestDataFliePath = pool["TestDataFliePath"];
		OutputResultFilePath = pool["OutputResultFilePath"];

		std::cout<<"SpliterThreadNum = "<<SpliterThreadNum
			<<"\nSearchSplitPointerThreadNum = "<<SearchSplitPointerThreadNum
			<<"\nresidualThreadNum = "<<ResidualThreadNum
			<<"\nLearningRate = "<<LearningRate
			<<"\nTreeNum = "<<TreeNum
			<<"\nFeatureNum = "<<FeatureNum
			<<"\nMaxDepth = "<<MaxDepth
			<<"\nMinSampleSplit = "<<MinSampleSplit
			<<"\nMinSampleLeaf = "<<MinSampleLeaf
			<<"\nSubSampleRate = "<<SubSampleRate
			<<"\nSubFeatureRate = "<<SubFeatureRate
			<<"\nMaxNodes = "<<MaxNodes
			<<"\nOutputModelFilePath = "<<OutputModelFilePath
			<<"\nLogLevel = "<<LogLevel
			<<"\nIsLearnNewInstances = "<<IsLearnNewInstances
			<<"\nIsPushBackOgX = "<<IsPushBackOgX
			<<"\nOutputNewInstancesFilePath = "<<OutputNewInstancesFilePath
			<<"\nInputDataFilePath = "<<InputDataFilePath
			<<"\nInputModelFilePath = "<<InputModelFilePath
			<<"\nTestDataFliePath = "<<TestDataFliePath
			<<"\nOutputResultFilePath = "<<OutputResultFilePath
			<<std::endl;

		return 0;
	}
}
