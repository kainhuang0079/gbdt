#pragma once
#include"unity.h"
#include<string>
namespace gbdt
{
	typedef unsigned int uint32; 
	typedef unsigned long long uint64;
	typedef float FloatT;
//	typedef float FloatT;
	const FloatT EPS = 1e-6;
	class GbdtConf:public Comm::Config
	{
		public:
			GbdtConf();
			~GbdtConf();
			int Init(const char * configFile);

		public:
			int SpliterThreadNum;
			int SearchSplitPointerThreadNum;
			int ResidualThreadNum;
			FloatT LearningRate;
			int FeatureNum;
			int TreeNum;
			int MaxDepth;
			int MinSampleSplit;
			int MinSampleLeaf;
			FloatT SubSampleRate;
			FloatT SubFeatureRate;
			int MaxNodes;
			std::string OutputModelFilePath;
			int LogLevel;
			int IsLearnNewInstances;
			int IsPushBackOgX;
			std::string OutputNewInstancesFilePath;
			std::string InputDataFilePath;
			std::string InputModelFilePath;
			std::string TestDataFliePath;
			std::string OutputResultFilePath;
	};
}
