#pragma once 
#include"threadpool.h"
#include"Log.h"
#include"unity.h"
#include"mempool.h"
#include"gbdtconf.h"
#include<cstdlib>
#include<cstdio>
#include<algorithm>
#include<vector>
#include<cstring>
#include<string>

namespace gbdt 
{
	class Instance
	{
		public:
			Instance();
			~Instance();
			std::string ToString();
			std::string DebugStr();
			void print();
		public:
			std::vector<FloatT> X;
			std::vector<int> X_BucketIndex;
			FloatT y;
			FloatT ys;
			FloatT weight;
			uint32 index;
	};


	class InstancePool
	{
		public:
			InstancePool(GbdtConf * pconfig);
			~InstancePool();
			int GetSubSamplesPtr(
					Instance ** & ppRetInstances, 
					int & RetInstanceNum
					);
			int GetSubSamplesPtr(
					FloatT SubSampleRate, 
					Instance ** & ppRetInstances, 
					int & RetInstanceNum
					);  
			int GetSubSamplesPtr(
					FloatT SubSampleRate, 
					Instance ** ppInstances,
					int instanceNum,
					Instance ** & ppRetInstances, 
					int & RetInstanceNum
					); 
			int GetSubFeatureIDs(std::vector<uint32> & SubFeatures);
			int GetSubFeatureIDs(FloatT SubFeatureRate, 
					std::vector<uint32> & SubFeatures);
			virtual int Input();
			virtual int Input(const std::string & InputDataFilePath);
			Instance & GetInstance(int index);
			Instance & operator [](int index);
			void AddInstance(const Instance & instance);
			int Size()const;
			void print();
			void MakeBucket();
		public:
			std::vector<int> m_FeatureBucketSize;
			std::vector<std::vector<FloatT> >m_FeatureBucketMap;
		private:
			int ProcessBucket(int FeatureId, std::vector<FloatT> & vecFeature);
		private:
			std::vector<Instance> m_Instances;
			GbdtConf * m_pconfig;

	};
}
