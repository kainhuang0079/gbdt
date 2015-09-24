#include"instancepool.h"
#include"gbdtconf.h"
#include"unity.h"
#include"instancepool.h"
#include"Log.h"
#include"threadpool.h"
#include"DecisionTree.h"
#include<cstdio>
#include<cstring>
#include<string>
#include<iostream>
#include <utility>

namespace gbdt
{
	class SparseInstance
	{
		public:
			SparseInstance();
			~SparseInstance();
			std::string ToString();
		public:
			std::vector< std::pair<int,FloatT> > X;
			FloatT y;
			FloatT ys;
			FloatT weight;
			uint32 index;
	};
	class GradientBoostingForest
	{
		public:
			GradientBoostingForest(GbdtConf * pconfig);
			~GradientBoostingForest();
			int Fit(InstancePool *pInstancepool);
			int SaveModel();
			int LoadModel();
			int Predict(const std::vector<FloatT> &X, FloatT &predict);
			FloatT FitError();
			void SetTestInstancePool(InstancePool * pTestInstancePool);
			FloatT TestError();
			int GetNewInstance(const Instance & instance, Instance & newInstance);
			int GetNewInstance(const Instance & instance, SparseInstance & newInstance);

			int GetNewInstancePool(InstancePool &instancepool,InstancePool &newInstancePool);
		private:
			int Residual();
			int LearnNewInstance();
			int SaveResult();
			void FeatureStat();

			int m_TotLeafCnt;
			GbdtConf * m_pconfig;
			InstancePool * m_pInstancePool;
			InstancePool * m_pTestInstancePool;
			std::vector<DecisionTree *> m_Forest;
	};
	class ResidualThreadWork : public Comm::Work
	{
		public:
			ResidualThreadWork(
					GbdtConf * pconfig,
					InstancePool *pInstancepool,
					GradientBoostingForest * pModel,
					uint32 begin,
					uint32 end
					);
			~ResidualThreadWork();
			bool NeedDelete()const;
			int DoWork();
		private:
			GbdtConf * m_pconfig;
			InstancePool * m_pInstancePool;
			GradientBoostingForest * m_pModel;
			uint32 m_begin;
			uint32 m_end;
	};

}
