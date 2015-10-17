#pragma once 
#include"threadpool.h"
#include"Log.h"
#include"unity.h"
#include"mempool.h"
#include"gbdtconf.h"
#include"instancepool.h"
#include<cstdlib>
#include<cstdio>
#include<algorithm>
#include<vector>
#include<cstring>
#include<string>

namespace gbdt
{ 
	


	enum DecisionTreeNodeStatus
	{
		UNKOWN = -1,
		ROOT = 0,
		INTERTOR = 1,
		LEAF = 2
	};


	class FeatureCmp
	{
		public:
			FeatureCmp(int featureID);
			~FeatureCmp();
			bool operator()(Instance * pInstance1, Instance * pInstance2)const;
		private:
			int m_featureID;
	};



	class DecisionTreeNode
	{
		public:
			DecisionTreeNode(
					DecisionTreeNodeStatus status,
					uint32 depth, 
					Instance ** ppInstances,
					int InstancesNum
					);
			DecisionTreeNode(
					DecisionTreeNodeStatus status,
					uint32 depth,
					int splitFeatureId,
					FloatT splitFeatureValue,
					FloatT sumWeight,
					FloatT sum_y_X_weight,
					int LeafIndex,
					int lson,
					int rson
					);

			~DecisionTreeNode();
			std::string ToString();
			std::string DebugStr() const;
			void print();
		public:
			DecisionTreeNodeStatus m_status; 
			uint32 m_depth;
			Instance ** m_ppInstances;
			int m_InstancesNum;
			
			uint32 m_InstancesHashCode;
			FloatT m_Error;
			int m_splitFeatureId;
			FloatT m_splitFeatureValue;
			FloatT m_sumWeight;
			FloatT m_sum_y_X_weight;
			int m_LeafIndex;
			int m_lson;
			int m_rson;
	};


	class SearchSplitPointerWorkInfo
	{
		public:
			SearchSplitPointerWorkInfo(
					bool IsDone,
					uint32 m_splitFeatureId, 
					FloatT m_splitFeatureValue
					);
			~SearchSplitPointerWorkInfo();
			std::string DebugStr();
			void print();
		public:
			bool m_IsDone;
			uint32 m_splitFeatureId;
			FloatT m_splitFeatureValue;
			uint32 m_splitIndex;
			FloatT m_target;
	};

	class SpliterWork : public Comm::Work
	{
		public:
			SpliterWork(
					Comm::WorkerThreadPool * pSpliterThreadPool,
					Comm::MemPool<DecisionTreeNode> * pmempool,
					GbdtConf * pconfig,
					InstancePool * pInstancepool,
					DecisionTreeNode * pSplitNode
					);
			~SpliterWork();
			bool NeedDelete()const;
			int DoWork();
		private:
			void WaitAllSearchSplitPointerWorkDone(const std::vector<SearchSplitPointerWorkInfo *> & vecpWorkInfo);
			bool IsAllSearchSplitPointerWorkDone(const std::vector<SearchSplitPointerWorkInfo *> & vecpWorkInfo);
			bool IsLeaf();
		private:
			Comm::WorkerThreadPool * m_pSpliterThreadPool;
			Comm::MemPool<DecisionTreeNode> * m_pmempool;
			GbdtConf * m_pconfig;
			InstancePool * m_pInstancePool;
			DecisionTreeNode * m_pSplitNode;
	};

	struct Bucket
	{
		Bucket();
		FloatT sum_y_X_weight;
		FloatT sumWeight;
		int num;
		FloatT value;
	};


	class SearchSplitPointerWork
	{
		public:
			SearchSplitPointerWork(
					GbdtConf * pconfig, 
					DecisionTreeNode * pSplitNode, 
					InstancePool * pInstancepool,
					SearchSplitPointerWorkInfo * pSearchSplitPointerWorkInfo
					);
			~SearchSplitPointerWork();
			bool NeedDelete()const;
			int DoWork(uint32 * Tmat, FloatT * Ty, FloatT * Tweight);
		private:
			GbdtConf * m_pconfig;
			const DecisionTreeNode * m_pSplitNode;
			InstancePool * m_pInstancePool;
			SearchSplitPointerWorkInfo * m_pSearchSplitPointerWorkInfo;
	};
	class DecisionTree
	{
		public:
			DecisionTree(GbdtConf * pconfig);
			~DecisionTree();
			int Fit(InstancePool * pInstancepool);
			int SaveModel();
			int SaveModel(const char * modelfile);
			int SaveModel(FILE * fp);
			int LoadModel();
			int LoadModel(const char * modelfile);
			int LoadModel(FILE * fp);
			int Predict(const std::vector<FloatT> &X, FloatT & predict);
			int Predict(const std::vector<FloatT> &X, FloatT & predict, int & fallLeafIndex);
			FloatT FitError();
			void print();
			DecisionTreeNode & GetNode(int index);
			int Size()const;
		private:
			Comm::WorkerThreadPool m_SpliterThreadPool;
			GbdtConf * m_pconfig; 
			int m_RootIndex;
			InstancePool * m_pInstancePool;
			Comm::MemPool<DecisionTreeNode> * m_pmempool;
	};

}
