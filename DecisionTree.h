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
                    InstancePool * pInstancepool,
                    DecisionTreeNodeStatus status,
                    uint32 depth, 
                    const std::vector<uint32> & InstanceIds
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

            inline DecisionTreeNode():m_pInstancePool(NULL), m_status(UNKOWN), 
            m_depth(0), m_InstanceIds(0), m_InstancesHashCode(0), m_Error(0),
            m_splitFeatureId(0), m_splitFeatureValue(0), m_sumWeight(0), m_sum_y_X_weight(0),
            m_LeafIndex(0), m_lson(-1), m_rson(-1)
            {
            }
            inline DecisionTreeNode(const DecisionTreeNode & r)
            {
                m_pInstancePool = r.m_pInstancePool;
                m_status = r.m_status;
                m_depth = r.m_depth;
                m_InstanceIds = r.m_InstanceIds;
                m_InstancesHashCode = r.m_InstancesHashCode;
                m_Error = r.m_Error;
                m_splitFeatureId = r.m_splitFeatureId;
                m_splitFeatureValue = r.m_splitFeatureValue;
                m_sumWeight = r.m_sumWeight;
                m_sum_y_X_weight = r.m_sum_y_X_weight;
                m_LeafIndex = r.m_LeafIndex;
                m_lson = r.m_lson;
                m_rson = r.m_rson;
            }

            inline DecisionTreeNode & operator=(const DecisionTreeNode & r)
            {
                m_pInstancePool = r.m_pInstancePool;
                m_status = r.m_status;
                m_depth = r.m_depth;
                m_InstanceIds = r.m_InstanceIds;
                m_InstancesHashCode = r.m_InstancesHashCode;
                m_Error = r.m_Error;
                m_splitFeatureId = r.m_splitFeatureId;
                m_splitFeatureValue = r.m_splitFeatureValue;
                m_sumWeight = r.m_sumWeight;
                m_sum_y_X_weight = r.m_sum_y_X_weight;
                m_LeafIndex = r.m_LeafIndex;
                m_lson = r.m_lson;
                m_rson = r.m_rson;
            } 

			~DecisionTreeNode();
			std::string ToString();
			std::string DebugStr() const;
			void print();
		public:
            InstancePool * m_pInstancePool;
			DecisionTreeNodeStatus m_status; 
			uint32 m_depth;
            std::vector<uint32> m_InstanceIds;
			
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
			int DoWork(const FloatT * Ty, const FloatT * Tweight);
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
