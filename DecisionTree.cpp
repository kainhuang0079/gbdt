#include<sstream>
#include<fstream>
#include<omp.h>
#include<time.h>
#include<memory>
#include"unity.h"
#include"DecisionTree.h"
#include"mempool.h" 
#include"threadpool.h"
#include"instancepool.h"
#include<cmath>

namespace gbdt
{
	FeatureCmp::FeatureCmp(int featureID):m_featureID(featureID)
	{}
	FeatureCmp::~FeatureCmp(){}
	bool FeatureCmp::operator () (Instance * pInstance1, Instance * pInstance2)const 
	{
		if(pInstance1->X[m_featureID] != pInstance2->X[m_featureID])
			return pInstance1->X[m_featureID] < pInstance2->X[m_featureID];
		else
			return pInstance1->index < pInstance2->index;
	}

	DecisionTreeNode::DecisionTreeNode(
			DecisionTreeNodeStatus status,
			uint32 depth,
			Instance ** ppInstances,
			int InstancesNum
			)
	{
		//Comm::TimeStat stat("DecisionTreeNode construct");
		m_status = status;
		m_depth = depth;
		m_ppInstances = ppInstances;
		m_InstancesNum = InstancesNum;

		m_InstancesHashCode = 0;
		m_Error = 0.0;
		m_splitFeatureId = -1;
		m_splitFeatureValue = 0.0;
		m_sumWeight = 0.0;
		m_sum_y_X_weight = 0.0;

		for(int i=0;i<m_InstancesNum;i++)
		{
			m_sumWeight += m_ppInstances[i]->weight;
			m_sum_y_X_weight += (m_ppInstances[i]->y * m_ppInstances[i]->weight); 
			m_InstancesHashCode = m_InstancesHashCode * 37 + m_ppInstances[i]->index;
		}
		FloatT Avg = m_sum_y_X_weight / m_sumWeight;
		for(int i=0;i<m_InstancesNum;i++)
		{
			m_Error += ((m_ppInstances[i]->y - Avg) *  (m_ppInstances[i]->y - Avg));
		}
		m_LeafIndex = -1;
		m_lson = -1;
		m_rson = -1;
		//stat.TimeMark("DecisionTreeNode construct " + DebugStr());
	}
	
	DecisionTreeNode::DecisionTreeNode(
			DecisionTreeNodeStatus status,
			uint32 depth,
			int splitFeatureId,
			FloatT splitFeatureValue,
			FloatT sumWeight,
			FloatT sum_y_X_weight,
			int LeafIndex,
			int lson,
			int rson
			)
	{
		m_status = status;
		m_depth = depth;
		m_ppInstances = NULL;
		m_InstancesNum = 0;
		m_InstancesHashCode = 0;
		m_Error = 0.0;
		m_splitFeatureId = splitFeatureId;
		m_splitFeatureValue = splitFeatureValue;
		m_sumWeight = sumWeight;
		m_sum_y_X_weight = sum_y_X_weight;
		m_LeafIndex = LeafIndex;
		m_lson = lson;
		m_rson = rson;
	}
	
	DecisionTreeNode::~DecisionTreeNode()
	{
	}
	
	std::string DecisionTreeNode::ToString()
	{
		return "";

	}
	std::string DecisionTreeNode::DebugStr() const
	{
		FloatT target = m_sum_y_X_weight * m_sum_y_X_weight / m_sumWeight;
		FloatT avg = m_sum_y_X_weight / m_sumWeight;
		std:: ostringstream oss;
		oss << "status:"<<m_status<<" depth:"<<m_depth<<" ppInstances:"<<m_ppInstances<<" InstancesNum:"<<m_InstancesNum<<" InstancesHashCode:"<<m_InstancesHashCode<<" Error:"<<m_Error<<" splitFeatureId:"<<m_splitFeatureId<<" splitFeatureValue:"<<m_splitFeatureValue<<" sumWeight:"<<m_sumWeight<<" sum_y_X_weight:"<<m_sum_y_X_weight<<" target:"<<target<<" avg:"<<avg<<" LeafIndex:"<<m_LeafIndex<<" lson:"<<m_lson<<" rson:"<<m_rson;
		return oss.str();
	}
	void DecisionTreeNode::print()
	{
		printf("%s\n",DebugStr().c_str());
	}
	SpliterWork::SpliterWork(
			Comm::WorkerThreadPool * pSpliterThreadPool,
			Comm::MemPool<DecisionTreeNode> * pmempool,
			GbdtConf * pconfig,
			InstancePool * pInstancepool,
			DecisionTreeNode * pSplitNode
			)
	{
		m_pSpliterThreadPool = pSpliterThreadPool;
		m_pmempool = pmempool;
		m_pconfig = pconfig;
		m_pInstancePool = pInstancepool;
		m_pSplitNode = pSplitNode;
	}
	
	SpliterWork::~SpliterWork()
	{
	}
	
	bool SpliterWork::NeedDelete()const
	{
		return true;
	}

	int SpliterWork::DoWork()
	{
		/*
		if(m_pSplitNode->m_InstancesHashCode == 195174386)
		{
			Comm::LogDebug("SpliterWork::DoWork m_pSplitNode->DebugStr() = <%s>",m_pSplitNode->DebugStr().c_str());
			for(int i=0;i<m_pSplitNode->m_InstancesNum;i++)
			{
				Comm::LogDebug("SpliterWork::DoWork Instance = %s",m_pSplitNode->m_ppInstances[i]->DebugStr().c_str());
			}
		}
		*/
		//Comm::TimeStat stat("SpliterWork " + m_pSplitNode->DebugStr());
		if(IsLeaf())
		{
			m_pSplitNode->m_status = LEAF;
		//	printf("SpliterWork::DoWork the node is leaf m_pSplitNode->DebugStr() = <%s>\n",m_pSplitNode->DebugStr().c_str());
			Comm::LogInfo("SpliterWork::DoWork the node is leaf m_pSplitNode->DebugStr() = <%s>",m_pSplitNode->DebugStr().c_str());
			m_pSplitNode->m_ppInstances = Comm::Free(m_pSplitNode->m_ppInstances);
			return 0;
		}
		std::vector<uint32> SubFeatures;
		int ret = m_pInstancePool->GetSubFeatureIDs(SubFeatures);
		if(ret != 0 || SubFeatures.size() <= 0)
		{
			m_pSplitNode->m_status = UNKOWN;
			Comm::LogErr("SpliterWork::DoWork m_pInstancePool->GetSubFeatureIDs fail! m_pSplitNode->DebugStr = <%s>",m_pSplitNode->DebugStr().c_str());
			m_pSplitNode->m_ppInstances = Comm::Free(m_pSplitNode->m_ppInstances);
			return -1;
		}
		std::vector<SearchSplitPointerWorkInfo *> vecpWorkInfo;
		for(uint32 i=0;i<SubFeatures.size();i++)
		{
			vecpWorkInfo.push_back(new SearchSplitPointerWorkInfo(false, SubFeatures[i],0.0));
		}
		
		//std::random_shuffle(m_pSplitNode->m_ppInstances, m_pSplitNode->m_ppInstances + m_pSplitNode->m_InstancesNum);
		//Instance ** ppInstances = (Instance **)malloc((m_pSplitNode->m_InstancesNum + 3) * sizeof(Instance *));
		Instance **& ppInstances = m_pSplitNode->m_ppInstances;
		int InstancesNum = m_pSplitNode->m_InstancesNum;
		if(!ppInstances)
		{
			m_pSplitNode->m_status = UNKOWN;
			Comm::LogErr("SpliterWork::DoWork ppInstances is NULL m_pSplitNode->DebugStr() = <%s>",m_pSplitNode->DebugStr().c_str());
			m_pSplitNode->m_ppInstances = Comm::Free(m_pSplitNode->m_ppInstances);
			for(uint32 i=0;i<vecpWorkInfo.size();i++)
			{
				vecpWorkInfo[i] = Comm::Delete(vecpWorkInfo[i]);
			}
			return -1;
		
		}
		/*
		for(int i=0;i<m_pSplitNode->m_InstancesNum;i++)
		{
			ppInstances[i] = m_pSplitNode->m_ppInstances[i];
		}
		*/

		//memcpy(ppInstances, m_pSplitNode->m_ppInstances, m_pSplitNode->m_InstancesNum * sizeof(Instance *));

		uint32 * Tmat = (uint32 *)malloc(m_pconfig->FeatureNum * m_pSplitNode->m_InstancesNum * sizeof(uint32));
		FloatT * Ty = (FloatT *)malloc(m_pSplitNode->m_InstancesNum * sizeof(FloatT));
		FloatT * Tweight = (FloatT *)malloc(m_pSplitNode->m_InstancesNum * sizeof(FloatT));

		int max_threads = std::min(omp_get_max_threads(), m_pconfig->SearchSplitPointerThreadNum);
		#pragma omp parallel for schedule(static, 100) num_threads(max_threads)
		for(int i=0;i<m_pSplitNode->m_InstancesNum;i++)
		{
			Instance * pInstance = m_pSplitNode->m_ppInstances[i];
			Ty[i] = pInstance->y;
			Tweight[i] = pInstance->weight;
			for(int j=0;j<pInstance->X_BucketIndex.size();j++)
			{
				Tmat[j * m_pSplitNode->m_InstancesNum + i] = pInstance->X_BucketIndex[j];
			}
		}
		//stat.TimeMark("make Tmat finish");

		//#pragma omp parallel for num_threads(m_pconfig->SearchSplitPointerThreadNum)
		#pragma omp parallel for schedule(dynamic, 1) num_threads(max_threads)
		for(uint32 i=0;i<SubFeatures.size();i++)
		{
			SearchSplitPointerWork * pwork = new SearchSplitPointerWork(
					m_pconfig,
					m_pSplitNode,
					m_pInstancePool,
					vecpWorkInfo[i]
					);
			int para_ret = pwork->DoWork(Tmat, Ty, Tweight);
			if(para_ret != 0)
				Comm::LogErr("SpliterWork::DoWork parallel for fail!!! SearchSplitPointerWork DoWork fail para_ret = %d", para_ret);
		}

		WaitAllSearchSplitPointerWorkDone(vecpWorkInfo);

		//stat.TimeMark("fine split point finish");

		Tmat = Comm::Free(Tmat);
		Ty = Comm::Free(Ty);
		Tweight = Comm::Free(Tweight);

		//stat.TimeMark("Free Tmat finish");

		SearchSplitPointerWorkInfo * pBestWorkInfo = NULL;
		FloatT BestTarget = m_pSplitNode->m_sum_y_X_weight * m_pSplitNode->m_sum_y_X_weight / m_pSplitNode->m_sumWeight;
		for(uint32 i=0;i<vecpWorkInfo.size();i++)
		{
			if(vecpWorkInfo[i]->m_target > BestTarget) //god!
			{
				BestTarget = vecpWorkInfo[i]->m_target;
				pBestWorkInfo = vecpWorkInfo[i];
			}
		}

		if(NULL == pBestWorkInfo)
		{
			m_pSplitNode->m_status = LEAF;
			Comm::LogInfo("SpliterWork::DoWork pBestWorkInfo = NULL m_pSplitNode->DebugStr() = <%s>",m_pSplitNode->DebugStr().c_str());
			m_pSplitNode->m_ppInstances = Comm::Free(m_pSplitNode->m_ppInstances);
			for(uint32 i=0;i<vecpWorkInfo.size();i++)
			{
				vecpWorkInfo[i] = Comm::Delete(vecpWorkInfo[i]);
			}
			return 0;
		}
		
		//std::sort(ppInstances, ppInstances + InstancesNum, FeatureCmp(pBestWorkInfo->m_splitFeatureId));

		//int i;
		//for(i = 0; ppInstances[i]->X[pBestWorkInfo->m_splitFeatureId] < pBestWorkInfo->m_splitFeatureValue; i++);
		//pBestWorkInfo->m_splitIndex = i;
		int LeftInstanceNum = pBestWorkInfo->m_splitIndex;
		int RightInstanceNum = m_pSplitNode->m_InstancesNum - pBestWorkInfo->m_splitIndex;
		if(LeftInstanceNum < m_pconfig->MinSampleLeaf || RightInstanceNum < m_pconfig->MinSampleLeaf)
		{
			m_pSplitNode->m_status = LEAF;
			m_pSplitNode->m_ppInstances = Comm::Free(m_pSplitNode->m_ppInstances);
			for(uint32 i=0;i<vecpWorkInfo.size();i++)
			{
				vecpWorkInfo[i] = Comm::Delete(vecpWorkInfo[i]);
			}
		//	ppInstances = Comm::Free(ppInstances);
			return 0;
		}
		Instance ** ppLeftInstances = (Instance **)malloc((LeftInstanceNum + 3) * sizeof(Instance *));
		Instance ** ppRightInstances = (Instance **)malloc((RightInstanceNum + 3) * sizeof(Instance *));

		if((!ppLeftInstances)||(!ppRightInstances))
		{
			m_pSplitNode->m_status = UNKOWN;
			Comm::LogErr("SpliterWork::DoWork ppLeftInstances or ppRightInstances is NULL m_pSplitNode->DebugStr() = <%s>",m_pSplitNode->DebugStr().c_str());
			m_pSplitNode->m_ppInstances = Comm::Free(m_pSplitNode->m_ppInstances);
			for(uint32 i=0;i<vecpWorkInfo.size();i++)
			{
				vecpWorkInfo[i] = Comm::Delete(vecpWorkInfo[i]);
			}
		//	ppInstances = Comm::Free(ppInstances);
			ppLeftInstances = Comm::Free(ppLeftInstances);
			ppRightInstances = Comm::Free(ppRightInstances);
			return -1;
		}

		int leaf = 0;
		int right = 0;
		for(int i = 0; i < m_pSplitNode->m_InstancesNum; i++)
		{
			if(ppInstances[i]->X[pBestWorkInfo->m_splitFeatureId] < pBestWorkInfo->m_splitFeatureValue)
			{
				ppLeftInstances[leaf++] = ppInstances[i];
			}
			else
			{
				ppRightInstances[right++] = ppInstances[i];
			}
		}
		//memcpy(ppLeftInstances, ppInstances, LeftInstanceNum * sizeof(Instance *));
		//memcpy(ppRightInstances, ppInstances + LeftInstanceNum, RightInstanceNum * sizeof(Instance *)); 


		int lson = m_pmempool->New(
				DecisionTreeNode(
					INTERTOR,
					m_pSplitNode->m_depth + 1,
					ppLeftInstances,
					LeftInstanceNum
					)
				);
		int rson = m_pmempool->New(
				DecisionTreeNode(
					INTERTOR,
					m_pSplitNode->m_depth + 1,
					ppRightInstances,
					RightInstanceNum
					)
				);
		if(lson < 0||rson < 0)
		{
			m_pSplitNode->m_status = LEAF;
			Comm::LogInfo("SpliterWork::DoWork lson < 0 or rson < 0 mempool maybe full m_pSplitNode->DebugStr() = <%s>",m_pSplitNode->DebugStr().c_str());
			m_pSplitNode->m_ppInstances = Comm::Free(m_pSplitNode->m_ppInstances);
			for(uint32 i=0;i<vecpWorkInfo.size();i++)
			{
				vecpWorkInfo[i] = Comm::Delete(vecpWorkInfo[i]);
			}
		//	ppInstances = Comm::Free(ppInstances);
			ppLeftInstances = Comm::Free(ppLeftInstances);
			ppRightInstances = Comm::Free(ppRightInstances);

			if(lson >= 0)
			{
				m_pmempool->Get(lson).m_ppInstances = NULL;
				m_pmempool->Get(lson).m_status = UNKOWN;
			}
			if(rson >= 0)
			{
				m_pmempool->Get(rson).m_ppInstances = NULL;
				m_pmempool->Get(rson).m_status = UNKOWN;
			}

			return 0;

		}
		
		m_pSplitNode->m_status = INTERTOR;
		m_pSplitNode->m_ppInstances = Comm::Free(m_pSplitNode->m_ppInstances);
		m_pSplitNode->m_splitFeatureId = pBestWorkInfo->m_splitFeatureId;
		m_pSplitNode->m_splitFeatureValue = pBestWorkInfo->m_splitFeatureValue;
		m_pSplitNode->m_lson = lson;
		m_pSplitNode->m_rson = rson;
		
		ret = m_pSpliterThreadPool->AddWork(
				new SpliterWork(
					m_pSpliterThreadPool,
					m_pmempool,
					m_pconfig,
					m_pInstancePool,
					&(m_pmempool->Get(lson))
					)
				);
		if(ret != 0)
		{
			Comm::LogErr("SpliterWork::DoWork Add lson Work fail! m_pSplitNode->DebugStr() = <%s>",m_pSplitNode->DebugStr().c_str());
		}

		ret = m_pSpliterThreadPool->AddWork(
				new SpliterWork(
					m_pSpliterThreadPool,
					m_pmempool,
					m_pconfig,
					m_pInstancePool,
					&(m_pmempool->Get(rson))
					)
				);

		if(ret !=0)
		{
			Comm::LogErr("SpliterWork::DoWork Add rson Work fail m_pSplitNode->DebugStr() = <%s>",m_pSplitNode->DebugStr().c_str());
		}
		//todo
		//ppInstances = Comm::Free(ppInstances);

		for(uint32 i=0;i<vecpWorkInfo.size();i++)
		{
			vecpWorkInfo[i] = Comm::Delete(vecpWorkInfo[i]);
		}

		return 0;
	}

	bool SpliterWork::IsLeaf()
	{
		if(LEAF == m_pSplitNode->m_status || m_pSplitNode->m_depth >= m_pconfig->MaxDepth || m_pSplitNode->m_InstancesNum < m_pconfig->MinSampleSplit || m_pSplitNode->m_InstancesNum < 2 || m_pmempool->IsFull())
			return true;
		return false;
	}

	bool SpliterWork::IsAllSearchSplitPointerWorkDone(const std::vector<SearchSplitPointerWorkInfo *> & vecpWorkInfo)
	{
		for(uint32 i=0;i<vecpWorkInfo.size();i++)
		{
			if(!vecpWorkInfo[i]->m_IsDone)
				return false;
		}
		return true;
	}

	void SpliterWork::WaitAllSearchSplitPointerWorkDone(const std::vector<SearchSplitPointerWorkInfo *> & vecpWorkInfo)
	{
		while(!IsAllSearchSplitPointerWorkDone(vecpWorkInfo))
		{

		}
	}

	Bucket::Bucket():sum_y_X_weight(0.0), sumWeight(0.0), num(0)
	{
	}

	SearchSplitPointerWorkInfo::SearchSplitPointerWorkInfo(
			bool IsDone,
			uint32 splitFeatureId,
			FloatT splitFeatureValue
			)
	{
		m_IsDone = IsDone;
		m_splitFeatureId = splitFeatureId;
		m_splitFeatureValue = splitFeatureValue;
		m_splitIndex = 0;

	}
	
	SearchSplitPointerWorkInfo::~SearchSplitPointerWorkInfo()
	{
	}

	std::string SearchSplitPointerWorkInfo::DebugStr()
	{
		std::ostringstream oss;
		oss<<"IsDone:"<<m_IsDone<<" splitFeatureId:"<<m_splitFeatureId<<" splitFeatureValue:"<<m_splitFeatureValue<<" m_target:"<<m_target;
		return oss.str();
	}

	void SearchSplitPointerWorkInfo::print()
	{
		std::string str = DebugStr();
		printf("%s\n",str.c_str());
	}
	
	SearchSplitPointerWork::SearchSplitPointerWork(
			GbdtConf * pconfig, 
			DecisionTreeNode * pSplitNode, 
			InstancePool * pInstancepool,
			SearchSplitPointerWorkInfo * pSearchSplitPointerWorkInfo
			)
	{
		m_pconfig = pconfig;
		m_pSplitNode = pSplitNode;
		m_pInstancePool = pInstancepool;
		m_pSearchSplitPointerWorkInfo = pSearchSplitPointerWorkInfo;
			
	}

	SearchSplitPointerWork::~SearchSplitPointerWork()
	{
	}	

	bool SearchSplitPointerWork::NeedDelete()const
	{
		return true;
	}

	int SearchSplitPointerWork::DoWork(const uint32 * Tmat, const FloatT * Ty, const FloatT * Tweight)
	{
		m_pSearchSplitPointerWorkInfo->m_target = (m_pSplitNode->m_sum_y_X_weight * m_pSplitNode->m_sum_y_X_weight)/m_pSplitNode->m_sumWeight;

		int InstancesNum = m_pSplitNode->m_InstancesNum;

		int featureID = m_pSearchSplitPointerWorkInfo->m_splitFeatureId;

		int log_num = log(InstancesNum * 1.0) / log(2.0);

		/*
		Instance ** ppInstances;

		{
			Comm::TimeStat stat("Sampling");
			ppInstances = (Instance **)malloc((InstancesNum + 3)*sizeof(Instance *));

			for(int i = 0; i < InstancesNum; i++)
			{
				int start = rand() % m_pSplitNode->m_InstancesNum;
				ppInstances[i] = m_pSplitNode->m_ppInstances[start];
			}
		}
		Instance ** ppInstances; 
		int ret_num;
		int ret = m_pInstancePool->GetSubSamplesPtr(1.0/1.0,
				m_pSplitNode->m_ppInstances, InstancesNum,
				ppInstances, ret_num);
		InstancesNum = ret_num;
		*/
		if(m_pInstancePool->m_FeatureBucketMap[featureID].size() >= InstancesNum * log_num * 5)
		//if(1)
		{
			//Comm::TimeStat stat("qsort split " + m_pSearchSplitPointerWorkInfo->DebugStr());
			
			Instance ** ppInstances = (Instance **)malloc((InstancesNum + 3)*sizeof(Instance *));
			
			if(!ppInstances)
			{
				Comm::LogErr("SearchSplitPointerWork::DoWork fail! ppInstances is NULL");
				return -1;
			}
			memcpy(ppInstances, m_pSplitNode->m_ppInstances, InstancesNum * sizeof(Instance *));
			
			FloatT left_sum_y_X_weight = 0.0;
			FloatT right_sum_y_X_weight = m_pSplitNode->m_sum_y_X_weight;
			FloatT left_sumWeight = 0.0;
			FloatT right_sumWeight = m_pSplitNode->m_sumWeight;

			std::sort(ppInstances, ppInstances + InstancesNum, FeatureCmp(featureID));
			for(int i=0;i<InstancesNum - 1;i++)
			{
				FloatT y = ppInstances[i]->y;
				FloatT weight = ppInstances[i]->weight;
				FloatT d = y * weight;
				left_sum_y_X_weight += d;
				right_sum_y_X_weight -= d;
				left_sumWeight += weight;
				right_sumWeight -= weight;

				if(ppInstances[i]->X[featureID] < ppInstances[i + 1]->X[featureID])
				{
					FloatT tmp_target = (left_sum_y_X_weight * left_sum_y_X_weight / left_sumWeight) + (right_sum_y_X_weight * right_sum_y_X_weight / right_sumWeight);
					if(tmp_target > m_pSearchSplitPointerWorkInfo->m_target)
					{
						m_pSearchSplitPointerWorkInfo->m_target = tmp_target;
						m_pSearchSplitPointerWorkInfo->m_splitFeatureValue = (ppInstances[i]->X[featureID] + ppInstances[i + 1]->X[featureID])/2.0;
						m_pSearchSplitPointerWorkInfo->m_splitIndex = i + 1;
					}
				}
			}
			ppInstances = Comm::Free(ppInstances);
			m_pSearchSplitPointerWorkInfo->m_IsDone = true;
			return 0;
		}
		else
		{
			//Comm::TimeStat stat("bucket sort split " + m_pSearchSplitPointerWorkInfo->DebugStr() + " " + this->m_pSplitNode->DebugStr());
			//puts("new>>>>>>");
			FloatT left_sum_y_X_weight = 0.0;
			FloatT right_sum_y_X_weight = m_pSplitNode->m_sum_y_X_weight;
			FloatT left_sumWeight = 0.0;
			FloatT right_sumWeight = m_pSplitNode->m_sumWeight;
			std::vector<Bucket> tmp_buckets(m_pInstancePool->m_FeatureBucketMap[featureID].size());
			//Bucket tmp_buckets[10086];
			Instance ** ppInstances = m_pSplitNode->m_ppInstances;
			//stat.TimeMark("====bucket sort split 0.5====" + m_pSearchSplitPointerWorkInfo->DebugStr() + " " + this->m_pSplitNode->DebugStr());
			for(int i = 0; i < InstancesNum; i++)
			{
				//int BucketIndex = ppInstances[i]->X_BucketIndex[featureID];
				int BucketIndex = Tmat[InstancesNum * featureID + i];
				Bucket & tmp = tmp_buckets[BucketIndex];
				//Instance & instance = *ppInstances[i];
				//tmp.sum_y_X_weight += instance.y * instance.weight;
				tmp.sum_y_X_weight += Ty[i] * Tweight[i];
				//tmp.sumWeight += instance.weight; 
				tmp.sumWeight += Tweight[i]; 
				tmp_buckets[BucketIndex].num++;
			}
			//stat.TimeMark("====bucket sort split 1====" + m_pSearchSplitPointerWorkInfo->DebugStr() + " " + this->m_pSplitNode->DebugStr());
			std::vector<Bucket> buckets;
			for(int i = 0; i< m_pInstancePool->m_FeatureBucketMap[featureID].size(); i++)
			{
				FloatT weight = tmp_buckets[i].sumWeight;
				if(weight <= 0)
					continue;
				tmp_buckets[i].value = m_pInstancePool->m_FeatureBucketMap[featureID][i];
				buckets.push_back(tmp_buckets[i]);
			}
			//stat.TimeMark("bucket sort split 2" + m_pSearchSplitPointerWorkInfo->DebugStr() + " " + this->m_pSplitNode->DebugStr());
			int num = 0;
			for(int i = 0; i < buckets.size() - 1; i++)
			{
				FloatT weight = buckets[i].sumWeight;
				FloatT d = buckets[i].sum_y_X_weight;
				left_sum_y_X_weight += d;
				right_sum_y_X_weight -= d;
				left_sumWeight += weight;
				right_sumWeight -= weight;
				FloatT tmp_target = (left_sum_y_X_weight * left_sum_y_X_weight / left_sumWeight) + (right_sum_y_X_weight * right_sum_y_X_weight / right_sumWeight);
				num += buckets[i].num;
				if(tmp_target > m_pSearchSplitPointerWorkInfo->m_target)
				{
					m_pSearchSplitPointerWorkInfo->m_target = tmp_target;
					m_pSearchSplitPointerWorkInfo->m_splitFeatureValue = (buckets[i].value + buckets[i + 1].value)/2.0;
					m_pSearchSplitPointerWorkInfo->m_splitIndex = num;
				}
				
			}
			//ppInstances = Comm::Free(ppInstances);
			m_pSearchSplitPointerWorkInfo->m_IsDone = true;
			//stat.TimeMark("bucket sort split 3" + m_pSearchSplitPointerWorkInfo->DebugStr() + " " + this->m_pSplitNode->DebugStr());
			return 0;
		}
	}

	DecisionTree::DecisionTree(GbdtConf * pconfig):m_pconfig(pconfig),m_pInstancePool(NULL),m_SpliterThreadPool("m_SpliterThreadPool")
	{
		m_pmempool = new Comm::MemPool<DecisionTreeNode>(m_pconfig->MaxNodes);	
	}

	DecisionTree::~DecisionTree()
	{
	//	print();
		for(int i=0;i<m_pmempool->Size();i++)
		{
//			m_pmempool->Get(i).m_ppInstances = Comm::Free(m_pmempool->Get(i).m_ppInstances);
		}
		m_pmempool = Comm::Delete(m_pmempool);
	}

	int DecisionTree::Fit(InstancePool * pInstancepool)
	{
		m_pInstancePool = pInstancepool;
		if(NULL == m_pInstancePool )
		{
			Comm::LogErr("DecisionTree::Fit fail! m_pInstancePool is NULL");
			return -1;
		}
		int ret = -1;
		ret = m_SpliterThreadPool.Start(m_pconfig->SpliterThreadNum);
		if(ret != 0)
		{
			Comm::LogErr("DecisionTree::Fit fail! m_SpliterThreadPool start fail!");
			return ret;
		}
		
		Instance ** ppInstances = NULL;
		int InstancesNum = 0;
		ret = m_pInstancePool->GetSubSamplesPtr(ppInstances, InstancesNum);
		if(ret != 0 && InstancesNum <= 0 && NULL == ppInstances)
		{
			Comm::LogErr("DecisionTree::Fit fail! m_pInstancePool GetSubSamplesPtr fail");
			return ret;
		}
		
		m_RootIndex = m_pmempool->New(DecisionTreeNode(ROOT,0,ppInstances,InstancesNum));
		if(m_RootIndex < 0)
		{
			Comm::LogErr("DecisionTree::Fit fail! m_pmempool->New fail! m_RootIndex = %d",m_RootIndex);
			return -1;
		}


		ret = m_SpliterThreadPool.AddWork(
				new SpliterWork(
					&m_SpliterThreadPool,
					m_pmempool,
					m_pconfig,
					m_pInstancePool,
					&(m_pmempool->Get(m_RootIndex))
					)
				);
		if(ret !=0)
		{
			Comm::LogErr("DecisionTree::Fit m_SpliterThreadPool AddWork fail!");
			return -1;
		}
		m_SpliterThreadPool.WaitAllWorkDone();

		m_SpliterThreadPool.Shutdown();
		ret = m_SpliterThreadPool.JoinAll();
		//printf("======\n");
		if(m_pconfig->LogLevel >= 2)printf("NodeNum = %d FitError = %f\n",m_pmempool->Size(),FitError());
		//printf("======\n");

		return 0;
	}
	int DecisionTree::SaveModel()
	{
		return SaveModel(m_pconfig->OutputModelFilePath.c_str());
	}
	
	int DecisionTree::SaveModel(const char *modelfile)
	{
		FILE * fp;
		fp = fopen(modelfile , "w");
		if(NULL == fp)
		{
			Comm::LogErr("DecisionTree::SaveModel fail! open %s fail!",modelfile);
			return -1;
		}
		int ret = SaveModel(fp);
		if(ret != 0)
		{
			Comm::LogErr("DecisionTree::SaveModel fail!");
			fclose(fp);
			return ret;
		}
		fclose(fp);
		return 0;
	}

	int DecisionTree::SaveModel(FILE * fp)
	{
		if(!fp)
		{
			Comm::LogErr("DecisionTree::SaveModel fp is NULL");
			return -1;
		}
		int ret;
		ret = fprintf(fp,"%d %d\n",m_pmempool->Size(),m_RootIndex);
	//	puts("aaaaaaaa");
		if(ret < 0)
		{
			Comm::LogErr("DecisionTree SaveModel fail fprintf fail! Size =%d  m_RootIndex = %d",m_pmempool->Size(),m_RootIndex);
			return -1;
		}
		for(int i=0;i<m_pmempool->Size();i++)
		{
			DecisionTreeNode & node = m_pmempool->Get(i);
	//		node.print();
	//		printf("bbbbb %d\n",i);
			ret = fprintf(fp,"%d %u %d %f %f %f %d %d %d\n",
					node.m_status,
					node.m_depth,
					node.m_splitFeatureId,
					node.m_splitFeatureValue,
					node.m_sumWeight,
					node.m_sum_y_X_weight,
					node.m_LeafIndex,
					node.m_lson,
					node.m_rson
					);
		//	printf("cccccc %d\n",i);
			if(ret < 0)
			{
				Comm::LogErr("DecisionTree::SaveModel fail!fprintf node i = %d fail!",i);
				return -1;
			}
		}
	//	puts("SaveModel Done");
		return 0;
	}

	int DecisionTree::LoadModel()
	{
		return LoadModel(m_pconfig->InputModelFilePath.c_str());
	}

	int DecisionTree::LoadModel(const char * modelfile)
	{
		FILE * fp = fopen(modelfile,"r");
		if(NULL == fp)
		{
			Comm::LogErr("DecisionTree::LoadModel fail! %s open fail!",modelfile);
			return -1;
		}
		int ret = LoadModel(fp);
		if(ret != 0)
		{
			Comm::LogErr("DecisionTree::LoadModel fail!");
			fclose(fp);
			return -1;
		}
		fclose(fp);
		return 0;
	}
	
	int DecisionTree::LoadModel(FILE * fp)
	{
		if(!fp)
		{
			Comm::LogErr("DecisionTree::LoadModel fail! fp is NULL");
			return -1;
		}
		int ret;
		m_pmempool->Clear();
		int nodeNum;
		ret = fscanf(fp,"%d%d",&nodeNum,&m_RootIndex);
		if(ret < 0)
		{
			Comm::LogErr("DecisionTree::LoadModel fscanf nodeNum and m_RootIndex fail!");
			return -1;
		}
		for(int i=0;i<nodeNum;i++)
		{
			DecisionTreeNodeStatus status;
			uint32 depth;
			int splitFeatureId;
			FloatT splitFeatureValue;
			FloatT sumWeight;
			FloatT sum_y_X_weight;
			int LeafIndex;
			int lson;
			int rson;
			
			ret = fscanf(fp, "%d%u%d%lf%lf%lf%d%d%d",
					&status,
					&depth,
					&splitFeatureId,
					&splitFeatureValue,
					&sumWeight,
					&sum_y_X_weight,
					&LeafIndex,
					&lson,
					&rson
					);
			if(ret < 0)
			{
				Comm::LogErr("DecisionTree::LoadModel fscanf node i = %d fail!",i);
				return -1;
			}
			ret = m_pmempool->Add(
					DecisionTreeNode(
						status,
						depth,
						splitFeatureId,
						splitFeatureValue,
						sumWeight,
						sum_y_X_weight,
						LeafIndex,
						lson,
						rson
						)
					);
	//		printf("index = %d ",i);
	//		m_pmempool->Get(i).print();
			if(ret != 0)
			{
				Comm::LogErr("DecisionTree::LoadModel fail! m_pmempool->Add fail!");
				return -1;
			}
		}

		return 0;
	}

	int DecisionTree::Predict(const std::vector<FloatT> &X, FloatT & predict)
	{
		int tmp;
		return Predict(X,predict,tmp);
	}


	int DecisionTree::Predict(const std::vector<FloatT> &X, FloatT & predict, int & fallLeafIndex)
	{
		int CurIndex = m_RootIndex;
		if(X.size() != m_pconfig->FeatureNum)
		{
			Comm::LogErr("DecisionTree::Predict fail! X size = %u FeatureNum = %u",X.size(),m_pconfig->FeatureNum);
			return -1;
		}
		while(!(-1 == CurIndex) && !(LEAF == m_pmempool->Get(CurIndex).m_status))
		{
			int featureID = m_pmempool->Get(CurIndex).m_splitFeatureId;
			FloatT splitValue = m_pmempool->Get(CurIndex).m_splitFeatureValue;
			int lson = m_pmempool->Get(CurIndex).m_lson;
			int rson = m_pmempool->Get(CurIndex).m_rson;
	//		printf("X[%d] = %f ",featureID,X[featureID]);
	//		m_pmempool->Get(CurIndex).print();
			CurIndex = X[featureID] < splitValue ? lson : rson;

	//		printf("CurIndex %d\n",CurIndex);
		}
//		m_pmempool->Get(CurIndex).print();
		if(-1 == CurIndex)
		{
			Comm::LogErr("DecisionTree::Predict fail! no such CurIndex!");
			return -1;
		}
		predict = m_pmempool->Get(CurIndex).m_sum_y_X_weight / m_pmempool->Get(CurIndex).m_sumWeight;
		fallLeafIndex = CurIndex;
		return 0;
	}

	FloatT DecisionTree::FitError()
	{
		
		FloatT ret =0.0;
		FloatT sum_weight = 0.0;
		#pragma omp parallel for schedule(static) reduction(+:sum_weight)
		for(int i=0;i<m_pInstancePool->Size();i++)
		{
			sum_weight += m_pInstancePool->GetInstance(i).weight;
		}
		#pragma omp parallel for schedule(static) reduction(+:ret)
		for(int i=0;i<m_pInstancePool->Size();i++)
		{
			FloatT predict;
			Predict(m_pInstancePool->GetInstance(i).X, predict);
			ret += ((m_pInstancePool->GetInstance(i).y - predict) * (m_pInstancePool->GetInstance(i).y - predict));
		//	printf("predict: %f ",predict);
		//	m_pInstancePool->GetInstance(i).print();
		}	

		//puts("zzzzzzz");
		return ret / sum_weight;
	}
	void DecisionTree::print()
	{

		for(int i=0;i<m_pmempool->Size();i++)
		{
			printf("index: %d ",i);
			m_pmempool->Get(i).print();
		}
	}

	DecisionTreeNode & DecisionTree::GetNode(int index)
	{
		return m_pmempool->Get(index);
	}

	int DecisionTree::Size()const 
	{
		return m_pmempool->Size();
	}
}
