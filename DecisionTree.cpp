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
            InstancePool * pInstancepool,
            DecisionTreeNodeStatus status,
            uint32 depth,
            const std::vector<uint32> & InstanceIds
            )
	{
		//Comm::TimeStat stat("DecisionTreeNode construct");
        m_pInstancePool = pInstancepool;
		m_status = status;
		m_depth = depth;
        m_InstanceIds = InstanceIds;

		m_InstancesHashCode = 0;
		m_Error = 0.0;
		m_splitFeatureId = -1;
		m_splitFeatureValue = 0.0;
		m_sumWeight = 0.0;
		m_sum_y_X_weight = 0.0;

		for(int i=0;i<m_InstanceIds.size();i++)
		{
            const Instance & instance = 
                m_pInstancePool->GetInstance(m_InstanceIds[i]);
			m_sumWeight += instance.weight;
			m_sum_y_X_weight += (instance.y * instance.weight); 
			m_InstancesHashCode = m_InstancesHashCode * 37 + instance.index;
		}
		FloatT Avg = m_sum_y_X_weight / m_sumWeight;
		for(int i=0;i<m_InstanceIds.size();i++)
		{
            const Instance & instance = 
                m_pInstancePool->GetInstance(m_InstanceIds[i]);
            m_Error += ((instance.y - Avg) *  (instance.y - Avg));
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
        m_pInstancePool = NULL;
		m_status = status;
		m_depth = depth;
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
		oss << "status:"<<m_status<<" depth:"<<
            m_depth<<" InstancesNum:"<<m_InstanceIds.size()<<
            " capacity:"<< m_InstanceIds.capacity()<<
            " InstancesHashCode:"<<m_InstancesHashCode<<" Error:"<<
            m_Error<<" splitFeatureId:"<<m_splitFeatureId<<" splitFeatureValue:"<<
            m_splitFeatureValue<<" sumWeight:"<<m_sumWeight<<" sum_y_X_weight:"<<
            m_sum_y_X_weight<<" target:"<<target<<" avg:"<<
            avg<<" LeafIndex:"<<m_LeafIndex<<" lson:"<<m_lson<<" rson:"<<m_rson;
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
		//Comm::TimeStat stat("SpliterWork " + m_pSplitNode->DebugStr());
		if(IsLeaf())
		{
			m_pSplitNode->m_status = LEAF;
			Comm::LogInfo("SpliterWork::DoWork the node is leaf m_pSplitNode->DebugStr() = <%s>",
                    m_pSplitNode->DebugStr().c_str());
            std::vector<uint32>().swap(m_pSplitNode->m_InstanceIds);
            //stat.TimeMark("\nSpliterWork::DoWork m_pSplitNode=" + m_pSplitNode->DebugStr() + "\n");
			return 0;
		}
		std::vector<uint32> SubFeatures;
		int ret = m_pInstancePool->GetSubFeatureIDs(SubFeatures);
		if(ret != 0 || SubFeatures.size() <= 0)
		{
			m_pSplitNode->m_status = UNKOWN;
			Comm::LogErr("SpliterWork::DoWork m_pInstancePool->GetSubFeatureIDs"
                    " fail! m_pSplitNode->DebugStr = <%s>",m_pSplitNode->DebugStr().c_str());
            std::vector<uint32>().swap(m_pSplitNode->m_InstanceIds);
			return -1;
		}
		std::vector<SearchSplitPointerWorkInfo *> vecpWorkInfo;
		for(uint32 i=0;i<SubFeatures.size();i++)
		{
			vecpWorkInfo.push_back(new SearchSplitPointerWorkInfo(false, SubFeatures[i],0.0));
		}
		
        const std::vector<uint32> & InstanceIds = m_pSplitNode->m_InstanceIds;
        int InstancesNum = InstanceIds.size();
        FloatT * Ty = (FloatT *)malloc(InstancesNum * sizeof(FloatT));
        FloatT * Tweight = (FloatT *)malloc(InstancesNum * sizeof(FloatT));

		int max_threads = std::min(omp_get_max_threads(), m_pconfig->SearchSplitPointerThreadNum);
		#pragma omp parallel for schedule(static, 100) num_threads(max_threads)
		for(int i=0;i<InstancesNum;i++)
		{
			const Instance & instance = m_pInstancePool->GetInstance(InstanceIds[i]);
            Ty[i] = instance.y;
            Tweight[i] = instance.weight;
        }
        //stat.TimeMark("make Tmat finish");
		#pragma omp parallel for schedule(dynamic, 1) num_threads(max_threads)
		for(uint32 i=0;i<SubFeatures.size();i++)
		{
			SearchSplitPointerWork * pwork = new SearchSplitPointerWork(
					m_pconfig,
					m_pSplitNode,
					m_pInstancePool,
					vecpWorkInfo[i]
					);
			int para_ret = pwork->DoWork(Ty, Tweight);
			if(para_ret != 0)
				Comm::LogErr("SpliterWork::DoWork parallel for fail!!! "
                        "SearchSplitPointerWork DoWork fail para_ret = %d", para_ret);
		}

		WaitAllSearchSplitPointerWorkDone(vecpWorkInfo);

		//stat.TimeMark("fine split point finish");

		Ty = Comm::Free(Ty);
		Tweight = Comm::Free(Tweight);

		SearchSplitPointerWorkInfo * pBestWorkInfo = NULL;
		FloatT BestTarget = 
            m_pSplitNode->m_sum_y_X_weight * m_pSplitNode->m_sum_y_X_weight / m_pSplitNode->m_sumWeight;
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
			Comm::LogInfo("SpliterWork::DoWork pBestWorkInfo = NULL"
                    " m_pSplitNode->DebugStr() = <%s>",m_pSplitNode->DebugStr().c_str());
            std::vector<uint32>().swap(m_pSplitNode->m_InstanceIds);
			for(uint32 i=0;i<vecpWorkInfo.size();i++)
			{
				vecpWorkInfo[i] = Comm::Delete(vecpWorkInfo[i]);
			}
			return 0;
		}
		//stat.TimeMark("find get split point finish");
        std::vector<uint32> LeftInstanceIds;
        std::vector<uint32> RightInstanceIds;
		for(int i = 0; i < InstanceIds.size(); i++)
		{
            int index = 
                m_pInstancePool->m_pTmat[pBestWorkInfo->m_splitFeatureId *
                m_pInstancePool->Size() + InstanceIds[i]];
            FloatT value = 
                m_pInstancePool->m_FeatureBucketMap[pBestWorkInfo->m_splitFeatureId][index];
            //printf("index %d value %f  best %f X %f\n", index, value, pBestWorkInfo->m_splitFeatureValue,
            //      m_pInstancePool->GetInstance(InstanceIds[i]).X[pBestWorkInfo->m_splitFeatureId]);
			if(value < pBestWorkInfo->m_splitFeatureValue)
			{
			    LeftInstanceIds.push_back(InstanceIds[i]);	
			}
			else
			{
			    RightInstanceIds.push_back(InstanceIds[i]);	
			}
		}
		//stat.TimeMark("splited point finish");
		if(LeftInstanceIds.size() < m_pconfig->MinSampleLeaf ||
                RightInstanceIds.size() < m_pconfig->MinSampleLeaf)
		{
			m_pSplitNode->m_status = LEAF;
            std::vector<uint32>().swap(m_pSplitNode->m_InstanceIds);
			for(uint32 i=0;i<vecpWorkInfo.size();i++)
			{
				vecpWorkInfo[i] = Comm::Delete(vecpWorkInfo[i]);
			}
            //stat.TimeMark("\nSpliterWork::DoWork m_pSplitNode=" + m_pSplitNode->DebugStr() + "\n");
			return 0;
		}
		//stat.TimeMark("splited point finish zzzzzzz");

		int lson = m_pmempool->New(
				DecisionTreeNode(
                    m_pInstancePool,
					INTERTOR,
					m_pSplitNode->m_depth + 1,
                    LeftInstanceIds
					)
				);
		//stat.TimeMark("splited lson point finish");
		int rson = m_pmempool->New(
				DecisionTreeNode(
                    m_pInstancePool,
					INTERTOR,
					m_pSplitNode->m_depth + 1,
                    RightInstanceIds
					)
				);
		//stat.TimeMark("splited rson point finish");
		if(lson < 0||rson < 0)
		{
			m_pSplitNode->m_status = LEAF;
			Comm::LogInfo("SpliterWork::DoWork lson < 0 or rson < 0 mempool "
                    "maybe full m_pSplitNode->DebugStr() = <%s>",m_pSplitNode->DebugStr().c_str());
            std::vector<uint32>().swap(m_pSplitNode->m_InstanceIds);
			for(uint32 i=0;i<vecpWorkInfo.size();i++)
			{
				vecpWorkInfo[i] = Comm::Delete(vecpWorkInfo[i]);
			}

			if(lson >= 0)
			{
				m_pmempool->Get(lson).m_status = UNKOWN;
                std::vector<uint32>().swap(m_pmempool->Get(lson).m_InstanceIds);
			}
			if(rson >= 0)
			{
				m_pmempool->Get(rson).m_status = UNKOWN;
                std::vector<uint32>().swap(m_pmempool->Get(rson).m_InstanceIds);
			}

            //stat.TimeMark("\nSpliterWork::DoWork m_pSplitNode=" + m_pSplitNode->DebugStr() + "\n");
			return 0;

		}
		
		m_pSplitNode->m_status = INTERTOR;
        std::vector<uint32>().swap(m_pSplitNode->m_InstanceIds);
		m_pSplitNode->m_splitFeatureId = pBestWorkInfo->m_splitFeatureId;
		m_pSplitNode->m_splitFeatureValue = pBestWorkInfo->m_splitFeatureValue;
		m_pSplitNode->m_lson = lson;
		m_pSplitNode->m_rson = rson;
		
        //stat.TimeMark("\nSpliterWork::DoWork m_pSplitNode=" + m_pSplitNode->DebugStr() + "\n");
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
			Comm::LogErr("SpliterWork::DoWork Add lson Work fail! m_pSplitNode->DebugStr() = <%s>"
                    ,m_pSplitNode->DebugStr().c_str());
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
			Comm::LogErr("SpliterWork::DoWork Add rson Work fail m_pSplitNode->DebugStr() = <%s>",
                    m_pSplitNode->DebugStr().c_str());
		}
		//todo

		for(uint32 i=0;i<vecpWorkInfo.size();i++)
		{
			vecpWorkInfo[i] = Comm::Delete(vecpWorkInfo[i]);
		}

		return 0;
	}

	bool SpliterWork::IsLeaf()
	{
		if(LEAF == m_pSplitNode->m_status ||
                m_pSplitNode->m_depth >= m_pconfig->MaxDepth ||
                m_pSplitNode->m_InstanceIds.size() < m_pconfig->MinSampleSplit || 
                m_pSplitNode->m_InstanceIds.size() < 2 || m_pmempool->IsFull())
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
		oss<<"IsDone:"<<m_IsDone<<" splitFeatureId:"<<
            m_splitFeatureId<<" splitFeatureValue:"<<
            m_splitFeatureValue<<" m_target:"<<m_target;
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

	int SearchSplitPointerWork::DoWork(const FloatT * Ty, const FloatT * Tweight)
	{
		m_pSearchSplitPointerWorkInfo->m_target = 
            (m_pSplitNode->m_sum_y_X_weight * m_pSplitNode->m_sum_y_X_weight)/m_pSplitNode->m_sumWeight;

        const uint32 * Tmat = m_pInstancePool->m_pTmat;
        const std::vector<uint32> & InstanceIds = m_pSplitNode->m_InstanceIds;
		const int InstancesNum = m_pInstancePool->Size();
		int featureID = m_pSearchSplitPointerWorkInfo->m_splitFeatureId;

   		{
			//Comm::TimeStat stat("bucket sort split " + m_pSearchSplitPointerWorkInfo->DebugStr() + " " + this->m_pSplitNode->DebugStr());
			FloatT left_sum_y_X_weight = 0.0;
			FloatT right_sum_y_X_weight = m_pSplitNode->m_sum_y_X_weight;
			FloatT left_sumWeight = 0.0;
			FloatT right_sumWeight = m_pSplitNode->m_sumWeight;
			std::vector<Bucket> tmp_buckets(m_pInstancePool->m_FeatureBucketMap[featureID].size());
			for(int i = 0; i < InstanceIds.size(); i++)
			{
				uint32 BucketIndex = Tmat[InstancesNum * featureID + InstanceIds[i]];
				Bucket & tmp = tmp_buckets[BucketIndex];
				tmp.sum_y_X_weight += Ty[i] * Tweight[i];
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
				FloatT tmp_target = 
                    (left_sum_y_X_weight * left_sum_y_X_weight / left_sumWeight)
                    + (right_sum_y_X_weight * right_sum_y_X_weight / right_sumWeight);
				num += buckets[i].num;
				if(tmp_target > m_pSearchSplitPointerWorkInfo->m_target)
				{
					m_pSearchSplitPointerWorkInfo->m_target = tmp_target;
					m_pSearchSplitPointerWorkInfo->m_splitFeatureValue = 
                        (buckets[i].value + buckets[i + 1].value)/2.0;
					m_pSearchSplitPointerWorkInfo->m_splitIndex = num;
				}
				
			}
			m_pSearchSplitPointerWorkInfo->m_IsDone = true;
			//stat.TimeMark("bucket sort split 3" + m_pSearchSplitPointerWorkInfo->DebugStr() + " " + this->m_pSplitNode->DebugStr());
			return 0;
		}
	}

	DecisionTree::DecisionTree(GbdtConf * pconfig):
        m_pconfig(pconfig),m_pInstancePool(NULL),m_SpliterThreadPool("m_SpliterThreadPool")
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
        //Comm::TimeStat stat("DecisionTree::Fit");
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
		
        std::vector<uint32> SubIDs;
		ret = m_pInstancePool->GetSubIDs(
                m_pInstancePool->Size() * m_pconfig->SubSampleRate,
                m_pInstancePool->Size(), SubIDs);
        std::sort(SubIDs.begin(), SubIDs.end());
        //stat.TimeMark("sorted");
        const DecisionTreeNode & tp = DecisionTreeNode(
                    m_pInstancePool, 
                    ROOT, 0, SubIDs);
		m_RootIndex = m_pmempool->New(tp);
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
        //stat.TimeMark("splited");
		if(ret !=0)
		{
			Comm::LogErr("DecisionTree::Fit m_SpliterThreadPool AddWork fail!");
			return -1;
		}
		m_SpliterThreadPool.WaitAllWorkDone();
        //stat.TimeMark("splited");
		m_SpliterThreadPool.Shutdown();
        //stat.TimeMark("Shutdowned");
		ret = m_SpliterThreadPool.JoinAll();
        //stat.TimeMark("JoinAlled");
		//printf("======\n");
        for (int i = 0; i < m_pmempool->Size(); i++)
        {
            if (m_pmempool->Get(i).m_InstanceIds.capacity() > 0)
            {
                printf("EEEEEEEEEEEEEEEE %s", m_pmempool->Get(i).DebugStr().c_str());
            }
        }
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
			ret += ((m_pInstancePool->GetInstance(i).y - predict)
                    * (m_pInstancePool->GetInstance(i).y - predict)) *
                   m_pInstancePool->GetInstance(i).weight;
		//	printf("predict: %f ",predict);
		//	m_pInstancePool->GetInstance(i).print();
		}	

		//puts("zzzzzzz");
		return sqrt(ret / sum_weight);
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
