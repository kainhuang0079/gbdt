#include<sstream>
#include<cmath>
#include<assert.h>
#include"GradientBoosting.h"

namespace gbdt
{

	SparseInstance::SparseInstance(){}
	SparseInstance::~SparseInstance(){}
	std::string SparseInstance::ToString()
	{
		std::ostringstream oss;
		for(int i=0;i<X.size();i++)
		{
			oss<<X[i].first<<":"<<X[i].second<<",";
		}
		oss<<ys;
		return oss.str();
	}


	GradientBoostingForest::GradientBoostingForest(GbdtConf * pconfig):m_pconfig(pconfig), m_pInstancePool(NULL), m_pTestInstancePool(NULL), m_TotLeafCnt(0){}
	GradientBoostingForest::~GradientBoostingForest() { 
		for(uint32 i=0;i<m_Forest.size();i++)
		{
			m_Forest[i] = Comm::Delete(m_Forest[i]);
		}
//		puts("GradientBoostingForest del done");
	}
	int GradientBoostingForest::Fit(InstancePool * pInstancepool)
	{
		m_pInstancePool = pInstancepool;
		m_pInstancePool->MakeBucket();
		if(NULL == m_pInstancePool)
		{
			Comm::LogErr("GradientBoostingForest::Fit pInstancepool is NULL");
			return -1;
		}
		int ret = -1;
		for(int i=0;i<m_pconfig->TreeNum;i++)
		{
			DecisionTree * pTree = new DecisionTree(m_pconfig);
			ret = pTree->Fit(m_pInstancePool);

		//	printf("i = %d Fited pTree->FitError = %f\n",i,pTree->FitError());
			if(ret != 0)
			{
				Comm::LogErr("GradientBoostingForest::Fit fail! tree i = %d Fit fail!",i);
				return -1;
			}
			m_Forest.push_back(pTree);
			if(m_pconfig->LogLevel >= 2)printf("i = %d FitError = %f TestError = %f\n",i,FitError(),TestError());
			ret = Residual();
			printf("i = %d Residualed\n",i);
			if(ret != 0)
			{
				Comm::LogErr("GradientBoostingForest::Fit fail! Residual fail!");
				return -1;
			}
		}
		if(m_pconfig->IsLearnNewInstances)
		{
			ret = LearnNewInstance();
			if(ret != 0)
			{
				Comm::LogErr("GradientBoostingForest::Fit fail! LearnNewInstance fail!");
				return -1;
			}
		}

		ret = SaveResult();
		if(ret != 0)
		{
			Comm::LogErr("GradientBoostingForest::Fit fail ! SaveResult fail!");
			return -1;
		}
		
		if(m_pconfig->LogLevel >= 2)FeatureStat();

		return 0;
	}
	int GradientBoostingForest::SaveModel()
	{
		if(m_pconfig->OutputModelFilePath == "null")
		{
			Comm::LogInfo("GradientBoostingForest::SaveModel OutputModelFilePath is null");
			return 0;
		}
		FILE * fp;
		fp = fopen(m_pconfig->OutputModelFilePath.c_str(),"w");
		if(!fp)
		{
			Comm::LogErr("GradientBoostingForest::SaveModel fp is NULL open file %s fail!",m_pconfig->OutputModelFilePath.c_str());
			return -1;
		}
		int ret;
		
	//	puts("fopened");

		ret = fprintf(fp,"%u %d\n",m_Forest.size(),m_TotLeafCnt);
		if(ret < 0)
		{
			Comm::LogErr("GradientBoostingForest::SaveModel fail! fprintf m_Forest size fail!");
			fclose(fp);
			return -1;
		}
	//	puts("fopened fprintf");
		for(int i=0;i<m_Forest.size();i++)
		{
			if(!m_Forest[i])
			{
				Comm::LogErr("GradientBoostingForest::SaveModel fail! m_Forest[%d] is NULL",i);
				fclose(fp);
				return -1;
			}
			printf("i = %d\n",i);
			ret = m_Forest[i]->SaveModel(fp);
			if(ret != 0)
			{
				Comm::LogErr("GradientBoostingForest::SaveModel fail! m_Forest[%d] SaveModel fail!",i);
				fclose(fp);
				return -1;
			}
		}

	//	puts("fuck");

		fclose(fp);

		
		std::string configfile = m_pconfig->OutputModelFilePath + ".conf";
		fp = fopen(configfile.c_str(),"w");
		if(!fp)
		{
			Comm::LogErr("GradientBoostingForest::SaveModel open %s fail!",configfile.c_str());
			return -1;
		}
	//	puts("fuck, fuck");
		ret = fprintf(fp,"%s",m_pconfig->ToString().c_str());
		if(ret < 0)
		{
			Comm::LogErr("GradientBoostingForest::SaveModel fail! fprintf config fail!");
			fclose(fp);
			return -1;
		}
	//	puts("fuck, fuck, fuck");
		fclose(fp);
	//	puts("fuck, fuck, fuck, fuck");
		
		return 0;
	}
	int GradientBoostingForest::LoadModel()
	{
		FILE * fp;
		fp = fopen(m_pconfig->InputModelFilePath.c_str(),"r");
		if(!fp)
		{
			Comm::LogErr("GradientBoostingForest::LoadModel fp is NULL");
			return -1;
		}
		int ret;
		int treeNum;
		ret = fscanf(fp,"%d %d",&treeNum,&m_TotLeafCnt);
		if(ret < 0)
		{
			Comm::LogErr("GradientBoostingForest::LoadModel fail! fscanf treeNum fail!");
			fclose(fp);
			return -1;
		}
		for(int i=0;i<m_Forest.size();i++)
		{
			m_Forest[i] = Comm::Delete(m_Forest[i]);
		}
		m_Forest.clear();
		for(int i=0;i<treeNum;i++)
		{
			DecisionTree * pTree = new DecisionTree(m_pconfig);
			ret = pTree->LoadModel(fp);
			if(ret != 0)
			{
				Comm::LogErr("GradientBoostingForest::LoadModel fail! Tree %d LoadModel",i);
				fclose(fp);
				return -1;
			}
			m_Forest.push_back(pTree);
		}
		fclose(fp);
		printf("GradientBoostingForest::LoadModel treeNum = %d m_TotLeafCnt = %d\n",treeNum, m_TotLeafCnt);
		return 0;
	}
	int GradientBoostingForest::Predict(const std::vector<FloatT> &X, FloatT & predict)
	{
		int ret = 0;
		predict = 0;
		for(uint32 i=0;i<m_Forest.size();i++)
		{
			FloatT tmpPredict;
			ret = m_Forest[i]->Predict(X,tmpPredict);
			if(ret !=0)
			{
				Comm::LogErr("GradientBoostingForest::Predict fail! m_Forest %d Predict fail!",i);
				return -1;
			}
			predict += m_pconfig->LearningRate * tmpPredict;
		}
		return 0;
	}
	FloatT GradientBoostingForest::FitError()
	{
//		puts("FitError do");
		FloatT ret =0.0;
		assert(NULL != m_pInstancePool);
#pragma omp parallel for reduction(+:ret)
		for(int i=0;i<m_pInstancePool->Size();i++)
		{
			FloatT predict;
			if(0 != Predict(m_pInstancePool->GetInstance(i).X, predict))
			{
				Comm::LogErr("GradientBoostingForest::FitError fail! Predict fail");
			}
			ret = ret + ((m_pInstancePool->GetInstance(i).ys - predict) * (m_pInstancePool->GetInstance(i).ys - predict));
		}
//		puts("FitError done");
		return ret / m_pInstancePool->Size();
	}

	void GradientBoostingForest::SetTestInstancePool(InstancePool * pTestInstancePool)
	{
		m_pTestInstancePool = pTestInstancePool;
	}

	FloatT GradientBoostingForest::TestError()
	{
//		puts("TestError do");
		FloatT ret =0.0;
		if(NULL == m_pTestInstancePool)
		{
			Comm::LogErr("GradientBoostingForest::TestError fail m_pTestInstancePool is NULL");
			return -1;
		}

		FloatT lim[8] = {0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99};
		int cnt[8] = {0,0,0,0,0,0,0,0};
		int tot[8] = {0,0,0,0,0,0,0,0};

#pragma omp parallel for reduction(+:ret)
		for(int i=0;i<m_pTestInstancePool->Size();i++)
		{
			FloatT predict;
			if(0 != Predict(m_pTestInstancePool->GetInstance(i).X, predict))
			{
				Comm::LogErr("GradientBoostingForest::TestError fail! Predict fail!");
			}
			FloatT tmp = (m_pTestInstancePool->GetInstance(i).ys - predict);
			
	//		printf("%s predict:%f\n",m_pTestInstancePool->GetInstance(i).DebugStr().c_str(),predict);
			for(int j=0;j<8;j++)
			{
				if( predict >= lim[j])
				{
					if(m_pTestInstancePool->GetInstance(i).ys == 1)
					#pragma omp atomic
						cnt[j]++;
					#pragma omp atomic
					tot[j]++;
				}
			}

			tmp = tmp * tmp;	
			
			ret = ret + tmp;
		}

		for(int i=0;i<8;i++)
		{
			printf("predict >= %f cnt = %d tot = %d cnt/tot = %f\n",lim[i], cnt[i], tot[i], cnt[i]*1.0/tot[i]);
		}
		//printf("cnt = %d tot = %d\n",cnt,tot);
//		puts("TestError done");
		return ret;
		
	}

	int GradientBoostingForest::Residual()
	{
		Comm::WorkerThreadPool ResidualThreadPool("ResidualThreadPool");		
		int ret = ResidualThreadPool.Start(m_pconfig->ResidualThreadNum);
		if(ret != 0)
		{
			Comm::LogErr("GradientBoostingForest::Residual fail! ResidualThreadPool Start fail!");
			return -1;
		}
		uint32 blocksize = 1000;
		uint32 begin = 0;
		uint32 end = begin + blocksize;
		while(begin < m_pInstancePool->Size())
		{
			end = begin + blocksize < m_pInstancePool->Size() ? begin + blocksize : m_pInstancePool->Size();
			ret = ResidualThreadPool.AddWork(
					new ResidualThreadWork(
						m_pconfig,
						m_pInstancePool,
						this,
						begin,
						end
						)
					);
			if(ret !=0 )
			{
				Comm::LogErr("GradientBoostingForest::Residual fail!ResidualThreadPool AddWork fail!");
				return -1;
			}
			begin = end;
		}
		ResidualThreadPool.WaitAllWorkDone();
		ResidualThreadPool.Shutdown();
		ResidualThreadPool.JoinAll();

		return 0;

	}

	int GradientBoostingForest::GetNewInstance(const Instance & instance, Instance & newInstance)
	{
		if(!m_pconfig->IsLearnNewInstances)
		{
			Comm::LogErr("GradientBoostingForest::GetNewInstance fail! m_pconfig->IsLearnNewInstances is 0 !");
			return -1;
		}

		if(!m_TotLeafCnt)
		{
			Comm::LogErr("GradientBoostingForest::GetNewInstance fail! m_TotLeafCnt is 0");
			return -1;
		}
		int ret;
		std::vector<FloatT> tmpX;
		for(uint32 i=0;i<m_Forest.size();i++)
		{
			int fallLeafIndex;
			FloatT predict;
			ret = m_Forest[i]->Predict(instance.X,predict,fallLeafIndex);
			if(ret != 0)
			{
				Comm::LogErr("GradientBoostingForest::GetNewInstance fail! m_Forest[%d] Predict fail!",i);
				return -1;
			}
			int LeafIndex = m_Forest[i]->GetNode(fallLeafIndex).m_LeafIndex;
			if(-1 == LeafIndex)
			{
				Comm::LogErr("GradientBoostingForest::GetNewInstance fail m_Forest[%d] LeafIndex is -1 DebugStr <%s>",i,m_Forest[i]->GetNode(fallLeafIndex).DebugStr().c_str());
				return -1;
			}
			tmpX.push_back(LeafIndex);
		}
		for(int i=0;i<tmpX.size();i++)
		{
			newInstance.X.push_back(tmpX[i]);
		}

		//IsPushBackOgX
		if(m_pconfig->IsPushBackOgX)
		{
			for(int i=0;i<instance.X.size();i++)
			{
				newInstance.X.push_back(instance.X[i]);
			}
		}
		newInstance.ys = instance.ys;
		newInstance.y = instance.ys;
		return 0;	
	}

	int GradientBoostingForest::GetNewInstance(const Instance & instance, SparseInstance &newInstance)
	{
		if(!m_pconfig->IsLearnNewInstances)
		{
			Comm::LogErr("GradientBoostingForest::GetNewInstance fail! m_pconfig->IsLearnNewInstances is 0 !");
			return -1;
		}

		if(!m_TotLeafCnt)
		{
			Comm::LogErr("GradientBoostingForest::GetNewInstance fail! m_TotLeafCnt is 0");
			return -1;
		}
		int ret;
		for(uint32 i=0;i<m_Forest.size();i++)
		{
			int fallLeafIndex;
			FloatT predict;
			ret = m_Forest[i]->Predict(instance.X,predict,fallLeafIndex);
			if(ret != 0)
			{
				Comm::LogErr("GradientBoostingForest::GetNewInstance fail! m_Forest[%d] Predict fail!",i);
				return -1;
			}
			int LeafIndex = m_Forest[i]->GetNode(fallLeafIndex).m_LeafIndex;
			if(-1 == LeafIndex)
			{
				Comm::LogErr("GradientBoostingForest::GetNewInstance fail m_Forest[%d] LeafIndex is -1 DebugStr <%s>",i,m_Forest[i]->GetNode(fallLeafIndex).DebugStr().c_str());
				return -1;
			}
			newInstance.X.push_back(std::make_pair(LeafIndex,1.0));
		}

		//IsPushBackOgX
		if(m_pconfig->IsPushBackOgX)
		{
			for(int i=0;i<instance.X.size();i++)
			{
				newInstance.X.push_back(std::make_pair(m_TotLeafCnt + i,instance.X[i]));
			}
		}
		newInstance.ys = instance.ys;
		newInstance.y = instance.ys;
		return 0;	

	}

	int GradientBoostingForest::GetNewInstancePool(InstancePool &instancepool,InstancePool &newInstancePool)
	{
		for(int i=0;i<instancepool.Size();i++)
		{
			int ret;
			Instance newInstance;
			ret = GetNewInstance(instancepool.GetInstance(i),newInstance);
			if(ret != 0)
			{
				return -1;
			}
			newInstancePool.AddInstance(newInstance);
		}
		return 0;
	}

	int GradientBoostingForest::LearnNewInstance()
	{
		if(!m_pconfig->IsLearnNewInstances)	
		{
			Comm::LogErr("GradientBoostingForest::LearnNewInstance IsLearnNewInstances is 0");
			return -1;
		}
		//	std::vector< std::vector<int> > vecTmpInstances(m_pInstancePool->Size());
		int LeafCnt = 0;
		for(int i=0;i<m_Forest.size();i++)
		{
			DecisionTree * pTree = m_Forest[i];
			for(int j=0;j<pTree->Size();j++)
			{
			//	printf("SizeSize %d\n",pTree->Size());
				DecisionTreeNode & node = pTree->GetNode(j);
				if(LEAF == node.m_status)
				{
					/*
					if(node.m_InstancesHashCode == 1150390567)
					{
						printf("fuck debug");
						node.print();
					}
					*/
					if(NULL == node.m_ppInstances)
					{
						Comm::LogErr("GradientBoostingForest::LearnNewInstance fail m_ppInstances is NULL DebugStr = %s",node.DebugStr().c_str());
						return -1;
					}
					node.m_LeafIndex = LeafCnt;
					LeafCnt++;
					node.m_ppInstances = Comm::Free(node.m_ppInstances);
				}
			}
		}
		
		m_TotLeafCnt = LeafCnt;
			
		if(m_pconfig->OutputNewInstancesFilePath == "null")
		{
			Comm::LogInfo("GradientBoostingForest::LearnNewInstance OutputNewInstancesFilePath is null");
			return 0;
		}
		std::ostringstream oss;
		oss.clear();
		oss.str("");
		FILE * fp = fopen(m_pconfig->OutputNewInstancesFilePath.c_str(),"w");
		if(NULL == fp)
		{
			Comm::LogErr("GradientBoostingForest::LearnNewInstance fail! open %s fail!",m_pconfig->OutputNewInstancesFilePath.c_str());
			return -1;
		}

		//IsPushBackOgX
		if(m_pconfig->IsPushBackOgX)
			oss<<"n_feature,"<<LeafCnt + m_pconfig->FeatureNum<<"\n";
		else 
			oss<<"n_feature,"<<LeafCnt<<"\n";
		
		int ret = fputs(oss.str().c_str(),fp);
		if(ret < 0)
		{
			Comm::LogErr("GradientBoostingForest::LearnNewInstance fail!fputs %s fail!",oss.str().c_str());
			fclose(fp);
			return -1;
		}

		for(int i=0;i<m_pInstancePool->Size();i++)
		{
			Instance newInstance;
			ret = GetNewInstance(m_pInstancePool->GetInstance(i), newInstance);
			if(ret != 0)
			{
				Comm::LogErr("GradientBoostingForest::LearnNewInstance GetNewInstance i = %d fail!",i);
				fclose(fp);
				return -1;
			}
			std::string str = newInstance.ToString() + "\n";
			ret = fputs(str.c_str(),fp);
			if(ret < 0)
			{
				Comm::LogErr("GradientBoostingForest::LearnNewInstance fputs %s fail!",str.c_str());
				fclose(fp);
				return -1;
			}
		}

		fclose(fp);
		return 0;
	}

	int GradientBoostingForest::SaveResult()
	{
		if(m_pconfig->OutputResultFilePath == "null")
		{
			Comm::LogInfo("GradientBoostingForest::SaveResult OutputResultFilePath is null");
			return 0;
		}

		if(NULL == m_pTestInstancePool)
		{
			Comm::LogInfo("GradientBoostingForest::SaveResult m_pTestInstancePool is NULL");
			return 0;
		}
		FILE * fp;
		fp = fopen(m_pconfig->OutputResultFilePath.c_str(),"w");
		if(NULL == fp)
		{
			Comm::LogErr("GradientBoostingForest::SaveResult fail! open %s fail!",m_pconfig->OutputResultFilePath.c_str());
			return -1;
		}
		
		int ret;

		for(int i=0;i<m_pTestInstancePool->Size();i++)
		{
			FloatT predict;
			if(0 != Predict(m_pTestInstancePool->GetInstance(i).X, predict))
			{
				Comm::LogErr("GradientBoostingForest::SaveResult fail! Predict fail!");
				fclose(fp);
				return -1;
			}
			ret = fprintf(fp,"%f\n",predict);
			if(ret < 0)
			{
				Comm::LogErr("GradientBoostingForest::SaveResult fprintf fail");
				fclose(fp);
				return -1;
			}
		}

		fclose(fp);
		return 0;
	}

	void GradientBoostingForest::FeatureStat()
	{
		std::vector< std::pair< FloatT ,int > > vecFeatureInfo(m_pconfig->FeatureNum);
		for(int i=0;i<vecFeatureInfo.size();i++)
		{
			vecFeatureInfo[i].first = 0.0;
			vecFeatureInfo[i].second = i;
		}
		
		for(int i=0;i<m_Forest.size();i++)
		{
			for(int j=0;j<m_Forest[i]->Size();j++)
			{
				DecisionTreeNode & node = m_Forest[i]->GetNode(j);
				if(-1 == node.m_splitFeatureId || -1 == node.m_lson || -1 == node.m_rson)
					continue;
				DecisionTreeNode & lson = m_Forest[i]->GetNode(node.m_lson);
				DecisionTreeNode & rson = m_Forest[i]->GetNode(node.m_rson);

				FloatT target1 = node.m_sum_y_X_weight * node.m_sum_y_X_weight / node.m_sumWeight;
				FloatT target2 = lson.m_sum_y_X_weight * lson.m_sum_y_X_weight / lson.m_sumWeight + rson.m_sum_y_X_weight * rson.m_sum_y_X_weight / rson.m_sumWeight;

				vecFeatureInfo[node.m_splitFeatureId].first += (target2 - target1);
			}
		}
		
		std::sort( vecFeatureInfo.begin() , vecFeatureInfo.end() );
	
		std::reverse(vecFeatureInfo.begin(), vecFeatureInfo.end());

		for(int i=0;i<vecFeatureInfo.size();i++)
		{
			printf("rank:%d:feature:%d:%f, ",i,vecFeatureInfo[i].second,vecFeatureInfo[i].first);
		}
		printf("\n");
		
	}


	ResidualThreadWork::ResidualThreadWork(
			GbdtConf * pconfig,
			InstancePool *pInstancepool,
			GradientBoostingForest * pModel,
			uint32 begin,
			uint32 end
			)
	{
		m_pconfig = pconfig;
		m_pInstancePool = pInstancepool;
		m_pModel = pModel;
		m_begin = begin;
		m_end = end;
	}

	ResidualThreadWork::~ResidualThreadWork()
	{
	}
	
	bool ResidualThreadWork::NeedDelete()const
	{
		return true;
	}

	int ResidualThreadWork::DoWork()
	{
		int ret = -1;
		if(!m_pconfig)
		{
			Comm::LogErr("ResidualThreadWork::DoWork fail! m_pconfig is NULL");
			return -1;
		}
		if(!m_pInstancePool)
		{
			Comm::LogErr("ResidualThreadWork::DoWork fail! m_pInstancePool is NULL!");
			return -1;
		}
		if(!m_pModel)
		{
			Comm::LogErr("ResidualThreadWork::DoWork fail! m_pModel is NULL");
			return -1;
		}
		for(int i = m_begin ;i < m_end; i++)
		{
			FloatT predict;
			ret = m_pModel->Predict(m_pInstancePool->GetInstance(i).X, predict);
			if(ret !=0)
			{
				Comm::LogErr("ResidualThreadWork::DoWork fail! m_pModel Predict fail!");
				return -1;
			}
			m_pInstancePool->GetInstance(i).y = m_pInstancePool->GetInstance(i).ys - predict;
		}
		return 0;	
	}

}
