#include"GradientBoosting.h"
#include"unity.h"
#include<iostream>
#include<cstdlib>
#include<cstdio>
#include<map>
#include<string>

using namespace Comm;
using namespace gbdt;
using namespace std;

void get_cmd_map(int argc, char * argv[], map<string, string> &cmd_map)
{
	for(int i = 0; i < argc; i++)
	{
		vector<string> col;
		stringHelper::split(argv[i], "=", col);
		if(col.size() == 2)
		{
			cmd_map[col[0]] = col[1];
		}
	}
}

bool in(string key, map<string, string> &mp)
{
	if(mp.find(key) != mp.end())
		return true;
	else
		return false;
}

int main(int argc,char * argv[])
{
	if(argc < 2)
	{
		printf("Usage: %s [--config=<must>] [--train=<option>] [--test=<option>] [--candidate=<option>] [--in_model=<option>] [--out_test=<option>] [--out_candidate=<option>] [--out_model=<option>] [--leaf_train=<option>] [--leaf_test=<option>] [--leaf_candidate=<option>]\n", argv[0]);
		return -1;
	}
	map<string, string> cmd_map;
	get_cmd_map(argc, argv, cmd_map);

	if(!in("--config", cmd_map))
	{
		puts("config is None");
		return 1;
	}
	

	GbdtConf config;
	int ret = -1;
	if(in("--config", cmd_map))
	{
		ret = config.Init(cmd_map["--config"].c_str());
		if(ret != 0)
		{
			puts("ERR:config Init fail");	
			return -1;
		}
	}

	LogInit("w",config.LogLevel);

	string candidate = "";
	string out_candidate = "";
	string leaf_train = "";
	string leaf_test = "";
	string leaf_candidate = "";


	for(map<string, string>::iterator it = cmd_map.begin(); it != cmd_map.end(); it++)
	{
		if("--train" == it->first)
		{
			config.InputDataFilePath = it->second;
		}
		if("--test" == it->first)
		{
			config.TestDataFliePath = it->second;
		}
		if("--candidate" == it->first)
		{
			candidate = it->second;
		}
		if("--in_model" == it->first)
		{
			config.InputModelFilePath = it->second;
		}
		if("--out_model" == it->first)
		{
			config.OutputModelFilePath = it->second;
		}
		if("--out_test" == it->first)
		{
			config.OutputResultFilePath = it->second;
		}
		if("--out_candidate" == it->first)
		{
			out_candidate = it->second;
		}
		if("--leaf_train" == it->first)
		{
			leaf_train = it->second;
		}
		if("--leaf_test" == it->first)
		{
			leaf_test = it->second;
		}
		if("--leaf_candidate" == it->first)
		{
			leaf_candidate = it->second;
		}
		cout << it->first << "=" << it->second << endl;
	}

	GradientBoostingForest forest(&config);
	
	if(in("--in_model", cmd_map))
	{
		ret = forest.LoadModel();
		if(ret != 0)
		{
			puts("ERR::LoadModel faol");
			return -1;
		}
		puts("LoadModel finish");
	}
	else if(in("--train", cmd_map))
	{
		InstancePool trainPool(&config);
		ret = trainPool.Input();
		if(ret !=0 )
		{
			puts("ERR:train data Input fail");
			return -1;
		}
		InstancePool testinstancepool(&config);
		if(in("--test", cmd_map) && config.TestDataFliePath != "null")
		{
			ret = testinstancepool.Input(config.TestDataFliePath);
			if(ret != 0)
			{
				puts("ERR:testinstancepool input fail");
				return -1;
			}
			forest.SetTestInstancePool(&testinstancepool);
		}
		TimeStat ts(argv[0]);
		ret = forest.Fit(&trainPool);
		ts.TimeMark("forest Fit finish");
		if(ret != 0)
		{
			puts("ERR:forest fit fail!");
			return -1;
		}
		if(in("--test", cmd_map) && config.TestDataFliePath != "null")
		{
			if(leaf_test != "")
			{
				FILE * fp;
				fp = fopen(leaf_test.c_str(), "w");
				if(!fp)
				{
					return -1;
				}
				for(int i = 0; i < testinstancepool.Size(); i++)
				{
					FloatT _;
					vector<int> leafs;
					ret = forest.Predict(testinstancepool[i].X, _, leafs);
					if(ret != 0)
					{
						return -1;
					}

					ostringstream oss;
					for(int j = 0; j < leafs.size(); j++)
					{
						if(j != 0)
						{
							oss << ",";
						}
						oss << leafs[j];
					}
					ret = fprintf(fp, "%s\n", oss.str().c_str());
					if(ret < 0)
					{
						return -1;
					}
				}
				puts("save leaf_test finish");
			}
		}
		if(leaf_train != "")
		{
			FILE * fp;
			fp = fopen(leaf_train.c_str(), "w");
			if(!fp)
			{
				return -1;
			}
			for(int i = 0; i < trainPool.Size(); i++)
			{
				FloatT _;
				vector<int> leafs;
				ret = forest.Predict(trainPool[i].X, _, leafs);
				if(ret != 0)
				{
					return -1;
				}

				ostringstream oss;
				for(int j = 0; j < leafs.size(); j++)
				{
					if(j != 0)
					{
						oss << ",";
					}
					oss << leafs[j];
				}
				ret = fprintf(fp, "%s\n", oss.str().c_str());
				if(ret < 0)
				{
					return -1;
				}
			}
			puts("save leaf_train finish");
		}

		if(in("--out_model", cmd_map) && config.OutputModelFilePath != "null")
		{
			ret = forest.SaveModel();
			if(ret != 0)
			{
				puts("ERR:forest SaveModel fail!");
				return -1;
			}
			puts("output model finish");
		}

	}
	if(candidate != "")
	{
		InstancePool candidate_set(&config);
		ret = candidate_set.Input(candidate);
		if(ret != 0)
		{
			puts("ERR:candidate_set input fail");
			return -1;
		}
		vector<FloatT> vecPredict;
		ret = forest.BatchPredict(&candidate_set, vecPredict);
		if(out_candidate != "")
		{
			FILE * fp;
			fp = fopen(out_candidate.c_str(), "w");
			if(!fp)
				return -1;
			for(int i = 0; i < vecPredict.size(); i++)
			{
				ret = fprintf(fp, "%f\n", vecPredict[i]);
				if(ret < 0)
					return -1;
			}
			fclose(fp);
			puts("save out_candidate finish");

		}
		if(leaf_candidate !="")
		{
			FILE * fp;
			fp = fopen(leaf_candidate.c_str(), "w");
			if(!fp)
			{
				return -1;
			}
			for(int i = 0; i < candidate_set.Size(); i++)
			{
				FloatT _;
				vector<int> leafs;
				ret = forest.Predict(candidate_set[i].X, _, leafs);
				if(ret != 0)
				{
					return -1;
				}

				ostringstream oss;
				for(int j = 0; j < leafs.size(); j++)
				{
					if(j != 0)
					{
						oss << ",";
					}
					oss << leafs[j];
				}
				ret = fprintf(fp, "%s\n", oss.str().c_str());
				if(ret < 0)
				{
					return -1;
				}
			}
			puts("save leaf_candidate finish");
		}
	}

	return 0;

}
