#include"GradientBoosting.h"

using namespace std;
using namespace Comm;
using namespace gbdt;

int main(int argc , char **argv)
{
	TimeStat ts(argv[0]);
	LogInit("w",1);
	GbdtConf config;
	config.Init("DT.conf");
	LogInfo("begin");
	InstancePool instancepool(&config);
	instancepool.Input();
	InstancePool testpool(&config);
	testpool.Input(config.TestDataFliePath);
	GradientBoostingForest forest(&config);
	forest.SetTestInstancePool(&testpool);
	forest.Fit(&instancepool);
	printf("FitError = %f TestError = %f\n",forest.FitError(),forest.TestError());
	forest.SaveModel();

	GbdtConf config2;
	string str = config.OutputModelFilePath+".conf";
	config2.Init(str.c_str());
	GradientBoostingForest forest2(&config2);
	forest2.SetTestInstancePool(&testpool);
	forest2.LoadModel();
	printf("TestError = %f\n",forest2.TestError());

	InstancePool newInstancePool(&config2);
	forest2.GetNewInstancePool(testpool,newInstancePool);
//	newInstancePool.print();
	
	GbdtConf config3;
	config3.Init("DT2.conf");
	InstancePool trainpool(&config3);
	trainpool.Input();
	GradientBoostingForest forest3(&config3);
	forest3.SetTestInstancePool(&newInstancePool);
	forest3.Fit(&trainpool);
//	cout<<"asasasasasasaas"<<endl;
	forest3.SaveModel();

/*	
	for(int i=0;i<instancepool.Size();i++)
	{
		Instance newInstance;
		forest2.GetNewInstance(instancepool.GetInstance(i),newInstance);
		printf("%s\n",newInstance.ToString().c_str());
	}
*/
	LogInfo("end");
	return 0;
}
