#include"instancepool.h"
#include"gbdtconf.h"
#include"unity.h"

namespace gbdt
{
	class Model
	{
		virtual ~Model(){}
		virtual int Fit(InstancePool *pInstancepool) = 0;
		virtual int Predict(const std::vector<FloatT> &X, FloatT & predict) = 0;
		virtual FitError();
	};
}
