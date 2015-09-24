CFLAGS =-O3 -Wall
FFLAGS =-lpthread -fopenmp
gbdt_train:gbdtmain.cpp mempool.h DecisionTree.o unity.o Log.o gbdtconf.o threadpool.o instancepool.o GradientBoosting.o
	g++ $(CFLAGS) -o gbdt_train gbdtmain.cpp DecisionTree.o unity.o Log.o gbdtconf.o threadpool.o instancepool.o GradientBoosting.o $(FFLAGS)
testGradientBoosting:testGradientBoosting.cpp mempool.h DecisionTree.o unity.o Log.o gbdtconf.o threadpool.o instancepool.o GradientBoosting.o 
	g++ $(CFLAGS) -o testGradientBoosting testGradientBoosting.cpp DecisionTree.o unity.o Log.o gbdtconf.o threadpool.o instancepool.o GradientBoosting.o $(FFLAGS)
test:test.cpp mempool.h Log.o threadpool.o unity.o 
	g++  $(CFLAGS) -o test test.cpp Log.o threadpool.o unity.o $(FFLAGS) 
threadpool.o:threadpool.cpp threadpool.h runnable.h
	g++ $(CFLAGS) -c threadpool.cpp $(FFLAGS)
clean:
	rm *.o gbdt_train test testunity testDecisionTree testGradientBoosting
testDecisionTree:testDecisionTree.cpp mempool.h DecisionTree.o unity.o Log.o gbdtconf.o threadpool.o instancepool.o
	g++ $(CFLAGS) -o testDecisionTree testDecisionTree.cpp DecisionTree.o unity.o Log.o gbdtconf.o threadpool.o instancepool.o $(FFLAGS)
DecisionTree.o:DecisionTree.cpp DecisionTree.h
	g++ $(CFLAGS) -c DecisionTree.cpp $(FFLAGS)
unity.o:unity.cpp unity.h
	g++ $(CFLAGS) -c unity.cpp
Log.o:Log.cpp Log.h
	g++ $(CFLAGS) -c Log.cpp
gbdtconf.o:gbdtconf.cpp gbdtconf.h
	g++ $(CFLAGS) -c gbdtconf.cpp
testunity:testunity.cpp unity.o Log.o gbdtconf.o
	g++ $(CFLAGS) -o testunity testunity.cpp unity.o Log.o gbdtconf.o
instancepool.o:instancepool.cpp instancepool.h
	g++ $(CFLAGS) -c instancepool.cpp $(FFLAGS)
GradientBoosting.o:GradientBoosting.cpp GradientBoosting.h
	g++ $(CFLAGS) -c GradientBoosting.cpp $(FFLAGS)
