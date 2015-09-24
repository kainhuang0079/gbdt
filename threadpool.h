// =====================================================================================
// 
//       Filename:  theadpool.h
// 
//    Description:  
// 
//        Version:  1.0
//        Created:  04/16/2014 11:35:37 AM //       Revision:  none //       Compiler:  g++
// 
//         Author:  YOUR NAME (), 
//        Company:  
// 
// =====================================================================================
#pragma once
#include<pthread.h>
#include<deque>
#include<vector>
#include<iostream>
#include<cstdio>
#include"runnable.h"
#include<queue>
#include<unistd.h>
#include"Log.h" 
namespace Comm{
	
	
	class Thread:public Runnable{
		public:
			Thread():running(false), target(NULL){}
			Thread(Runnable & t): running(false),target(&t){}

			virtual ~Thread()
			{
				Comm::LogInfo("exit tid= %u",tid);
			}
			int Run();
			virtual int Start();	
			int Join();
			inline pthread_t GetId()const 
			{
				return tid;
			}
			inline bool IsRunning() const
			{
				return running;
			}
			static void * thread_proxy_func(void *args)
			{
				Thread *pThread = static_cast<Thread*>(args);
				pThread->Run();
				return NULL;
			}

		protected:
			virtual int DoRun()
			{
				return 0;
			}

			bool running;
			pthread_t tid;
			Runnable * target;
	};




	class Work{
		public:
			virtual ~Work(){
			}
			virtual bool NeedDelete()const =0;
			virtual int DoWork()=0;		
	};


	class WorkQueue{

		public:
			WorkQueue();

			~WorkQueue();

			int AddWork(Work * work);

			Work *GetWork();

			int Shutdown();

			bool IsShutdown();

			inline int Size()const
			{
				return works.size();
			}

			inline bool Empty()const
			{
				return works.empty();
			}
		private:
			typedef std:: queue<Work*> Queue;
			pthread_mutex_t mutex;
			pthread_cond_t cond;
			bool _shut_down;
			Queue works;
	};
	


	class ThreadPool
	{
		public:
			ThreadPool();
			virtual ~ThreadPool();
			virtual int Start(int threadCount,Runnable &target);
			int JoinAll();
		protected:
			typedef std :: vector<Thread*> ThreadVec;
			ThreadVec threads;
	};

	
	class Worker :public Runnable
	{
		public:
			Worker(WorkQueue &queue);
			~Worker();
			int Run();
			inline int GetDoingWork()const
			{
				return doing_work;
			}
		private:
			void IncDoingWork();
			void DecDoingWork();
			WorkQueue & workQueue;
			int doing_work;
			pthread_mutex_t mutex;
	};

	class WorkerThreadPool
	{
		public:
			WorkerThreadPool(const std::string &name);
			~WorkerThreadPool();
			int Start(int threadCount);
			int AddWork(Work *work);
			int Shutdown();
			int JoinAll();
			bool IsAllWorkDone()const;
			void WaitAllWorkDone()const;				
			inline int WorkQueueSize()const
			{
				return workQueue.Size();
			}
		protected:
			int Start(int threadCount,Runnable & target);
			WorkQueue workQueue;
			Worker worker;
			ThreadPool pool;
			std::string logName;
	};

	
	
}



