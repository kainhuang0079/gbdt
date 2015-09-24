/*************************************************************************
    > File Name: threadpool.cpp
    > Author: ma6174
    > Mail: ma6174@163.com 
    > Created Time: Mon 21 Apr 2014 12:44:38 PM CST
 ************************************************************************/

#include<iostream>
#include"threadpool.h"
#include"Log.h"
namespace Comm{


	int Thread::Run()
	{
		running =true;
		int ret=-1;
		if(target!=NULL)
		{
			ret=target->Run();
		}
		else
		{
			ret=this->DoRun();
		}
		running =false;
//		puts("runed");
		return ret;
	}

	int Thread::Start()
	{
		
		int ret = pthread_create(&tid,NULL,thread_proxy_func,this);
		while(!running)
		{
			usleep(1);
		}
		return ret;
	}


	int Thread::Join()
	{
		if(tid >0 )
		{
			int ret=pthread_join(tid,NULL);	
	//		printf("join tid = %u\n", tid);
			Comm::LogInfo("join tid = %u",tid);
			running=false;
			return ret;
		}
		Comm::LogErr("Thread::Join fail");
		return -1;
	}

	WorkQueue::WorkQueue():_shut_down(false)
	{
		pthread_mutex_init(&mutex,NULL);
		pthread_cond_init(&cond,NULL);
	}
	WorkQueue::~WorkQueue(){
		pthread_mutex_destroy(&mutex);
		pthread_cond_destroy(&cond);
	}
	int WorkQueue:: AddWork(Work *work)
	{
		if( NULL == work )
		{
			Comm::LogErr( "WorkQueue::add work fail,the work is null, ret = -1" );
			return -1;
		}
		if(_shut_down)
		{
			Comm::LogErr( "WorkQueue::add work fail , queue had shut down");
			return -1;
		}
		pthread_mutex_lock(&mutex);
		works.push(work);
		pthread_cond_signal(&cond);
		pthread_mutex_unlock(&mutex);
		return 0;
	}
	Work * WorkQueue:: GetWork()
	{
		if(_shut_down)
		{
			Comm::LogInfo( "WorkQueue::get work fail, had shut down" );
			return NULL;
		}
		pthread_mutex_lock(&mutex);
		Work * tmp=NULL;
		if(!works.empty())
		{
			tmp =works.front();
			works.pop();
		}
		else
		{
			while( works.empty() )
			{
				if( _shut_down )
				{
					pthread_mutex_unlock(&mutex);
					Comm::LogInfo( "WorkQueue::get work fail, queue had shut down" );
					return NULL;
				}
				pthread_cond_wait(&cond,&mutex);
			}
			tmp = works.front();
			works.pop();
		}
		pthread_mutex_unlock(&mutex);
		if( NULL == tmp )Comm::LogInfo( "WorkQueue::get work fail" );
		return tmp;
	}

	int WorkQueue:: Shutdown()
	{
		int ret=-1;
		pthread_mutex_lock(&mutex);
		_shut_down=true;
		ret=pthread_cond_broadcast(&cond);
		while(!works.empty())works.pop();
		pthread_mutex_unlock(&mutex);
		return ret;
	}

	bool WorkQueue:: IsShutdown()
	{
		return _shut_down;
	}




	Worker :: Worker(WorkQueue &queue):workQueue(queue), doing_work(0)
	{
		pthread_mutex_init(&mutex,NULL);
	}
	Worker:: ~Worker()
	{
		pthread_mutex_destroy(&mutex);
	}
	int Worker:: Run()
	{
		while(true)
		{
			if(workQueue.IsShutdown())return 0;
			Work * tmp=workQueue.GetWork();
			if(tmp!=NULL)
			{
				IncDoingWork();
				int ret = tmp->DoWork();
				if(ret !=0)
				{
					Comm::LogErr("Worker::Run Dowork fail!");
				}
				DecDoingWork();
			}
			else Comm::LogInfo( "Worker::Run work is null(workQueue is Shutdown?)" );
			if(tmp!=NULL&&tmp->NeedDelete())
			{
				delete tmp;
				tmp=NULL;
			}
			else
				Comm::LogInfo( "Worker::Run work is null before delete(workQueue is Shutdown?)" );
		}
		return -1;
	}

	void Worker::IncDoingWork()
	{
		pthread_mutex_lock(&mutex);
		doing_work++;
		pthread_mutex_unlock(&mutex);
	}
	void Worker::DecDoingWork()
	{
		pthread_mutex_lock(&mutex);
		doing_work--;
		pthread_mutex_unlock(&mutex);
	}

	ThreadPool::ThreadPool(){}
	ThreadPool:: ~ThreadPool(){
		for(int i=0;i<threads.size();i++)
			delete threads[i];
	}
	int ThreadPool::Start(int threadCount,Runnable &target)
	{
		for(int i=0;i<threadCount;i++)
		{
			Thread *tmp=new Thread(target);
			threads.push_back(tmp);
			if(tmp->Start()!=0)return -1;
		}
		return 0;
	}

	int ThreadPool::JoinAll()
	{
		for(int i=0;i<threads.size();i++)
		{
			if(threads[i]->Join()!=0)return -1;
		}
		return 0;
	}



	WorkerThreadPool::WorkerThreadPool(const std::string &name):worker(workQueue),logName(name){
	}
	WorkerThreadPool:: ~WorkerThreadPool(){
	}
	int WorkerThreadPool::Start(int threadCount)
	{
		return pool.Start(threadCount,worker);
	}
	int WorkerThreadPool::Start(int threadCount,Runnable &target)
	{
		return pool.Start(threadCount ,target);
	}
	int WorkerThreadPool::AddWork(Work *work)
	{
		return workQueue.AddWork(work);
	}
	int WorkerThreadPool::Shutdown()
	{
		return workQueue.Shutdown();
	}
	int WorkerThreadPool:: JoinAll()
	{
		return pool.JoinAll();
	}
	bool WorkerThreadPool::IsAllWorkDone()const
	{
		for(int i=0; i < 30; i++)
		{
			if( !( workQueue.Empty() && ( 0 == worker.GetDoingWork() ) ) )
			{
//				printf("Empty = %d , doing_work = %d\n", workQueue.Empty(),worker.GetDoingWork());
				return false;
			}
			usleep(5);
		}
		return true;
	}
	void WorkerThreadPool::WaitAllWorkDone()const
	{
		int i = 0;
		while(!IsAllWorkDone())
		{
			i++;
			if( i % 10000 == 0 )
				printf("WorkerThreadPool::WaitAllWorkDone %s WorkQueueSize = %d DoingWork %d\n", logName.c_str(),WorkQueueSize(),worker.GetDoingWork());
			usleep(100);
		}
	}
}

