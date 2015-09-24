#include<sys/types.h>
#include<unistd.h>
#include<iostream>
#include<cstdlib>
#include<cstring>
#include<string>
#include<time.h>
#include<cstdio>
#include"threadpool.h"
#include"unity.h"
#include"Log.h"
#include"mempool.h"
using namespace std;
using namespace Comm;
/*
class node
{
	public:
		node(int a, int b, int c, int d)
		{
			u = a;
			v = b;
			val = c;
			next = d;
		}
		void print()
		{
			cout<<u<<" "<<v<<" "<<val<<" "<<next<<endl;
		}
	public:
		int u, v, val, next;
};

class Graph
{
	public:
		Graph():mempool(500000)
		{
			memset(eh ,-1, sizeof(eh));
			pthread_mutex_init(&m_mutex, NULL);
		}
		~Graph()
		{
			pthread_mutex_destroy(&m_mutex);
		}
		void add(int u, int v ,int val)
		{
			pthread_mutex_lock(&m_mutex);
			eh[u] = mempool.New(node(u,v,val,eh[u]));
			pthread_mutex_unlock(&m_mutex);
		}
		void print(int limit)
		{
			for(int u = 0; u <= limit; u++)
			{
				printf("%d " , u);
				for(int v = eh[u]; v != -1; v = mempool[v].next)
				{
					printf("%d ",mempool[v].v);
				}
				puts("");
			}
			printf("mempool size %d\n",mempool.Size());
			for(int i=0;i<mempool.Size();i++)
				mempool[i].print();
		}
	public:
		int eh[10086];
		pthread_mutex_t m_mutex;
		MemPool<node> mempool;
}G;
*/
class TestWork :public Work
{
	public:
		TestWork(int u,int v,int val){
			m_u = u;
			m_v = v;
			m_val = val;
		}

		bool NeedDelete()const
		{
			return true;
		}
		int DoWork() {
			int a = 1;
			for(int i=0;i<100000;i++)
			{
//				G.add(m_u,m_v,m_val);
//				Comm::LogErr("aaaaa");
//				Comm::LogInfo("bbbbb");
//				Comm::LogDebug("ccccc");
				a += a * a / a + 1 + m_val + m_u + m_v * rand();
			}
//			printf("%d\n",a);
			return 0;
		}

	private:
		int m_u,m_v,m_val;
};

int main()
{

	puts("zzzzzzzzzzzzzzzzzzzz");
	string strSet[3] = {
		"aaaaaa",
		"bbbbbb",
		"cccccc"
	};
	for(int i=0;i<3 ;i++)
		cout<<strSet[i]<<endl;
	srand(time(NULL));
	for(int i=0;i<10;i++)
		cout<<rand()<<endl;
	Comm::LogInit("w",3);
	WorkerThreadPool pool("sssss");
	freopen("in","r",stdin);
	int n , m;
	cin>>n>>m;
	puts("waiting");
//	sleep(10);
	Comm::LogInfo("begin");
	while(m--)
	{
		int u,v,val;
		cin>>u>>v>>val;
		for(int i = 0 ; i<1200 ; i++ )
		{
			pool.AddWork(new TestWork(u,v,val));
		}
	}
	TimeStat Ti("kain");
	pool.Start(1);
	pool.WaitAllWorkDone();

	Comm::LogInfo("end");

	printf("zzzzzzzz\n");
	pool.Shutdown();
	pool.JoinAll();
	return 0;
}
