#pragma once 
#include<unistd.h>
#include<pthread.h>
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<vector>
#include"Log.h"
#include"unity.h"
namespace Comm
{
	template<class T> class MemPool
	{
		public:
			MemPool(unsigned int maxSize)
			{
				m_num = 0;
				m_maxSize = maxSize;
				LogInfo("MemPool::MemPool() malloc memory size =  %uB", (unsigned int)(maxSize + 107) *sizeof(T));
				m_pMem = (T *)malloc( (maxSize + 107) * sizeof(T) ); 	
				if(NULL != m_pMem)LogInfo("MemPool::MemPool() malloc memory ok");
				pthread_mutex_init(&m_mutex, NULL);
			}

			~MemPool()
			{
				free(m_pMem); 
				m_pMem = NULL;
				pthread_mutex_destroy(&m_mutex);
			}

			int ReSet(int maxSize)
			{
				m_num = 0;
				m_maxSize = maxSize;
				free(m_pMem);
				m_pMem = NULL;
				printf("MemPool::ReSet() malloc memory size = %uB\n", (unsigned int)(maxSize + 107) *sizeof(T));
				m_pMem = (T *)malloc( (maxSize + 107) * sizeof(T) ); 	
				if(NULL != m_pMem)printf("MemPool::ReSet() malloc memory ok\n");
				else 
				{
					printf("MemPool::ReSet() malloc memory fail!\n");
					return -1;
				}
				return 0;

			}

			int Add(const T &item)
			{
				if(m_num >= m_maxSize)
				{
//					printf("MemPool::Add mempool had full! m_num = %d ,maxSize = %d\n",m_num, m_maxSize);
					LogErr("MemPool::Add mempool had full! m_num = %d ,maxSize = %d",m_num, m_maxSize);
					return -1;
				}
				m_pMem[m_num++] = item;
				return 0;
			}
			int New(const T &item)
			{
				int index = -1;
				pthread_mutex_lock(&m_mutex);
				int ret = Add(item);
				if(ret != 0)
				{
//					puts("MemPool::New new item fail,maybe the mempool had full");
					LogErr("MemPool::New new item fail,maybe the mempool had full");
					pthread_mutex_unlock(&m_mutex);
					return -1;
				}
				index = m_num - 1;
				pthread_mutex_unlock(&m_mutex);
				return index;
			}

			T & Get(int index) 
			{
				if(index >= m_num)
				{
					printf("MemPool::Get index overflow index = %d mempool side = %d\n", index, m_num);
					LogErr("MemPool::Get index overflow index = %d mempool side = %d", index, m_num);
				}
				return m_pMem[index];
			}

			T & operator [](int index) 
			{
				return Get(index);
			}
			int Size()const
			{
				return m_num;
			}

			int GetNum()const
			{
				return m_num;
			}

			bool IsFull()const
			{
				return m_num >= m_maxSize;
			}

			void Clear()
			{
				m_num = 0;
			}
		private:
			T * m_pMem;
			pthread_mutex_t m_mutex;
			unsigned int m_maxSize;
			int m_num;
	};
}
