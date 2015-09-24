#pragma once
#include<map>
#include<iostream>
#include<cstdlib>
#include<cstdio>
#include<cstring>
#include<string>
#include<vector>
#include<algorithm>
#include<sstream>
#include<cmath>
#include<assert.h>
#include<time.h>
#include<pthread.h>
namespace Comm
{
	
	class TimeStat
	{
		public:
			TimeStat(const std::string & prefix)
			{
				m_start = time(NULL);
				m_prefix = prefix;
				printf("%s begin\n",m_prefix.c_str());
				puts("---------------------------------------------------------------------------------------------------------\n");
			}

			~TimeStat()
			{
				printf("%s  TimeCost : %ds\n",m_prefix.c_str(),(time(NULL) - m_start));
				puts("----------------------------------------------------------------------------------------------------------\n");
			}
		private:
			time_t m_start;
			std::string m_prefix;
	};

	class ThreadLock
	{
		public:
			ThreadLock();
			~ThreadLock();
			int Lock();
			int UnLock();
		private:
			pthread_mutex_t m_mutex;
	};

	class ThreadLockGuard
	{
		public:
			ThreadLockGuard(ThreadLock * plock);
			~ThreadLockGuard();
			int Lock();
			int UnLock();
		private:
			ThreadLock * m_plock;
			bool m_islock;
		
	};


	template<typename ClassT> 
		ClassT* Delete(ClassT* p) {
			if ( NULL != p && p) {
				delete p; 
			}  
			p = NULL;
			return p; 
		}

	template<typename TypeT> 
		TypeT* Free(TypeT* p) {
			if ( NULL != p && p) {
				free(p);
			}  
			p = NULL; 
			return p; 
		}
	class stringHelper
	{
		public:
			static int split(const char *str,const char *spset,std::vector<std::string>&RetSet);
			static bool isInSpset(const char c,const char *spset);
	};

	class Config
	{
		public:
			Config();
			virtual ~Config();
			virtual int Init(const char * configFile) = 0;
			int makePool( const char * configFile);
			std::string ToString();
			std::map<std::string,std::string> pool;
		private:

	};

}
