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
#include <sys/time.h> 
#include"Log.h"
namespace Comm
{
	
	class TimeStat
	{
		public:
			TimeStat(const std::string & prefix)
			{
				m_start = getCurrentTime();
				m_prefix = prefix;
				//LogDebug("%s begin",m_prefix.c_str());
				printf("%s begin\n",m_prefix.c_str());
				puts("---------------------------------------------------------------------------------------------------------\n");
			}

			void TimeMark(const std::string & minstr)
			{
				//LogDebug("%s %s TimeCost : %dms",m_prefix.c_str(), minstr.c_str(), (getCurrentTime() - m_start));
				printf("%s %s TimeCost : %dms\n",m_prefix.c_str(), minstr.c_str(), (getCurrentTime() - m_start));
				puts("----------------------------------------------------------------------------------------------------------\n");
			}

			~TimeStat()
			{
				//LogDebug("%s  TimeCost : %dms",m_prefix.c_str(),(getCurrentTime() - m_start));
				printf("%s  TimeCost : %dms\n",m_prefix.c_str(),(getCurrentTime() - m_start));
				puts("----------------------------------------------------------------------------------------------------------\n");
			}
			long getCurrentTime()  
			{  
				struct timeval tv;  
				gettimeofday(&tv,NULL);  
				return tv.tv_sec * 1000 + tv.tv_usec / 1000;  
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
