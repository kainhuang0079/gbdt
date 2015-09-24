#include<sys/stat.h>
#include<sys/types.h>
#include<unistd.h>
#include<cstdlib>
#include<cstdio>
#include<string>
#include<iostream>
#include<time.h>
#include<stdarg.h>
#include<pthread.h>
#include"Log.h"
namespace Comm
{
	static FILE * THE_ONE_LOG_FILE;
	static std::string THE_ONE_LOG_TIME_HEAD;
	static pthread_mutex_t LOG_MUTEX;
	static std::string THE_PROCCESS_CMD;
	static bool IS_INIT = 0;
	static int LOG_LEVEL = 3;
	std::string GetDate()
	{
		time_t t= time(NULL);
		struct tm * p_sys_tm = localtime(&t);
		char date[96];
		snprintf(date , 96, "%d%02d%02d",p_sys_tm->tm_year+1900 , p_sys_tm->tm_mon+1, p_sys_tm->tm_mday );
		std::string ret =date;
		return ret;
	}
	std::string GetTime()
	{
		time_t t = time(NULL);
		struct tm * p_sys_tm  = localtime( &t );
		char time[96];
		snprintf( time , 96 ,"%02d-%02d-%02d",p_sys_tm->tm_hour , p_sys_tm->tm_min,p_sys_tm->tm_sec);
		std::string ret  = time ; 
		return ret;
	}
	int GetPathNameByPid(const pid_t pid, std::string &sName)
	{
		FILE * fPtr;
		char cmd[255] = {'\0'};
		char name[256];
		sprintf(cmd , "readlink /proc/%d/exe" , pid);
		if((fPtr = popen(cmd , "r")) != NULL)
		{
			fscanf(fPtr,"%s",name);
			sName = name;
			return 0;
		}
		else 
		{
			printf("ERR:%s GetPathNameByPid fail! popen fail!\n",__func__);
			return -1;
		}
	}				

	int LogInit(const char mo[])
	{
		return LogInit(mo,LOG_LEVEL);
	}
	int LogInit(const char mo[], int Level )
	{
		LOG_LEVEL = Level;
		pthread_mutex_init(&LOG_MUTEX,NULL);
		mkdir("log",S_IRWXU);
		if( GetPathNameByPid(getpid(),THE_PROCCESS_CMD) != 0)
		{
			printf("ERR:%s GetPathNameByPid fail!\n",__func__ );
			return -1;
		}
		std::string sFileName = GetDate();
		THE_ONE_LOG_TIME_HEAD = sFileName;
		sFileName ="log/"+sFileName+ ".log";

		THE_ONE_LOG_FILE = fopen(sFileName.c_str() , mo);
		if(THE_ONE_LOG_FILE == NULL )
		{
			printf("ERR:%s LogInit fail!!\n",__func__);
			return -1;
		}
		else return 0;
	}
	int LogInfo(const char *format , ...)
	{
		pthread_mutex_lock( &LOG_MUTEX );
		if(!IS_INIT)
		{
			LogInit("a");
			IS_INIT = 1;
		}
		if(LOG_LEVEL < 2)
		{
			pthread_mutex_unlock(&LOG_MUTEX);
			return 0;
		}
		va_list valist;
		va_start(valist , format);
		char buff[4096];
		vsprintf(buff,format,valist);
		std::string date = GetDate();
		if(date != THE_ONE_LOG_TIME_HEAD)
		{
			fflush( THE_ONE_LOG_FILE);
			fclose( THE_ONE_LOG_FILE);
			std::string sLogFileName = "log/"+date + ".log";
			THE_ONE_LOG_FILE = fopen(sLogFileName.c_str() , "a");
			THE_ONE_LOG_TIME_HEAD = date;
			if(THE_ONE_LOG_FILE == NULL )
			{
				printf("ERR:%s open log file fail!\n",__func__);
				pthread_mutex_unlock(&LOG_MUTEX);
				return -1;
			}
		}
		char tmp[4096];
		snprintf(tmp ,4096,"<%s(%u:%u)>INFO:\t%s\t%s\n",THE_PROCCESS_CMD.c_str() ,(unsigned int)getpid(),(unsigned int)pthread_self(),GetTime().c_str(),buff );
		fprintf(THE_ONE_LOG_FILE,"%s",tmp);
		fflush(THE_ONE_LOG_FILE);
		pthread_mutex_unlock(&LOG_MUTEX);
		return  0;
	}
	int LogErr(const char *format , ...)
	{
		pthread_mutex_lock( &LOG_MUTEX );
		if(!IS_INIT)
		{
			LogInit("a");
			IS_INIT = 1;
		}
		if(LOG_LEVEL < 1)
		{
			pthread_mutex_unlock(&LOG_MUTEX);
			return 0;
		}
		va_list valist;
		va_start(valist , format);
		char buff[4096];
		vsprintf(buff,format,valist);
		std::string date = GetDate();
		if(date != THE_ONE_LOG_TIME_HEAD)
		{
			fflush( THE_ONE_LOG_FILE);
			fclose( THE_ONE_LOG_FILE);
			std::string sLogFileName = "log/"+date + ".log";
			THE_ONE_LOG_FILE = fopen(sLogFileName.c_str() , "a");
			THE_ONE_LOG_TIME_HEAD = date;
			if(THE_ONE_LOG_FILE == NULL )
			{
				printf("ERR:%s open log file fail!\n",__func__);
				pthread_mutex_unlock(&LOG_MUTEX);
				return -1;
			}
		}
		char tmp[4096];
		snprintf(tmp ,4096,"<%s(%u:%u)>ERR:\t%s\t%s\n",THE_PROCCESS_CMD.c_str(),(unsigned int)getpid(),(unsigned int)pthread_self(),GetTime().c_str(),buff );
		fprintf(THE_ONE_LOG_FILE,"%s",tmp);
		fflush(THE_ONE_LOG_FILE);
		pthread_mutex_unlock(&LOG_MUTEX);
		return  0;
	}
	int LogDebug(const char *format , ...)
	{
		pthread_mutex_lock( &LOG_MUTEX );
		if(!IS_INIT)
		{
			LogInit("a");
			IS_INIT = 1;
		}
		if(LOG_LEVEL < 3)
		{
			pthread_mutex_unlock(&LOG_MUTEX);
			return 0;
		}
		va_list valist;
		va_start(valist , format);
		char buff[4096];
		vsprintf(buff,format,valist);
		std::string date = GetDate();
		if(date != THE_ONE_LOG_TIME_HEAD)
		{
			fflush( THE_ONE_LOG_FILE);
			fclose( THE_ONE_LOG_FILE);
			std::string sLogFileName = "log/"+date + ".log";
			THE_ONE_LOG_FILE = fopen(sLogFileName.c_str() , "a");
			THE_ONE_LOG_TIME_HEAD = date;
			if(THE_ONE_LOG_FILE == NULL )
			{
				printf("ERR:%s open log file fail!\n",__func__);
				pthread_mutex_unlock(&LOG_MUTEX);
				return -1;
			}
		}
		char tmp[4096];
		snprintf(tmp ,4096,"<%s(%u:%u)>Debug:\t%s\t%s\n",THE_PROCCESS_CMD.c_str() ,(unsigned int)getpid(),(unsigned int)pthread_self(),GetTime().c_str(),buff );
		fprintf(THE_ONE_LOG_FILE,"%s",tmp);
		fflush(THE_ONE_LOG_FILE);
		pthread_mutex_unlock(&LOG_MUTEX);
		return  0;
	}
}


