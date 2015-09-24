#pragma once
#include<sys/types.h>
#include<unistd.h>
#include<cstdlib>
#include<cstdio>
#include<string>
#include<iostream>
#include<time.h>
#include<stdarg.h>
#include<pthread.h>
namespace Comm
{
	std::string GetDate();
	std::string GetTime();
	int GetPathNameByPid(pid_t pid,std::string &sName);
	int LogInit(const char mo[]);
	int LogInit(const char mo[], int Level);
	int LogInfo(const char * format , ...);
	int LogErr(const char * format , ...);
	int LogDebug(const char * format , ...);
	
}
