#include<fstream>
#include"sstream"
#include"unity.h"
#include"Log.h"
namespace Comm
{
	
	ThreadLock::ThreadLock()
	{
		assert(0 == pthread_mutex_init(&m_mutex,NULL));
	}

	ThreadLock::~ThreadLock()
	{
		assert(0 == pthread_mutex_destroy(&m_mutex));
	}

	int ThreadLock::Lock()
	{
		int ret = pthread_mutex_lock(&m_mutex);
		return ret;
	}

	int ThreadLock::UnLock()
	{
		return pthread_mutex_unlock(&m_mutex);
	}

	ThreadLockGuard::ThreadLockGuard(ThreadLock * plock)
	{
		m_plock = plock;
		m_islock = false;
	}

	ThreadLockGuard::~ThreadLockGuard()
	{
		UnLock();
	}

	int ThreadLockGuard::Lock()
	{
		int ret = m_plock->Lock();
		if(0 == ret)
		{
			m_islock = true;
		}
		return ret;
	}

	int ThreadLockGuard::UnLock()
	{
		int ret = 0;
		if(m_islock)
		{
			ret = m_plock->UnLock();
			if(0 == ret) m_islock = false;
		}
		return ret;
	}
	int stringHelper::split(const char *str,const char *spset,std::vector<std::string>&RetSet)
	{
		int len=strlen(str);
		char *tmp=new char[len+5];
		int i=0;
		int j=0;
		int cnt=0;
		while(i<len)
		{
			while(i<len&&isInSpset(str[i],spset))i++;
			while(i<len&&!isInSpset(str[i],spset))
			{
				tmp[j]=str[i];
				j++;
				i++;
			}
			if(j!=0)
			{
				tmp[j]='\0';
				std::string buf=tmp;
				RetSet.push_back(buf);
				cnt++;
				j=0;
			}
		}

		delete[] tmp;
		return cnt;
	}
	bool stringHelper::isInSpset(char c,const char *spset)
	{
		if(spset==NULL)return false;
		for(int i=0;spset[i];i++)
		{
			if(c==spset[i])return true;
		}
		return false;
	}

	Config::Config(){}
	Config::~Config(){}
	int Config::makePool(const char *configFile)
	{
		std::ifstream fpconfig(configFile);
		if( !fpconfig )
		{
			LogErr("Config::makePool %s not exit!",configFile);
			return -1;
		}
		pool.clear();
		std::string line;
		while( getline(fpconfig , line) != NULL )
		{
			if('#'==line[0])continue;
			std::vector<std::string> col;
			stringHelper::split(line.c_str()," =\r\n",col);
			if(col.size() < 2)
			{
				continue;
			}
			pool[col[0]]=col[1];
		}
		fpconfig.close();
		return 0;
	}

	std::string Config::ToString()
	{
		std::ostringstream oss;
		for(std::map<std::string,std::string>::iterator i = pool.begin(); i != pool.end();i++)
		{
			oss<<(*i).first<<"="<<(*i).second<<"\n";
		}
		return oss.str();
	}



}
