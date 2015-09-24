// =====================================================================================
// 
// SWEQWE
//       Filename:  runnable.h
// 
//    Description:  
// 
//        Version:  1.0
//        Created:  04/16/2014 11:23:36 AM
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  YOUR NAME (), 
//        Company:  
// 
// =====================================================================================

#pragma once
namespace Comm{
    class Runnable{
        public:
            virtual ~Runnable(){}
            virtual int Run()=0;
    };
}
