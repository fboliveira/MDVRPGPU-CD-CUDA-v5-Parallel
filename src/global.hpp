/* 
 * File:   global.h
 * Author: fernando
 *
 * Created on May 15, 2013, 9:30 PM
 */

#include <vector>
#include <list>
#include <iterator>
#include <string>
#include <time.h>

using namespace std;

#ifndef GLOBAL_H

#define	GLOBAL_H

// INSTANCE TEST
#define INST_TEST "pfbo02"

#define SOURCE 3

#if SOURCE==1
// LOCAL
#define BASE_DIR_DAT "/Users/fernando/Temp/MDVRP/dat/" 
#define BASE_DIR_SOL "/Users/fernando/Temp/MDVRP/sol/" 
//#define LOG_RUN_FILE "/Users/fernando/Temp/MDVRP/experiments/mdvrpcpu-teste.txt"
#define LOG_RUN_FILE "/Users/fernando/Temp/MDVRP/mdvrpcpu-20.txt"

#elif SOURCE==2

// UFMG
#define BASE_DIR_DAT "/home/fernando/experiments/instances/dat/" 
#define BASE_DIR_SOL "/home/fernando/experiments/instances/sol/" 
#define LOG_RUN_FILE "/home/fernando/experiments/mdvrpcpu-97-01-01.txt"

#else

// HUGO
#define BASE_DIR_DAT "/home/fernando/Temp/MDVRP/dat/"
#define BASE_DIR_SOL "/home/fernando/Temp/MDVRP/sol/"
#define LOG_RUN_FILE "/home/fernando/Temp/MDVRP/mdvrpGPU-hugo-01-01.txt"

#endif

// CIRRELT LAB
//#define BASE_DIR_DAT "/home/berfer/Experiments/instances/dat/" 
//#define BASE_DIR_SOL "/home/berfer/Experiments/instances/sol/" 
//#define LOG_RUN_FILE "/home/berfer/Experiments/experiments/mdvrpcpu-56.txt"

// AWS
//#define BASE_DIR_DAT "/home/ubuntu/experiments/instances/dat/" 
//#define BASE_DIR_SOL "/home/ubuntu/experiments/instances/sol/" 
//#define LOG_RUN_FILE "/home/ubuntu/experiments/mdvrpcpu-13.txt"

#define DEPOT_DELIM -1
#define ROUTE_DELIM -2

#define MIN_ELEM_IND 5

#define NUM_MAX_DEP 9
#define NUM_MAX_CLI 360

//#define MANAGED 0

enum class Enum_Algorithms {
    SSGA,
    SSGPU,
    ES
};

enum Enum_StopCriteria
{
    NUM_GER,
    TEMPO
};

enum Enum_Process_Type
{
    MONO_THREAD,
    MULTI_THREAD
};

enum Enum_Local_Search_Type
{
    RANDOM,
    SEQUENTIAL,
    NOT_APPLIED            
};

template<class T>
using typedef_vectorMatrix = vector<vector<T>>;

using typedef_vectorIntIterator = vector<int>::iterator;
using typedef_vectorIntSize = vector<int>::size_type;
using typedef_listIntIterator = list<int>::iterator;

typedef struct {
    int i; // id, indice, ...
    int x;
    int y;
} typedef_point;

typedef struct {
    int index;
    float cost;
} typedef_order;

typedef struct {
    double time;
    float cost;
} typedef_evolution;

typedef struct {
    double time;
    string text;
} typedef_log;

typedef struct {
    int depot;
    typedef_vectorIntIterator position;
    float cost;
} typedef_location;

#endif	/* GLOBAL_H */
