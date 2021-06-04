#define DEBUG
//#define HDEBUG

#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <utility>
#include <ctime>
#include <unistd.h>
#include <sys/time.h>
#include <random>
#include "mmio.h"
//#include "patoh.h"
#include <assert.h>
//#include "gpu_partition.cu"
#include <omp.h>

using namespace std;

extern void gpu_part(int m, int* ord, int* ptrs, int* js, int* pv, int no_parts, double& runtime, int* xpins, int* pins, int n);

char filename[2048] = {'\0'};  
int no_parts = -1, model = -1, seed = 11;
double  buffer_mul = 0;
double cut = -1, algo_param_1 = -1, algo_param_2 = -1, algo_param_3 = -1, algo_param_4 = -1, algo_param_5 = -1;
double imbal = -1;

int parse_args(int argc, char *argv[]){
  int opt;
  if(argc == 1) {
    cout << "usage: \n";
    cout << "\t-f input filename in mtx format - req\n";		
    cout << "\t-m hypergraph model: - req\n";    
    cout << "\t\t0 col-net\n";    
    cout << "\t\t1 row-net\n";    
    cout << "\t-k number of parts - req\n";    
    cout << "\t-i imbalance - req\n";
    cout << "\t-x algo_param_1\n";
    cout << "\t-y algo_param_2\n";
    cout << "\t-z algo_param_3\n";
    cout << "\t\tfor patoh: x = metric(con = 1, cut = 2), y = speed(def = 0, speed = 1, quality = 2)\n";
    cout << "\t\tfor random\n";
    cout << "\t\tfor lsh x = no_hash\n";
    cout << "\t\tfor mmgood/mmbad: x = penalty(none = 0 (default), classic = 1)\n";
    cout << "\t\tfor mmrefine: x = penalty(none = 0 (default), classic = 1), y = no passes over buffer\n";
    cout << "\t\tfor nk x = penalty(none = 0 (default), classic = 1), y = memory for net\n";
    cout << "\t\tfor bloom x = penalty(none = 0, classic 1), y = bitsize in K, z = noBFhash\n";
    cout << "\t -b buffer_mul: 0 for not buffer (default), otherwise (buffer x nnz) will be the buffer size (i.e., %10 of nnz for 0.1)\n";
    cout << "\t\tonly works for mmgood, mmrefine and nk" << endl;
    cout << "\t-s random seed\n";
    cout << "gpu set interaction";
    exit(1);
  }

  bool fname = false;  
  while((opt = getopt(argc, argv, "f:m:k:i:x:y:z:w:b:s:g:")) != -1){
    switch(opt){
    case 'f':
      strcpy(filename,optarg);
      fname = true;
      break;
    case 'i':
      imbal = atof(optarg);
      if(!(imbal > 0 && imbal < 1)) {
	cout << "invalid imbal" << endl;
	return -1;
      }
      break;
    case 'm':
      model = atoi(optarg);
      if(!(model >= 0 && model <= 1)) {
	cout << "invalid model" << endl;
	return -1;
      }
      break;
    case 'k':
      no_parts = atoi(optarg);
      if(no_parts < 2) {
	cout << "invalid no_parts" << endl;
	return -1;
      }
      break;
    case 'b':
      buffer_mul = atof(optarg);
      if(buffer_mul < 0) {
	cout << "invalid buffer" << endl;
	return -1;
      }
      break;
    case 'x':
      algo_param_1 = atof(optarg);
      break;
    case 'y':
      algo_param_2 = atof(optarg);
      break;
    case 'z':
      algo_param_3 = atof(optarg);
      break;
    case 'g':
      algo_param_5 = atof(optarg);
      break;
    case 'w':
      algo_param_4 = atof(optarg);
    case 's':
      seed = atoi(optarg);
      break;
    default:
      cout << "unknown parameter" << endl;
      return -1;
    }
  }

  cout << "hfile: " << filename << endl;
  cout << "model: " << model << endl;
  cout << "no parts: " << no_parts << endl;
  cout << "imbal: " << imbal << endl;      
  cout << "buffer_mul: " << buffer_mul << endl;
  cout << "seed: " << seed << endl;
  cout << "params: " << algo_param_1 << ", " << algo_param_2 << ", " << algo_param_3 << " " << algo_param_4 << "\n";


  if(imbal == -1 || no_parts == -1 || model == -1 || fname == false) {
    cout << "missing required parameter" << endl;
    return -1;
  }
  return 1;
}

int computeCut(int* pv, int no_parts, int n, int* xpins, int* pins) {
  int cut = 0; 
  int* mark = new int[no_parts];  for(int i = 0; i < no_parts; i++) mark[i] = -1;
  for(int i = 0; i < n; i++) {
    int con = 0;
    for(int j = xpins[i]; j < xpins[i+1]; j++) {
      int cell = pins[j];
      if(pv[cell] != -1) {
  	if(mark[pv[cell]] != i) {
  	  mark[pv[cell]] = i;
   	  con++;
     	}
      }
    }
    if(con != 0 && xpins[i] != xpins[i+1]) cut += (con - 1);
  }	
  delete [] mark;
  return cut;
}

bool sortpair(const pair<int,int> &a, const pair<int,int> &b) { 
  if(a.first != b.first) {
    return (a.first < b.first); 
  }
  return (a.second < b.second); 
} 

const char *get_filename_ext(const char *filename) {
  const char *dot = strrchr(filename, '.');
  if(!dot || dot == filename) return "";
  return dot + 1;
}

const char *get_filename(const char *path) {
  const char *fnm = strrchr(path, '/');
  if(!fnm || fnm == path) return path;
  return fnm + 1;
}

int read_hypergraph(int& m, int& n, int nz, int*& ptrs, int*& js) {
  FILE *f;
  char binaryfilename[4096];
  sprintf(binaryfilename, "./%s.bin", get_filename(filename)); 
  f = fopen(binaryfilename, "rb");
  
  if(f == NULL) {
    cout << "Reading from matrix market" << endl;
    if((f = fopen(filename, "r")) == NULL) {
      cout << "Invalid file\n";
      return -1;
    }
    
    MM_typecode matcode;  
    if (mm_read_banner(f, &matcode) != 0) {
      cout << "Could not process Matrix Market banner\n" << endl;
    return -1;
    }
    
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) ) {
      cout << "Sorry, this application does not support\n";
      cout << "Market Market type: " << mm_typecode_to_str(matcode) << endl;
      return -1;
    }
    
    int tnz;
    if (mm_read_mtx_crd_size(f, &m, &n, &tnz) != 0 ) {
      cout << "Matrix size could not be read" << endl;
      return -1;
    }
    

    pair<int, int>* crds;
    if(mm_is_symmetric(matcode)) {
      crds = new pair<int, int>[2 * tnz];
    } else {
      crds = new pair<int, int>[tnz];
    }
    
    nz = 0;
    double val;
    for (int i = 0; i < tnz; i++) {
      if(mm_is_pattern(matcode)) {
	fscanf(f, "%d %d\n", &crds[nz].first, &crds[nz].second);      
      } else {
	fscanf(f, "%d %d %lf\n", &crds[nz].first, &crds[nz].second, &val);      
      }
      
      crds[nz].first--;  /* adjust from 1-based to 0-based */
      crds[nz].second--;
      nz++;    
      
      if(mm_is_symmetric(matcode)) {
	if(crds[nz-1].first != crds[nz-1].second) {			
	  crds[nz].second = crds[nz-1].first;
	  crds[nz].first = crds[nz-1].second;
	  nz++;      
	}		
      }      
    }
    fclose(f);    
    cout << "file read: " << m << " " <<  n << " " << nz << " (from " << tnz << " lines)" << "\n";    

    //remove duplicates if any
    sort(crds, crds + nz,  sortpair); 
    int prev = 0;
    for(int i = 1; i < nz; i++) {
      if(crds[i].first != crds[prev].first || crds[i].second != crds[prev].second) {
	prev++;
	crds[prev].first = crds[i].first;
	crds[prev].second = crds[i].second;
      }
    }
    nz = prev + 1;
    cout << "dup elim: " << m << " " <<  n << " " << nz << "\n";

    mm_write_banner(stdout, matcode);
    fflush(0);
    
    if(model == 0) {//row-net model, columns are vertices so transpose
      for(int i = 0; i < nz; i++) {
	int temp = crds[i].first;
	crds[i].first = crds[i].second;
	crds[i].second = temp;
      }
      
      int temp = m;
      m = n;
      n = temp;
    }
    sort(crds, crds + nz,  sortpair); 
    
    js = new int[nz];
    ptrs = new int[m+1];
    memset(ptrs, 0, sizeof(int) * (m + 1));
    for(int i = 0; i < nz; i++) {
      ptrs[crds[i].first + 1]++;
      js[i] = crds[i].second;
    }
    for(int i = 1; i <= m; i++) {
      ptrs[i] += ptrs[i-1];
    }
    if(ptrs[m] != nz) cout << "Problem in ptrs - js" << endl; 

    //write in binary
    cout << "writing in binary to " << binaryfilename << endl;
    FILE* bf = fopen(binaryfilename, "wb");
    fwrite(&m, sizeof(int), 1, bf);
    fwrite(&n, sizeof(int), 1, bf);
    fwrite(ptrs, sizeof(int), m+1, bf);
    fwrite(js, sizeof(int), nz, bf);
    fclose(bf);
  } else {
    cout << "Reading from binary file " << binaryfilename << endl;
    fread(&m, sizeof(int), 1, f); 
    fread(&n, sizeof(int), 1, f);

    ptrs = new int[m+1];
    fread(ptrs, sizeof(int), m+1, f);

    nz = ptrs[m];
    js = new int[nz];
    fread(js, sizeof(int), nz, f);
    fclose(f);
  }

  cout << "----------------------------------" << endl;
  cout << m << " " << n << " " << nz << endl;
  float avdeg = (1.0f * nz) / m;
  int maxdeg = 0;
  int mindeg = m;
  int zerodeg = 0;
  float sum = 0;
  int deg;

  for(int i = 0; i < m; i++) {
    deg = ptrs[i+1] - ptrs[i];
    if(deg == 0) zerodeg++;   
    maxdeg = max(maxdeg, deg);
    mindeg = min(mindeg, deg);
    sum += (deg - avdeg) * (deg - avdeg);
  }
  cout << "#0 deg: " <<  zerodeg << endl;
  cout << "Deg. min " <<  mindeg << endl;
  cout << "Deg. max " <<  maxdeg << endl;
  cout << "Deg. avg " <<  avdeg << endl;
  cout << "Deg. variance " << sqrt(sum/m) << endl;
  cout << "----------------------------------" << endl;

  return 1;
}



double penfunc(int part, int min_part, int* pw, int slack, int pen_type) {
  if(pen_type == 0) return 1.0f;
  else if(pen_type == 1) return (1 - ((pw[part] - pw[min_part] + .0f) / (2.0f * slack)));
  else {
    cout << "unknown pen_type" << endl;
    exit(1);
  }
  return -1;
}

int streaming(int m, int* ord, int* ptrs, int* js, int* pv, int no_parts, double& runtime, int* xpins, int* pins, int n) {
  std::random_device rd;     
  std::mt19937 gen(seed*7654321+54321);    
  int* pw = new int[no_parts]; memset(pw, 0, sizeof(int) * no_parts);
  int min_part = 0;
  int slack = 1;
  for(int i = 0; i < m;i ++) pv[i] = -1;
  //PRE PARTITIONING ----------------------------------------
  std::uniform_int_distribution<> pdistrib(0, no_parts-1);
  
  int buffer = buffer_mul * ptrs[m];
  bool from_buffer = false;
  int buffer_size = 0;
  int part_start_index = 0;
  int refine_start_index = 0;
  bool last_fix = false;

  int *part_mark = new int[no_parts]; for(int i = 0; i < no_parts; i++) part_mark[i] = -1;
  int marker = 0;
  int *part_index = new int[no_parts];
  int *part_id = new int[no_parts];
  int *saved = new int[no_parts];
  int pen_type = (int)algo_param_1;
  vector<vector<int> > n2p; 
  n2p.reserve(100000);
  //------------------------------------------------------------------------------------  
  clock_t start, end;
  start = clock();
  int i = 0; 
  for(; i < m; i++) {
    int vtx = ord[i];

    if(((1.5 * imbal * i) / no_parts) > slack) {
      slack = (1.5 * imbal * i) / no_parts;
    }

    //PARTITIONING ---------------------------------------------------------------------  
    if(pv[vtx] != -1) continue;
    
    if((buffer == 0) || !from_buffer) {
      int maxvn = 0; for(int p = ptrs[vtx]; p < ptrs[vtx + 1]; p++) {if(maxvn < js[p]) {maxvn = js[p];}}   
      if((uint)maxvn >= n2p.size()) {n2p.resize(maxvn + 1);}
    }
  
    int part_cnt = 0;
    for(int p = ptrs[vtx]; p < ptrs[vtx + 1]; p++) {
      int nx = js[p];
      for(const int& k : n2p[nx]) {
	if(pw[k] - pw[min_part] < slack) {
	  if(part_mark[k] != marker) {
	    part_mark[k] = marker; saved[part_cnt] = 1; part_id[part_cnt] = k; part_index[k] = part_cnt++;
	  } else {
	    saved[part_index[k]]++;
	  }
	}
      }
    }
    marker++;
        
    int best_part;
    double best_cost;
    if(part_cnt == 0) {
      int k = pdistrib(gen);
      for(; k < no_parts; k++) {if(pw[k] - pw[min_part] < slack) {best_part = k; best_cost = 0; break;}}
      if(k == no_parts) {
	for(k = 0; k < no_parts; k++) {if(pw[k] - pw[min_part] < slack) {best_part = k; best_cost = 0; break;}}
      }
    } else {
      best_part = part_id[0];
      best_cost = saved[0];// * penfunc(0, min_part, pw, slack, pen_type); 
      for(int k = 1; k < part_cnt; k++) {
	int part = part_id[k];
	double cost = saved[k];// * penfunc(part, min_part, pw, slack, pen_type); 
	if((cost > best_cost) || (cost == best_cost && pw[best_part] > pw[part])) {
	  best_part = part; best_cost = cost;
	}
      }
    }

    if(buffer > 0 && !last_fix) {
      if(!from_buffer) {
	if(i == m-1) last_fix = true; 
	//double ratio = ((best_cost + 0.0f) / penfunc(best_part, min_part, pw, slack, pen_type)) / (ptrs[vtx + 1] -  ptrs[vtx]);
	double ratio = (best_cost + 0.0f) / (ptrs[vtx + 1] -  ptrs[vtx]);
	if((i > 1000 && ratio < 0.3) || (i == (m - 1))) {
	  buffer_size += ptrs[vtx + 1] -  ptrs[vtx];
	  if(buffer_size > buffer || (i == (m - 1))) {
	    from_buffer = true;
	    int temp = part_start_index-1;	 
	    part_start_index = i+1;
	    i = temp;
	  }
	  continue;
	} 
      } else {
	if(i == part_start_index - 1) {
	  from_buffer = false;
	  buffer_size = 0;
	}
      }
    }
  
    for(int p = ptrs[vtx]; p < ptrs[vtx + 1]; p++) {
      int nx = js[p];
      bool found = false;
      for(const int& k : n2p[nx]) {
	if(k == best_part) {
	  found = true; 
	  break;
	}
      }
      if(!found) {
	n2p[nx].push_back(best_part);
      }
    }
    //-----------------------------------------------------------------------------------

    //POST PARTITIONING -----------------------------------------------------------------  
    //cout << i << " " << vtx << " " << best_part << " " << fromBuffer << " (" << bufferSize << " " << buffer << ")" << endl;
    //    if(vtx == 738086) cout << i << " " << ord[i] << " " << vtx << " " << best_part << "--- " << is_refining << endl;
    pv[vtx] = best_part;
    pw[best_part]++;
    if(best_part == min_part) {min_part = 0; for(int k = 1; k < no_parts; k++) {if(pw[min_part] > pw[k]) {min_part = k;}}}
  
    
    /*#ifdef MM_REFINE 
    if(is_refining) {
      if(best_part != current_part) {
    	int newCut = computeCut(pv, no_parts, n, xpins, pins);	
	cout << "hola " << " " << ptrs[vtx+1] - ptrs[vtx] << " " << best_cost << " " << best_part << " " << endl;
	if(i != -1 && old_cut - newCut != move_gain) {
	  cout << "problem: " << i << ": part id of " << vtx << " changed: " << best_part << " " << current_part << " | " << old_cut << " " << newCut << " | " << move_gain << " " << leave_gain << " " << arrival_loss << endl;
	  pv[vtx] = -1;
	  cout << "when removed " << computeCut(pv, no_parts, n, xpins, pins) << endl;
	  pv[vtx] = current_part;
	  cout << "when put back to old part " << computeCut(pv, no_parts, n, xpins, pins) << endl;
	  exit(1);
      	}
      }
    }
    #endif*/
    //----------------------------------------------------------------------------------  
  }
  end = clock();
  runtime = ((double) (end - start)) / CLOCKS_PER_SEC;
  delete [] pw;

  bool err = false;
  for(int i = 0; i < m; i++) {
    if(pv[i] == -1) {
      err = true;
      cout << "vtx " << i << " is not partitioned " << endl;
      exit(1);
    }
  }
  if(err) exit(1);

  //CLEANING MEMORY --------------------------------------------------------------------  
  delete [] part_mark;
  delete [] part_id;
  delete [] saved;
  delete [] part_index;
  //---------------------------------------------------------

  return 1;
}


int main(int argc, char *argv[]) {
  cout << "reading parameters" << endl;     
  if(parse_args(argc, argv) == -1) {
    cout << "type executable name to learn more" << endl;
    return 0;
  }

  cout << "reading file " << endl;

  int m = -1, n = -1, nz = -1, *ptrs = nullptr, *js = nullptr;
  read_hypergraph(m, n, nz, ptrs, js);

  int* xpins = new int[n+1]; memset(xpins, 0, sizeof(int) * (n+1));
  int* pins = new int[ptrs[m]]; 
  
  for(int i = 0; i < ptrs[m]; i++) xpins[js[i] + 1]++;  
  for(int i = 1; i <= n; i++) xpins[i] += xpins[i-1];
  for(int i = 0; i < m; i++) {
    for(int j = ptrs[i]; j < ptrs[i+1]; j++) {
      pins[xpins[js[j]]++] = i;  
    }
  }
  for(int i = n; i > 0; i--) xpins[i] = xpins[i-1]; 
  xpins[0] = 0;

  //#define HDEBUG
  int* pv = new int[m]; for(int i = 0; i < m; i++) pv[i] = -1;
  double runtime = 0;
    int* ord = new int[m];
    for(int i = 0; i < m; i++) ord[i] = i;
    std::srand(seed*1234567+432101);
    std::random_shuffle (ord, ord + m);

    double startt = omp_get_wtime();
    
    if(algo_param_5 == -1)
      streaming(m, ord, ptrs, js, pv, no_parts, runtime, xpins, pins, n);
    else{
      gpu_part(m, ord, ptrs, js, pv, no_parts, runtime, xpins, pins, n);
    }

    double endt = omp_get_wtime();

    
    int* pw = new int[no_parts]; 
    memset(pw, 0, sizeof(int) * no_parts);
    for(int i = 0; i < m; i++) {
      pw[pv[i]]++;
    }
    
  double avgwght = (m + 0.0f) / no_parts;
  int maxpw = 0, minpw = m;
  for(int i = 0; i < no_parts; i++) {
    if(pw[i] > maxpw) maxpw = pw[i];
    if(pw[i] < minpw) minpw = pw[i];
  }

  double a_imbal = 0;
  for(int i = 0; i < no_parts; i++) {
    a_imbal = max(a_imbal, pw[i]/avgwght);
  }
  a_imbal -= 1;
  //delete [] pw;

  cout << "Will compute cut: " << std::endl;
  int cut = computeCut(pv, no_parts, n, xpins, pins);

  cout << "ttt: " << filename << " " << m << " " << n << " " << nz << " | " << no_parts << " " << imbal << " | " << 
    "mm-good"
       << " " << algo_param_1 << " " << algo_param_2 << " " << algo_param_3 << " " << algo_param_4 << 
    " | " << endt-startt << " " << a_imbal << " " << maxpw - minpw  << " " << cut << endl;
  
  delete [] ptrs;
  delete [] js;
  delete [] xpins;
  delete [] pins;
  delete [] pv;
}

