#include <cuda.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include "/home/users/ftasyaran/thrust/thrust/universal_allocator.h"
#include "/home/users/ftasyaran/thrust/thrust/universal_vector.h"
#include <algorithm>
#include <cstdlib>


__host__ void print_two_sets(thrust::universal_vector<int> set1, thrust::universal_vector<int> set2){

  std::cout << "######SET 1######" <<std::endl; 
  for(int i = 0; i < set1.size(); i++){
    std::cout << set1[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "######SET 1######" <<std::endl;

  std::cout << "######SET 2######" <<std::endl; 
  for(int i = 0; i < set2.size(); i++){
    std::cout << set2[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "######SET 2######" <<std::endl; 
  
}

void gpu_part(int m, int* ord, int* ptrs, int* js, int* pv, int no_parts, double& runtime, int* xpins, int* pins, int n){

  thrust::universal_vector<thrust::universal_vector<int>> p2n(no_parts);
  //for(int i = 0; i < no_parts; i++){
  //p2n[i].resize(1000);
  //}
  
  //PRE-PARTITIONING
  //Put first no_parts vertices to each part to get a comparable set

  double slack = 0;
  int min_part = 0;
  int* pw = new int[no_parts];

  double imbal = 1;
  
  for(int i = 0; i < no_parts; i++){
    pw[i] = 1;
  }

  for(int i = 0; i < no_parts; i++){

    int vertex = ord[i]; //Choose a vertex, simulate streaming setting
    pv[vertex] = i;

    for(int p = ptrs[vertex]; p < ptrs[vertex + 1]; p++){
      int nx = js[p];
      p2n[i].push_back(nx);
    }
    
  }
  
  for(int i = 0; i < no_parts; i++){
    printf("Part %d : ", i);
    for(int j = 0; j < p2n[i].size(); j++){
      printf(" %d ", p2n[i][j]);
    }
    printf("\n");
  }
  
  thrust::discard_iterator<> C_begin, C_end;
  thrust::universal_vector<int> nets;

  double pend1, pstart1, pend2, pstart2, vtotal;

  
  int my_part = 0;
  for(int v = no_parts; v < m; v++){//m
    int vertex = ord[v];
    int my_max = 0;
    
    if(((1.5 * imbal * v) / no_parts) > slack) {
      slack = (1.5 * imbal * v) / no_parts ;
    }
    
    printf("Vertex: %d \n", v);
    
    for(int p = ptrs[vertex]; p < ptrs[vertex + 1]; p++){
      nets.push_back(js[p]);
    } //This could be a pre-processing step
    
    for(int part = 0; part < no_parts; part++){
      
      C_end = thrust::set_intersection(p2n[part].begin(), p2n[part].end(), nets.begin(), nets.end(), C_begin);
      
      int size_intersect = C_end - C_begin;
      
      if(size_intersect > my_max && (pw[part] - pw[min_part] < slack)){
	my_part = part;
      }

      if(size_intersect > my_max)
	my_max = part;

      std::cout << "my_part: " << my_part << " part: " << part <<" slack: " << slack << " size_intersect: " << size_intersect <<std::endl
		<< "pw[part]: " << pw[part] << " my_max: " << my_max << std::endl;;

      print_two_sets(p2n[part], nets);
      
    }
    
    pv[vertex] = my_part;
    pw[my_part] += 1;
    
    if(my_part == min_part){
      for(int k = 0; k < no_parts; k++) {
	if(pw[min_part] > pw[k]) {
	  min_part = k;
	}
      }
    }
    
    for(int p = 0; p < nets.size(); p++){
      p2n[my_part].push_back(nets[p]);
    }
    
    nets.resize(0);
    
    for(int i = 0; i < no_parts; i++){
      thrust::sort(p2n[i].begin(), p2n[i].end());
    }
    
  }

  for(int i = 0; i < 100; i++){
    std::cout << "pv[i]: " << pv[i] << std::endl;
  }
  
}

 

