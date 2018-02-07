
#include "SAX.h"
#include <cuda.h>

//#define THREADS 250
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


uchar* d_collisionTable;
uchar* d_cols;
CudaWord* d_words;


void startEvent(cudaEvent_t &start, cudaEvent_t &stop){
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
}

float endEvent(cudaEvent_t &start, cudaEvent_t &stop){
	float elapsedTime;
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );    
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
	return elapsedTime;
}

void devInit(){
	int size0 = NUM_WORDS * NUM_WORDS * sizeof(uchar);
	int size1 = MASK_SIZE * ITERATIONS * sizeof(uchar);
	int size2 = NUM_WORDS * sizeof(CudaWord);

	clock_t start, end;
	start = clock();
	cudaMalloc( &d_collisionTable, size0 );
	cudaMalloc( &d_cols, size1 );
	cudaMalloc( &d_words, size2 );	
	end = clock();
	printf("\tExec time cudaMallocs: %d ms\n", end - start );
	
	start = clock();
	cudaMemset( d_collisionTable, 0,  size0 );
	end = clock();
	printf("\tExec time cudaMemset: %d ms\n", end - start );

	start = clock();
	cudaMemcpy( d_words, h_words, size2, cudaMemcpyHostToDevice );
	end = clock();
	printf("\tExec time cudaMemcpy: %d ms\n", end - start );
}


void devFree() {
	cudaFree( d_collisionTable );
	cudaFree( d_cols );
	cudaFree( d_words );
}


__global__
void cudaRP_1D_OneIteration(uchar* collisionTable, CudaWord* words, uchar* cols){
	uint index_i = blockIdx.x * blockDim.x + threadIdx.x;	

	if( index_i >= NUM_WORDS ) return;

	uint index_j = blockIdx.y * blockDim.y;

	__shared__ CudaWord _words [ THREADS ];
	__shared__ CudaWord _maskedWords [ THREADS ];

	_maskedWords [ threadIdx.x ] = words [ index_i ];
	_words [ threadIdx.x ] = words [ index_j + threadIdx.x ];

	__syncthreads();	
	
	bool flag;
	for( uint i = 0; i < blockDim.x; i++ )
	{
		if ( index_i != index_j && index_j < NUM_WORDS ) 
		{
			flag = true;
			for (uchar j = 0; j < MASK_SIZE; j++) {
				if ( _words[ threadIdx.x ].values[ cols[j] ] != _maskedWords[ i ].values[ cols[j] ] ) {
					flag = false;
					break;
				}
			}

			if ( flag ) {
				collisionTable[index_j * NUM_WORDS + index_i] += 1;
			}
		}
		index_j++;
	}
}


__global__
void cudaRP_1D_AllIterations(uchar* collisionTable, CudaWord* words, uchar* cols){
	uint index_i = blockIdx.x * blockDim.x + threadIdx.x;	

	if( index_i >= NUM_WORDS ) return;

	uint index_j = blockIdx.y * blockDim.y;

	__shared__ CudaWord _words [ THREADS ];
	__shared__ CudaWord _maskedWords [ THREADS ];

	_maskedWords [ threadIdx.x ] = words [ index_i ];
	_words [ threadIdx.x ] = words [ index_j + threadIdx.x ];

	__syncthreads();	
	
	bool flag;
	uchar count;
	for( uint i = 0; i < blockDim.x; i++ )
	{
		if ( index_i != index_j && index_j < NUM_WORDS ) 
		{
			count = 0;
			for( int iter = 0; iter < ITERATIONS; iter++ )
			{
				flag = true;
				for (uchar j = iter * MASK_SIZE; j < iter * MASK_SIZE + MASK_SIZE; j++) {
					if ( _words[ threadIdx.x ].values[ cols[j] ] != _maskedWords[ i ].values[ cols[j] ] ) {
						flag = false;
						break;
					}
				}

				if ( flag ) {
					count++;
				}
			}

			collisionTable[index_j * NUM_WORDS + index_i] = count;
		}
		index_j++;
	}
}


__global__
void cudaRP_2D_OneIteration(uchar* collisionTable, CudaWord* words, uchar* cols){
	uint index_i = blockIdx.x * blockDim.x + threadIdx.x;
	uint index_j = blockIdx.y * blockDim.y + threadIdx.y;

	if( index_i == index_j || index_i >= NUM_WORDS || index_j >= NUM_WORDS ) return;

	uchar count = 0;
	uint index = index_j * NUM_WORDS + index_i;
// 	CudaWord word_i = words[ index_i ];
// 	CudaWord word_j = words[ index_j ];

	bool flag = true;
	for (uchar c = 0; c < MASK_SIZE; c++) {
//		if ( word_i.values[ cols[c] ] != word_j.values[ cols[c] ] ) {
		if ( words[ index_j ].values[ cols[c] ] != words[ index_i ].values[ cols[c] ] ) {
			flag = false;
			break;
		}
	}

	if ( flag ) {
		collisionTable[index] += 1;
	}
}


__global__
void cudaRP_2D_AllIterations(uchar* collisionTable, CudaWord* words, uchar* cols){
	uint index_i = blockIdx.x * blockDim.x + threadIdx.x;
	uint index_j = blockIdx.y * blockDim.y + threadIdx.y;

	if( index_i == index_j || index_i >= NUM_WORDS || index_j >= NUM_WORDS ) return;

	uchar count = 0;
	uint index = index_j * NUM_WORDS + index_i;
// 	CudaWord word_i = words[ index_i ];
// 	CudaWord word_j = words[ index_j ];

	for( int iter = 0; iter < ITERATIONS; iter++ ){
		bool flag = true;
		for (uchar c = iter * MASK_SIZE; c < iter * MASK_SIZE + MASK_SIZE; c++) {
//			if ( word_i.values[ cols[c] ] != word_j.values[ cols[c] ] ) {
			if ( words[ index_j ].values[ cols[c] ] != words[ index_i ].values[ cols[c] ] ) {
				flag = false;
				break;
			}
		}

		if ( flag ) {
			count++;
		}
	}

	collisionTable[index] = count;
}


void devRandomProjection(){

	clock_t start, end;
	start = clock();
	for (int it = 0; it < ITERATIONS; it++) {
		std::set<int> cols_set;
		for (int i = 0; i < MASK_SIZE; i++) {
			int tentativeColumn = rand() % WORD_SIZE;
			if ( cols_set.find(tentativeColumn) == cols_set.end() )
				cols_set.insert(tentativeColumn);
			else
				i--;
		}
		std::copy( cols_set.begin(), cols_set.end (), &h_cols[ MASK_SIZE * it ] ); 

// 		printf("Mask columns (%d): [", it);
// 		for (int i = 0; i < cols_set.size(); i++) {
// 			printf("%d, ", h_cols[it * MASK_SIZE + i]);
// 		}
// 		printf("]\nWORD LIST SIZE: %d \n", cols_set.size());

		cudaMemcpy(d_cols, h_cols, MASK_SIZE * ITERATIONS, cudaMemcpyHostToDevice);
	}
	end = clock();
	printf("\tExec time cols generation: %d ms\n", (end - start));
	
	int b = (NUM_WORDS + THREADS - 1) / THREADS;
	dim3 blocks ( b , b );
	dim3 threads ( THREADS , THREADS );
	
	cudaEvent_t e_start, e_stop;
	startEvent(e_start, e_stop);
 	for (int i = 0; i < ITERATIONS; i++ ) {	
 		cudaRP_1D_OneIteration<<<blocks , THREADS >>>( d_collisionTable, d_words, &d_cols[i*MASK_SIZE] );
//		cudaRP_2D_OneIteration<<<blocks , threads >>>( d_collisionTable, d_words, &d_cols[i*MASK_SIZE] );
//		cudaRP_2D_AllIterations<<<blocks , threads >>>( d_collisionTable, d_words, d_cols );
//		cudaRP_1D_AllIterations<<<blocks , THREADS >>>( d_collisionTable, d_words, d_cols );
 	}
 	printf("\tExec time #%d cudaRP_1D_OneIteration: %lf ms\n", ITERATIONS, endEvent(e_start, e_stop));
//	printf("\tEXEC TIME cudaRP_2D_AllIterations: %3.1f ms\n", endEvent(e_start, e_stop));
	printf("\tConfiguration: <<< (%d,%d) , (%d,%d) >>>\n", b, b, THREADS, 1 );
	
	startEvent(e_start, e_stop);
	cudaMemcpy(h_collisionTable, d_collisionTable, NUM_WORDS * NUM_WORDS * sizeof(uchar), cudaMemcpyDeviceToHost);
	printf("\tExec time cudaMemcpy collision_table: %3.1f ms\n", endEvent(e_start, e_stop));
}


void show1MotifResult() 
{
	int bestMotifSoFar = 0;
	std::vector<int> bestMotifLocationSoFar;
	for (int i = 0; i < NUM_WORDS; i++) {
		int counter = 0;
		std::multimap<uchar, int, std::greater<uchar> > pointers;
		for(int j = 0; j < NUM_WORDS; j++) {
			int index = i * NUM_WORDS + j;

			uchar count = h_collisionTable[index];
			counter += count; 
			pointers.insert( std::make_pair(count, j) );
		}
		if( counter > bestMotifSoFar ) {
			bestMotifSoFar = counter;
			bestMotifLocationSoFar.clear();

			bestMotifLocationSoFar.push_back(i);
			std::multimap<uchar, int, std::greater<uchar>>::iterator iter = pointers.begin();
			for ( ; iter != pointers.end(); iter++) {
				if (iter->first > 0)
					bestMotifLocationSoFar.push_back(iter->second);
			}
		}
	}
	int topK = 25;
	printf("SIZE: %d\n", NUM_WORDS * NUM_WORDS);
	printf("1-MOTIF:\n ");
	for (int t = 0; t < min(topK, (int)bestMotifLocationSoFar.size()); t++) {
		printf("%d, ",  bestMotifLocationSoFar[t]);
	}
 	printf("\n");
}

void testCudaRandomProjection(){	
	clock_t start = clock();
	std::string train = DATASET;
	SAX::loadData(train);
	clock_t end = clock();
	printf("Exec time loadData(): %d ms\n", (end - start));
	
	start = clock();
	devInit();
	end = clock();
	printf("Exec time devInit(): %d ms\n", (end - start));

	start = clock();
	devRandomProjection();
	end = clock();
	printf("Exec time devRandomProjection(): %d ms\n", (end - start));

	devFree();

	start = clock();
	show1MotifResult();
	end = clock();
	printf("Exec time show1MotifResult(): %d ms\n", (end - start));
}


void main(){
	printf("\nWORDS = %d\n", NUM_WORDS);	
	clock_t begin = clock();
	testCudaRandomProjection();
	clock_t end = clock();
	printf("Total exec time was: %d ms\n\n", end - begin);
}