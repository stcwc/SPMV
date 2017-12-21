
/*
//Since it must be a undirected graph, we ONLY use half of it (x > y part).
*/
typedef struct {
    int M;
    int * adj;
} Graph;

Graph * initGraph(int M){
    Graph * graph = (Graph*)malloc(sizeof(Graph));
    graph->M = M;
    int * adj = (int*)malloc(M*M*sizeof(int));
    memset(adj, 0, M*M*sizeof(int));
    graph->adj = adj;
    return graph;
}

int getWeight(Graph * graph, int x, int y){
    int M = graph->M;
    int temp = graph->adj[x*M+y];
    if(temp<0)
        printf("ERROR - (%d,%d)=%d.\n", x, y, graph->adj[x*M+y]);
    return temp;
}

void addEdge(Graph * graph, int x, int y){
    int M = graph->M;
    //printf("Before addEdge : (%d, %d) = %d. (%d, %d) = %d.\n", x, y ,graph->adj[x*M+y], y, x, graph->adj[y*M+x]);
    if(x==y)
        printf("ERROR1 when adding an edge (x cannot == y).\n");
    if((graph->adj[x*M+y])<0 || (graph->adj[y*M+x])<0){
        printf("ERROR2 when adding - graph(%d,%d)=%d. graph(%d,%d)=%d.\n", x, y, graph->adj[x*M+y], y, x, graph->adj[y*M+x]);
        exit(0);
    }
    (graph->adj[x*M+y])++;
    (graph->adj[y*M+x])++;
}

void removeEdge(Graph * graph, int x, int y){
    int M = graph->M;
    //printf("Before removeEdge : (%d, %d) = %d. (%d, %d) = %d.\n", x, y ,graph->adj[x*M+y], y, x, graph->adj[y*M+x]);
    if(x==y)
        printf("ERROR1 when removing an edge (x cannot == y).\n");
    if((graph->adj[x*M+y])<=0 || (graph->adj[y*M+x])<=0){
        printf("ERROR2 when removing - graph(%d,%d)=%d. graph(%d,%d)=%d.\n", x, y, graph->adj[x*M+y], y, x, graph->adj[y*M+x]);
        exit(0);
    }
    graph->adj[x*M+y]--;
    graph->adj[y*M+x]--;
}

void printGraph(Graph * graph){
    int m = graph->M;
    for(int i=0;i<m;i++){
        for(int j=0;j<m;j++){
            printf("%d, ", graph->adj[i*m+j]);
        }
        printf("\n");
    }
}

int checkGraphSymmetric(Graph * graph){
    int m = graph->M;
    int * adj = graph->adj;
    for(int i=0;i<m;i++){
        for(int j=i;j<m;j++){
            if(i==j && adj[i*m+j]!=0){
                printf("ERROR when checking - graph(%d, %d) = %d.\n", i, j, adj[i*m+i]);
                return 0;
            } else if(i!=j && adj[i*m+j] != adj[j*m+i]){
                printf("ERROR when checking - graph(%d,%d)=%d. graph(%d,%d)=%d.\n", i, j, adj[i*m+j], j, i, adj[j*m+i]);
                return 0;
            }
        }
    }
    return 1;
}

void generateGraph(int * cIndex, Graph * graph, int elePerWarp, int nz){
    int index = 0, m = graph->M;
    int * fullset = (int*)malloc(m*sizeof(int));
    while(index<nz){
        memset(fullset, 0, m*sizeof(int));
        int size = 0;                               // set size
        int * set = (int*)malloc(m*sizeof(int));
        memset(set, 0, m*sizeof(int));
        for(int i=0;i<elePerWarp && index<nz;i++){
            if(fullset[cIndex[index]]==0){
                fullset[cIndex[index]] = 1;
                set[size++] = cIndex[index];
            }
            index++;
        }
        //set = (int*)realloc(set, size*sizeof(int));
        for(int i=0;i<size;i++){
            for(int j=i+1;j<size;j++){
                addEdge(graph, set[i], set[j]);
            }
        }
        //free(set);
    }
    if(checkGraphSymmetric(graph)==0)
        exit(0);
    //free(fullset);
}

void getMaxEdge(Graph * graph, int * a, int * b, int * chosen, int isNewCacheLine){
    int maxTemp = 0;
    int m = graph->M;
    for(int i=0;i<m;i++){
        for(int j=0;j<m;j++){
            if((isNewCacheLine || chosen[i]==1 || chosen[j]==1) && graph->adj[i*m+j]>maxTemp){
                *a = i;
                *b = j;
                maxTemp = graph->adj[i*m+j];
            }
        }
    }
    if(maxTemp<=0)
        printf("ERROR - maxTemp = %d.\n", maxTemp);
}


