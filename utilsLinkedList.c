

#define WARP_SIZE 32

typedef struct LinkedListNode{
    int index, weight;
    struct LinkedListNode * next, * pre;
} LinkedListNode;

typedef struct{
    int M;
    LinkedListNode ** adj;
} Graph;

LinkedListNode * initDummyNode(){
    LinkedListNode * dummyNode = (LinkedListNode*)malloc(sizeof(LinkedListNode));
    dummyNode->index = -1;
    dummyNode->weight = -1;
    dummyNode->next = NULL;
    dummyNode->pre = NULL;
    return dummyNode;
}

Graph * initGraph(int M){
    Graph * graph = (Graph*)malloc(sizeof(Graph));
    graph->M = M;
    LinkedListNode ** adj = (LinkedListNode**)malloc(M*sizeof(LinkedListNode*));
    for(int i=0;i<M;i++){
        adj[i] = initDummyNode();
    }
    graph->adj = adj;
    return graph;
}

void addEdgeSpecifically(Graph * graph, int from, int to){
    //int M = graph->M;
    //printf("[%d, %d]\n", from, to);
    LinkedListNode ** adj = graph->adj;
    LinkedListNode * cur = adj[from];
    while(cur->next != NULL){
        cur = cur->next;
        //printf("\nCurrent index: %d, weight: %d, pre: %d, next: %d.\n", cur->index, cur->weight, cur->pre==NULL?-1:cur->pre->index, cur->next==NULL?-1:cur->next->index);
        if(cur->index == to){
            cur->weight++;
            while(cur->pre->index!=-1 && cur->weight>cur->pre->weight){     // Move the node ahead to sort the list.
                //printf("1 move.\n");
                LinkedListNode * pre_temp = cur->pre;
                pre_temp->pre->next = cur;
                if(cur->next!=NULL)
                    cur->next->pre = pre_temp;
                pre_temp->next = cur->next;
                cur->next = cur->pre;
                cur->pre = pre_temp->pre;
                pre_temp->pre = cur;
                //printf("1 move done.\n");
            }
            return;
        }
    }
    // If node "to" wasn't in the linkedlist, add it to the end.
    LinkedListNode * node = (LinkedListNode*)malloc(sizeof(LinkedListNode));
    node->index = to;
    node->weight = 1;
    node->next = NULL;
    node->pre = cur;
    cur->next = node;
}

void addEdge(Graph * graph, int x, int y){
    if(x==y)
        return;
        //printf("ERROR1 when adding an edge (x cannot == y).\n");
    if(x >= graph->M || y >= graph->M){
        printf("ERROR - out of bound request when adding edge from %d to %d, when M = %d.\n", x, y, graph->M);
        exit(0);
    }
    addEdgeSpecifically(graph, x, y);
    addEdgeSpecifically(graph, y, x);
}

void removeEdgeSpecifically(Graph * graph, int from, int to){
    //int M = graph->M;
    LinkedListNode ** adj = graph->adj;
    LinkedListNode * cur = adj[from];
    while(cur->next != NULL){
        cur = cur->next;
        if(cur->index == to){
            cur->pre->next = cur->next;
            if(cur->next != NULL)
                cur->next->pre = cur->pre;
            return;
        }
    }
    // This edge doesn't exist, error.
    printf("ERROR - edge (%d, %d) doesn't exist.\n", from, to);
    exit(0);
}

void removeEdge(Graph * graph, int x, int y){
    if(x==y)
        printf("ERROR1 when removing an edge (x cannot == y).\n");
    if(x >= graph->M || y >= graph->M){
        printf("ERROR - out of bound request when removing edge from %d to %d, when M = %d.\n", x, y, graph->M);
        exit(0);
    }
    removeEdgeSpecifically(graph, x, y);
    removeEdgeSpecifically(graph, y, x);
}

void generateGraph(int * cIndex, Graph * graph, int elePerWarp, int nz){
    printf("Generating graph......\n");
    int index = 0, M = graph->M;
    int * fullset = (int*)malloc(M*sizeof(int));
    while(index<nz){
        //printf("index: %d\n", index);
        int isBeginning = 1;                        // meaning index is the beginning of a segment of 32
        memset(fullset, 0, M*sizeof(int));
        int size = 0;                               // set size
        int * set = (int*)malloc(M*sizeof(int));
        memset(set, 0, M*sizeof(int));
        while((isBeginning || (index%WARP_SIZE!=0 && index%elePerWarp!=0)) && index<nz){
            if(fullset[cIndex[index]]==0){
                fullset[cIndex[index]] = 1;
                set[size++] = cIndex[index];
            }
            index++;
            isBeginning = 0;
        }
        //printf("b\n");
        //set = (int*)realloc(set, size*sizeof(int));
        for(int i=0;i<size;i++){
            for(int j=i+1;j<size;j++){
                //printf("(%d, %d) \n", set[i], set[j]);
                addEdge(graph, set[i], set[j]);
                //printf("adddone\n");
            }
        }
        //printf(", index: %d\n", index);
        //free(set);
    }
    printf("Finish generating graph.\n");
    //free(fullset);
}

void getMaxEdgeAndRemove(Graph * graph, int * a, int * b, int * chosen, int isNewCacheLine){
    int maxTemp = 0;
    int m = graph->M;
    LinkedListNode ** adj = graph->adj;
    LinkedListNode * cur = NULL;
    for(int i=0;i<m;i++){
        cur = adj[i];
        if(cur->next==NULL)
            continue;
        cur = cur->next;                // Since each linkedlist is sorted from big weight to low weight, comparing the first is enough.
        int j = cur->index;
        if((isNewCacheLine || chosen[i]==1 || chosen[j]==1) && cur->weight>maxTemp){
            *a = i;
            *b = j;
            maxTemp = cur->weight;
        }
    }
    if(maxTemp<=0){
        printf("ERROR - maxTemp = %d.\n", maxTemp);
        exit(0);
    }
    removeEdge(graph, *a, *b);
}

void printGraph(Graph * graph){
    printf("Printing graph......\n");
    int M = graph->M;
    for(int i=0;i<M;i++){
        LinkedListNode * cur = graph->adj[i];
        printf("\nfrom %d to ", i);
        while(cur->next!=NULL){
            cur = cur->next;
            printf("%d: %d, ", cur->index, cur->weight);
        }
    }
}



