#include <stdio.h>
#include <stdlib.h>

/*
Most inside Node, with int key and int value.
*/
struct Node{
    int key;
    int val;
    struct Node *next;
};

/*
The Map corresponding to Node.
*/
struct Map{
    int n_slots;
    int size;
    struct Node **list;
};


struct Map *createMap(int n_slots){
    struct Map *t = (struct Map*)malloc(sizeof(struct Map));
    t->n_slots = n_slots;
    t->size = 0;
    t->list = (struct Node**)malloc(sizeof(struct Node*)*n_slots);
    int i;
    for(i=0;i<n_slots;i++)
        t->list[i] = NULL;
    return t;
}

int hashCodeMap(struct Map *t,int key){
    if(key<0)
        return -(key%t->n_slots);
    return key%t->n_slots;
}

void putIntoMap(struct Map *t,int key,int val){
    int pos = hashCodeMap(t,key);
    struct Node *list = t->list[pos];
    struct Node *newNode = (struct Node*)malloc(sizeof(struct Node));
    struct Node *temp = list;
    while(temp){
        if(temp->key==key){
            temp->val = val;
            return;
        }
        temp = temp->next;
    }
    newNode->key = key;
    newNode->val = val;
    newNode->next = list;
    t->list[pos] = newNode;
    t->size++;
}

int getInMap(struct Map *t,int key){
    int pos = hashCodeMap(t,key);
    struct Node *list = t->list[pos];
    struct Node *temp = list;
    while(temp){
        if(temp->key==key){
            return temp->val;
        }
        temp = temp->next;
    }
    return -1;
}



/*
RowNode, with int key (vertex index) and hashtable value.
*/
struct RowNode{
    int key;
    struct Map * map;
    struct RowNode *next;
};

/*
The Map corresponding to RowNode.
*/
struct WholeMap{
    int n_slots;
    int size;
    struct RowNode **list;
};

struct WholeMap *createWholeMap(int n_slots){
    struct WholeMap *t = (struct WholeMap*)malloc(sizeof(struct WholeMap));
    t->n_slots = n_slots;
    t->size = 0;
    t->list = (struct RowNode**)malloc(sizeof(struct RowNode*)*n_slots);
    int i;
    for(i=0;i<n_slots;i++)
        t->list[i] = NULL;
    return t;
}

int hashCodeWholeMap(struct WholeMap *t,int key){
    if(key<0)
        return -(key%t->n_slots);
    return key%t->n_slots;
}

void putIntoWholeMap(struct WholeMap *t,int key,Map *map){
    int pos = hashCodeWholeMap(t,key);
    struct RowNode *list = t->list[pos];
    struct RowNode *newNode = (struct RowNode*)malloc(sizeof(struct RowNode));
    struct RowNode *temp = list;
    while(temp){
        if(temp->key==key){
            temp->map = map;
            return;
        }
        temp = temp->next;
    }
    newNode->key = key;
    newNode->map = map;
    newNode->next = list;
    t->list[pos] = newNode;
    t->size++;
}

Map * getInWholeMap(struct WholeMap *t,int key){
    int pos = hashCodeWholeMap(t,key);
    struct RowNode *list = t->list[pos];
    struct RowNode *temp = list;
    while(temp){
        if(temp->key==key){
            return temp->map;
        }
        temp = temp->next;
    }
    return NULL;
}

typedef struct LinkedNode {
    int val, weight;
    LinkedNode * next;
}LinkedNode;


int compareNode(const void * a, const void * b){
    LinkedNode ** aa = (LinkedNode **)a;
    LinkedNode ** bb = (LinkedNode **)b;
    return (*bb)->weight - (*aa)->weight;


    // int weight_a = ((LinkedNode*)a)->weight;
    // int weight_b = ((LinkedNode*)b)->weight;
    // return weight_b-weight_a;
}

LinkedNode * sortMap(Map * map){
    if(map==NULL || map->size==0)
        return NULL;
    int size = map->size;
    //LinkedNode * linkedNodeList = (LinkedNode*)malloc(size*sizeof(LinkedNode));
    LinkedNode ** linkedNodeList = (LinkedNode**)malloc(size*sizeof(LinkedNode*));
    int index = 0;
    // printf("aaaaaaaaaaaaa\n");
    for(int i=0;i<map->n_slots;i++){
        struct Node *temp = map->list[i];
        while(temp){
            // linkedNodeList[index].val = temp->key;
            // linkedNodeList[index].weight = temp->val;
            // linkedNodeList[index].next = NULL;
           //  printf("bbbbbbbb\n");
            linkedNodeList[index] = (LinkedNode*)malloc(sizeof(LinkedNode));
            linkedNodeList[index]->val = temp->key;
            linkedNodeList[index]->weight = temp->val;
            linkedNodeList[index]->next = NULL;
            index++;
            // printf("ccccccccccccc\n");
            temp = temp->next;
        }
    }
    if(index!=size){
        //printf("ERROR when sorting map - index = %d, size = %d.\n", index, size);
        exit(0);
    }
    // printf("ddddddddd\n");
    //qsort((void*)linkedNodeList, size, sizeof(LinkedNode), compareNode);
    qsort((void*)linkedNodeList, size, sizeof(LinkedNode*), &compareNode);
    // printf("eeeeeeeeeeee\n");
    for(int i=0;i<size-1;i++){
        //linkedNodeList[i].next = &(linkedNodeList[i+1]);
        linkedNodeList[i]->next = (linkedNodeList[i+1]);
    }
    free(map);      // TODO
    //return linkedNodeList;
    return linkedNodeList[0];
}

struct LinkedNodeWrap{
    int index;
    LinkedNode * linkedNode;
};

int compareNodeWrap(const void * a, const void * b){
    int weight_a = ((LinkedNodeWrap*)a)->linkedNode->weight;
    int weight_b = ((LinkedNodeWrap*)b)->linkedNode->weight;
    return weight_b-weight_a;
}

LinkedNode ** sortWholeGraph(WholeMap * wholeMap, Map * mapping, int * ssize){
    if(wholeMap == NULL || wholeMap->size == 0)
        return NULL;
    int size = wholeMap->size, index = 0;
    // printf("111111111111\n");
    //LinkedNode * list = (LinkedNode*)malloc(size*sizeof(LinkedNode));
    LinkedNode ** list = (LinkedNode**)malloc(size*sizeof(LinkedNode*));
    // printf("2222222222\n");
    for(int i=0;i<wholeMap->n_slots;i++){
        struct RowNode * temp = wholeMap->list[i];
        while(temp){

            // printf("3333333333333\n");
            LinkedNode * largest = sortMap(temp->map);

            // list[index].val = temp->key;
            // list[index].weight = largest->weight;
            // list[index].next = largest;
            list[index] = (LinkedNode*)malloc(sizeof(LinkedNode));
            list[index]->val = temp->key;
            list[index]->weight = largest->weight;
            list[index]->next = largest;
            index++;
            temp = temp->next;
            // printf("444444444444\n");





        }
        //wrapList[i].index = (wholeMap->list[i])->key;
        //wrapList[i].linkedNode = sortMap((wholeMap->list[i])->map);
    }
    // printf("55555555555\n");
    //qsort((void*)list, size, sizeof(LinkedNode), compareNode);
    qsort((void*)list, size, sizeof(LinkedNode*), &compareNode);

        LinkedNode * subGraph = list[2];
        // printf("*****************\n");
        while (subGraph!=NULL) {
           //  printf("location inner b \n");
            // printf(" %d : %d \n", subGraph->val, subGraph->weight);
            // printf("location innder c \n");
            subGraph = subGraph->next;
        }


    // printf("6666666666666\n");
    for(int i=0;i<size;i++){
        //putIntoMap(mapping, list[i].val, i);
        putIntoMap(mapping, list[i]->val, i);
    }
    *ssize = size;
    return list;
}





