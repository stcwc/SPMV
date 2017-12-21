#include <stdio.h>
#include <stdlib.h>
#include "hashmap.c"

int main(){

    Map * map = createMap(11);
    putIntoMap(map, 0,3);
    putIntoMap(map, 2,5);
    putIntoMap(map, 1,9);
    putIntoMap(map, 13,8);
    printf("get(0): %d, get(1): %d, get(2): %d, get(-1): %d, get(13): %d, map size: %d.\n", getInMap(map, 0), getInMap(map, 1), getInMap(map, 2), getInMap(map, -1), getInMap(map, 13), map->size);

    Map * map2 = createMap(7);
    putIntoMap(map2, 1,4);
    putIntoMap(map2, 4,5);
    putIntoMap(map2, 8,6);

    WholeMap * wholeMap = createWholeMap(499999);
    putIntoWholeMap(wholeMap, 1, map);
    putIntoWholeMap(wholeMap, 500000, map2);
    Map * tmp1 = getInWholeMap(wholeMap, 1);
    if(tmp1!=NULL)
        printf("get(0): %d, get(1): %d, get(2): %d, get(-1): %d, get(13): %d, map size: %d.\n", getInMap(tmp1, 0), getInMap(tmp1, 1), getInMap(tmp1, 2), getInMap(tmp1, -1), getInMap(tmp1, 13), tmp1->size);
    Map * tmp2 = getInWholeMap(wholeMap, 500000);
    if(tmp2!=NULL)
        printf("get(1): %d, get(4): %d, get(8): %d, get(-1): %d, get(8): %d, map size: %d.\n", getInMap(tmp2, 1), getInMap(tmp2, 4), getInMap(tmp2, 8), getInMap(tmp2, -1), getInMap(tmp2, 8), tmp2->size);
    printf("wholeMap size: %d.\n", wholeMap->size);

    exit(0);

}
