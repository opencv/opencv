#include <stdio.h>
#include "string.h"
#include "_lsvmparser.h"

int isMODEL    (char *str){
    char stag [] = "<Model>";
    char etag [] = "</Model>";
    if(strcmp(stag, str) == 0)return  MODEL;
    if(strcmp(etag, str) == 0)return EMODEL;
    return 0;
}
int isP        (char *str){
    char stag [] = "<P>";
    char etag [] = "</P>";
    if(strcmp(stag, str) == 0)return  P;
    if(strcmp(etag, str) == 0)return EP;
    return 0;
}
int isSCORE        (char *str){
    char stag [] = "<ScoreThreshold>";
    char etag [] = "</ScoreThreshold>";
    if(strcmp(stag, str) == 0)return  SCORE;
    if(strcmp(etag, str) == 0)return ESCORE;
    return 0;
}
int isCOMP     (char *str){
    char stag [] = "<Component>";
    char etag [] = "</Component>";
    if(strcmp(stag, str) == 0)return  COMP;
    if(strcmp(etag, str) == 0)return ECOMP;
    return 0;
}
int isRFILTER  (char *str){
    char stag [] = "<RootFilter>";
    char etag [] = "</RootFilter>";
    if(strcmp(stag, str) == 0)return  RFILTER;
    if(strcmp(etag, str) == 0)return ERFILTER;
    return 0;
}
int isPFILTERs (char *str){
    char stag [] = "<PartFilters>";
    char etag [] = "</PartFilters>";
    if(strcmp(stag, str) == 0)return  PFILTERs;
    if(strcmp(etag, str) == 0)return EPFILTERs;
    return 0;
}
int isPFILTER  (char *str){
    char stag [] = "<PartFilter>";
    char etag [] = "</PartFilter>";
    if(strcmp(stag, str) == 0)return  PFILTER;
    if(strcmp(etag, str) == 0)return EPFILTER;
    return 0;
}
int isSIZEX    (char *str){
    char stag [] = "<sizeX>";
    char etag [] = "</sizeX>";
    if(strcmp(stag, str) == 0)return  SIZEX;
    if(strcmp(etag, str) == 0)return ESIZEX;
    return 0;
}
int isSIZEY    (char *str){
    char stag [] = "<sizeY>";
    char etag [] = "</sizeY>";
    if(strcmp(stag, str) == 0)return  SIZEY;
    if(strcmp(etag, str) == 0)return ESIZEY;
    return 0;
}
int isWEIGHTS  (char *str){
    char stag [] = "<Weights>";
    char etag [] = "</Weights>";
    if(strcmp(stag, str) == 0)return  WEIGHTS;
    if(strcmp(etag, str) == 0)return EWEIGHTS;
    return 0;
}
int isV        (char *str){
    char stag [] = "<V>";
    char etag [] = "</V>";
    if(strcmp(stag, str) == 0)return  TAGV;
    if(strcmp(etag, str) == 0)return ETAGV;
    return 0;
}
int isVx       (char *str){
    char stag [] = "<Vx>";
    char etag [] = "</Vx>";
    if(strcmp(stag, str) == 0)return  Vx;
    if(strcmp(etag, str) == 0)return EVx;
    return 0;
}
int isVy       (char *str){
    char stag [] = "<Vy>";
    char etag [] = "</Vy>";
    if(strcmp(stag, str) == 0)return  Vy;
    if(strcmp(etag, str) == 0)return EVy;
    return 0;
}
int isD        (char *str){
    char stag [] = "<Penalty>";
    char etag [] = "</Penalty>";
    if(strcmp(stag, str) == 0)return  TAGD;
    if(strcmp(etag, str) == 0)return ETAGD;
    return 0;
}
int isDx       (char *str){
    char stag [] = "<dx>";
    char etag [] = "</dx>";
    if(strcmp(stag, str) == 0)return  Dx;
    if(strcmp(etag, str) == 0)return EDx;
    return 0;
}
int isDy       (char *str){
    char stag [] = "<dy>";
    char etag [] = "</dy>";
    if(strcmp(stag, str) == 0)return  Dy;
    if(strcmp(etag, str) == 0)return EDy;
    return 0;
}
int isDxx      (char *str){
    char stag [] = "<dxx>";
    char etag [] = "</dxx>";
    if(strcmp(stag, str) == 0)return  Dxx;
    if(strcmp(etag, str) == 0)return EDxx;
    return 0;
}
int isDyy      (char *str){
    char stag [] = "<dyy>";
    char etag [] = "</dyy>";
    if(strcmp(stag, str) == 0)return  Dyy;
    if(strcmp(etag, str) == 0)return EDyy;
    return 0;
}
int isB      (char *str){
    char stag [] = "<LinearTerm>";
    char etag [] = "</LinearTerm>";
    if(strcmp(stag, str) == 0)return  BTAG;
    if(strcmp(etag, str) == 0)return EBTAG;
    return 0;
}

int getTeg(char *str){
    int sum = 0;
    sum = isMODEL (str)+
    isP        (str)+
    isSCORE    (str)+
    isCOMP     (str)+
    isRFILTER  (str)+
    isPFILTERs (str)+
    isPFILTER  (str)+
    isSIZEX    (str)+
    isSIZEY    (str)+
    isWEIGHTS  (str)+
    isV        (str)+
    isVx       (str)+
    isVy       (str)+
    isD        (str)+
    isDx       (str)+
    isDy       (str)+
    isDxx      (str)+
    isDyy      (str)+
    isB        (str);

    return sum;
}

void addFilter(filterObject *** model, int *last, int *max){
    filterObject ** nmodel;
    int i;
    (*last) ++;
    if((*last) >= (*max)){
        (*max) += 10;
        nmodel = (filterObject **)malloc(sizeof(filterObject *) * (*max));
        for(i = 0; i < *last; i++){
            nmodel[i] = (* model)[i];
        }
        free(* model);
        (*model) = nmodel;
    }
    (*model) [(*last)] = (filterObject *)malloc(sizeof(filterObject));
}

void parserRFilter  (FILE * xmlf, int p, filterObject * model, float *b){
    int st = 0;
    int sizeX, sizeY;
    int tag;
    int tagVal;
    char ch;
    int i,j,ii;
    char buf[1024];
    char tagBuf[1024];
    double *data;
    //printf("<RootFilter>\n");

    model->V.x = 0;
    model->V.y = 0;
    model->V.l = 0;
    model->fineFunction[0] = 0.0;
    model->fineFunction[1] = 0.0;
    model->fineFunction[2] = 0.0;
    model->fineFunction[3] = 0.0;

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == ERFILTER){
                    //printf("</RootFilter>\n");
                    return;
                }
                if(tagVal == SIZEX){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEX){
                    st = 0;
                    buf[i] = '\0';
                    sizeX = atoi(buf);
                    model->sizeX = sizeX;
                    //printf("<sizeX>%d</sizeX>\n", sizeX);
                }
                if(tagVal == SIZEY){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEY){
                    st = 0;
                    buf[i] = '\0';
                    sizeY = atoi(buf);
                    model->sizeY = sizeY;
                    //printf("<sizeY>%d</sizeY>\n", sizeY);
                }
                if(tagVal == WEIGHTS){
                    data = (double *)malloc( sizeof(double) * p * sizeX * sizeY);
                    fread(data, sizeof(double), p * sizeX * sizeY, xmlf);
                    model->H = (float *)malloc(sizeof(float)* p * sizeX * sizeY);
                    for(ii = 0; ii < p * sizeX * sizeY; ii++){
                        model->H[ii] = (float)data[ii];
                    }
                    free(data);
                }
                if(tagVal == EWEIGHTS){
                    //printf("WEIGHTS OK\n");
                }
                if(tagVal == BTAG){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EBTAG){
                    st = 0;
                    buf[i] = '\0';
                    *b =(float) atof(buf);
                    //printf("<B>%f</B>\n", *b);
                }

                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}

void parserV  (FILE * xmlf, int p, filterObject * model){
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j;
    char buf[1024];
    char tagBuf[1024];
    //printf("    <V>\n");

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == ETAGV){
                    //printf("    </V>\n");
                    return;
                }
                if(tagVal == Vx){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EVx){
                    st = 0;
                    buf[i] = '\0';
                    model->V.x = atoi(buf);
                    //printf("        <Vx>%d</Vx>\n", model->V.x);
                }
                if(tagVal == Vy){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EVy){
                    st = 0;
                    buf[i] = '\0';
                    model->V.y = atoi(buf);
                    //printf("        <Vy>%d</Vy>\n", model->V.y);
                }
                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}
void parserD  (FILE * xmlf, int p, filterObject * model){
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j;
    char buf[1024];
    char tagBuf[1024];
    //printf("    <D>\n");

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == ETAGD){
                    //printf("    </D>\n");
                    return;
                }
                if(tagVal == Dx){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDx){
                    st = 0;
                    buf[i] = '\0';
                    
                    model->fineFunction[0] = (float)atof(buf);
                    //printf("        <Dx>%f</Dx>\n", model->fineFunction[0]);
                }
                if(tagVal == Dy){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDy){
                    st = 0;
                    buf[i] = '\0';
                    
                    model->fineFunction[1] = (float)atof(buf);
                    //printf("        <Dy>%f</Dy>\n", model->fineFunction[1]);
                }
                if(tagVal == Dxx){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDxx){
                    st = 0;
                    buf[i] = '\0';
                    
                    model->fineFunction[2] = (float)atof(buf);
                    //printf("        <Dxx>%f</Dxx>\n", model->fineFunction[2]);
                }
                if(tagVal == Dyy){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EDyy){
                    st = 0;
                    buf[i] = '\0';
                    
                    model->fineFunction[3] = (float)atof(buf);
                    //printf("        <Dyy>%f</Dyy>\n", model->fineFunction[3]);
                }

                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}

void parserPFilter  (FILE * xmlf, int p, int N_path, filterObject * model){
    int st = 0;
    int sizeX, sizeY;
    int tag;
    int tagVal;
    char ch;
    int i,j, ii;
    char buf[1024];
    char tagBuf[1024];
    double *data;
    //printf("<PathFilter> (%d)\n", N_path);

    model->V.x = 0;
    model->V.y = 0;
    model->V.l = 0;
    model->fineFunction[0] = 0.0f;
    model->fineFunction[1] = 0.0f;
    model->fineFunction[2] = 0.0f;
    model->fineFunction[3] = 0.0f;

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == EPFILTER){
                    //printf("</PathFilter>\n");
                    return;
                }

                if(tagVal == TAGV){
                    parserV(xmlf, p, model);
                }
                if(tagVal == TAGD){
                    parserD(xmlf, p, model);
                }
                if(tagVal == SIZEX){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEX){
                    st = 0;
                    buf[i] = '\0';
                    sizeX = atoi(buf);
                    model->sizeX = sizeX;
                    //printf("<sizeX>%d</sizeX>\n", sizeX);
                }
                if(tagVal == SIZEY){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESIZEY){
                    st = 0;
                    buf[i] = '\0';
                    sizeY = atoi(buf);
                    model->sizeY = sizeY;
                    //printf("<sizeY>%d</sizeY>\n", sizeY);
                }
                if(tagVal == WEIGHTS){
                    data = (double *)malloc( sizeof(double) * p * sizeX * sizeY);
                    fread(data, sizeof(double), p * sizeX * sizeY, xmlf);
                    model->H = (float *)malloc(sizeof(float)* p * sizeX * sizeY);
                    for(ii = 0; ii < p * sizeX * sizeY; ii++){
                        model->H[ii] = (float)data[ii];
                    }
                    free(data);
                }
                if(tagVal == EWEIGHTS){
                    //printf("WEIGHTS OK\n");
                }
                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}
void parserPFilterS (FILE * xmlf, int p, filterObject *** model, int *last, int *max){
    int st = 0;
    int N_path = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j;
    char buf[1024];
    char tagBuf[1024];
    //printf("<PartFilters>\n");

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == EPFILTERs){
                    //printf("</PartFilters>\n");
                    return;
                }
                if(tagVal == PFILTER){
                    addFilter(model, last, max);
                    parserPFilter  (xmlf, p, N_path, (*model)[*last]);
                    N_path++;
                }
                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}
void parserComp (FILE * xmlf, int p, int *N_comp, filterObject *** model, float *b, int *last, int *max){
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j;
    char buf[1024];
    char tagBuf[1024];
    //printf("<Component> %d\n", *N_comp);

    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == ECOMP){
                    (*N_comp) ++;
                    return;
                }
                if(tagVal == RFILTER){
                    addFilter(model, last, max);
                    parserRFilter   (xmlf, p, (*model)[*last],b);
                }
                if(tagVal == PFILTERs){
                    parserPFilterS  (xmlf, p, model, last, max);
                }
                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}
void parserModel(FILE * xmlf, filterObject *** model, int *last, int *max, int **comp, float **b, int *count, float * score){
    int p = 0;
    int N_comp = 0;
    int * cmp;
    float *bb;
    int st = 0;
    int tag;
    int tagVal;
    char ch;
    int i,j, ii = 0;
    char buf[1024];
    char tagBuf[1024];
    
    //printf("<Model>\n");
    
    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                
                tagVal = getTeg(tagBuf);
               
                if(tagVal == EMODEL){
                    //printf("</Model>\n");
                    for(ii = 0; ii <= *last; ii++){
                        (*model)[ii]->p = p;
                        (*model)[ii]->xp = 9;
                    }
                    * count = N_comp;
                    return;
                }
                if(tagVal == COMP){
                    if(N_comp == 0){
                        cmp = (int    *)malloc(sizeof(int));
                        bb  = (float *)malloc(sizeof(float));
                        * comp = cmp;
                        * b    = bb;
                        * count = N_comp + 1; 
                    } else {
                        cmp = (int    *)malloc(sizeof(int)    * (N_comp + 1));
                        bb  = (float *)malloc(sizeof(float) * (N_comp + 1));
                        for(ii = 0; ii < N_comp; ii++){
                            cmp[i] = (* comp)[ii];
                            bb [i] = (* b   )[ii];
                        }
                        free(* comp);
                        free(* b   );
                        * comp = cmp;
                        * b    = bb;
                        * count = N_comp + 1; 
                    }
                    parserComp(xmlf, p, &N_comp, model, &((*b)[N_comp]), last, max);
                    cmp[N_comp - 1] = *last;
                }
                if(tagVal == P){
                    st = 1;
                    i = 0;
                }
                if(tagVal == EP){
                    st = 0;
                    buf[i] = '\0';
                    p = atoi(buf);
                    //printf("<P>%d</P>\n", p);
                }
                if(tagVal == SCORE){
                    st = 1;
                    i = 0;
                }
                if(tagVal == ESCORE){
                    st = 0;
                    buf[i] = '\0';
                    *score = (float)atof(buf);
                    //printf("<ScoreThreshold>%f</ScoreThreshold>\n", score);
                }
                tag = 0;
                i   = 0;                
            }else{
                if((tag == 0)&& (st == 1)){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}

void LSVMparser(const char * filename, filterObject *** model, int *last, int *max, int **comp, float **b, int *count, float * score){
    int st = 0;
    int tag;
    char ch;
    int i,j;
    FILE *xmlf;
    char buf[1024];
    char tagBuf[1024];

    (*max) = 10;
    (*last) = -1;
    (*model) = (filterObject ** )malloc((sizeof(filterObject * )) * (*max));

    //printf("parse : %s\n", filename);
    xmlf = fopen(filename, "rb");
    
    i   = 0;
    j   = 0;
    st  = 0;
    tag = 0;
    while(!feof(xmlf)){
        ch = fgetc( xmlf );
        if(ch == '<'){
            tag = 1;
            j   = 1;
            tagBuf[j - 1] = ch;
        }else {
            if(ch == '>'){
                tag = 0;
                i   = 0;
                tagBuf[j    ] = ch;
                tagBuf[j + 1] = '\0';
                if(getTeg(tagBuf) == MODEL){
                    parserModel(xmlf, model, last, max, comp, b, count, score);
                }
            }else{
                if(tag == 0){
                    buf[i] = ch; i++;
                }else{
                    tagBuf[j] = ch; j++;
                }
            }
        }        
    }
}

int loadModel(
             // Входные параметры
              const char *modelPath,// - путь до файла с моделью
             
              // Выходные параметры
              filterObject ***filters,// - массив указателей на фильтры компонент
              int *kFilters, //- общее количество фильтров во всех моделях
              int *kComponents, //- количество компонент
              int **kPartFilters, //- массив, содержащий количество точных фильтров в каждой компоненте
              float **b, //- массив линейных членов в оценочной функции
              float *scoreThreshold){ //- порог для score)
    int last;
    int max;
    int *comp;
    int count;
    int i;
    float score;
    //printf("start_parse\n\n");

    LSVMparser(modelPath, filters, &last, &max, &comp, b, &count, &score);
    (*kFilters)       = last + 1;
    (*kComponents)    = count;
    (*scoreThreshold) = (float) score;

    (*kPartFilters) = (int *)malloc(sizeof(int) * count);

    for(i = 1; i < count;i++){
        (*kPartFilters)[i] = (comp[i] - comp[i - 1]) - 1;
    }
    (*kPartFilters)[0] = comp[0];

    //printf("end_parse\n");
    return 0;
}