namespace cv{namespace optim{
#ifdef ALEX_DEBUG
#define dprintf(x) printf x
static void print_matrix(const Mat& x){
    printf("\ttype:%d vs %d,\tsize: %d-on-%d\n",x.type(),CV_64FC1,x.rows,x.cols);
    for(int i=0;i<x.rows;i++){
        printf("\t[");
        for(int j=0;j<x.cols;j++){
            printf("%g, ",x.at<double>(i,j));
        }
        printf("]\n");
    }
}
#else
#define dprintf(x)
#define print_matrix(x)
#endif
}}
