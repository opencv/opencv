#include "opencv2/ts.hpp"
#include "precomp.hpp"
#include <climits>
#include <algorithm>

namespace cv{namespace optim{
using std::vector;

double LPSolver::solve(const Function& F,const Constraints& C, OutputArray result)const{

    return 0.0;
}

double LPSolver::LPFunction::calc(InputArray args)const{
    printf("call to LPFunction::calc()\n");
    return 0.0;
}
void print_matrix(const Mat& X){
    printf("\ttype:%d vs %d,\tsize: %d-on-%d\n",X.type(),CV_64FC1,X.rows,X.cols);
    for(int i=0;i<X.rows;i++){
      printf("\t[");
      for(int j=0;j<X.cols;j++){
          printf("%g, ",X.at<double>(i,j));
      }
      printf("]\n");
    } 
}
namespace solveLP_aux{
    //return -1 if problem is unfeasible, 0 if feasible
    //in latter case it returns feasible solution in z with homogenised b's and v
    int initialize_simplex(const Mat& c, Mat& b, Mat& z,double& v);
}
int solveLP(const Mat& Func, const Mat& Constr, Mat& z){
    printf("call to solveLP\n");//-3(incorrect),-2 (no_sol - unbdd),-1(no_sol - unfsbl), 0(single_sol), 1(multiple_sol=>least_l2_norm)

    //sanity check (size, type, no. of channels) (and throw exception, if appropriate)
    if(Func.type()!=CV_64FC1 || Constr.type()!=CV_64FC1){
        printf("both Func and Constr should be one-channel matrices of double's\n");
        return -3;
    }
    if(Func.rows!=1){
        printf("Func should be row-vector\n");
        return -3;
    }
    vector<int> N(Func.cols);
    N[0]=1;
    for (std::vector<int>::iterator it = N.begin()+1 ; it != N.end(); ++it){
        *it=it[-1]+1;
    }
    if((Constr.cols-1)!=Func.cols){
        printf("Constr should have one more column when compared to Func\n");
        return -3;
    }
    vector<int> B(Constr.rows);
    B[0]=Func.cols+1;
    for (std::vector<int>::iterator it = B.begin()+1 ; it != B.end(); ++it){
        *it=it[-1]+1;
    }

    //copy arguments for we will shall modify them
    Mat c=Func.clone(),
        b=Constr.clone();
    double v=0;

    solveLP_aux::initialize_simplex(c,b,z,v);

    int count=0;
    while(1){
        printf("iteration #%d\n",count++);

        MatIterator_<double> pos_ptr;
        int e=0;
        for(pos_ptr=c.begin<double>();(*pos_ptr<=0) && pos_ptr!=c.end<double>();pos_ptr++,e++);
        if(pos_ptr==c.end<double>()){
            break;
        }
        printf("offset of first nonneg coef is %d\n",e);//TODO: choose the var with the smallest index

        int l=-1;
        double min=DBL_MAX;
        int row_it=0;
        double ite=0;
        MatIterator_<double> min_row_ptr=b.begin<double>();
        for(MatIterator_<double> it=b.begin<double>();it!=b.end<double>();it+=b.cols,row_it++){
            double myite=0;
            //check constraints, select the tightest one, TODO: smallest index
            if((myite=it[e])>0){
                double val=it[b.cols-1]/myite;
                if(val<min){
                    min_row_ptr=it;
                    ite=myite;
                    min=val;
                    l=row_it;
                }
            }
        }
        if(l==-1){
            //unbounded
            return -2;
        }
        printf("the tightest constraint is in row %d with %g\n",l,min);

        //pivoting:
        {
            int col_count=0;
            for(MatIterator_<double> it=min_row_ptr;col_count<b.cols;col_count++,it++){
                if(col_count==e){
                    *it=1/ite;
                }else{
                    *it/=ite;
                }
            }
        }
        int row_count=0;
        for(MatIterator_<double> it=b.begin<double>();row_count<b.rows;row_count++){
            printf("offset: %d\n",it-b.begin<double>());
            if(row_count==l){
                it+=b.cols;
            }else{
                //remaining constraints
                double coef=it[e];
                MatIterator_<double> shadow_it=min_row_ptr;
                for(int col_it=0;col_it<(b.cols);col_it++,it++,shadow_it++){
                    if(col_it==e){
                        *it=-coef*(*shadow_it);
                    }else{
                        *it-=coef*(*shadow_it);
                    }
                }
            }
        }
        //objective function
        double coef=*pos_ptr;
        MatIterator_<double> shadow_it=min_row_ptr;
        MatIterator_<double> it=c.begin<double>();
        for(int col_it=0;col_it<(b.cols-1);col_it++,it++,shadow_it++){
            if(col_it==e){
                *it=-coef*(*shadow_it);
            }else{
                *it-=coef*(*shadow_it);
            }
        }
        v+=coef*(*shadow_it);
        
        //new basis and nonbasic sets
        int tmp=N[e];
        N[e]=B[l];
        B[l]=tmp;

        printf("objective, v=%g\n",v);
        print_matrix(c);
        printf("constraints\n");
        print_matrix(b);
        printf("non-basic: ");
        for (std::vector<int>::iterator it = N.begin() ; it != N.end(); ++it){
            printf("%d, ",*it);
        }
        printf("\nbasic: ");
        for (std::vector<int>::iterator it = B.begin() ; it != B.end(); ++it){
            printf("%d, ",*it);
        }
        printf("\n");
    }

    //return the optimal solution
    //z=cv::Mat_<double>(1,c.cols,0);
    MatIterator_<double> it=z.begin<double>();
    for(int i=1;i<=c.cols;i++,it++){
        std::vector<int>::iterator pos=B.begin();
        if((pos=std::find(B.begin(),B.end(),i))==B.end()){
            *it+=0;
        }else{
            *it+=b.at<double>(pos-B.begin(),b.cols-1);
        }
    }

    return 0;
}
int solveLP_aux::initialize_simplex(const Mat& c, Mat& b, Mat& z,double& v){//TODO
    z=Mat_<double>(1,c.cols,0.0);
    v=0;
    return 0;

    cv::Mat mod_b=(cv::Mat_<double>(1,b.rows));
    bool gen_new_sol_flag=false,hom_sol_given=false;
    if(z.type()!=CV_64FC1 || z.rows!=1 || z.cols!=c.cols || (hom_sol_given=(countNonZero(z)==0))){
        printf("line %d\n",__LINE__);
        if(hom_sol_given==false){
            printf("line %d, %d\n",__LINE__,hom_sol_given);
            z=cv::Mat_<double>(1,c.cols,(double)0);
        }
        //check homogeneous solution
        printf("line %d\n",__LINE__);
        for(MatIterator_<double> b_it=b.begin<double>()+b.cols-1,mod_b_it=mod_b.begin<double>();mod_b_it!=mod_b.end<double>();
                b_it+=b.cols,mod_b_it++){
            if(*b_it<0){
                //if no - we need feasible solution
                gen_new_sol_flag=true;
                break;
            }
        }
        printf("line %d, gen_new_sol_flag=%d - I've got here!!!\n",__LINE__,gen_new_sol_flag);
        //if yes - we have feasible solution!
    }else{
        //check for feasibility
        MatIterator_<double> it=b.begin<double>();
        for(MatIterator_<double> mod_b_it=mod_b.begin<double>();it!=b.end<double>();mod_b_it++){
            double sum=0;
            for(MatIterator_<double> z_it=z.begin<double>();z_it!=z.end<double>();z_it++,it++){
                sum+=(*it)*(*z_it);
            }
            if((*mod_b_it=(*it-sum))<0){
                break;
            }
            it++;
        }
        if(it==b.end<double>()){
            //z contains feasible solution - just homogenise b's TODO: and v
            gen_new_sol_flag=false;
            for(MatIterator_<double> b_it=b.begin<double>()+b.cols-1,mod_b_it=mod_b.begin<double>();mod_b_it!=mod_b.end<double>();
                    b_it+=b.cols,mod_b_it++){
                *b_it=*mod_b_it;
            }
        }else{
            //if no - we need feasible solution
            gen_new_sol_flag=true;
        }
    }
    if(gen_new_sol_flag==true){
        //we should generate new solution - TODO
        printf("we should generate new solution\n");
        Mat new_c=Mat_<double>(1,c.cols+1,0.0),
            new_b=Mat_<double>(b.rows,b.cols+1,-1.0),
            new_z=Mat_<double>(1,c.cols+1,0.0);

        new_c.end<double>()[-1]=-1;
        c.copyTo(new_c.colRange(0,new_c.cols-1));

        b.col(b.cols-1).copyTo(new_b.col(new_b.cols-1));
        b.colRange(0,b.cols-1).copyTo(new_b.colRange(0,new_b.cols-2));

        Mat b_slice=b.col(b.cols-1);
        new_z.end<double>()[-1]=-*(std::min_element(b_slice.begin<double>(),b_slice.end<double>()));

        /*printf("matrix new_c\n");
        print_matrix(new_c);
        printf("matrix new_b\n");
        print_matrix(new_b);
        printf("matrix new_z\n");
        print_matrix(new_z);*/
        
        printf("run for the second time!\n");
        solveLP(new_c,new_b,new_z);
        printf("original z was\n");
        print_matrix(z);
        printf("that's what I've got\n");
        print_matrix(new_z);
        printf("for the constraints\n");
        print_matrix(b);
        return 0;
    }
    
}

}}
