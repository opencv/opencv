//
// From OpenCV's samples/c directory
//   Example 13-2. Creating and training a decision tree
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/* License:
   July 20, 2011
   Standard BSD

   BOOK: It would be nice if you cited it:
   Learning OpenCV 2: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media
 
   AVAILABLE AT: 
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130    

   Main OpenCV site
   http://opencv.willowgarage.com/wiki/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://pr.willowgarage.com/wiki/OpenCV
*/

/*
The sample demonstrates how to build a decision tree for classifying mushrooms.
It uses the sample base agaricus-lepiota.data from UCI Repository, here is the link:

Newman, D.J. & Hettich, S. & Blake, C.L. & Merz, C.J. (1998).
UCI Repository of machine learning databases
[http://www.ics.uci.edu/~mlearn/MLRepository.html].
Irvine, CA: University of California, Department of Information and Computer Science.
*/

// loads the mushroom database, which is a text file, containing
// one training sample per row, all the input variables and the output variable are categorical,
// the values are encoded by characters.
bool mushroom_read_database( const char* filename, Mat& data, Mat& responses, Mat& missing )
{
    const int M = 1024;
    FILE* f = fopen( filename, "rt" );
    char buf[M+2], *ptr;
    int i, j, var_count = 0;
    
    if( !f )
        return false;
    
    // read the first line and determine the number of variables
    if( !fgets( buf, M, f ))
    {
        fclose(f);
        return false;
    }
    
    for( ptr = buf; *ptr != '\0'; ptr++ )
        var_count += *ptr == ',';
    assert( ptr - buf == (var_count+1)*2 );
    
    // create temporary memory storage to store the whole database
    vector<float> datarow(var_count+1);
    vector<float> alldata;
    
    for(;;)
    {
        for( i = 0; i <= var_count; i++ )
        {
            int c = buf[i*2];
            datarow[i] = c == '?' ? -1.f : (float)c;
        }
        if( i != var_count+1 )
            break;
        copy(datarow.begin(), datarow.end(), back_inserter(alldata));
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
    }
    fclose(f);
    
    // allocate the output matrices and copy the base there
    Mat alldatam((int)(alldata.size()/datarow.size()), (int)datarow.size(), CV_32F, &alldata[0]);
    alldatam.colRange(1, alldatam.cols).copyTo(data);
    missing = alldatam.colRange(1, alldatam.cols) == -1;
    alldatam.col(0).copyTo(responses);
    
    return true;
}


Ptr<DecisionTree> mushroom_create_dtree( Mat& data, Mat& responses, Mat& missing, float p_weight )
{
    Ptr<DecisionTree> dtree = new DecisionTree;
    
    int i, hr1 = 0, hr2 = 0, p_total = 0;
    float priors[] = { 1, p_weight };

    Mat vartypes(data.cols+1, 1, CV_8U);
    vartypes = Scalar::all(CV_VAR_CATEGORICAL);
    
    dtree->train( data, CV_ROW_SAMPLE, responses, Mat(), Mat(), vartypes, missing,
                  DTreeParams( 8, // max depth
                                 10, // min sample count
                                 0, // regression accuracy: N/A here
                                 true, // compute surrogate split, as we have missing data
                                 15, // max number of categories (use sub-optimal algorithm for larger numbers)
                                 10, // the number of cross-validation folds
                                 true, // use 1SE rule => smaller tree
                                 true, // throw away the pruned tree branches
                                 priors // the array of priors, the bigger p_weight, the more attention
                                        // to the poisonous mushrooms
                                        // (a mushroom will be judjed to be poisonous with bigger chance)
                                 ));
    
    

    // compute hit-rate on the training database, demonstrates predict usage.
    for( i = 0; i < data.rows; i++ )
    {
        Mat sample = data.row(i), mask = missing.row(i);
        double r = dtree->predict( sample, mask )->value;
        int d = fabs(r - responses.at<float>(i)) >= FLT_EPSILON;
        if( d )
        {
            if( r != 'p' )
                hr1++;
            else
                hr2++;
        }
        p_total += responses.at<float>(i) == 'p';
    }

    cout << "Results on the training database:\n"
         << "\tPoisonous mushrooms mis-predicted: "
         << hr1 << "(" << (double)hr1*100/p_total << ")\n" <<
         "\tFalse-alarms: " << hr2 << "(" << (double)hr2*100/(data.rows - p_total) << ")\n";

    return dtree;
}


static const char* var_desc[] =
{
    "cap shape (bell=b,conical=c,convex=x,flat=f)",
    "cap surface (fibrous=f,grooves=g,scaly=y,smooth=s)",
    "cap color (brown=n,buff=b,cinnamon=c,gray=g,green=r,\n\tpink=p,purple=u,red=e,white=w,yellow=y)",
    "bruises? (bruises=t,no=f)",
    "odor (almond=a,anise=l,creosote=c,fishy=y,foul=f,\n\tmusty=m,none=n,pungent=p,spicy=s)",
    "gill attachment (attached=a,descending=d,free=f,notched=n)",
    "gill spacing (close=c,crowded=w,distant=d)",
    "gill size (broad=b,narrow=n)",
    "gill color (black=k,brown=n,buff=b,chocolate=h,gray=g,\n\tgreen=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y)",
    "stalk shape (enlarging=e,tapering=t)",
    "stalk root (bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r)",
    "stalk surface above ring (ibrous=f,scaly=y,silky=k,smooth=s)",
    "stalk surface below ring (ibrous=f,scaly=y,silky=k,smooth=s)",
    "stalk color above ring (brown=n,buff=b,cinnamon=c,gray=g,orange=o,\n\tpink=p,red=e,white=w,yellow=y)",
    "stalk color below ring (brown=n,buff=b,cinnamon=c,gray=g,orange=o,\n\tpink=p,red=e,white=w,yellow=y)",
    "veil type (partial=p,universal=u)",
    "veil color (brown=n,orange=o,white=w,yellow=y)",
    "ring number (none=n,one=o,two=t)",
    "ring type (cobwebby=c,evanescent=e,flaring=f,large=l,\n\tnone=n,pendant=p,sheathing=s,zone=z)",
    "spore print color (black=k,brown=n,buff=b,chocolate=h,green=r,\n\torange=o,purple=u,white=w,yellow=y)",
    "population (abundant=a,clustered=c,numerous=n,\n\tscattered=s,several=v,solitary=y)",
    "habitat (grasses=g,leaves=l,meadows=m,paths=p\n\turban=u,waste=w,woods=d)",
    0
};


void print_variable_importance( Ptr<DecisionTree> dtree, const char** var_desc )
{
    Mat var_importance(dtree->get_var_importance());
    int i;

    if( var_importance.empty() )
    {
        cout << "Error: Variable importance can not be retrieved\n";
        return;
    }

    cout << "Print variable importance information? (y/n) ";
    cout.flush();
    char input[100];
    cin.getline(input, sizeof(input)-2);
    if( input[0] != 'y' && input[0] != 'Y' )
        return;

    for( i = 0; i < var_importance.cols*var_importance.rows; i++ )
    {
        double val = var_importance.at<double>(i);
        if( var_desc )
        {
            char buf[100];
            int len = strchr( var_desc[i], '(' ) - var_desc[i] - 1;
            strncpy( buf, var_desc[i], len );
            buf[len] = '\0';
            cout << buf;
        }
        else
            cout << "var #" << i;
        cout << ": " << val*100. << endl;
    }
}

void interactive_classification( Ptr<DecisionTree> dtree, const char** var_desc )
{
    if( dtree.empty() )
        return;

    const CvDTreeNode* root = dtree->get_root();
    CvDTreeTrainData* data = dtree->get_data();

    for(;;)
    {
        const CvDTreeNode* node;
        char input[100];
        cout << "Start/Proceed with interactive mushroom classification (y/n): ";
        cout.flush();
        cin.getline(input, sizeof(input)-2);
        if( input[0] != 'y' && input[0] != 'Y' )
            break;
        cout << "Enter 1-letter answers, '?' for missing/unknown value...\n"; 

        // custom version of predict
        node = root;
        for(;;)
        {
            CvDTreeSplit* split = node->split;
            int dir = 0;
            
            if( !node->left || node->Tn <= dtree->get_pruned_tree_idx() || !node->split )
                break;

            for( ; split != 0; )
            {
                int vi = split->var_idx, j;
                int count = data->cat_count->data.i[vi];
                const int* catmap = data->cat_map->data.i + data->cat_ofs->data.i[vi];

                cout << var_desc[vi] << ": ";
                cout.flush();
                cin.getline(input, sizeof(input)-2);

                if( input[0] == '?' )
                {
                    split = split->next;
                    continue;
                }

                // convert the input character to the normalized value of the variable
                for( j = 0; j < count; j++ )
                    if( catmap[j] == input[0] )
                        break;
                if( j < count )
                {
                    dir = (split->subset[j>>5] & (1 << (j&31))) ? -1 : 1;
                    if( split->inversed )
                        dir = -dir;
                    break;
                }
                else
                    cout << "Error: unrecognized value\n";
            }
            
            if( !dir )
            {
                cout << "Impossible to classify the sample\n";
                node = 0;
                break;
            }
            node = dir < 0 ? node->left : node->right;
        }

        if( node )
            cout << "Prediction result: the mushroom is " <<
                (node->class_idx == 0 ? "EDIBLE" : "POISONOUS");
        cout << "\n-----------------------------\n";
    }
}

void help()
{
	cout << "\nThe sample demonstrates how to build a decision tree for classifying mushrooms.\n"
            " It uses the sample base agaricus-lepiota.data from UCI Repository\n\n"
			"Usage:\n ./ch13_ex13_2 [path/file-name_data]\n"
			"where [] means optional. If you don't give it a data file, it will try to load\n"
			"agaricus-lepiota.data\n" << endl;
}
int main( int argc, char** argv )
{
    Ptr<DecisionTree> dtree;
    TrainData mldata;
    const char* base_path = argc >= 2 ? argv[1] : "agaricus-lepiota.data";
    Mat data, responses, missing;
    help();
    if( !mushroom_read_database( base_path, data, responses, missing ) )
    {
        cout << "Unable to load the training database\n" << base_path << endl;
        return -1;
    }

    dtree = mushroom_create_dtree( data, responses, missing,
        10 // poisonous mushrooms will have 10x higher weight in the decision tree
        );

    print_variable_importance( dtree, var_desc );
    interactive_classification( dtree, var_desc );

    return 0;
}
