/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <fstream>

using namespace std;

namespace cv
{

ERStat::ERStat(int init_level, int init_pixel, int init_x, int init_y) : pixel(init_pixel), 
               level(init_level), area(0), perimeter(0), euler(0), probability(1.0), 
               parent(0), child(0), next(0), prev(0), local_maxima(0), 
               max_probability_ancestor(0), min_probability_ancestor(0)
{
    rect = Rect(init_x,init_y,1,1);
    raw_moments[0] = 0.0;
    raw_moments[1] = 0.0;
    central_moments[0] = 0.0;
    central_moments[1] = 0.0;
    central_moments[2] = 0.0;
    crossings = new std::deque<int>();
    crossings->push_back(0);
}


// derivative classes


// the classe implementing the interface for the 1st and 2nd stages of Neumann and Matas algorithm
class CV_EXPORTS ERFilterNM : public ERFilter
{
public:
    //Constructor
    ERFilterNM();
    //Destructor
    ~ERFilterNM() {};

    float minProbability;   
    bool  nonMaxSuppression;
    float minProbabilityDiff;

    // the key method. Takes image on input, vector of ERStat is output for the first stage, 
    // input/output - for the second one.
    void run( InputArray image, std::vector<ERStat>& regions );

protected:
    int thresholdDelta;
    float maxArea; 
    float minArea;

    Ptr<ERFilter::Callback> classifier;

    // count of the rejected/accepted regions
    int num_rejected_regions;
    int num_accepted_regions;

public:

    // set/get methods to set the algorithm properties,
    void setCallback(const Ptr<ERFilter::Callback>& cb);
    void setThresholdDelta(int thresholdDelta);
    void setMinArea(float minArea);
    void setMaxArea(float maxArea);
    void setMinProbability(float minProbability);
    void setMinProbabilityDiff(float minProbabilityDiff);
    void setNonMaxSuppression(bool nonMaxSuppression);
    int  getNumRejected();

private:
    // pointer to the input/output regions vector
    std::vector<ERStat> *regions;
    // image mask used for feature calculations
    Mat region_mask;

    // extract the component tree and store all the ER regions
    void er_tree_extract( InputArray image );
    // accumulate a pixel into an ER
    void er_add_pixel( ERStat *parent, int x, int y, int non_boundary_neighbours, 
                       int non_boundary_neighbours_horiz, 
                       int d_C1, int d_C2, int d_C3 );
    // merge an ER with its nested parent
    void er_merge( ERStat *parent, ERStat *child );
    // recursively walk the tree and clean memory
    void er_tree_clean( ERStat *er );
    // copy extracted regions into the output vector
    ERStat* er_save( ERStat *er, ERStat *parent, ERStat *prev );
    // recursively walk the tree and filter (remove) regions using the callback classifier
    ERStat* er_tree_filter( InputArray image, ERStat *stat, ERStat *parent, ERStat *prev );
    // recursively walk the tree selecting only regions with local maxima probability
    ERStat* er_tree_nonmax_suppression( ERStat *er, ERStat *parent, ERStat *prev );
};


// default 1st stage classifier
class CV_EXPORTS ERClassifierNM1 : public ERFilter::Callback 
{
public:
    //Constructor
    ERClassifierNM1();
    // Destructor
    ~ERClassifierNM1() {};

    // The classifier must return probability measure for the region.
    double eval(const ERStat& stat); 

private:
    CvBoost boost;
};

// default 2nd stage classifier
class CV_EXPORTS ERClassifierNM2 : public ERFilter::Callback 
{
public:
    //constructor
    ERClassifierNM2();
    // Destructor
    ~ERClassifierNM2() {};

    // The classifier must return probability measure for the region.
    double eval(const ERStat& stat); 

private:
    CvBoost boost;
};





// default constructor
ERFilterNM::ERFilterNM()
{
    thresholdDelta = 1;
    minArea = 0.;
    maxArea = 1.;
    minProbability = 0.;
    nonMaxSuppression = false;
    minProbabilityDiff = 1.;
    num_accepted_regions = 0;
    num_rejected_regions = 0;
    classifier = NULL;
}

// the key method. Takes image on input, vector of ERStat is output for the first stage, 
// input/output for the second one.
void ERFilterNM::run( InputArray image, std::vector<ERStat>& _regions )
{

    // assert correct image type
    CV_Assert( image.getMat().type() == CV_8UC1 );

    regions = &_regions;
    region_mask = Mat::zeros(image.getMat().rows+2, image.getMat().cols+2, CV_8UC1);
    
    // if regions vector is empty we must extract the entire component tree
    if ( regions->size() == 0 )
    {
        er_tree_extract( image );
        if (nonMaxSuppression)
        {
            vector<ERStat> aux_regions;
            regions->swap(aux_regions);
            regions->reserve(aux_regions.size());
            er_tree_nonmax_suppression( &aux_regions.front(), NULL, NULL );
            aux_regions.clear();
        }
    }
    else // if regions vector is already filled we'll just filter the current regions
    {
        // the tree root must have no parent
        CV_Assert( regions->front().parent == NULL );

        vector<ERStat> aux_regions;
        regions->swap(aux_regions);
        regions->reserve(aux_regions.size());
        er_tree_filter( image, &aux_regions.front(), NULL, NULL );
        aux_regions.clear();
    }
}

// extract the component tree and store all the ER regions
// uses the algorithm described in
// Linear time maximally stable extremal regions, D Nistér, H Stewénius – ECCV 2008
void ERFilterNM::er_tree_extract( InputArray image )
{

    Mat src = image.getMat();
    // assert correct image type
    CV_Assert( src.type() == CV_8UC1 );

    if (thresholdDelta > 1)
    {
        Mat tmp;
        src.copyTo(tmp);
        src.release();
        src = (image.getMat() / thresholdDelta) -1;
    }

    const unsigned char * image_data = src.data; 
    int width = src.cols, height = src.rows; 

    // the component stack
    vector<ERStat*> er_stack;

    //the quads for euler number calculation 
    unsigned char quads[3][4];
    quads[0][0] = 1 << 3;
    quads[0][1] = 1 << 2;
    quads[0][2] = 1 << 1;
    quads[0][3] = 1;
    quads[1][0] = (1<<2)|(1<<1)|(1);
    quads[1][1] = (1<<3)|(1<<1)|(1);
    quads[1][2] = (1<<3)|(1<<2)|(1);
    quads[1][3] = (1<<3)|(1<<2)|(1<<1);
    quads[2][0] = (1<<2)|(1<<1);
    quads[2][1] = (1<<3)|(1);
    quads[2][3] = 255;


    // masks to know if a pixel is accessible and if it has been already added to some region
    vector<bool> accessible_pixel_mask(width * height);
    vector<bool> accumulated_pixel_mask(width * height);

    // heap of boundary pixels
    vector<int> boundary_pixes[256];
    vector<int> boundary_edges[256];

    // add a dummy-component before start
    er_stack.push_back(new ERStat);

    // we'll look initially for all pixels with grey-level lower than a grey-level higher than any allowed in the image
    int threshold_level = (255/thresholdDelta)+1;
    
    // starting from the first pixel (0,0)
    int current_pixel = 0;
    int current_edge = 0;
    int current_level = image_data[0];
    accessible_pixel_mask[0] = true;
    
    bool push_new_component = true;
    
    for (;;) {

        int x = current_pixel % width;
        int y = current_pixel / width;

        // push a component with current level in the component stack
        if (push_new_component) 
            er_stack.push_back(new ERStat(current_level, current_pixel, x, y));
        push_new_component = false;
        
        // explore the (remaining) edges to the neighbors to the current pixel
        for (current_edge = current_edge; current_edge < 4; current_edge++) 
        {

            int neighbour_pixel = current_pixel;
            
            switch (current_edge) 
            {
                    case 0: if (x < width - 1) neighbour_pixel = current_pixel + 1;  break;
                    case 1: if (y < height - 1) neighbour_pixel = current_pixel + width; break;
                    case 2: if (x > 0) neighbour_pixel = current_pixel - 1; break;
                    default: if (y > 0) neighbour_pixel = current_pixel - width; break;
            }

            // if neighbour is not accessible, mark it accessible and retreive its grey-level value
            if ( !accessible_pixel_mask[neighbour_pixel] && (neighbour_pixel != current_pixel) ) 
            {

                int neighbour_level = image_data[neighbour_pixel];
                accessible_pixel_mask[neighbour_pixel] = true;

                // if neighbour level is not lower than current level add neighbour to the boundary heap         
                if (neighbour_level >= current_level) 
                {

                    boundary_pixes[neighbour_level].push_back(neighbour_pixel);
                    boundary_edges[neighbour_level].push_back(0);
                    
                    // if neighbour level is lower than our threshold_level set threshold_level to neighbour level
                    if (neighbour_level < threshold_level)
                        threshold_level = neighbour_level;

                } 
                else // if neighbour level is lower than current add current_pixel (and next edge) 
                     // to the boundary heap for later processing
                {
                    
                    boundary_pixes[current_level].push_back(current_pixel);
                    boundary_edges[current_level].push_back(current_edge + 1);
                    
                    // if neighbour level is lower than threshold_level set threshold_level to neighbour level
                    if (current_level < threshold_level)
                        threshold_level = current_level;
                    
                    // consider the new pixel and its grey-level as current pixel
                    current_pixel = neighbour_pixel;
                    current_edge = 0;
                    current_level = neighbour_level;
                    
                    // and push a new component
                    push_new_component = true;
                    break; 
                }
            }
                
        } // else neigbor was already accessible

        if (push_new_component) continue;


        // once here we can add the current pixel to the component at the top of the stack
        // but first we find how many of its neighbours are part of the region boundary (needed for
        // perimeter and crossings calc.) and the increment in quads counts for euler number calc.
        int non_boundary_neighbours = 0;
        int non_boundary_neighbours_horiz = 0;

        unsigned char quad_before[4] = {0,0,0,0};
        unsigned char quad_after[4] = {0,0,0,0};
        quad_after[0] = 1<<1;
        quad_after[1] = 1<<3;
        quad_after[2] = 1<<2;
        quad_after[3] = 1;

        for (int edge = 0; edge < 8; edge++) 
        {
            int neighbour4 = -1;
            int neighbour8 = -1;
            int cell = 0;
            switch (edge) 
            {
                    case 0: if (x < width - 1) { neighbour4 = neighbour8 = current_pixel + 1;} cell = 5; break;
                    case 1: if ((x < width - 1)&&(y < height - 1)) { neighbour8 = current_pixel + 1 + width;} cell = 8; break;
                    case 2: if (y < height - 1) { neighbour4 = neighbour8 = current_pixel + width;} cell = 7; break;
                    case 3: if ((x > 0)&&(y < height - 1)) { neighbour8 = current_pixel - 1 + width;} cell = 6; break;
                    case 4: if (x > 0) { neighbour4 = neighbour8 = current_pixel - 1;} cell = 3; break;
                    case 5: if ((x > 0)&&(y > 0)) { neighbour8 = current_pixel - 1 - width;} cell = 0; break;
                    case 6: if (y > 0) { neighbour4 = neighbour8 = current_pixel - width;} cell = 1; break;
                    default: if ((x < width - 1)&&(y > 0)) { neighbour8 = current_pixel + 1 - width;} cell = 2; break;
            }
            if ((neighbour4 != -1)&&(accumulated_pixel_mask[neighbour4])&&(image_data[neighbour4]<=image_data[current_pixel]))
            {
                non_boundary_neighbours++;
                if ((edge == 0) || (edge == 4))
                    non_boundary_neighbours_horiz++;
            }

            int pix_value = image_data[current_pixel] + 1;
            if (neighbour8 != -1)
            {
                if (accumulated_pixel_mask[neighbour8])
                    pix_value = image_data[neighbour8];
            } 

            if (pix_value<=image_data[current_pixel])
            {
                switch(cell)
                {
                    case 0:
                        quad_before[3] = quad_before[3] | (1<<3);
                        quad_after[3]  = quad_after[3]  | (1<<3);
                        break;
                    case 1:
                        quad_before[3] = quad_before[3] | (1<<2);
                        quad_after[3]  = quad_after[3]  | (1<<2);
                        quad_before[0] = quad_before[0] | (1<<3);
                        quad_after[0]  = quad_after[0]  | (1<<3);
                        break;
                    case 2:
                        quad_before[0] = quad_before[0] | (1<<2);
                        quad_after[0]  = quad_after[0]  | (1<<2);
                        break;
                    case 3:
                        quad_before[3] = quad_before[3] | (1<<1);
                        quad_after[3]  = quad_after[3]  | (1<<1);
                        quad_before[2] = quad_before[2] | (1<<3);
                        quad_after[2]  = quad_after[2]  | (1<<3);
                        break;
                    case 5:
                        quad_before[0] = quad_before[0] | (1);
                        quad_after[0]  = quad_after[0]  | (1);
                        quad_before[1] = quad_before[1] | (1<<2);
                        quad_after[1]  = quad_after[1]  | (1<<2);
                        break;
                    case 6:
                        quad_before[2] = quad_before[2] | (1<<1);
                        quad_after[2]  = quad_after[2]  | (1<<1);
                        break;
                    case 7:
                        quad_before[2] = quad_before[2] | (1);
                        quad_after[2]  = quad_after[2]  | (1);
                        quad_before[1] = quad_before[1] | (1<<1);
                        quad_after[1]  = quad_after[1]  | (1<<1);
                        break;
                    default:
                        quad_before[1] = quad_before[1] | (1);
                        quad_after[1]  = quad_after[1]  | (1);
                        break;
                }
            }

        }

        int C_before[3] = {0, 0, 0};
        int C_after[3] = {0, 0, 0};

        for (int p=0; p<3; p++)
        {
            for (int q=0; q<4; q++)
            {
                if ( (quad_before[0] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if ( (quad_before[1] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;
                if ( (quad_before[2] == quads[p][q]) && ((p<2)||(q<2)) ) 
                    C_before[p]++;
                if ( (quad_before[3] == quads[p][q]) && ((p<2)||(q<2)) )
                    C_before[p]++;

                if ( (quad_after[0] == quads[p][q]) && ((p<2)||(q<2)) ) 
                    C_after[p]++;
                if ( (quad_after[1] == quads[p][q]) && ((p<2)||(q<2)) ) 
                    C_after[p]++;
                if ( (quad_after[2] == quads[p][q]) && ((p<2)||(q<2)) ) 
                    C_after[p]++;
                if ( (quad_after[3] == quads[p][q]) && ((p<2)||(q<2)) ) 
                    C_after[p]++;
            }
        }

        int d_C1 = C_after[0]-C_before[0];
        int d_C2 = C_after[1]-C_before[1];
        int d_C3 = C_after[2]-C_before[2];

        er_add_pixel(er_stack.back(), x, y, non_boundary_neighbours, non_boundary_neighbours_horiz, d_C1, d_C2, d_C3);
        accumulated_pixel_mask[current_pixel] = true;
        
        // if we have processed all the possible threshold levels (the hea is empty) we are done!
        if (threshold_level == (255/thresholdDelta)+1) 
        {

            // save the extracted regions into the output vector
            regions->reserve(num_accepted_regions+1);
            er_save(er_stack.back(), NULL, NULL);

            // clean memory
            er_tree_clean(er_stack.back());
            er_stack.clear();

            return;
        }
            
        
        // pop the heap of boundary pixels
        current_pixel = boundary_pixes[threshold_level].back();
        boundary_pixes[threshold_level].erase(boundary_pixes[threshold_level].end()-1);
        current_edge  = boundary_edges[threshold_level].back();
        boundary_edges[threshold_level].erase(boundary_edges[threshold_level].end()-1);
        
        while (boundary_pixes[threshold_level].empty() && (threshold_level < (255/thresholdDelta)+1))
            threshold_level++;

        
        int new_level = image_data[current_pixel];

        // if the new pixel has higher grey value than the current one
        if (new_level != current_level) {

            current_level = new_level;

            // process components on the top of the stack until we reach the higher grey-level
            while (er_stack.back()->level < new_level)
            {
                ERStat* er = er_stack.back();
                er_stack.erase(er_stack.end()-1);
                
                if (new_level < er_stack.back()->level) 
                {
                    er_stack.push_back(new ERStat(new_level, current_pixel, current_pixel%width, current_pixel/width));
                    er_merge(er_stack.back(), er); 
                    break;
                }

                er_merge(er_stack.back(), er);
            }

        }

    }
}

// accumulate a pixel into an ER
void ERFilterNM::er_add_pixel(ERStat *parent, int x, int y, int non_border_neighbours, 
                                                            int non_border_neighbours_horiz, 
                                                            int d_C1, int d_C2, int d_C3)
{
    parent->area++;
    parent->perimeter += 4 - 2*non_border_neighbours;

    if (parent->crossings->size()>0)
    {
        if (y<parent->rect.y) parent->crossings->push_front(2);
        else if (y>parent->rect.br().y-1) parent->crossings->push_back(2);
        else {
            parent->crossings->at(y - parent->rect.y) += 2-2*non_border_neighbours_horiz;
        }
    } else {
        parent->crossings->push_back(2);
    }

    parent->euler += (d_C1 - d_C2 + 2*d_C3) / 4;

    int new_x1 = min(parent->rect.x,x);
    int new_y1 = min(parent->rect.y,y);
    int new_x2 = max(parent->rect.br().x-1,x);
    int new_y2 = max(parent->rect.br().y-1,y);
    parent->rect.x = new_x1;
    parent->rect.y = new_y1;
    parent->rect.width  = new_x2-new_x1+1;
    parent->rect.height = new_y2-new_y1+1;

    parent->raw_moments[0] += x;
    parent->raw_moments[1] += y;

    parent->central_moments[0] += x * x;
    parent->central_moments[1] += x * y;
    parent->central_moments[2] += y * y;
}

// merge an ER with its nested parent
void ERFilterNM::er_merge(ERStat *parent, ERStat *child)
{

    parent->area += child->area;

    parent->perimeter += child->perimeter;
    

    for (int i=parent->rect.y; i<=min(parent->rect.br().y-1,child->rect.br().y-1); i++)
        if (i-child->rect.y >= 0)
            parent->crossings->at(i-parent->rect.y) += child->crossings->at(i-child->rect.y);

    for (int i=parent->rect.y-1; i>=child->rect.y; i--)
        if (i-child->rect.y < (int)child->crossings->size())
            parent->crossings->push_front(child->crossings->at(i-child->rect.y));
        else 
            parent->crossings->push_front(0);

    for (int i=parent->rect.br().y; i<child->rect.y; i++)
        parent->crossings->push_back(0);
        
    for (int i=max(parent->rect.br().y,child->rect.y); i<=child->rect.br().y-1; i++)
        parent->crossings->push_back(child->crossings->at(i-child->rect.y));

    parent->euler += child->euler;

    int new_x1 = min(parent->rect.x,child->rect.x);
    int new_y1 = min(parent->rect.y,child->rect.y);
    int new_x2 = max(parent->rect.br().x-1,child->rect.br().x-1);
    int new_y2 = max(parent->rect.br().y-1,child->rect.br().y-1);
    parent->rect.x = new_x1;
    parent->rect.y = new_y1;
    parent->rect.width  = new_x2-new_x1+1;
    parent->rect.height = new_y2-new_y1+1;

    parent->raw_moments[0] += child->raw_moments[0];
    parent->raw_moments[1] += child->raw_moments[1];

    parent->central_moments[0] += child->central_moments[0];
    parent->central_moments[1] += child->central_moments[1];
    parent->central_moments[2] += child->central_moments[2];

    vector<int> m_crossings;
    m_crossings.push_back(child->crossings->at((int)(child->rect.height)/6));
    m_crossings.push_back(child->crossings->at((int)3*(child->rect.height)/6));
    m_crossings.push_back(child->crossings->at((int)5*(child->rect.height)/6));
    std::sort(m_crossings.begin(), m_crossings.end());
    child->med_crossings = (float)m_crossings.at(1);

    // free unnecessary mem 
    child->crossings->clear(); 
    delete(child->crossings);
    child->crossings = NULL;

    // recover the original grey-level
    child->level = child->level*thresholdDelta;

    // before saving calculate P(child|character) and filter if possible
    if (classifier != NULL)
    {
        child->probability = classifier->eval(*child);
    }

    if ( ((classifier!=NULL)?(child->probability >= minProbability):true) && 
         ((child->area >= (minArea*region_mask.rows*region_mask.cols)) && 
          (child->area <= (maxArea*region_mask.rows*region_mask.cols))) )
    {

        num_accepted_regions++;

        child->next = parent->child;
        if (parent->child) 
            parent->child->prev = child;
        parent->child = child;
        child->parent = parent;

    } else {

        num_rejected_regions++;

        if (child->prev !=NULL)
            child->prev->next = child->next;

        ERStat *new_child = child->child;
        if (new_child != NULL)
        {
            while (new_child->next != NULL)
                new_child = new_child->next;
            new_child->next = parent->child;
            if (parent->child) 
                parent->child->prev = new_child;
            parent->child   = child->child;
            child->child->parent = parent;
        }

        // free mem
        if(child->crossings)
        {
            child->crossings->clear();
            delete(child->crossings);
            child->crossings = NULL;
        }
        delete(child);
    } 
    
}

// recursively walk the tree and clean memory
void ERFilterNM::er_tree_clean( ERStat *stat )
{
        for (ERStat * child = stat->child; child; child = child->next)
        {
            er_tree_clean(child);
        }
        if (stat->crossings)
        {
            stat->crossings->clear();
            delete(stat->crossings);
            stat->crossings = NULL;
        }
        delete stat;
}
    
// copy extracted regions into the output vector
ERStat* ERFilterNM::er_save( ERStat *er, ERStat *parent, ERStat *prev )
{
    
    regions->push_back(*er);

    regions->back().parent = parent;
    if (prev != NULL)
      prev->next = &(regions->back());
    else if (parent != NULL)
      parent->child = &(regions->back());

    ERStat *old_prev = NULL;
    ERStat *this_er  = &regions->back();

    if (nonMaxSuppression)
    {
        if (this_er->parent == NULL)
        {
            this_er->probability = 0; //TODO this makes sense in order to select at least one region in short tree's but is it really necessary?
            this_er->max_probability_ancestor = this_er;
            this_er->min_probability_ancestor = this_er;
        } 
        else
        {
            this_er->max_probability_ancestor = (this_er->probability > parent->max_probability_ancestor->probability)? this_er :  parent->max_probability_ancestor;

            this_er->min_probability_ancestor = (this_er->probability < parent->min_probability_ancestor->probability)? this_er :  parent->min_probability_ancestor;

            if ( (this_er->max_probability_ancestor->probability > minProbability) && (this_er->max_probability_ancestor->probability - this_er->min_probability_ancestor->probability > minProbabilityDiff))
            {

                this_er->max_probability_ancestor->local_maxima = true;
                //TODO check here if the last local_maxima can be also suppressed, is the following correct?
                //if (this_er->min_probability_ancestor->local_maxima)
                //  this_er->min_probability_ancestor->local_maxima = false;

                this_er->max_probability_ancestor = this_er;
                this_er->min_probability_ancestor = this_er;		
            }
        }
    }
        
    for (ERStat * child = er->child; child; child = child->next)
    {
        old_prev = er_save(child, this_er, old_prev);
    }

    return this_er;
}

// recursively walk the tree and filter (remove) regions using the callback classifier
ERStat* ERFilterNM::er_tree_filter ( InputArray image, ERStat * stat, ERStat *parent, ERStat *prev )
{
    Mat src = image.getMat();
    // assert correct image type
    CV_Assert( src.type() == CV_8UC1 );
    
    //Fill the region and calculate 2nd stage features
    Mat region = region_mask(Rect(Point(stat->rect.x,stat->rect.y),Point(stat->rect.br().x+2,stat->rect.br().y+2)));
    region = Scalar(0);
    int newMaskVal = 255;
    int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
    Rect rect;
    
    floodFill( src(Rect(Point(stat->rect.x,stat->rect.y),Point(stat->rect.br().x,stat->rect.br().y))), 
               region, Point(stat->pixel%src.cols - stat->rect.x, stat->pixel/src.cols - stat->rect.y), 
               Scalar(255), &rect, Scalar(stat->level), Scalar(0), flags );
    rect.width += 2;
    rect.height += 2;
    region = region(rect);

    vector<vector<Point> > contours;
    vector<Point> contour_poly;
    vector<Vec4i> hierarchy;
    findContours( region, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0) );
    //TODO check epsilon parameter of approxPolyDP (set empirically) : we want more precission 
    //     if the region is very small because otherwise we'll loose all the convexities
    approxPolyDP( Mat(contours[0]), contour_poly, max(rect.width,rect.height)/25, true ); 


    bool was_convex = false;
    int  num_inflexion_points = 0;

    for (int p = 0 ; p<(int)contour_poly.size(); p++)
    {
        int p_prev = p-1;
        int p_next = p+1;
        if (p_prev == -1)
            p_prev = contour_poly.size()-1;
        if (p_next == (int)contour_poly.size())
            p_next = 0;

        double angle_next = atan2((double)(contour_poly[p_next].y-contour_poly[p].y),
                                  (double)(contour_poly[p_next].x-contour_poly[p].x));
        double angle_prev = atan2((double)(contour_poly[p_prev].y-contour_poly[p].y),
                                  (double)(contour_poly[p_prev].x-contour_poly[p].x));
        if ( angle_next < 0 )
            angle_next = 2.*CV_PI + angle_next;

        double angle = (angle_next - angle_prev);
        if (angle > 2.*CV_PI)
            angle = angle - 2.*CV_PI;
        else if (angle < 0)
            angle = 2.*CV_PI + std::abs(angle);

        if (p>0)
        {
            if ( ((angle > CV_PI)&&(!was_convex)) || ((angle < CV_PI)&&(was_convex)) )
                num_inflexion_points++;
        }
        was_convex = (angle > CV_PI);

    }

    floodFill(region, Point(0,0), Scalar(255), 0);
    int holes_area = region.cols*region.rows-countNonZero(region);

    int hull_area = 0;

    {

        vector<Point> hull;
        convexHull(contours[0], hull, false);
        hull_area = (int)contourArea(hull);
    }


    stat->hole_area_ratio = (float)holes_area / stat->area;
    stat->convex_hull_ratio = (float)hull_area / (float)contourArea(contours[0]);
    stat->num_inflexion_points = (float)num_inflexion_points;


    // calculate P(child|character) and filter if possible
    if ( (classifier != NULL) && (stat->parent != NULL) )
    {
        stat->probability = classifier->eval(*stat);
    } 

    if ( ( ((classifier != NULL)?(stat->probability >= minProbability):true) && 
          ((stat->area >= minArea*region_mask.rows*region_mask.cols) && 
           (stat->area <= maxArea*region_mask.rows*region_mask.cols)) ) || 
        (stat->parent == NULL) )
    {

        num_accepted_regions++;
        regions->push_back(*stat);

        regions->back().parent = parent;
        regions->back().next   = NULL;
        regions->back().child  = NULL;

        if (prev != NULL)
            prev->next = &(regions->back());
        else if (parent != NULL)
            parent->child = &(regions->back());

        ERStat *old_prev = NULL;
        ERStat *this_er  = &regions->back();

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_filter(image, child, this_er, old_prev);
        }

        return this_er;

    } else {

        num_rejected_regions++;

        ERStat *old_prev = prev;

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_filter(image, child, parent, old_prev);
        }

        return old_prev;
    }

}

// recursively walk the tree selecting only regions with local maxima probability
ERStat* ERFilterNM::er_tree_nonmax_suppression ( ERStat * stat, ERStat *parent, ERStat *prev )
{

    if ( ( stat->local_maxima ) || ( stat->parent == NULL ) )
    {

        regions->push_back(*stat);

        regions->back().parent = parent;
        regions->back().next   = NULL;
        regions->back().child  = NULL;

        if (prev != NULL)
            prev->next = &(regions->back());
        else if (parent != NULL)
            parent->child = &(regions->back());

        ERStat *old_prev = NULL;
        ERStat *this_er  = &regions->back();

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_nonmax_suppression( child, this_er, old_prev );
        }

        return this_er;

    } else {

        num_rejected_regions++;
        num_accepted_regions--;

        ERStat *old_prev = prev;

        for (ERStat * child = stat->child; child; child = child->next)
        {
            old_prev = er_tree_nonmax_suppression( child, parent, old_prev );
        }

        return old_prev;
    }

}

void ERFilterNM::setCallback(const Ptr<ERFilter::Callback>& cb)
{
    classifier = cb;
};

void ERFilterNM::setMinArea(float _minArea)
{
    CV_Assert( (_minArea >= 0) && (_minArea < maxArea) );
    minArea = _minArea;
    return;
};

void ERFilterNM::setMaxArea(float _maxArea)
{
    CV_Assert(_maxArea <= 1);
    CV_Assert(minArea < _maxArea);
    maxArea = _maxArea;
    return;
};

void ERFilterNM::setThresholdDelta(int _thresholdDelta)
{
    CV_Assert( (_thresholdDelta > 0) && (_thresholdDelta <= 128) );
    thresholdDelta = _thresholdDelta;
    return;
};

void ERFilterNM::setMinProbability(float _minProbability)
{
    CV_Assert( (_minProbability >= 0.0) && (_minProbability <= 1.0) );
    minProbability = _minProbability;
    return;
};

void ERFilterNM::setMinProbabilityDiff(float _minProbabilityDiff)
{
    CV_Assert( (_minProbabilityDiff >= 0.0) && (_minProbabilityDiff <= 1.0) );
    minProbabilityDiff = _minProbabilityDiff;
    return;
};

void ERFilterNM::setNonMaxSuppression(bool _nonMaxSuppression)
{
    nonMaxSuppression = _nonMaxSuppression;
    return;
};

int ERFilterNM::getNumRejected()
{
    return num_rejected_regions;
};




// load default 1st stage classifier if found
ERClassifierNM1::ERClassifierNM1()
{

    if (ifstream("./trained_classifierNM1.xml")) 
    {
        // The file with default classifier exists
        boost.load("./trained_classifierNM1.xml", "boost");
    } 
    else if (ifstream("./training/trained_classifierNM1.xml")) 
    {
        // The file with default classifier exists
        boost.load("./training/trained_classifierNM1.xml", "boost");
    } 
    else 
    {
        // File not found 
        CV_Error(CV_StsBadArg, "Default classifier ./trained_classifierNM1.xml not found!");
    }
};

double ERClassifierNM1::eval(const ERStat& stat)
{
    //Classify
    float arr[] = {0,(float)(stat.rect.width)/(stat.rect.height), // aspect ratio
                     sqrt((float)(stat.area))/stat.perimeter, // compactness
                     (float)(1-stat.euler), //number of holes
                     stat.med_crossings};

    vector<float> sample (arr, arr + sizeof(arr) / sizeof(arr[0]) );

    float votes = boost.predict( Mat(sample), Mat(), Range::all(), false, true );

    // Logistic Correction returns a probability value (in the range(0,1))
    return (double)1-(double)1/(1+exp(-2*votes));
};


// load default 2nd stage classifier if found
ERClassifierNM2::ERClassifierNM2()
{

    if (ifstream("./trained_classifierNM2.xml")) 
    {
        // The file with default classifier exists
        boost.load("./trained_classifierNM2.xml", "boost");
    } 
    else if (ifstream("./training/trained_classifierNM2.xml")) 
    {
        // The file with default classifier exists
        boost.load("./training/trained_classifierNM2.xml", "boost");
    } 
    else 
    {
        // File not found 
        CV_Error(CV_StsBadArg, "Default classifier ./trained_classifierNM2.xml not found!");
    }
};

double ERClassifierNM2::eval(const ERStat& stat)
{
    //Classify
    float arr[] = {0,(float)(stat.rect.width)/(stat.rect.height), // aspect ratio
                     sqrt((float)(stat.area))/stat.perimeter, // compactness
                     (float)(1-stat.euler), //number of holes
                     stat.med_crossings, stat.hole_area_ratio, 
                     stat.convex_hull_ratio, stat.num_inflexion_points};

    vector<float> sample (arr, arr + sizeof(arr) / sizeof(arr[0]) );

    float votes = boost.predict( Mat(sample), Mat(), Range::all(), false, true );

    // Logistic Correction returns a probability value (in the range(0,1))
    return (double)1-(double)1/(1+exp(-2*votes));
};


/*!
    Create an Extremal Region Filter for the 1st stage classifier of N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
 
    The component tree of the image is extracted by a threshold increased step by step
    from 0 to 255, incrementally computable descriptors (aspect_ratio, compactness, 
    number of holes, and number of horizontal crossings) are computed for each ER 
    and used as features for a classifier which estimates the class-conditional 
    probability P(er|character). The value of P(er|character) is tracked using the inclusion 
    relation of ER across all thresholds and only the ERs which correspond to local maximum 
    of the probability P(er|character) are selected (if the local maximum of the
    probability is above a global limit pmin and the difference between local maximum and 
    local minimum is greater than minProbabilityDiff).

    \param  cb                Callback with the classifier.
                              if omitted tries to load a default classifier from file trained_classifierNM1.xml
    \param  thresholdDelta    Threshold step in subsequent thresholds when extracting the component tree
    \param  minArea           The minimum area (% of image size) allowed for retreived ER's
    \param  minArea           The maximum area (% of image size) allowed for retreived ER's
    \param  minProbability    The minimum probability P(er|character) allowed for retreived ER's
    \param  nonMaxSuppression Whenever non-maximum suppression is done over the branch probabilities  
    \param  minProbability    The minimum probability difference between local maxima and local minima ERs
*/
Ptr<ERFilter> createERFilterNM1(const Ptr<ERFilter::Callback>& cb, int thresholdDelta, 
                                float minArea, float maxArea, float minProbability, 
                                bool nonMaxSuppression, float minProbabilityDiff)
{

    CV_Assert( (minProbability >= 0.) && (minProbability <= 1.) );
    CV_Assert( (minArea < maxArea) && (minArea >=0.) && (maxArea <= 1.) );
    CV_Assert( (thresholdDelta >= 0) && (thresholdDelta <= 128) );
    CV_Assert( (minProbabilityDiff >= 0.) && (minProbabilityDiff <= 1.) );

    Ptr<ERFilterNM> filter = new ERFilterNM();
    
    if (cb == NULL)
        filter->setCallback(new ERClassifierNM1());
    else
        filter->setCallback(cb);

    filter->setThresholdDelta(thresholdDelta);
    filter->setMinArea(minArea);
    filter->setMaxArea(maxArea);
    filter->setMinProbability(minProbability);
    filter->setNonMaxSuppression(nonMaxSuppression);
    filter->setMinProbabilityDiff(minProbabilityDiff);
    return (Ptr<ERFilter>)filter;
}

/*!
    Create an Extremal Region Filter for the 2nd stage classifier of N&M algorithm
    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012

    In the second stage, the ERs that passed the first stage are classified into character 
    and non-character classes using more informative but also more computationally expensive
    features. The classifier uses all the features calculated in the first stage and the following 
    additional features: hole area ratio, convex hull ratio, and number of outer inflexion points.

    \param  cb             Callback with the classifier
                           if omitted tries to load a default classifier from file trained_classifierNM2.xml
    \param  minProbability The minimum probability P(er|character) allowed for retreived ER's
*/
Ptr<ERFilter> createERFilterNM2(const Ptr<ERFilter::Callback>& cb, float minProbability)
{

    CV_Assert( (minProbability >= 0.) && (minProbability <= 1.) );

    Ptr<ERFilterNM> filter = new ERFilterNM();

    
    if (cb == NULL)
        filter->setCallback(new ERClassifierNM2());
    else
        filter->setCallback(cb);

    filter->setMinProbability(minProbability);
    return (Ptr<ERFilter>)filter;
}

}
