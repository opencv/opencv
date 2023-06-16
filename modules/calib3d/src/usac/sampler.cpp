// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
/*
* Uniform Sampler:
* Choose uniformly m (sample size) points from N (points size).
* Uses Fisher-Yates shuffle.
*/
class UniformSamplerImpl : public UniformSampler {
private:
    std::vector<int> points_random_pool;
    int sample_size, points_size = 0;
    RNG rng;
public:

    UniformSamplerImpl (int state, int sample_size_, int points_size_)
        : rng(state)
    {
        sample_size = sample_size_;
        setPointsSize (points_size_);
    }
    void setNewPointsSize (int points_size_) override {
        setPointsSize(points_size_);
    }
    void generateSample (std::vector<int> &sample) override {
        int random_pool_size = points_size; // random points of entire range
        for (int i = 0; i < sample_size; i++) {
            // get random point index
            const int array_random_index = rng.uniform(0, random_pool_size);
            // get point by random index
            // store sample
            sample[i] = points_random_pool[array_random_index];
            // swap random point with the end of random pool
            std::swap(points_random_pool[array_random_index],
                      points_random_pool[--random_pool_size]);
        }
    }
private:
    void setPointsSize (int points_size_) {
        CV_Assert (sample_size <= points_size_);

        if (points_size_ > points_size)
            points_random_pool = std::vector<int>(points_size_);

        if (points_size != points_size_) {
            points_size  = points_size_;

            for (int i = 0; i < points_size; i++)
                points_random_pool[i] = i;
        }
    }
};
Ptr<UniformSampler> UniformSampler::create(int state, int sample_size_, int points_size_) {
    return makePtr<UniformSamplerImpl>(state, sample_size_, points_size_);
}

/////////////////////////////////// PROSAC (SIMPLE) SAMPLER ///////////////////////////////////////
/*
* PROSAC (simple) sampler does not use array of precalculated T_n (n is subset size) samples, but computes T_n for
* specific n directy in generateSample() function.
* Also, the stopping length (or maximum subset size n*) by default is set to points_size (N) and does not updating
* during computation.
*/
class ProsacSimpleSamplerImpl : public ProsacSimpleSampler {
protected:
    int points_size, subset_size, t_n_prime, kth_sample_number,
        max_prosac_samples_count, largest_sample_size, sample_size;
    double t_n;
    Ptr<UniformRandomGenerator> random_gen;
public:
    ProsacSimpleSamplerImpl (int state, int points_size_, int sample_size_,
            int max_prosac_samples_count_) : random_gen(UniformRandomGenerator::create(state)) {

        CV_Assert(sample_size_ <= points_size_);
        sample_size = sample_size_;
        points_size = points_size_;
        max_prosac_samples_count = max_prosac_samples_count_;
        initialize ();
    }

    void generateSample (std::vector<int> &sample) override {
        if (kth_sample_number > max_prosac_samples_count) {
            // do uniform sampling, if prosac has not found solution
            random_gen->generateUniqueRandomSet(sample, sample_size, points_size);
            return;
        }

        kth_sample_number++; // t := t + 1

        // Choice of the hypothesis generation set
        if (kth_sample_number >= t_n_prime && subset_size < largest_sample_size) {
            // do not use array of growth sample, calculate it directly
            double t_n_plus1 = (subset_size + 1) * t_n / (subset_size + 1 - sample_size);
            t_n_prime += static_cast<int>(ceil(t_n_plus1 - t_n));
            t_n = t_n_plus1;
            subset_size++;
        }

        // Semi-random sample Mt of size m
        if (t_n_prime < kth_sample_number) {
            random_gen->generateUniqueRandomSet(sample, sample_size, subset_size);
        } else {
            random_gen->generateUniqueRandomSet(sample, sample_size-1, subset_size-1);
            sample[sample_size-1] = subset_size-1; // Make the last point from the nth position.
        }
    }

    // Set the sample such that you are sampling the kth prosac sample (Eq. 6).
    void setSampleNumber (int k) {
        kth_sample_number = k;

        // If the method should act exactly like RANSAC
        if (kth_sample_number > max_prosac_samples_count)
            return;
        else { // Increment the size of the sampling pool while required
            t_n = max_prosac_samples_count;
            t_n_prime = 1; // reset growth function
            subset_size = sample_size; // reset subset size as from the beginning
            for (int i = 0; i < sample_size; i++)
                t_n *= static_cast<double>(subset_size - i) / (points_size - i);

            while (kth_sample_number > t_n_prime) { // t_n_prime == growth_function
                double t_n_plus1 = static_cast<double>(subset_size + 1) * t_n / (subset_size + 1 - sample_size);
                t_n_prime += static_cast<int>(ceil(t_n_plus1 - t_n));
                t_n = t_n_plus1;
                subset_size++;
            }
            if (subset_size > points_size)
                subset_size = points_size;
        }
    }

    void setNewPointsSize (int points_size_) override {
        CV_Assert(sample_size <= points_size_);
        points_size = points_size_;
        initialize ();
    }
private:
    void initialize () {
        largest_sample_size = points_size; // termination length, n*
        subset_size = sample_size; // n
        t_n = max_prosac_samples_count;
        t_n_prime = 1;

        // From Equations leading up to Eq 3 in Chum et al.
        // t_n samples containing only data points from U_n and
        // t_n+1 samples containing only data points from U_n+1
        for (int i = 0; i < sample_size; i++)
            t_n *= static_cast<double>(subset_size - i) / (points_size - i);

        kth_sample_number = 0;
    }
};
Ptr<ProsacSimpleSampler> ProsacSimpleSampler::create(int state, int points_size_, int sample_size_,
               int max_prosac_samples_count_) {
    return makePtr<ProsacSimpleSamplerImpl>(state, points_size_, sample_size_,
            max_prosac_samples_count_);
}

////////////////////////////////////// PROSAC SAMPLER ////////////////////////////////////////////
class ProsacSamplerImpl : public ProsacSampler {
protected:
    std::vector<int> growth_function;

    // subset_size = size of sampling range (subset of good sorted points)
    // termination_length = n*, maximum sampling range (largest subset size)
    int points_size, sample_size, subset_size, termination_length;

    // it is T_N
    // Imagine standard RANSAC drawing T_N samples of size m out of N data points
    // In our experiments, the parameter was set to T_N = 200000
    int growth_max_samples;

    // how many time PROSAC generateSample() was called
    int kth_sample_number;
    Ptr<UniformRandomGenerator> random_gen;
public:
    void setTerminationLength (int termination_length_) override {
        termination_length = termination_length_;
    }

    // return constant reference to prosac termination criteria
    int getKthSample () const override {
        return kth_sample_number;
    }

    // return constant points of growth function to prosac termination criteria
    const std::vector<int> & getGrowthFunction () const override {
        return growth_function;
    }

    ProsacSamplerImpl (int state, int points_size_, int sample_size_,
            int growth_max_samples_) : random_gen(UniformRandomGenerator::create(state)) {
        CV_Assert(sample_size_ <= points_size_);

        sample_size = sample_size_;
        points_size = points_size_;

        growth_max_samples = growth_max_samples_;
        growth_function = std::vector<int>(points_size);

        kth_sample_number = 0;

        // The data points in U_N are sorted in descending order w.r.t. the quality function q.
        // Let {Mi}i = 1...T_N denote the sequence of samples Mi c U_N that are uniformly drawn by Ransac.

        // Let T_n be an average number of samples from {Mi}i=1...T_N that contain data points from U_n only.
        // compute initial value for T_n
        //                                  n - i
        // T_n = T_N * Product i = 0...m-1 -------, n >= sample size, N = points size
        //                                  N - i
        double T_n = growth_max_samples;
        for (int i = 0; i < sample_size; i++)
            T_n *= static_cast<double> (sample_size-i) / (points_size-i);

        int T_n_prime = 1;

        // fill growth function with T'_n until sample_size
        for (int n = 0; n < sample_size; n++)
            growth_function[n] =  T_n_prime;

        // compute values using recurrent relation
        //             n + 1
        // T(n+1) = --------- T(n), m is sample size.
        //           n + 1 - m

        // growth function is defined as
        // g(t) = min {n, T'_(n) >= t}
        // T'_(n+1) = T'_(n) + (T_(n+1) - T_(n))
        // T'_m = 1
        for (int n = sample_size; n < points_size; n++) {
            double Tn_plus1 = static_cast<double>(n + 1) * T_n / (n + 1 - sample_size);
            growth_function[n] = T_n_prime + (int) ceil(Tn_plus1 - T_n); // T'_{n+1}

            // update
            T_n = Tn_plus1;
            T_n_prime = growth_function[n]; // T'_{n+1}
        }

        // other initializations
        termination_length = points_size; // n* = N, largest set sampled in PROSAC (termination length)
        subset_size = sample_size;      // n,  size of the current sampling pool
        kth_sample_number = 0; // t (iteration)
    }

    void generateSample (std::vector<int> &sample) override {
        if (kth_sample_number > growth_max_samples) {
            // if PROSAC has not converged to solution then do uniform sampling.
            random_gen->generateUniqueRandomSet(sample, sample_size, points_size);
            return;
        }

        kth_sample_number++; // t := t + 1

        // Choice of the hypothesis generation set
        // if (t = T'_n) & (n < n*) then n = n + 1 (eqn. 4)
        if (kth_sample_number >= growth_function[subset_size-1] && subset_size < termination_length)
            subset_size++;

        // Semi-random sample M_t of size m
        // if T'n < t   then
        if (growth_function[subset_size-1] < kth_sample_number) {
            if (subset_size >= termination_length) {
                random_gen->generateUniqueRandomSet(sample, sample_size, subset_size);
            } else {
                // The sample contains m-1 points selected from U_(n-1) at random and u_n
                random_gen->generateUniqueRandomSet(sample, sample_size-1, subset_size-1);
                sample[sample_size-1] = subset_size-1;
            }
        } else {
            // Select m points from U_n at random.
            random_gen->generateUniqueRandomSet(sample, sample_size, subset_size);
        }
    }

    // Set the sample such that you are sampling the kth prosac sample (Eq. 6).
    void setSampleNumber (int k) {
        kth_sample_number = k;

        // If the method should act exactly like RANSAC
        if (kth_sample_number > growth_max_samples)
            return;
        else { // Increment the size of the sampling pool while required
            subset_size = sample_size; // reset subset size as from the beginning
            while (kth_sample_number > growth_function[subset_size-1]) {
                subset_size++;
                if (subset_size >= points_size){
                    subset_size = points_size;
                    break;
                }
            }
            if (termination_length < subset_size)
                termination_length = subset_size;
        }
    }

    void setNewPointsSize (int /*points_size_*/) override {
        CV_Error(cv::Error::StsError, "Changing points size in PROSAC requires to change also "
                    "termination criteria! Use PROSAC simpler version");
    }
};

Ptr<ProsacSampler> ProsacSampler::create(int state, int points_size_, int sample_size_,
                             int growth_max_samples_) {
    return makePtr<ProsacSamplerImpl>(state, points_size_, sample_size_, growth_max_samples_);
}

////////////////////////////////////// P-NAPSAC SAMPLER ////////////////////////////////////////////
class ProgressiveNapsacImpl : public ProgressiveNapsac {
private:
    int max_progressive_napsac_iterations, points_size;
    // how many times generateSample() was called.
    int kth_sample_number, grid_layers_number, sample_size, sampler_length;

    const Ptr<UniformRandomGenerator> random_generator;
    ProsacSamplerImpl one_point_prosac, prosac_sampler;

    // The overlapping neighborhood layers
    const std::vector<Ptr<NeighborhoodGraph>> * layers;

    std::vector<int> growth_function;
    std::vector<int> hits_per_point; // number of iterations, t
    std::vector<int> subset_size_per_point; // k
    std::vector<int> current_layer_per_point; // layer of grid neighborhood graph
public:

    // points must be sorted
    ProgressiveNapsacImpl (int state,int points_size_, int sample_size_,
            const std::vector<Ptr<NeighborhoodGraph>> &layers_, int sampler_length_) :
            // initialize one-point prosac sampler and global prosac sampler
            random_generator (UniformRandomGenerator::create(state)),
            one_point_prosac (random_generator->getRandomNumber(INT_MAX), points_size_,
                        1 /* sample_size*/,points_size_),
            prosac_sampler (random_generator->getRandomNumber(INT_MAX), points_size_,
                        sample_size_, 200000), layers(&layers_) {
        CV_Assert(sample_size_ <= points_size_);
        sample_size = sample_size_;
        points_size = points_size_;
        sampler_length = sampler_length_;
        grid_layers_number = static_cast<int>(layers_.size());

        // Create growth function for P-NAPSAC
        growth_function = std::vector<int>(points_size);

        // 20 is sampler_length = The length of fully blending to global sampling
        max_progressive_napsac_iterations = sampler_length * points_size;

        const int local_sample_size = sample_size - 1; // not including initial point
        double T_n = max_progressive_napsac_iterations;
        for (int i = 0; i < local_sample_size; i++)
            T_n *= static_cast<double> (local_sample_size - i) / (points_size - i);

        // calculate growth function by recurrent relation (see PROSAC)
        int T_n_prime = 1;
        for (int n = 0; n < points_size; n++) {
            if (n + 1 <= local_sample_size) {
                growth_function[n] = T_n_prime;
                continue;
            }
            double Tn_plus1 = (n+1) * T_n / (n + 1 - local_sample_size);
            growth_function[n] = T_n_prime + static_cast<int>(ceil(Tn_plus1 - T_n));
            T_n = Tn_plus1;
            T_n_prime = growth_function[n];
        }

        subset_size_per_point = std::vector<int>(points_size, sample_size); // subset size
        hits_per_point = std::vector<int>(points_size, 0); // 0 hits
        current_layer_per_point = std::vector<int>(points_size, 0); // 0-th layer

        kth_sample_number = 0; // iteration
    }

    void generateSample (std::vector<int> &sample) override {
        // Do completely global sampling (PROSAC is used now), instead of Progressive NAPSAC,
        // if the maximum iterations has been done without finding the sought model.
        if (kth_sample_number > max_progressive_napsac_iterations) {
            prosac_sampler.generateSample(sample);
            return;
        }

        kth_sample_number++;

        // get PROSAC one-point sample (initial point)
        one_point_prosac.generateSample(sample);
        const int initial_point = sample[0];

        // get hits number and subset size (i.e., the size of the neighborhood sphere)
        // of initial point (note, get by reference)
        int &iters_of_init_pt = ++hits_per_point[initial_point]; // t := t + 1, increase iteration
        int &subset_size_of_init_pt = subset_size_per_point[initial_point];

        while (iters_of_init_pt > growth_function[subset_size_of_init_pt - 1] && subset_size_of_init_pt < points_size)
            subset_size_of_init_pt++;

        // Get layer of initial point (note, get by reference)
        int &current_layer = current_layer_per_point[initial_point];

        bool is_last_layer = false;
        do {// Try to find the grid which contains enough points
            // In the case when the grid with a single cell is used,
            // apply PROSAC.
            if (current_layer >= grid_layers_number) {
                is_last_layer = true;
                break;
            }

            // If there are not enough points in the cell, start using a
            // less fine grid.
            if ((int)layers->at(current_layer)->getNeighbors(initial_point).size() < subset_size_of_init_pt) {
                ++current_layer; // Jump to the next layer with bigger cells.
                continue;
            }
            // If the procedure got to this point, there is no reason to choose a different layer of grids
            // since the current one has enough points.
            break;
        } while (true);

        // If not the last layer has been chosen, sample from the neighbors of the initially selected point.
        if (!is_last_layer) {
            // The indices of the points which are in the same cell as the
            // initially selected one.
            const std::vector<int> &neighbors = layers->at(current_layer)->getNeighbors(initial_point);

            // Put the selected point to the end of the sample array to avoid
            // being overwritten when sampling the remaining points.
            sample[sample_size - 1] = initial_point;

            // The next point should be the farthest one from the initial point. Note that the points in the grid cell are
            // not ordered w.r.t. to their distances from the initial point. However, they are ordered as in PROSAC.
            sample[sample_size - 2] = neighbors[subset_size_of_init_pt - 1];

            // Select n - 2 points randomly
            random_generator->generateUniqueRandomSet(sample, sample_size - 2, subset_size_of_init_pt - 1);

            for (int i = 0; i < sample_size - 2; i++) {
                sample[i] = neighbors[sample[i]];  // Replace the neighbor index by the index of the point
                ++hits_per_point[sample[i]]; // Increase the hit number of each selected point
            }
            ++hits_per_point[sample[sample_size - 2]]; // Increase the hit number of each selected point
        }
            // If the last layer (i.e., the layer with a single cell) has been chosen, do global sampling
            // by PROSAC sampler.
        else {
            // last layer, all points are neighbors
            // If local sampling
            prosac_sampler.setSampleNumber(kth_sample_number);
            prosac_sampler.generateSample (sample);
            sample[sample_size - 1] = initial_point;
        }
    }

    void setNewPointsSize (int /*points_size_*/) override {
        CV_Error(cv::Error::StsError, "Changing points size requires changing neighborhood graph! "
                    "You must reinitialize P-NAPSAC!");
    }
};
Ptr<ProgressiveNapsac> ProgressiveNapsac::create(int state, int points_size_, int sample_size_,
        const std::vector<Ptr<NeighborhoodGraph>> &layers, int sampler_length_) {
    return makePtr<ProgressiveNapsacImpl>(state, points_size_, sample_size_,
                                          layers, sampler_length_);
}

////////////////////// N adjacent points sample consensus (NAPSAC) SAMPLER ////////////////////////
class NapsacSamplerImpl : public NapsacSampler {
private:
    const Ptr<NeighborhoodGraph> neighborhood_graph;
    const Ptr<UniformRandomGenerator> random_generator;
    bool do_uniform = false;
    std::vector<int> points_large_neighborhood;
    int points_large_neighborhood_size, points_size, sample_size;
public:

    NapsacSamplerImpl (int state, int points_size_, int sample_size_,
            const Ptr<NeighborhoodGraph> &neighborhood_graph_) :
            neighborhood_graph (neighborhood_graph_),
            random_generator(UniformRandomGenerator::create(state, points_size_, sample_size_)) {

        CV_Assert(points_size_ >= sample_size_);

        points_size = points_size_;
        sample_size = sample_size_;
        points_large_neighborhood = std::vector<int>(points_size);

        points_large_neighborhood_size = 0;

        // find indicies of points that have sufficient neighborhood (at least sample_size-1)
        for (int pt_idx = 0; pt_idx < points_size; pt_idx++)
            if ((int)neighborhood_graph->getNeighbors(pt_idx).size() >= sample_size-1)
                points_large_neighborhood[points_large_neighborhood_size++] = pt_idx;

        // if no points with sufficient neighborhood then do only uniform sampling
        if (points_large_neighborhood_size == 0)
            do_uniform = true;

        // set random generator to generate random points of sample_size-1
        random_generator->setSubsetSize(sample_size-1);
    }

    void generateSample (std::vector<int> &sample) override {
        if (do_uniform)
            // uniform sampling
            random_generator->generateUniqueRandomSet(sample, points_size);
        else {
            // Take uniformly one initial point from points with sufficient neighborhood
            int initial_point = points_large_neighborhood
                    [random_generator->getRandomNumber(points_large_neighborhood_size)];

            const std::vector<int> &neighbors = neighborhood_graph->getNeighbors(initial_point);

            // select random neighbors of initial point
            random_generator->generateUniqueRandomSet(sample, (int)neighbors.size());
            for (int i = 0; i < sample_size-1; i++)
                sample[i] = neighbors[sample[i]];

            // sample includes initial point too.
            sample[sample_size-1] = initial_point;
        }
    }

    void setNewPointsSize (int /*points_size_*/) override {
        CV_Error(cv::Error::StsError, "Changing points size requires changing neighborhood graph!"
                    " You must reinitialize NAPSAC!");
    }
};
Ptr<NapsacSampler> NapsacSampler::create(int state, int points_size_, int sample_size_,
                             const Ptr<NeighborhoodGraph> &neighborhood_graph_) {
    return makePtr<NapsacSamplerImpl>(state, points_size_, sample_size_, neighborhood_graph_);
}
}}
