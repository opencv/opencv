namespace cv { namespace cuda { namespace device
{
    namespace stereocsbp
    {
        template<class T>
        void init_data_cost(const uchar *left, const uchar *right, uchar *ctemp, size_t cimg_step, int rows, int cols, T* disp_selected_pyr, T* data_cost_selected, size_t msg_step,
                    int h, int w, int level, int nr_plane, int ndisp, int channels, float data_weight, float max_data_term, int min_disp, bool use_local_init_data_cost, cudaStream_t stream);

        template<class T>
        void compute_data_cost(const uchar *left, const uchar *right, size_t cimg_step, const T* disp_selected_pyr, T* data_cost, size_t msg_step,
                               int rows, int cols, int h, int w, int h2, int level, int nr_plane, int channels, float data_weight, float max_data_term,
                               int min_disp, cudaStream_t stream);

        template<class T>
        void init_message(uchar *ctemp, T* u_new, T* d_new, T* l_new, T* r_new,
                          const T* u_cur, const T* d_cur, const T* l_cur, const T* r_cur,
                          T* selected_disp_pyr_new, const T* selected_disp_pyr_cur,
                          T* data_cost_selected, const T* data_cost, size_t msg_step,
                          int h, int w, int nr_plane, int h2, int w2, int nr_plane2, cudaStream_t stream);

        template<class T>
        void calc_all_iterations(uchar *ctemp, T* u, T* d, T* l, T* r, const T* data_cost_selected,
            const T* selected_disp_pyr_cur, size_t msg_step, int h, int w, int nr_plane, int iters, int max_disc_term, float disc_single_jump, cudaStream_t stream);

        template<class T>
        void compute_disp(const T* u, const T* d, const T* l, const T* r, const T* data_cost_selected, const T* disp_selected, size_t msg_step,
            const PtrStepSz<short>& disp, int nr_plane, cudaStream_t stream);
    }
}}}
