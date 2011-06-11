
#if __CUDA_ARCH__ >= 200

	// for Fermi memory space is detected automatically
	template <typename T> struct ForceGlobLoad
	{
		__device__ __forceinline__ static void Ld(T* ptr, int offset, T& val)  { val = d_ptr[offset];  }
	};
		
#else


	#if defined(_WIN64) || defined(__LP64__)		
		// 64-bit register modifier for inlined asm
		#define _OPENCV_ASM_PTR_ "l"
	#else	
		// 32-bit register modifier for inlined asm
		#define _OPENCV_ASM_PTR_ "r"
	#endif

	template<class T> struct ForceGlobLoad;


#define DEFINE_FORCE_GLOB_LOAD(base_type, ptx_type, reg_mod)											      \
	template <> struct ForceGlobLoad<base_type> 															  \
	{                                                                                                         \
		__device__ __forceinline__ static void Ld(type* ptr, int offset, type& val)                           \ 
        {                                                                                                     \
			asm("ld.global."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _OPENCV_ASM_PTR_(d_ptr + offset));	  \
		}																									  \
	};					
	
	DEFINE_FORCE_GLOB_LOAD(int,   s32, r)	
	DEFINE_FORCE_GLOB_LOAD(float, f32, f)	
		

#undef DEFINE_FORCE_GLOB_LOAD
	
#endif
