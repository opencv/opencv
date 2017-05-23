#ifndef CVVISUAL_DEBUG_MODE_HPP
#define CVVISUAL_DEBUG_MODE_HPP

#if __cplusplus >= 201103L && defined CVVISUAL_USE_THREAD_LOCAL
#define CVVISUAL_THREAD_LOCAL thread_local
#else
#define CVVISUAL_THREAD_LOCAL
#endif

namespace cvv
{

namespace impl
{

/**
 * The debug-flag-singleton
 */
static inline bool &getDebugFlag()
{
	CVVISUAL_THREAD_LOCAL static bool flag = true;
	return flag;
}

} // namespace impl

/**
 * @brief Returns whether debug-mode is active for this TU and thread.
 */
static inline bool debugMode()
{
	return impl::getDebugFlag();
}

/**
 * @brief Set the debug-mode for this TU and thread.
 */
static inline void setDebugFlag(bool active)
{
	impl::getDebugFlag() = active;
}

} // namespace cvv

#endif
