#include <unordered_map>
#include <string>
#include <bridge>

typedef std::unordered_map Map;

/*! @brief Hash from strings to OpenCV enums
 *
 * This is a translation map for strings to OpenCV constants (enums).
 * When an int is requested from the bridge, and the the mxArray storage 
 * type is a string, this map is invoked. Thus functions can be called
 * from Matlab as, e.g.
 *    cv.dft(x, xf, "DFT_FORWARD");
 *
 * Note that an alternative MAtlab class exists as well, so that functions
 * can be called as, e.g.
 *    cv.dft(x, xf, cv.DFT_FORWARD);
 *
 * This string to int map tends to be faster than its Matlab companion,
 * but there is no direct access to the value of the constants. It also
 * enables different error reporting properties.
 */
Map<std::string, int> constants = {
  {% for key, val in constants.items() %}
  { "{{key}}", {{val}} },
  {% endfor %}
};
