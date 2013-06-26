#include <unordered_map>
#include <string>
#include <bridge>

typedef std::unordered_map Map;

Map<std::string, int> constants = {
  {% for key, val in constants.items() %}
  { "{{key}}", {{val}} },
  {% endfor %}
};
