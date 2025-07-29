/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef VAS_COMMON_PROF_HPP
#define VAS_COMMON_PROF_HPP

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <ostream>
#include <stack>
#include <vector>

#define PROF_COMP_NAME(comp) vas::Prof::Component::comp

#ifdef BUILD_OPTION_PROFILING
#define PROF_INIT(component) vas::Prof::Init(PROF_COMP_NAME(component))
#define PROF_START(tag) vas::Prof::Start(tag, __FUNCTION__, __LINE__)
#define PROF_END(tag) vas::Prof::End(tag)
#define PROF_EXTRA(tag, value) vas::Prof::SetExtra(tag, value)
#define PROF_FLUSH(component) vas::Prof::GetInstance(PROF_COMP_NAME(component)).Flush()
#else
#define PROF_INIT(tag)
#define PROF_START(tag)
#define PROF_END(tag)
#define PROF_EXTRA(tag, value)
#define PROF_FLUSH(component)
#endif

#define PROF_TAG_GENERATE(component, group_id, description)                                                            \
    { PROF_COMP_NAME(component), group_id, description }

namespace vas {

/**
 * @class Prof
 *
 * Global Prof instance accumulates all ProfData in a Tree structure.
 * Parallel codes within sigle vas component (ex. STKCF TBB) creates wrong profile result.
 */
class Prof {
  public:
    enum class Component : int32_t { FD, FR, PVD, CD, FAC, OT, PAC, HD, REID, BASE, KN, kCount };

    typedef uint64_t GroupId;
    typedef uint64_t UniqueId;

    /**
     * @class Prof::ProfData
     *
     * Data Node withtin Prof class
     * Accumulates elapsed times between PROF_START / PROF_END
     */
    class ProfData {
      public:
        ProfData(UniqueId id, GroupId group_id, size_t depth, const char *function_name, const int64_t line,
                 const char *description);
        ProfData(const ProfData &other);
        ~ProfData() = default;
        ProfData &operator=(const ProfData &) = delete;
        bool operator==(const ProfData &) const;
        ProfData *clone();

        std::vector<int64_t> accum_time;
        std::list<ProfData *> children;

        const UniqueId id;
        const GroupId group_id;
        const size_t depth;

        const char *function_name;
        const int64_t line;
        const char *description;
        int64_t start_time;
    };

    typedef struct _ProfTag {
        vas::Prof::Component component;
        vas::Prof::GroupId group_id;
        const char *description;
    } ProfTag;

  public:
    Prof();
    ~Prof() = default;

    static void Init(Component comp);
    static void Start(const ProfTag &tag, const char *function_name, int64_t line);
    static void End(const ProfTag &tag);

    static Prof &GetInstance(Component comp);

    static void SetExtra(const ProfTag &tag, int32_t value);

    void StartProfile(GroupId group_id, const char *function_name, int64_t line, const char *description);
    void EndProfile();
    void SetExtraData(const std::string &key, int32_t value);
    void Flush();

    void MergeToMainInstance(Prof *in);

  private:
    const char *GetComponentName(Component comp);
    void Clear();

    // Print detailed prof data.
    void PrintSummary1(std::ostream *out);

    // Print prof data merged in same stack.
    void PrintSummary2(std::ostream *out);

    // Print prof data merged with the same group-id.
    void PrintSummary3(std::ostream *out);

    void PrintSummary1ToCSV(std::ostream *out);

    void PrintExtra(std::ostream *out);

    void PrintAllData(std::ostream *out);

    void Traverse(const ProfData *root, const std::list<ProfData *> &data_list,
                  void (*print_function)(const ProfData *, const ProfData &, std::ostream *), std::ostream *out);
    void TraverseMergeSameStackGroups(const std::list<ProfData *> &in_data_list, std::list<ProfData *> *out_data_list);
    void TraverseMergeAllGroups(const std::list<ProfData *> &in_data_list, std::list<ProfData *> *out_data_list);

    void MergeProfDataList(std::list<Prof::ProfData *> *mergeList, const std::list<Prof::ProfData *> &addList);

  private:
    std::string outdir_;
    std::string out_prof_file_;
    Component component_;
    std::list<ProfData *> root_data_list_;
    std::stack<ProfData *> current_data_;
    std::map<std::string, std::vector<int32_t>> extra_data_;
};

} // namespace vas

#endif // VAS_COMMON_PROF_HPP
