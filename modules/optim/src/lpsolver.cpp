#include "opencv2/opencv.hpp"

namespace cv {
       	namespace optim {

class Solver : public Algorithm /* Algorithm is base OpenCV class */
{
      class Function
      {
      public:
            virtual ~Function() {}
            virtual double calc(InputArray args) const = 0;
            virtual double calc(InputArgs, OutputArray grad) const = 0;
      };

      // could be reused for all the generic algorithms like downhill simplex.
      virtual void solve(InputArray x0, OutputArray result) const = 0;

      virtual void setTermCriteria(const TermCriteria& criteria) = 0;
      virtual TermCriteria getTermCriteria() = 0;

      // more detailed API to be defined later ...
};

class LPSolver : public Solver
{
public:
     virtual void solve(InputArray coeffs, InputArray constraints, OutputArray result) const = 0;
     // ...
};

Ptr<LPSolver> createLPSimplexSolver();

}}

/*===============
Hill climbing solver is more generic one:*/
/*
class DownhillSolver : public Solver
{
public:
      // various setters and getters, if needed
};

Ptr<DownhillSolver> createDownhillSolver(const Ptr<Solver::Function>& func);*/
