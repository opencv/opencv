Understanding SVM {#tutorial_py_svm_basics}
=================

Goal
----

In this chapter
    -   We will see an intuitive understanding of SVM

Theory
------

### Linearly Separable Data

Consider the image below which has two types of data, red and blue. In kNN, for a test data, we used
to measure its distance to all the training samples and take the one with minimum distance. It takes
plenty of time to measure all the distances and plenty of memory to store all the training-samples.
But considering the data given in image, should we need that much?

![image](images/svm_basics1.png)

Consider another idea. We find a line, \f$f(x)=ax_1+bx_2+c\f$ which divides both the data to two
regions. When we get a new test_data \f$X\f$, just substitute it in \f$f(x)\f$. If \f$f(X) > 0\f$, it belongs
to blue group, else it belongs to red group. We can call this line as **Decision Boundary**. It is
very simple and memory-efficient. Such data which can be divided into two with a straight line (or
hyperplanes in higher dimensions) is called **Linear Separable**.

So in above image, you can see plenty of such lines are possible. Which one we will take? Very
intuitively we can say that the line should be passing as far as possible from all the points. Why?
Because there can be noise in the incoming data. This data should not affect the classification
accuracy. So taking a farthest line will provide more immunity against noise. So what SVM does is to
find a straight line (or hyperplane) with largest minimum distance to the training samples. See the
bold line in below image passing through the center.

![image](images/svm_basics2.png)

So to find this Decision Boundary, you need training data. Do you need all? NO. Just the ones which
are close to the opposite group are sufficient. In our image, they are the one blue filled circle
and two red filled squares. We can call them **Support Vectors** and the lines passing through them
are called **Support Planes**. They are adequate for finding our decision boundary. We need not
worry about all the data. It helps in data reduction.

What happened is, first two hyperplanes are found which best represents the data. For eg, blue data
is represented by \f$w^Tx+b_0 > 1\f$ while red data is represented by \f$w^Tx+b_0 < -1\f$ where \f$w\f$ is
**weight vector** ( \f$w=[w_1, w_2,..., w_n]\f$) and \f$x\f$ is the feature vector
(\f$x = [x_1,x_2,..., x_n]\f$). \f$b_0\f$ is the **bias**. Weight vector decides the orientation of decision
boundary while bias point decides its location. Now decision boundary is defined to be midway
between these hyperplanes, so expressed as \f$w^Tx+b_0 = 0\f$. The minimum distance from support vector
to the decision boundary is given by, \f$distance_{support \, vectors}=\frac{1}{||w||}\f$. Margin is
twice this distance, and we need to maximize this margin. i.e. we need to minimize a new function
\f$L(w, b_0)\f$ with some constraints which can expressed below:

\f[\min_{w, b_0} L(w, b_0) = \frac{1}{2}||w||^2 \; \text{subject to} \; t_i(w^Tx+b_0) \geq 1 \; \forall i\f]

where \f$t_i\f$ is the label of each class, \f$t_i \in [-1,1]\f$.

### Non-Linearly Separable Data

Consider some data which can't be divided into two with a straight line. For example, consider an
one-dimensional data where 'X' is at -3 & +3 and 'O' is at -1 & +1. Clearly it is not linearly
separable. But there are methods to solve these kinds of problems. If we can map this data set with
a function, \f$f(x) = x^2\f$, we get 'X' at 9 and 'O' at 1 which are linear separable.

Otherwise we can convert this one-dimensional to two-dimensional data. We can use \f$f(x)=(x,x^2)\f$
function to map this data. Then 'X' becomes (-3,9) and (3,9) while 'O' becomes (-1,1) and (1,1).
This is also linear separable. In short, chance is more for a non-linear separable data in
lower-dimensional space to become linear separable in higher-dimensional space.

In general, it is possible to map points in a d-dimensional space to some D-dimensional space
\f$(D>d)\f$ to check the possibility of linear separability. There is an idea which helps to compute the
dot product in the high-dimensional (kernel) space by performing computations in the low-dimensional
input (feature) space. We can illustrate with following example.

Consider two points in two-dimensional space, \f$p=(p_1,p_2)\f$ and \f$q=(q_1,q_2)\f$. Let \f$\phi\f$ be a
mapping function which maps a two-dimensional point to three-dimensional space as follows:

\f[\phi (p) = (p_{1}^2,p_{2}^2,\sqrt{2} p_1 p_2)
\phi (q) = (q_{1}^2,q_{2}^2,\sqrt{2} q_1 q_2)\f]

Let us define a kernel function \f$K(p,q)\f$ which does a dot product between two points, shown below:

\f[
\begin{aligned}
K(p,q)  = \phi(p).\phi(q) &= \phi(p)^T \phi(q) \\
                          &= (p_{1}^2,p_{2}^2,\sqrt{2} p_1 p_2).(q_{1}^2,q_{2}^2,\sqrt{2} q_1 q_2) \\
                          &= p_{1}^2 q_{1}^2 + p_{2}^2 q_{2}^2 + 2 p_1 q_1 p_2 q_2 \\
                          &= (p_1 q_1 + p_2 q_2)^2 \\
          \phi(p).\phi(q) &= (p.q)^2
\end{aligned}
\f]

It means, a dot product in three-dimensional space can be achieved using squared dot product in
two-dimensional space. This can be applied to higher dimensional space. So we can calculate higher
dimensional features from lower dimensions itself. Once we map them, we get a higher dimensional
space.

In addition to all these concepts, there comes the problem of misclassification. So just finding
decision boundary with maximum margin is not sufficient. We need to consider the problem of
misclassification errors also. Sometimes, it may be possible to find a decision boundary with less
margin, but with reduced misclassification. Anyway we need to modify our model such that it should
find decision boundary with maximum margin, but with less misclassification. The minimization
criteria is modified as:

\f[min \; ||w||^2 + C(distance \; of \; misclassified \; samples \; to \; their \; correct \; regions)\f]

Below image shows this concept. For each sample of the training data a new parameter \f$\xi_i\f$ is
defined. It is the distance from its corresponding training sample to their correct decision region.
For those who are not misclassified, they fall on their corresponding support planes, so their
distance is zero.

![image](images/svm_basics3.png)

So the new optimization problem is :

\f[\min_{w, b_{0}} L(w,b_0) = ||w||^{2} + C \sum_{i} {\xi_{i}} \text{ subject to } y_{i}(w^{T} x_{i} + b_{0}) \geq 1 - \xi_{i} \text{ and } \xi_{i} \geq 0 \text{ } \forall i\f]

How should the parameter C be chosen? It is obvious that the answer to this question depends on how
the training data is distributed. Although there is no general answer, it is useful to take into
account these rules:

-   Large values of C give solutions with less misclassification errors but a smaller margin.
    Consider that in this case it is expensive to make misclassification errors. Since the aim of
    the optimization is to minimize the argument, few misclassifications errors are allowed.
-   Small values of C give solutions with bigger margin and more classification errors. In this
    case the minimization does not consider that much the term of the sum so it focuses more on
    finding a hyperplane with big margin.

Additional Resources
--------------------

-#  [NPTEL notes on Statistical Pattern Recognition, Chapters
    25-29](http://www.nptel.ac.in/courses/106108057/26).

Exercises
---------
