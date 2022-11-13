//
// Created by yuval on 6/10/20.
//

#include "NPnpProblemSolver.h"
#include "DualVar.h"
#include "../Utils_Npnp/Parsing.h"
#include <iostream>
#include <utility>

namespace NPnP {
PnpProblemSolver::PnpProblemSolver(
    const SparseRowMatrix &A_rows, const SparseColMatrix &A_cols,
    const SparseColMatrix &c_vec,
    RowMatrix<NUM_CONSTRAINTS, Y_SIZE> zero_sub_A_rows,
    ColMatrix<NUM_CONSTRAINTS, Y_SIZE> zero_sub_A_cols,
    ColMatrix<NUM_CONSTRAINTS, NUM_CONSTRAINTS> zero_mat_15_15,
    ColVector<Y_SIZE> init_y, std::shared_ptr<DualVar> dual_var)
    : A_rows(A_rows), A_cols(A_cols), c_vec(c_vec),
      zero_sub_A_rows(std::move(zero_sub_A_rows)),
      zero_sub_A_cols(std::move(zero_sub_A_cols)),
      zero_mat_15_15(std::move(zero_mat_15_15)), init_y(std::move(init_y)),
      dual_var(std::move(dual_var)) {}

std::shared_ptr<PnpProblemSolver> PnpProblemSolver::init() {
  // init c vector
  std::vector<Eigen::Triplet<double>> c_vec_triplet_list;
  c_vec_triplet_list.reserve(2);
  c_vec_triplet_list.emplace_back(0, 0, -1);
  c_vec_triplet_list.emplace_back(15, 0, 1);
  SparseColMatrix c_vec(A_ROWS, 1);
  c_vec.setFromTriplets(c_vec_triplet_list.begin(), c_vec_triplet_list.end());

  // init y vector
  ColVector<Y_SIZE> init_y_vec;
  init_y_vec << 1.81708374227065E-07, -2.71468522392971E-10,
      -1.49378319259558E-10, -2.77091740805445E-11, 0.249631001517346,
      4.96940105201532E-05, -0.000129984428061, 5.97892049448515E-05,
      0.249893826536401, -0.000282031359397, -0.000299838600691,
      0.250010662193677, -0.000119648361021, 0.250464509752576,
      1.81036335999704E-07, -2.7244555135617E-10, -1.44912520866555E-10,
      -3.09767807222993E-11, 2.26443899887141E-10, -3.1024811831443E-12,
      3.18281863003413E-13, 2.14461800733467E-10, -1.33638362717957E-12,
      2.31132526688904E-10, 7.61682308171569E-13, -8.51485831453974E-13,
      7.23456344454997E-13, 5.18662287311713E-13, -9.181167075483E-14,
      -3.03315632232526E-13, -3.06071262491456E-12, 8.84104200363805E-13,
      -5.53599936640479E-13, 1.66004609693395E-12, 0.12485918729074,
      1.24357475655128E-05, -1.06930867828931E-06, -3.41617427196685E-05,
      0.04161036487279, 1.32703449912371E-05, -7.18816799089538E-06,
      0.041597999991592, -3.58128226124226E-06, 0.041563449362224,
      5.72613189555295E-05, -1.80142323379959E-05, 3.18277376965713E-05,
      6.54613708082515E-05, -1.75831016069252E-05, -8.54644268091407E-05,
      -0.000152813856463, 6.69489404481575E-05, 4.19129694179984E-05,
      -4.82573048020879E-06, 0.124762538478383, -3.70010098743364E-05,
      2.6077418953879E-05, 0.041753219462236, -0.000129437659595,
      0.041767703722993, -0.000320047212861, 0.000129331848335,
      6.17465183469158E-05, -0.000448059699989, 0.125204467737999,
      -0.000108969941535, 0.04145497500185, 0.00012234052237, 0.125678381665509;

  // init A matrix

  std::vector<Eigen::Triplet<double>> A_mat_triplet_list;
  A_mat_triplet_list.reserve(300);
  A_mat_triplet_list.emplace_back(0, 4, -1.0);
  A_mat_triplet_list.emplace_back(0, 8, -1.0);
  A_mat_triplet_list.emplace_back(0, 11, -1.0);
  A_mat_triplet_list.emplace_back(0, 13, -1.0);
  A_mat_triplet_list.emplace_back(1, 0, 1.0);
  A_mat_triplet_list.emplace_back(1, 14, -1.0);
  A_mat_triplet_list.emplace_back(1, 18, -1.0);
  A_mat_triplet_list.emplace_back(1, 21, -1.0);
  A_mat_triplet_list.emplace_back(1, 23, -1.0);
  A_mat_triplet_list.emplace_back(2, 1, 1.0);
  A_mat_triplet_list.emplace_back(2, 15, -1.0);
  A_mat_triplet_list.emplace_back(2, 24, -1.0);
  A_mat_triplet_list.emplace_back(2, 27, -1.0);
  A_mat_triplet_list.emplace_back(2, 29, -1.0);
  A_mat_triplet_list.emplace_back(3, 2, 1.0);
  A_mat_triplet_list.emplace_back(3, 16, -1.0);
  A_mat_triplet_list.emplace_back(3, 25, -1.0);
  A_mat_triplet_list.emplace_back(3, 30, -1.0);
  A_mat_triplet_list.emplace_back(3, 32, -1.0);
  A_mat_triplet_list.emplace_back(4, 3, 1.0);
  A_mat_triplet_list.emplace_back(4, 17, -1.0);
  A_mat_triplet_list.emplace_back(4, 26, -1.0);
  A_mat_triplet_list.emplace_back(4, 31, -1.0);
  A_mat_triplet_list.emplace_back(4, 33, -1.0);
  A_mat_triplet_list.emplace_back(5, 4, 1.0);
  A_mat_triplet_list.emplace_back(5, 34, -1.0);
  A_mat_triplet_list.emplace_back(5, 38, -1.0);
  A_mat_triplet_list.emplace_back(5, 41, -1.0);
  A_mat_triplet_list.emplace_back(5, 43, -1.0);
  A_mat_triplet_list.emplace_back(6, 5, 1.0);
  A_mat_triplet_list.emplace_back(6, 35, -1.0);
  A_mat_triplet_list.emplace_back(6, 44, -1.0);
  A_mat_triplet_list.emplace_back(6, 47, -1.0);
  A_mat_triplet_list.emplace_back(6, 49, -1.0);
  A_mat_triplet_list.emplace_back(7, 6, 1.0);
  A_mat_triplet_list.emplace_back(7, 36, -1.0);
  A_mat_triplet_list.emplace_back(7, 45, -1.0);
  A_mat_triplet_list.emplace_back(7, 50, -1.0);
  A_mat_triplet_list.emplace_back(7, 52, -1.0);
  A_mat_triplet_list.emplace_back(8, 7, 1.0);
  A_mat_triplet_list.emplace_back(8, 37, -1.0);
  A_mat_triplet_list.emplace_back(8, 46, -1.0);
  A_mat_triplet_list.emplace_back(8, 51, -1.0);
  A_mat_triplet_list.emplace_back(8, 53, -1.0);
  A_mat_triplet_list.emplace_back(9, 8, 1.0);
  A_mat_triplet_list.emplace_back(9, 38, -1.0);
  A_mat_triplet_list.emplace_back(9, 54, -1.0);
  A_mat_triplet_list.emplace_back(9, 57, -1.0);
  A_mat_triplet_list.emplace_back(9, 59, -1.0);
  A_mat_triplet_list.emplace_back(10, 9, 1.0);
  A_mat_triplet_list.emplace_back(10, 39, -1.0);
  A_mat_triplet_list.emplace_back(10, 55, -1.0);
  A_mat_triplet_list.emplace_back(10, 60, -1.0);
  A_mat_triplet_list.emplace_back(10, 62, -1.0);
  A_mat_triplet_list.emplace_back(11, 10, 1.0);
  A_mat_triplet_list.emplace_back(11, 40, -1.0);
  A_mat_triplet_list.emplace_back(11, 56, -1.0);
  A_mat_triplet_list.emplace_back(11, 61, -1.0);
  A_mat_triplet_list.emplace_back(11, 63, -1.0);
  A_mat_triplet_list.emplace_back(12, 11, 1.0);
  A_mat_triplet_list.emplace_back(12, 41, -1.0);
  A_mat_triplet_list.emplace_back(12, 57, -1.0);
  A_mat_triplet_list.emplace_back(12, 64, -1.0);
  A_mat_triplet_list.emplace_back(12, 66, -1.0);
  A_mat_triplet_list.emplace_back(13, 12, 1.0);
  A_mat_triplet_list.emplace_back(13, 42, -1.0);
  A_mat_triplet_list.emplace_back(13, 58, -1.0);
  A_mat_triplet_list.emplace_back(13, 65, -1.0);
  A_mat_triplet_list.emplace_back(13, 67, -1.0);
  A_mat_triplet_list.emplace_back(14, 13, 1.0);
  A_mat_triplet_list.emplace_back(14, 43, -1.0);
  A_mat_triplet_list.emplace_back(14, 59, -1.0);
  A_mat_triplet_list.emplace_back(14, 66, -1.0);
  A_mat_triplet_list.emplace_back(14, 68, -1.0);
  A_mat_triplet_list.emplace_back(16, 0, -1.0);
  A_mat_triplet_list.emplace_back(17, 1, -1.0);
  A_mat_triplet_list.emplace_back(18, 2, -1.0);
  A_mat_triplet_list.emplace_back(19, 3, -1.0);
  A_mat_triplet_list.emplace_back(20, 4, -1.0);
  A_mat_triplet_list.emplace_back(21, 5, -1.0);
  A_mat_triplet_list.emplace_back(22, 6, -1.0);
  A_mat_triplet_list.emplace_back(23, 7, -1.0);
  A_mat_triplet_list.emplace_back(24, 8, -1.0);
  A_mat_triplet_list.emplace_back(25, 9, -1.0);
  A_mat_triplet_list.emplace_back(26, 10, -1.0);
  A_mat_triplet_list.emplace_back(27, 11, -1.0);
  A_mat_triplet_list.emplace_back(28, 12, -1.0);
  A_mat_triplet_list.emplace_back(29, 13, -1.0);
  A_mat_triplet_list.emplace_back(30, 0, -1.0);
  A_mat_triplet_list.emplace_back(31, 4, -1.0);
  A_mat_triplet_list.emplace_back(32, 5, -1.0);
  A_mat_triplet_list.emplace_back(33, 6, -1.0);
  A_mat_triplet_list.emplace_back(34, 7, -1.0);
  A_mat_triplet_list.emplace_back(35, 14, -1.0);
  A_mat_triplet_list.emplace_back(36, 15, -1.0);
  A_mat_triplet_list.emplace_back(37, 16, -1.0);
  A_mat_triplet_list.emplace_back(38, 17, -1.0);
  A_mat_triplet_list.emplace_back(39, 18, -1.0);
  A_mat_triplet_list.emplace_back(40, 19, -1.0);
  A_mat_triplet_list.emplace_back(41, 20, -1.0);
  A_mat_triplet_list.emplace_back(42, 21, -1.0);
  A_mat_triplet_list.emplace_back(43, 22, -1.0);
  A_mat_triplet_list.emplace_back(44, 23, -1.0);
  A_mat_triplet_list.emplace_back(45, 1, -1.0);
  A_mat_triplet_list.emplace_back(46, 5, -1.0);
  A_mat_triplet_list.emplace_back(47, 8, -1.0);
  A_mat_triplet_list.emplace_back(48, 9, -1.0);
  A_mat_triplet_list.emplace_back(49, 10, -1.0);
  A_mat_triplet_list.emplace_back(50, 15, -1.0);
  A_mat_triplet_list.emplace_back(51, 18, -1.0);
  A_mat_triplet_list.emplace_back(52, 19, -1.0);
  A_mat_triplet_list.emplace_back(53, 20, -1.0);
  A_mat_triplet_list.emplace_back(54, 24, -1.0);
  A_mat_triplet_list.emplace_back(55, 25, -1.0);
  A_mat_triplet_list.emplace_back(56, 26, -1.0);
  A_mat_triplet_list.emplace_back(57, 27, -1.0);
  A_mat_triplet_list.emplace_back(58, 28, -1.0);
  A_mat_triplet_list.emplace_back(59, 29, -1.0);
  A_mat_triplet_list.emplace_back(60, 2, -1.0);
  A_mat_triplet_list.emplace_back(61, 6, -1.0);
  A_mat_triplet_list.emplace_back(62, 9, -1.0);
  A_mat_triplet_list.emplace_back(63, 11, -1.0);
  A_mat_triplet_list.emplace_back(64, 12, -1.0);
  A_mat_triplet_list.emplace_back(65, 16, -1.0);
  A_mat_triplet_list.emplace_back(66, 19, -1.0);
  A_mat_triplet_list.emplace_back(67, 21, -1.0);
  A_mat_triplet_list.emplace_back(68, 22, -1.0);
  A_mat_triplet_list.emplace_back(69, 25, -1.0);
  A_mat_triplet_list.emplace_back(70, 27, -1.0);
  A_mat_triplet_list.emplace_back(71, 28, -1.0);
  A_mat_triplet_list.emplace_back(72, 30, -1.0);
  A_mat_triplet_list.emplace_back(73, 31, -1.0);
  A_mat_triplet_list.emplace_back(74, 32, -1.0);
  A_mat_triplet_list.emplace_back(75, 3, -1.0);
  A_mat_triplet_list.emplace_back(76, 7, -1.0);
  A_mat_triplet_list.emplace_back(77, 10, -1.0);
  A_mat_triplet_list.emplace_back(78, 12, -1.0);
  A_mat_triplet_list.emplace_back(79, 13, -1.0);
  A_mat_triplet_list.emplace_back(80, 17, -1.0);
  A_mat_triplet_list.emplace_back(81, 20, -1.0);
  A_mat_triplet_list.emplace_back(82, 22, -1.0);
  A_mat_triplet_list.emplace_back(83, 23, -1.0);
  A_mat_triplet_list.emplace_back(84, 26, -1.0);
  A_mat_triplet_list.emplace_back(85, 28, -1.0);
  A_mat_triplet_list.emplace_back(86, 29, -1.0);
  A_mat_triplet_list.emplace_back(87, 31, -1.0);
  A_mat_triplet_list.emplace_back(88, 32, -1.0);
  A_mat_triplet_list.emplace_back(89, 33, -1.0);
  A_mat_triplet_list.emplace_back(90, 4, -1.0);
  A_mat_triplet_list.emplace_back(91, 14, -1.0);
  A_mat_triplet_list.emplace_back(92, 15, -1.0);
  A_mat_triplet_list.emplace_back(93, 16, -1.0);
  A_mat_triplet_list.emplace_back(94, 17, -1.0);
  A_mat_triplet_list.emplace_back(95, 34, -1.0);
  A_mat_triplet_list.emplace_back(96, 35, -1.0);
  A_mat_triplet_list.emplace_back(97, 36, -1.0);
  A_mat_triplet_list.emplace_back(98, 37, -1.0);
  A_mat_triplet_list.emplace_back(99, 38, -1.0);
  A_mat_triplet_list.emplace_back(100, 39, -1.0);
  A_mat_triplet_list.emplace_back(101, 40, -1.0);
  A_mat_triplet_list.emplace_back(102, 41, -1.0);
  A_mat_triplet_list.emplace_back(103, 42, -1.0);
  A_mat_triplet_list.emplace_back(104, 43, -1.0);
  A_mat_triplet_list.emplace_back(105, 5, -1.0);
  A_mat_triplet_list.emplace_back(106, 15, -1.0);
  A_mat_triplet_list.emplace_back(107, 18, -1.0);
  A_mat_triplet_list.emplace_back(108, 19, -1.0);
  A_mat_triplet_list.emplace_back(109, 20, -1.0);
  A_mat_triplet_list.emplace_back(110, 35, -1.0);
  A_mat_triplet_list.emplace_back(111, 38, -1.0);
  A_mat_triplet_list.emplace_back(112, 39, -1.0);
  A_mat_triplet_list.emplace_back(113, 40, -1.0);
  A_mat_triplet_list.emplace_back(114, 44, -1.0);
  A_mat_triplet_list.emplace_back(115, 45, -1.0);
  A_mat_triplet_list.emplace_back(116, 46, -1.0);
  A_mat_triplet_list.emplace_back(117, 47, -1.0);
  A_mat_triplet_list.emplace_back(118, 48, -1.0);
  A_mat_triplet_list.emplace_back(119, 49, -1.0);
  A_mat_triplet_list.emplace_back(120, 6, -1.0);
  A_mat_triplet_list.emplace_back(121, 16, -1.0);
  A_mat_triplet_list.emplace_back(122, 19, -1.0);
  A_mat_triplet_list.emplace_back(123, 21, -1.0);
  A_mat_triplet_list.emplace_back(124, 22, -1.0);
  A_mat_triplet_list.emplace_back(125, 36, -1.0);
  A_mat_triplet_list.emplace_back(126, 39, -1.0);
  A_mat_triplet_list.emplace_back(127, 41, -1.0);
  A_mat_triplet_list.emplace_back(128, 42, -1.0);
  A_mat_triplet_list.emplace_back(129, 45, -1.0);
  A_mat_triplet_list.emplace_back(130, 47, -1.0);
  A_mat_triplet_list.emplace_back(131, 48, -1.0);
  A_mat_triplet_list.emplace_back(132, 50, -1.0);
  A_mat_triplet_list.emplace_back(133, 51, -1.0);
  A_mat_triplet_list.emplace_back(134, 52, -1.0);
  A_mat_triplet_list.emplace_back(135, 7, -1.0);
  A_mat_triplet_list.emplace_back(136, 17, -1.0);
  A_mat_triplet_list.emplace_back(137, 20, -1.0);
  A_mat_triplet_list.emplace_back(138, 22, -1.0);
  A_mat_triplet_list.emplace_back(139, 23, -1.0);
  A_mat_triplet_list.emplace_back(140, 37, -1.0);
  A_mat_triplet_list.emplace_back(141, 40, -1.0);
  A_mat_triplet_list.emplace_back(142, 42, -1.0);
  A_mat_triplet_list.emplace_back(143, 43, -1.0);
  A_mat_triplet_list.emplace_back(144, 46, -1.0);
  A_mat_triplet_list.emplace_back(145, 48, -1.0);
  A_mat_triplet_list.emplace_back(146, 49, -1.0);
  A_mat_triplet_list.emplace_back(147, 51, -1.0);
  A_mat_triplet_list.emplace_back(148, 52, -1.0);
  A_mat_triplet_list.emplace_back(149, 53, -1.0);
  A_mat_triplet_list.emplace_back(150, 8, -1.0);
  A_mat_triplet_list.emplace_back(151, 18, -1.0);
  A_mat_triplet_list.emplace_back(152, 24, -1.0);
  A_mat_triplet_list.emplace_back(153, 25, -1.0);
  A_mat_triplet_list.emplace_back(154, 26, -1.0);
  A_mat_triplet_list.emplace_back(155, 38, -1.0);
  A_mat_triplet_list.emplace_back(156, 44, -1.0);
  A_mat_triplet_list.emplace_back(157, 45, -1.0);
  A_mat_triplet_list.emplace_back(158, 46, -1.0);
  A_mat_triplet_list.emplace_back(159, 54, -1.0);
  A_mat_triplet_list.emplace_back(160, 55, -1.0);
  A_mat_triplet_list.emplace_back(161, 56, -1.0);
  A_mat_triplet_list.emplace_back(162, 57, -1.0);
  A_mat_triplet_list.emplace_back(163, 58, -1.0);
  A_mat_triplet_list.emplace_back(164, 59, -1.0);
  A_mat_triplet_list.emplace_back(165, 9, -1.0);
  A_mat_triplet_list.emplace_back(166, 19, -1.0);
  A_mat_triplet_list.emplace_back(167, 25, -1.0);
  A_mat_triplet_list.emplace_back(168, 27, -1.0);
  A_mat_triplet_list.emplace_back(169, 28, -1.0);
  A_mat_triplet_list.emplace_back(170, 39, -1.0);
  A_mat_triplet_list.emplace_back(171, 45, -1.0);
  A_mat_triplet_list.emplace_back(172, 47, -1.0);
  A_mat_triplet_list.emplace_back(173, 48, -1.0);
  A_mat_triplet_list.emplace_back(174, 55, -1.0);
  A_mat_triplet_list.emplace_back(175, 57, -1.0);
  A_mat_triplet_list.emplace_back(176, 58, -1.0);
  A_mat_triplet_list.emplace_back(177, 60, -1.0);
  A_mat_triplet_list.emplace_back(178, 61, -1.0);
  A_mat_triplet_list.emplace_back(179, 62, -1.0);
  A_mat_triplet_list.emplace_back(180, 10, -1.0);
  A_mat_triplet_list.emplace_back(181, 20, -1.0);
  A_mat_triplet_list.emplace_back(182, 26, -1.0);
  A_mat_triplet_list.emplace_back(183, 28, -1.0);
  A_mat_triplet_list.emplace_back(184, 29, -1.0);
  A_mat_triplet_list.emplace_back(185, 40, -1.0);
  A_mat_triplet_list.emplace_back(186, 46, -1.0);
  A_mat_triplet_list.emplace_back(187, 48, -1.0);
  A_mat_triplet_list.emplace_back(188, 49, -1.0);
  A_mat_triplet_list.emplace_back(189, 56, -1.0);
  A_mat_triplet_list.emplace_back(190, 58, -1.0);
  A_mat_triplet_list.emplace_back(191, 59, -1.0);
  A_mat_triplet_list.emplace_back(192, 61, -1.0);
  A_mat_triplet_list.emplace_back(193, 62, -1.0);
  A_mat_triplet_list.emplace_back(194, 63, -1.0);
  A_mat_triplet_list.emplace_back(195, 11, -1.0);
  A_mat_triplet_list.emplace_back(196, 21, -1.0);
  A_mat_triplet_list.emplace_back(197, 27, -1.0);
  A_mat_triplet_list.emplace_back(198, 30, -1.0);
  A_mat_triplet_list.emplace_back(199, 31, -1.0);
  A_mat_triplet_list.emplace_back(200, 41, -1.0);
  A_mat_triplet_list.emplace_back(201, 47, -1.0);
  A_mat_triplet_list.emplace_back(202, 50, -1.0);
  A_mat_triplet_list.emplace_back(203, 51, -1.0);
  A_mat_triplet_list.emplace_back(204, 57, -1.0);
  A_mat_triplet_list.emplace_back(205, 60, -1.0);
  A_mat_triplet_list.emplace_back(206, 61, -1.0);
  A_mat_triplet_list.emplace_back(207, 64, -1.0);
  A_mat_triplet_list.emplace_back(208, 65, -1.0);
  A_mat_triplet_list.emplace_back(209, 66, -1.0);
  A_mat_triplet_list.emplace_back(210, 12, -1.0);
  A_mat_triplet_list.emplace_back(211, 22, -1.0);
  A_mat_triplet_list.emplace_back(212, 28, -1.0);
  A_mat_triplet_list.emplace_back(213, 31, -1.0);
  A_mat_triplet_list.emplace_back(214, 32, -1.0);
  A_mat_triplet_list.emplace_back(215, 42, -1.0);
  A_mat_triplet_list.emplace_back(216, 48, -1.0);
  A_mat_triplet_list.emplace_back(217, 51, -1.0);
  A_mat_triplet_list.emplace_back(218, 52, -1.0);
  A_mat_triplet_list.emplace_back(219, 58, -1.0);
  A_mat_triplet_list.emplace_back(220, 61, -1.0);
  A_mat_triplet_list.emplace_back(221, 62, -1.0);
  A_mat_triplet_list.emplace_back(222, 65, -1.0);
  A_mat_triplet_list.emplace_back(223, 66, -1.0);
  A_mat_triplet_list.emplace_back(224, 67, -1.0);
  A_mat_triplet_list.emplace_back(225, 13, -1.0);
  A_mat_triplet_list.emplace_back(226, 23, -1.0);
  A_mat_triplet_list.emplace_back(227, 29, -1.0);
  A_mat_triplet_list.emplace_back(228, 32, -1.0);
  A_mat_triplet_list.emplace_back(229, 33, -1.0);
  A_mat_triplet_list.emplace_back(230, 43, -1.0);
  A_mat_triplet_list.emplace_back(231, 49, -1.0);
  A_mat_triplet_list.emplace_back(232, 52, -1.0);
  A_mat_triplet_list.emplace_back(233, 53, -1.0);
  A_mat_triplet_list.emplace_back(234, 59, -1.0);
  A_mat_triplet_list.emplace_back(235, 62, -1.0);
  A_mat_triplet_list.emplace_back(236, 63, -1.0);
  A_mat_triplet_list.emplace_back(237, 66, -1.0);
  A_mat_triplet_list.emplace_back(238, 67, -1.0);
  A_mat_triplet_list.emplace_back(239, 68, -1.0);

  SparseColMatrix A_mat_cols(A_ROWS, Y_SIZE);
  A_mat_cols.setFromTriplets(A_mat_triplet_list.begin(),
                             A_mat_triplet_list.end());

  SparseRowMatrix A_mat_rows(A_ROWS, Y_SIZE);
  A_mat_rows.setFromTriplets(A_mat_triplet_list.begin(),
                             A_mat_triplet_list.end());

  auto zero_sub_A_cols = A_mat_cols.block(0, 0, NUM_CONSTRAINTS, Y_SIZE).eval();
  auto zero_sub_A_rows = A_mat_rows.block(0, 0, NUM_CONSTRAINTS, Y_SIZE).eval();
  SparseColMatrix zero_mat_15_15(NUM_CONSTRAINTS, NUM_CONSTRAINTS);

  A_mat_rows.makeCompressed();
  A_mat_cols.makeCompressed();
  c_vec.makeCompressed();
  zero_sub_A_cols.makeCompressed();
  zero_sub_A_rows.makeCompressed();
  zero_mat_15_15.makeCompressed();
  auto dual_var = std::shared_ptr<DualVar>(nullptr);

  return std::make_shared<PnpProblemSolver>(
      A_mat_rows, A_mat_cols, c_vec, zero_sub_A_rows, zero_sub_A_cols,
      zero_mat_15_15, init_y_vec, dual_var);
}
} // namespace NPnP
