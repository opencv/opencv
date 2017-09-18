// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2016, Itseez, Inc, all rights reserved.

#include "test_precomp.hpp"
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;


TEST(Imgproc_FitEllipseDirect_Issue_1, accuracy) {
    vector<Point2f>pts;
    pts.push_back(Point2f(173.41854895999165, 125.84473135880411));
    pts.push_back(Point2f(180.63769498640912, 130.960006577589));
    pts.push_back(Point2f(174.99173759130173, 137.34265632926764));
    pts.push_back(Point2f(170.9044645313217, 141.68017556480243));
    pts.push_back(Point2f(163.48965388499656, 141.9404438924043));
    pts.push_back(Point2f(159.37687818401147, 148.60835331594876));
    pts.push_back(Point2f(150.38917629356735, 155.68825577720446));
    pts.push_back(Point2f(147.16319653316862, 157.06039984963923));
    pts.push_back(Point2f(141.73118707843207, 157.2570155198414));
    pts.push_back(Point2f(130.61569602948597, 159.40742182929364));
    pts.push_back(Point2f(127.00573042229027, 161.34430232187867));
    pts.push_back(Point2f(120.49383815053747, 163.72610883128334));
    pts.push_back(Point2f(114.62383760040998, 162.6788666385239));
    pts.push_back(Point2f(108.84871269183333, 161.90597054388132));
    pts.push_back(Point2f(103.04574087829076, 167.44352944383985));
    pts.push_back(Point2f(96.31623870161255, 163.71641295746116));
    pts.push_back(Point2f(89.86174417295126, 157.2967811253635));
    pts.push_back(Point2f(84.27940674801192, 168.6331304010667));
    pts.push_back(Point2f(76.61995117937661, 159.4445412678832));
    pts.push_back(Point2f(72.22526316142418, 154.60770776728293));
    pts.push_back(Point2f(64.97742405067658, 152.3687174339018));
    pts.push_back(Point2f(58.34612797237003, 155.61116802371583));
    pts.push_back(Point2f(55.59089117268539, 148.56245696566418));
    pts.push_back(Point2f(45.22711195983706, 145.6713241271927));
    pts.push_back(Point2f(40.090542298840234, 142.36141304004002));
    pts.push_back(Point2f(31.788996807277414, 136.26164877915585));
    pts.push_back(Point2f(27.27613006088805, 137.46860042141503));
    pts.push_back(Point2f(23.972392188502226, 129.17993872328594));
    pts.push_back(Point2f(20.688046711616977, 121.52750840733087));
    pts.push_back(Point2f(14.635115184257643, 115.36942800110485));
    pts.push_back(Point2f(14.850919318756809, 109.43609786936987));
    pts.push_back(Point2f(7.476847697758103, 102.67657265589285));
    pts.push_back(Point2f(1.8896944088091914, 95.78878215565676));
    pts.push_back(Point2f(1.731997022935417, 88.17674033990495));
    pts.push_back(Point2f(1.6780841363402033, 80.65581939883002));
    pts.push_back(Point2f(0.035330281415411946, 73.1088693846768));
    pts.push_back(Point2f(0.14652518786238033, 65.42769523404296));
    pts.push_back(Point2f(6.99914645302843, 58.436451064804245));
    pts.push_back(Point2f(6.719616410428614, 50.15263031354927));
    pts.push_back(Point2f(5.122267598477748, 46.03603214691343));
    
    bool directGoodQ;
    float tol = 0.01;
    
    RotatedRect     ellipseDirectTrue = cv::RotatedRect(Point2f(91.3256, 90.4668),Size2f(187.211, 140.031), 21.5808);
    RotatedRect     ellipseDirectTest = fitEllipseDirect(pts);
    Point2f         ellipseDirectTrueVertices[4];
    Point2f         ellipseDirectTestVertices[4];
    ellipseDirectTest.points(ellipseDirectTestVertices);
    ellipseDirectTrue.points(ellipseDirectTrueVertices);
    float directDiff = 0.0;
    for (size_t i=0; i <=3; i++) {
        Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[0];
        float d = diff.x * diff.x + diff.y * diff.y;
        for (size_t j=1; i <=3; i++) {
            Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[j];
            float dd = diff.x * diff.x + diff.y * diff.y;
            if(dd<d){d=dd;}
        }
        directDiff += std::sqrt(d);
    }
    directGoodQ = directDiff < tol;
    
    EXPECT_TRUE(directGoodQ);
}

TEST(Imgproc_FitEllipseDirect_Issue_2, accuracy) {
    vector<Point2f>pts;
    pts.push_back(Point2f(436.59985753246326, 99.52113368023126));
    pts.push_back(Point2f(454.40214161915856, 160.47565296546912));
    pts.push_back(Point2f(406.01996690372687, 215.41999534561575));
    pts.push_back(Point2f(362.8738685722881, 262.1842668997318));
    pts.push_back(Point2f(300.72864073265407, 290.8182699272777));
    pts.push_back(Point2f(247.62963883830972, 311.383137106776));
    pts.push_back(Point2f(194.15394659099445, 313.30260991427565));
    pts.push_back(Point2f(138.934393338296, 310.50203123324223));
    pts.push_back(Point2f(91.66999301197541, 300.57303988670515));
    pts.push_back(Point2f(28.286233855826133, 268.0670159317756));
    
    bool directGoodQ;
    float tol = 0.01;
    
    RotatedRect     ellipseDirectTrue = cv::RotatedRect(Point2f(228.232, 174.879),Size2f(450.68, 265.556), 166.181);
    RotatedRect     ellipseDirectTest = fitEllipseDirect(pts);
    Point2f         ellipseDirectTrueVertices[4];
    Point2f         ellipseDirectTestVertices[4];
    ellipseDirectTest.points(ellipseDirectTestVertices);
    ellipseDirectTrue.points(ellipseDirectTrueVertices);
    float directDiff = 0.0;
    for (size_t i=0; i <=3; i++) {
        Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[0];
        float d = diff.x * diff.x + diff.y * diff.y;
        for (size_t j=1; i <=3; i++) {
            Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[j];
            float dd = diff.x * diff.x + diff.y * diff.y;
            if(dd<d){d=dd;}
        }
        directDiff += std::sqrt(d);
    }
    directGoodQ = directDiff < tol;
    
    EXPECT_TRUE(directGoodQ);
}


TEST(Imgproc_FitEllipseDirect_Issue_3, accuracy) {
    vector<Point2f>pts;
    pts.push_back(Point2f(459.59217920219083, 480.1054989283611));
    pts.push_back(Point2f(427.2759071813645, 501.82653857689616));
    pts.push_back(Point2f(388.35145730295574, 520.9488690267101));
    pts.push_back(Point2f(349.53248668650656, 522.9153107979839));
    pts.push_back(Point2f(309.56018996762094, 527.449631776843));
    pts.push_back(Point2f(272.07480726768665, 508.12367135706165));
    pts.push_back(Point2f(234.69230939247115, 519.8943877180591));
    pts.push_back(Point2f(201.65185545142472, 509.47870288702813));
    pts.push_back(Point2f(169.37222144138462, 498.2681549419808));
    pts.push_back(Point2f(147.96233740677815, 467.0923094529034));
    pts.push_back(Point2f(109.68331701139209, 433.39069422941986));
    pts.push_back(Point2f(81.95454413977822, 397.34325168750087));
    pts.push_back(Point2f(63.74923800767195, 371.939105294963));
    pts.push_back(Point2f(39.966434417279885, 329.9581349942296));
    pts.push_back(Point2f(21.581668415402532, 292.6692716276865));
    pts.push_back(Point2f(13.687334926511767, 248.91164234903772));
    pts.push_back(Point2f(0., 201.25693715845716));
    pts.push_back(Point2f(3.90259455356599, 155.68155247210575));
    pts.push_back(Point2f(39.683930802331844, 110.26290871953987));
    pts.push_back(Point2f(47.85826684019932, 70.82454140948524));
    
    bool directGoodQ;
    float tol = 0.01;
    
    RotatedRect     ellipseDirectTrue = cv::RotatedRect(Point2f(255.326, 272.626),Size2f(570.999, 434.23), 49.0265);
    RotatedRect     ellipseDirectTest = fitEllipseDirect(pts);
    Point2f         ellipseDirectTrueVertices[4];
    Point2f         ellipseDirectTestVertices[4];
    ellipseDirectTest.points(ellipseDirectTestVertices);
    ellipseDirectTrue.points(ellipseDirectTrueVertices);
    float directDiff = 0.0;
    for (size_t i=0; i <=3; i++) {
        Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[0];
        float d = diff.x * diff.x + diff.y * diff.y;
        for (size_t j=1; i <=3; i++) {
            Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[j];
            float dd = diff.x * diff.x + diff.y * diff.y;
            if(dd<d){d=dd;}
        }
        directDiff += std::sqrt(d);
    }
    directGoodQ = directDiff < tol;
    
    EXPECT_TRUE(directGoodQ);
}

TEST(Imgproc_FitEllipseDirect_Issue_4, accuracy) {
    vector<Point2f>pts;
    pts.push_back(Point2f(461.1761758124861, 79.55196261616746));
    pts.push_back(Point2f(470.5034888757249, 100.56760245239015));
    pts.push_back(Point2f(470.7814479849749, 127.45783922150272));
    pts.push_back(Point2f(465.214384653262, 157.51792078285405));
    pts.push_back(Point2f(465.3739691861813, 185.89204350118942));
    pts.push_back(Point2f(443.36043162278366, 214.43399982709002));
    pts.push_back(Point2f(435.04682693174095, 239.2657073987589));
    pts.push_back(Point2f(444.48553588292697, 262.0816619678671));
    pts.push_back(Point2f(407.1290185495328, 285.07828783776347));
    pts.push_back(Point2f(397.71436554935804, 304.782713567108));
    pts.push_back(Point2f(391.65678619785854, 323.6809382153118));
    pts.push_back(Point2f(366.3904205781036, 328.09416679736563));
    pts.push_back(Point2f(341.7656517790918, 346.9672607008338));
    pts.push_back(Point2f(335.8021864809171, 358.22416661090296));
    pts.push_back(Point2f(313.29224574204227, 373.3267160317279));
    pts.push_back(Point2f(291.121216115417, 377.3339312050791));
    pts.push_back(Point2f(284.20367595990547, 389.5930108233698));
    pts.push_back(Point2f(270.9682061106809, 388.4352006517971));
    pts.push_back(Point2f(253.10188273008825, 392.35120876055373));
    pts.push_back(Point2f(234.2306946938868, 407.0773705761117));
    pts.push_back(Point2f(217.0544384092144, 407.54850609237235));
    pts.push_back(Point2f(198.40910966657933, 423.7008860314684));
    pts.push_back(Point2f(175.47011114845057, 420.4223434173364));
    pts.push_back(Point2f(154.92083551695902, 418.5288198459268));
    pts.push_back(Point2f(136.52988517939698, 417.8311217226818));
    pts.push_back(Point2f(114.74657291069317, 410.1534699388714));
    pts.push_back(Point2f(78.9220388330042, 397.6266608135022));
    pts.push_back(Point2f(76.82658673144391, 404.27399269891055));
    pts.push_back(Point2f(50.953595435605116, 386.3824077178053));
    pts.push_back(Point2f(43.603489077456985, 368.7894972436907));
    pts.push_back(Point2f(19.37402592752713, 343.3511017547511));
    pts.push_back(Point2f(8.714663367287343, 322.2148323327599));
    pts.push_back(Point2f(0., 288.7836318007535));
    pts.push_back(Point2f(3.98686689837605, 263.1748167870333));
    pts.push_back(Point2f(9.536389714519785, 233.02995195684738));
    pts.push_back(Point2f(17.83246556512455, 205.6536519851621));
    pts.push_back(Point2f(33.00593702846919, 180.52628138608327));
    pts.push_back(Point2f(41.572400996463394, 153.95185568689314));
    pts.push_back(Point2f(54.55733659450332, 136.54322891729444));
    pts.push_back(Point2f(78.60990563833005, 112.76538180538182));
    
    bool directGoodQ;
    float tol = 0.01;
    
    RotatedRect     ellipseDirectTrue = cv::RotatedRect(Point2f(236.836, 208.089),Size2f(515.893, 357.166), -35.9996);
    RotatedRect     ellipseDirectTest = fitEllipseDirect(pts);
    Point2f         ellipseDirectTrueVertices[4];
    Point2f         ellipseDirectTestVertices[4];
    ellipseDirectTest.points(ellipseDirectTestVertices);
    ellipseDirectTrue.points(ellipseDirectTrueVertices);
    float directDiff = 0.0;
    for (size_t i=0; i <=3; i++) {
        Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[0];
        float d = diff.x * diff.x + diff.y * diff.y;
        for (size_t j=1; i <=3; i++) {
            Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[j];
            float dd = diff.x * diff.x + diff.y * diff.y;
            if(dd<d){d=dd;}
        }
        directDiff += std::sqrt(d);
    }
    directGoodQ = directDiff < tol;
    
    EXPECT_TRUE(directGoodQ);
}



TEST(Imgproc_FitEllipseDirect_Issue_5, accuracy) {
    vector<Point2f>pts;
    pts.push_back(Point2f(509.60609444351917, 484.8233016998119));
    pts.push_back(Point2f(508.55357451809846, 498.61004779125176));
    pts.push_back(Point2f(495.59325478416525, 507.9238702677585));
    pts.push_back(Point2f(455.32905012177747, 517.7518674113691));
    pts.push_back(Point2f(461.24821761238667, 524.2115477440211));
    pts.push_back(Point2f(438.8983455906825, 528.424911702069));
    pts.push_back(Point2f(425.9259699875303, 532.5700430134499));
    pts.push_back(Point2f(405.77496728300616, 535.7295008444993));
    pts.push_back(Point2f(384.31968113982475, 536.3076260371831));
    pts.push_back(Point2f(381.5356536818977, 540.183355729414));
    pts.push_back(Point2f(378.2530503455792, 540.2871855284832));
    pts.push_back(Point2f(357.7242088314752, 543.473075733281));
    pts.push_back(Point2f(339.27871831324853, 541.2099003613087));
    pts.push_back(Point2f(339.22481874867435, 541.1105421426018));
    pts.push_back(Point2f(331.50337377509396, 539.7296050163102));
    pts.push_back(Point2f(317.8306501537862, 540.9077275195326));
    pts.push_back(Point2f(304.9192648323086, 541.3434792768918));
    pts.push_back(Point2f(297.33855427908617, 543.0590309600501));
    pts.push_back(Point2f(288.95330515997694, 543.8756702506837));
    pts.push_back(Point2f(278.5850913122515, 538.1343888329859));
    pts.push_back(Point2f(266.05355938101724, 538.4115695907074));
    pts.push_back(Point2f(255.30186994366096, 534.2459272411796));
    pts.push_back(Point2f(238.52054973466758, 537.5007401480628));
    pts.push_back(Point2f(228.444463024996, 533.8992361116678));
    pts.push_back(Point2f(217.8111623149833, 538.2269193558991));
    pts.push_back(Point2f(209.43502138981037, 532.8057062984569));
    pts.push_back(Point2f(193.33570716763276, 527.2038128630041));
    pts.push_back(Point2f(172.66725340039625, 526.4020881005537));
    pts.push_back(Point2f(158.33654199771337, 525.2093856704676));
    pts.push_back(Point2f(148.65905485249067, 521.0146762179431));
    pts.push_back(Point2f(147.6615365176719, 517.4315201992808));
    pts.push_back(Point2f(122.43568509949394, 514.2089723387337));
    pts.push_back(Point2f(110.88482982039073, 509.14004840857046));
    pts.push_back(Point2f(107.10516681523065, 502.49943180234266));
    pts.push_back(Point2f(82.66611013934804, 494.0581153893113));
    pts.push_back(Point2f(63.573319848965966, 485.6772487054385));
    pts.push_back(Point2f(47.65729058071245, 475.4468806518075));
    pts.push_back(Point2f(19.96819458379347, 463.98285210241943));
    pts.push_back(Point2f(27.855803175234342, 450.2298664426336));
    pts.push_back(Point2f(12.832198085636549, 435.6317753810441));
    
    bool directGoodQ;
    float tol = 0.01;
    
    RotatedRect     ellipseDirectTrue = cv::RotatedRect(Point2f(264.354, 457.336),Size2f(493.728, 162.9), 5.36186);
    RotatedRect     ellipseDirectTest = fitEllipseDirect(pts);
    Point2f         ellipseDirectTrueVertices[4];
    Point2f         ellipseDirectTestVertices[4];
    ellipseDirectTest.points(ellipseDirectTestVertices);
    ellipseDirectTrue.points(ellipseDirectTrueVertices);
    float directDiff = 0.0;
    for (size_t i=0; i <=3; i++) {
        Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[0];
        float d = diff.x * diff.x + diff.y * diff.y;
        for (size_t j=1; i <=3; i++) {
            Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[j];
            float dd = diff.x * diff.x + diff.y * diff.y;
            if(dd<d){d=dd;}
        }
        directDiff += std::sqrt(d);
    }
    directGoodQ = directDiff < tol;
    
    EXPECT_TRUE(directGoodQ);
}

TEST(Imgproc_FitEllipseDirect_Issue_6, accuracy) {
    vector<Point2f>pts;
    pts.push_back(Point2f(414.90156479295905, 29.063453659930833));
    pts.push_back(Point2f(393.79576036337977, 58.59512774879134));
    pts.push_back(Point2f(387.9100725249931, 94.65067695657254));
    pts.push_back(Point2f(351.6987114318621, 124.6049267560123));
    pts.push_back(Point2f(335.3270519942532, 154.52182750730412));
    pts.push_back(Point2f(329.2955843262556, 179.38031343427303));
    pts.push_back(Point2f(322.7316812937696, 201.88774427737036));
    pts.push_back(Point2f(301.48326350826585, 217.63331351026562));
    pts.push_back(Point2f(287.4603938315088, 228.68790184154113));
    pts.push_back(Point2f(273.36617750656023, 234.48397257849905));
    pts.push_back(Point2f(270.7787206270782, 242.85279436204632));
    pts.push_back(Point2f(268.6973828073692, 246.10891460870312));
    pts.push_back(Point2f(261.60715070464255, 252.65744793902192));
    pts.push_back(Point2f(262.9041824871923, 257.1813047575656));
    pts.push_back(Point2f(263.3210079177046, 260.0532193246593));
    pts.push_back(Point2f(248.49568488533242, 264.56723557175013));
    pts.push_back(Point2f(245.4134174127509, 264.87259401292));
    pts.push_back(Point2f(244.73208618171216, 272.32307359830884));
    pts.push_back(Point2f(232.82093196087555, 272.0239734764616));
    pts.push_back(Point2f(235.28539413113458, 276.8668447478244));
    pts.push_back(Point2f(231.9766571511147, 277.71179872893083));
    pts.push_back(Point2f(227.23880706209866, 284.5588878789101));
    pts.push_back(Point2f(222.53202223537826, 282.2293154479012));
    pts.push_back(Point2f(217.27525654729595, 297.42961148365725));
    pts.push_back(Point2f(212.19490057230672, 294.5344078014253));
    pts.push_back(Point2f(207.47417472945446, 301.72230412668307));
    pts.push_back(Point2f(202.11143229969164, 298.8588627545512));
    pts.push_back(Point2f(196.62967096845824, 309.39738607353223));
    pts.push_back(Point2f(190.37809841992106, 318.3250479151242));
    pts.push_back(Point2f(183.1296129732803, 322.35242231955453));
    pts.push_back(Point2f(171.58530535265993, 330.4981441404153));
    pts.push_back(Point2f(160.40092880652247, 337.47275990208226));
    pts.push_back(Point2f(149.44888762618092, 343.42296086656717));
    pts.push_back(Point2f(139.7923528305302, 353.4821948045352));
    pts.push_back(Point2f(121.08414969113318, 359.7010225709457));
    pts.push_back(Point2f(100.10629739219641, 375.3155744055458));
    pts.push_back(Point2f(78.15715630786733, 389.0311284319413));
    pts.push_back(Point2f(51.22820988075294, 396.98646504159547));
    pts.push_back(Point2f(30.71132492338431, 402.85098740402844));
    pts.push_back(Point2f(10.994737323179852, 394.6764602972333));
    
    bool directGoodQ;
    float tol = 0.01;
    
    RotatedRect     ellipseDirectTrue = cv::RotatedRect(Point2f(207.145, 223.308),Size2f(499.583, 117.473), -42.6851);
    RotatedRect     ellipseDirectTest = fitEllipseDirect(pts);
    Point2f         ellipseDirectTrueVertices[4];
    Point2f         ellipseDirectTestVertices[4];
    ellipseDirectTest.points(ellipseDirectTestVertices);
    ellipseDirectTrue.points(ellipseDirectTrueVertices);
    float directDiff = 0.0;
    for (size_t i=0; i <=3; i++) {
        Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[0];
        float d = diff.x * diff.x + diff.y * diff.y;
        for (size_t j=1; i <=3; i++) {
            Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[j];
            float dd = diff.x * diff.x + diff.y * diff.y;
            if(dd<d){d=dd;}
        }
        directDiff += std::sqrt(d);
    }
    directGoodQ = directDiff < tol;
    
    EXPECT_TRUE(directGoodQ);
}

TEST(Imgproc_FitEllipseDirect_Issue_7, accuracy) {
    vector<Point2f>pts;
    pts.push_back(Point2f(386.7497806918209, 119.55623710363142));
    pts.push_back(Point2f(399.0712613744503, 132.61095972401034));
    pts.push_back(Point2f(400.3582576852657, 146.71942033652573));
    pts.push_back(Point2f(383.31046706707906, 160.13631428164982));
    pts.push_back(Point2f(387.1626582455823, 173.82700569763574));
    pts.push_back(Point2f(378.88843308401425, 186.10333319745317));
    pts.push_back(Point2f(367.55061701208, 201.41492900400164));
    pts.push_back(Point2f(360.3254967185148, 209.03834085076022));
    pts.push_back(Point2f(346.2645164278429, 222.03214282040395));
    pts.push_back(Point2f(342.3483403634167, 230.58290419787073));
    pts.push_back(Point2f(326.2900969991908, 240.23679566682756));
    pts.push_back(Point2f(324.5622396580625, 249.56961396707823));
    pts.push_back(Point2f(304.23417130914095, 259.6693711280021));
    pts.push_back(Point2f(295.54035697534675, 270.82284542557704));
    pts.push_back(Point2f(291.7403057147348, 276.1536825048371));
    pts.push_back(Point2f(269.19344116558665, 287.1705579044651));
    pts.push_back(Point2f(256.5350613899267, 274.91264707500943));
    pts.push_back(Point2f(245.93644351417183, 286.12398028743064));
    pts.push_back(Point2f(232.40892420943732, 282.73986583867065));
    pts.push_back(Point2f(216.17957969101082, 293.22229708237705));
    pts.push_back(Point2f(205.66843722622573, 295.7032575625158));
    pts.push_back(Point2f(192.219969335765, 302.6968969534755));
    pts.push_back(Point2f(178.37758801730416, 295.56656776633287));
    pts.push_back(Point2f(167.60089103756644, 301.4629292267722));
    pts.push_back(Point2f(157.44802813915317, 298.90830855734504));
    pts.push_back(Point2f(138.44311818820313, 293.951927187897));
    pts.push_back(Point2f(128.92747660038592, 291.4122695492978));
    pts.push_back(Point2f(119.75160909865994, 282.5809454721714));
    pts.push_back(Point2f(98.48443737042328, 290.39938776333247));
    pts.push_back(Point2f(88.05275635126131, 280.11156058895745));
    pts.push_back(Point2f(82.45799026448167, 271.46668468419773));
    pts.push_back(Point2f(68.04031962064084, 267.8136468580707));
    pts.push_back(Point2f(58.99967170878713, 263.8859310392943));
    pts.push_back(Point2f(41.256097220823484, 260.6041605773932));
    pts.push_back(Point2f(40.66198797608645, 246.64973068177196));
    pts.push_back(Point2f(31.085484380646008, 239.28615601336074));
    pts.push_back(Point2f(24.069417111444253, 225.2228746297288));
    pts.push_back(Point2f(22.10122953275156, 212.75509683149195));
    pts.push_back(Point2f(9.929991244497518, 203.20662088477752));
    pts.push_back(Point2f(0., 190.04891498441148));
    
    bool directGoodQ;
    float tol = 0.01;
    
    RotatedRect     ellipseDirectTrue = cv::RotatedRect(Point2f(199.463, 150.997),Size2f(390.341, 286.01), -12.9696);
    RotatedRect     ellipseDirectTest = fitEllipseDirect(pts);
    Point2f         ellipseDirectTrueVertices[4];
    Point2f         ellipseDirectTestVertices[4];
    ellipseDirectTest.points(ellipseDirectTestVertices);
    ellipseDirectTrue.points(ellipseDirectTrueVertices);
    float directDiff = 0.0;
    for (size_t i=0; i <=3; i++) {
        Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[0];
        float d = diff.x * diff.x + diff.y * diff.y;
        for (size_t j=1; i <=3; i++) {
            Point2f diff = ellipseDirectTrueVertices[i] - ellipseDirectTestVertices[j];
            float dd = diff.x * diff.x + diff.y * diff.y;
            if(dd<d){d=dd;}
        }
        directDiff += std::sqrt(d);
    }
    directGoodQ = directDiff < tol;
    
    EXPECT_TRUE(directGoodQ);
}
