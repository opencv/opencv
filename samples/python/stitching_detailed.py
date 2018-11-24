"""Rotation model images stitcher.
stitching_detailed img1 img2 [...imgN] [flags]
Flags:
    --preview
        Run stitching in the preview mode. Works faster than usual mode,
        but output image will have lower resolution.
    --try_cuda (yes|no)
        Try to use CUDA. The default value is 'no'. All default values
        are for CPU mode.
\nMotion Estimation Flags:
    --work_megapix <float>
        Resolution for image registration step. The default is 0.6 Mpx.
    --features (surf|orb|sift)
        Type of features used for images matching. The default is surf.
    --matcher (homography|affine)
        Matcher used for pairwise image matching.
    --estimator (homography|affine)
        Type of estimator used for transformation estimation.
    --match_conf <float>
        Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.
    --conf_thresh <float>
        Threshold for two images are from the same panorama confidence.
        The default is 1.0.
    --ba (no|reproj|ray|affine)
        Bundle adjustment cost function. The default is ray.
    --ba_refine_mask (mask)
        Set refinement mask for bundle adjustment. It looks like 'x_xxx',
        where 'x' means refine respective parameter and '_' means don't
        refine one, and has the following format:
        <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle
        adjustment doesn't support estimation of selected parameter then
        the respective flag is ignored.
    --wave_correct (no|horiz|vert)
        Perform wave effect correction. The default is 'horiz'.
    --save_graph <file_name>
        Save matches graph represented in DOT language to <file_name> file.
        Labels description: Nm is number of matches, Ni is number of inliers,
        C is confidence.
\nCompositing Flags:
    --warp (affine|plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)
        Warp surface type. The default is 'spherical'.
    --seam_megapix <float>
        Resolution for seam estimation step. The default is 0.1 Mpx.
    --seam (no|voronoi|gc_color|gc_colorgrad)
        Seam estimation method. The default is 'gc_color'.
    --compose_megapix <float>
        Resolution for compositing step. Use -1 for original resolution.
        The default is -1.
    --expos_comp (no|gain|gain_blocks)
        Exposure compensation method. The default is 'gain_blocks'.
    --blend (no|feather|multiband)
        Blending method. The default is 'multiband'.
    --blend_strength <float>
        Blending strength from [0,100] range. The default is 5.
    --output <result_img>
        The default is 'result.jpg'.
    --timelapse (as_is|crop)
        Output warped images separately as frames of a time lapse movie, with 'fixed_' prepended to input file names.
    --rangewidth <int>
        uses range_width to limit number of images to match with.\n
        """
import numpy as np
import cv2 as cv
import sys
import argparse

parser = argparse.ArgumentParser(description='stitching_detailed')
parser.add_argument('img_names', nargs='+',help='files to stitch',type=str)
parser.add_argument('--preview',help='Run stitching in the preview mode. Works faster than usual mode but output image will have lower resolution.',type=bool,dest = 'preview' )
parser.add_argument('--try_cuda',action = 'store', default = False,help='Try to use CUDA. The default value is no. All default values are for CPU mode.',type=bool,dest = 'try_cuda' )
parser.add_argument('--work_megapix',action = 'store', default = 0.6,help=' Resolution for image registration step. The default is 0.6 Mpx',type=float,dest = 'work_megapix' )
parser.add_argument('--features',action = 'store', default = 'orb',help='Type of features used for images matching. The default is orb.',type=str,dest = 'features' )
parser.add_argument('--matcher',action = 'store', default = 'homography',help='Matcher used for pairwise image matching.',type=str,dest = 'matcher' )
parser.add_argument('--estimator',action = 'store', default = 'homography',help='Type of estimator used for transformation estimation.',type=str,dest = 'estimator' )
parser.add_argument('--match_conf',action = 'store', default = 0.3,help='Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.',type=float,dest = 'match_conf' )
parser.add_argument('--conf_thresh',action = 'store', default = 1.0,help='Threshold for two images are from the same panorama confidence.The default is 1.0.',type=float,dest = 'conf_thresh' )
parser.add_argument('--ba',action = 'store', default = 'ray',help='Bundle adjustment cost function. The default is ray.',type=str,dest = 'ba' )
parser.add_argument('--ba_refine_mask',action = 'store', default = 'xxxxx',help='Set refinement mask for bundle adjustment.  mask is "xxxxx"',type=str,dest = 'ba_refine_mask' )
parser.add_argument('--wave_correct',action = 'store', default = 'horiz',help='Perform wave effect correction. The default is "horiz"',type=str,dest = 'wave_correct' )
parser.add_argument('--save_graph',action = 'store', default = None,help='Save matches graph represented in DOT language to <file_name> file.',type=str,dest = 'save_graph' )
parser.add_argument('--warp',action = 'store', default = 'plane',help='Warp surface type. The default is "spherical".',type=str,dest = 'warp' )
parser.add_argument('--seam_megapix',action = 'store', default = 0.1,help=' Resolution for seam estimation step. The default is 0.1 Mpx.',type=float,dest = 'seam_megapix' )
parser.add_argument('--seam',action = 'store', default = 'no',help='Seam estimation method. The default is "gc_color".',type=str,dest = 'seam' )
parser.add_argument('--compose_megapix',action = 'store', default = -1,help='Resolution for compositing step. Use -1 for original resolution.',type=float,dest = 'compose_megapix' )
parser.add_argument('--expos_comp',action = 'store', default = 'no',help='Exposure compensation method. The default is "gain_blocks".',type=str,dest = 'expos_comp' )
parser.add_argument('--blend',action = 'store', default = 'multiband',help='Blending method. The default is "multiband".',type=str,dest = 'blend' )
parser.add_argument('--blend_strength',action = 'store', default = 5,help='Blending strength from [0,100] range.',type=int,dest = 'blend_strength' )
parser.add_argument('--output',action = 'store', default = 'result.jpg',help='The default is "result.jpg"',type=str,dest = 'output' )
parser.add_argument('--timelapse',action = 'store', default = None,help='Output warped images separately as frames of a time lapse movie, with "fixed_" prepended to input file names.',type=str,dest = 'timelapse' )
parser.add_argument('--rangewidth',action = 'store', default = -1,help='uses range_width to limit number of images to match with.',type=int,dest = 'rangewidth' )
args = parser.parse_args()
img_names=args.img_names
print(img_names)
preview = args.preview
try_cuda = args.try_cuda
work_megapix = args.work_megapix
seam_megapix = args.seam_megapix
compose_megapix = args.compose_megapix
conf_thresh = args.conf_thresh
features_type = args.features
matcher_type = args.matcher
estimator_type = args.estimator
ba_cost_func = args.ba
ba_refine_mask = args.ba_refine_mask
wave_correct = args.wave_correct
if wave_correct=='no':
    do_wave_correct= False
else:
    do_wave_correct=True
if args.save_graph is None:
    save_graph = False
else:
    save_graph =True
    save_graph_to = args.save_graph
warp_type = args.warp
if args.expos_comp=='no':
    expos_comp_type = cv.detail.ExposureCompensator_NO
elif  args.expos_comp=='gain':
    expos_comp_type = cv.detail.ExposureCompensator_GAIN
elif  args.expos_comp=='gain_blocks':
    expos_comp_type = cv.detail.ExposureCompensator_GAIN_BLOCKS
else:
    print("Bad exposure compensation method")
    exit

match_conf = args.match_conf
seam_find_type = args.seam
blend_type = args.blend
blend_strength = args.blend_strength
result_name = args.output
if args.timelapse is not None:
    timelapse = True
    if args.timelapse=="as_is":
        timelapse_type = cv.detail.Timelapser_AS_IS
    elif args.timelapse=="crop":
        timelapse_type = cv.detail.Timelapser_CROP
    else:
        print("Bad timelapse method")
        exit()
else:
    timelapse= False
range_width = args.rangewidth
if features_type=='orb':
    finder= cv.ORB.create()
elif features_type=='surf':
    finder= cv.xfeatures2d_SURF.create()
elif features_type=='sift':
    finder= cv.xfeatures2d_SIFT.create()
else:
    print ("Unknown descriptor type")
    exit()
seam_work_aspect = 1
full_img_sizes=[]
features=[]
images=[]
is_work_scale_set = False
is_seam_scale_set = False
is_compose_scale_set = False;
for name in img_names:
    full_img = cv.imread(name)
    if full_img is None:
        print("Cannot read image ",name)
        exit()
    full_img_sizes.append((full_img.shape[1],full_img.shape[0]))
    if work_megapix < 0:
        img = full_img
        work_scale = 1
        is_work_scale_set = True
    else:
        if is_work_scale_set is False:
            work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0]*full_img.shape[1])))
            is_work_scale_set = True
        img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
    if is_seam_scale_set is False:
        seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0]*full_img.shape[1])))
        seam_work_aspect = seam_scale / work_scale
        is_seam_scale_set = True
    imgFea= cv.detail.computeImageFeatures2(finder,img)
    features.append(imgFea)
    img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
    images.append(img)
if matcher_type== "affine":
    matcher = cv.detail.AffineBestOf2NearestMatcher_create(False, try_cuda, match_conf)
elif range_width==-1:
    matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
else:
    matcher = cv.detail.BestOf2NearestRangeMatcher_create(range_width, try_cuda, match_conf)
p=matcher.apply2(features)
matcher.collectGarbage()
if save_graph:
    f = open(save_graph_to,"w")
#        f.write(matchesGraphAsString(img_names, pairwise_matches, conf_thresh))
    f.close()
indices=cv.detail.leaveBiggestComponent(features,p,0.3)
img_subset =[]
img_names_subset=[]
full_img_sizes_subset=[]
num_images=len(indices)
for i in range(0,num_images):
    img_names_subset.append(img_names[indices[i,0]])
    img_subset.append(images[indices[i,0]])
    full_img_sizes_subset.append(full_img_sizes[indices[i,0]])
images = img_subset;
img_names = img_names_subset;
full_img_sizes = full_img_sizes_subset;
num_images = len(img_names)
if num_images < 2:
    print("Need more images")
    exit()

if estimator_type == "affine":
    estimator = cv.detail_AffineBasedEstimator()
else:
    estimator = cv.detail_HomographyBasedEstimator()
b, cameras =estimator.apply(features,p,None)
if not b:
    print("Homography estimation failed.")
    exit()
for cam in cameras:
    cam.R=cam.R.astype(np.float32)

if ba_cost_func == "reproj":
    adjuster = cv.detail_BundleAdjusterReproj()
elif ba_cost_func == "ray":
    adjuster = cv.detail_BundleAdjusterRay()
elif ba_cost_func == "affine":
    adjuster = cv.detail_BundleAdjusterAffinePartial()
elif ba_cost_func == "no":
    adjuster = cv.detail_NoBundleAdjuster()
else:
    print( "Unknown bundle adjustment cost function: ", ba_cost_func )
    exit()
adjuster.setConfThresh(1)
refine_mask=np.zeros((3,3),np.uint8)
if ba_refine_mask[0] == 'x':
    refine_mask[0,0] = 1
if ba_refine_mask[1] == 'x':
    refine_mask[0,1] = 1
if ba_refine_mask[2] == 'x':
    refine_mask[0,2] = 1
if ba_refine_mask[3] == 'x':
    refine_mask[1,1] = 1
if ba_refine_mask[4] == 'x':
    refine_mask[1,2] = 1
adjuster.setRefinementMask(refine_mask)
b,cameras = adjuster.apply(features,p,cameras)
if not b:
    print("Camera parameters adjusting failed.")
    exit()
focals=[]
for cam in cameras:
    focals.append(cam.focal)
sorted(focals)
if len(focals)%2==1:
    warped_image_scale = focals[len(focals) // 2]
else:
    warped_image_scale = (focals[len(focals) // 2]+focals[len(focals) // 2-1])/2
if do_wave_correct:
    rmats=[]
    for cam in cameras:
        rmats.append(np.copy(cam.R))
    rmats	=	cv.detail.waveCorrect(	rmats,  cv.detail.WAVE_CORRECT_HORIZ)
    for idx,cam in enumerate(cameras):
        cam.R = rmats[idx]
corners=[]
mask=[]
masks_warped=[]
images_warped=[]
sizes=[]
masks=[]
for i in range(0,num_images):
    um=cv.UMat(255*np.ones((images[i].shape[0],images[i].shape[1]),np.uint8))
    masks.append(um)

warper = cv.PyRotationWarper(warp_type,warped_image_scale*seam_work_aspect) # warper peut etre nullptr?
for i in range(0,num_images):
    K = cameras[i].K().astype(np.float32)
    swa = seam_work_aspect
    K[0,0] *= swa
    K[0,2] *= swa
    K[1,1] *= swa
    K[1,2] *= swa
    corner,image_wp =warper.warp(images[i],K,cameras[i].R,cv.INTER_LINEAR, cv.BORDER_REFLECT)
    corners.append(corner)
    sizes.append((image_wp.shape[1],image_wp.shape[0]))
    images_warped.append(image_wp)

    p,mask_wp =warper.warp(masks[i],K,cameras[i].R,cv.INTER_NEAREST, cv.BORDER_CONSTANT)
    masks_warped.append(mask_wp)
images_warped_f=[]
for img in images_warped:
    imgf=img.astype(np.float32)
    images_warped_f.append(imgf)
compensator=cv.detail.ExposureCompensator_createDefault(expos_comp_type)
compensator.feed(corners, images_warped, masks_warped)
if seam_find_type == "no":
    seam_finder = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)
elif seam_find_type == "voronoi":
    seam_finder = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM);
elif seam_find_type == "gc_color":
    seam_finder = cv.detail_GraphCutSeamFinder("COST_COLOR")
elif seam_find_type == "gc_colorgrad":
    seam_finder = cv.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
elif seam_find_type == "dp_color":
    seam_finder = cv.detail_DpSeamFinder("COLOR")
elif seam_find_type == "dp_colorgrad":
    seam_finder = cv.detail_DpSeamFinder("COLOR_GRAD")
if seam_finder is None:
    print("Can't create the following seam finder ",seam_find_type)
    exit()
seam_finder.find(images_warped_f, corners,masks_warped )
imgListe=[]
compose_scale=1
corners=[]
sizes=[]
images_warped=[]
images_warped_f=[]
masks=[]
blender= None
timelapser=None
compose_work_aspect=1
for idx,name in enumerate(img_names): # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
    full_img  = cv.imread(name)
    if not is_compose_scale_set:
        if compose_megapix > 0:
            compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0]*full_img.shape[1])))
        is_compose_scale_set = True;
        compose_work_aspect = compose_scale / work_scale;
        warped_image_scale *= compose_work_aspect
        warper =  cv.PyRotationWarper(warp_type,warped_image_scale)
        for i in range(0,len(img_names)):
            cameras[i].focal *= compose_work_aspect
            cameras[i].ppx *= compose_work_aspect
            cameras[i].ppy *= compose_work_aspect
            sz = (full_img.shape[1] * compose_scale,full_img.shape[0] * compose_scale)
            K = cameras[i].K().astype(np.float32)
            roi = warper.warpRoi(sz, K, cameras[i].R);
            corners.append(roi[0:2])
            sizes.append(roi[2:4])
    if abs(compose_scale - 1) > 1e-1:
        img =cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale, interpolation=cv.INTER_LINEAR_EXACT)
    else:
        img = full_img;
    img_size = (img.shape[1],img.shape[0]);
    K=cameras[idx].K().astype(np.float32)
    corner,image_warped =warper.warp(img,K,cameras[idx].R,cv.INTER_LINEAR, cv.BORDER_REFLECT)
    mask =255*np.ones((img.shape[0],img.shape[1]),np.uint8)
    p,mask_warped =warper.warp(mask,K,cameras[idx].R,cv.INTER_NEAREST, cv.BORDER_CONSTANT)
    compensator.apply(idx,corners[idx],image_warped,mask_warped)
    image_warped_s = image_warped.astype(np.int16)
    image_warped=[]
    dilated_mask = cv.dilate(masks_warped[idx],None)
    seam_mask = cv.resize(dilated_mask,(mask_warped.shape[1],mask_warped.shape[0]),0,0,cv.INTER_LINEAR_EXACT)
    mask_warped = cv.bitwise_and(seam_mask,mask_warped)
    if blender==None and not timelapse:
        blender = cv.detail.Blender_createDefault(1)
        dst_sz = cv.detail.resultRoi(corners,sizes)
        blend_strength=1
        blend_width = np.sqrt(dst_sz[2]*dst_sz[3]) * blend_strength / 100
        if blend_width < 1:
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
        elif blend_type == "MULTI_BAND":
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_MULTIBAND)
            blender.setNumBands((np.log(blend_width)/np.log(2.) - 1.).astype(np.int))
        elif blend_type == "FEATHER":
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_FEATHER)
            blender.setSharpness(1./blend_width)
        blender.prepare(corners, sizes)
    elif timelapser==None  and timelapse:
        timelapser = cv.detail.createDefault(timelapse_type);
        timelapser.initialize(corners, sizes)
    if timelapse:
        matones=np.ones((image_warped_s.shape[0],image_warped_s.shape[1]), np.uint8)
        timelapser.process(image_warped_s, matones, corners[idx])
        pos_s = img_names[idx].rfind("/");
        if pos_s == -1:
            fixedFileName = "fixed_" + img_names[idx];
        else:
            fixedFileName = img_names[idx][:pos_s + 1 ]+"fixed_" + img_names[idx][pos_s + 1: ]
        cv.imwrite(fixedFileName, timelapser.getDst())
    else:
        blender.feed(image_warped_s, mask_warped, corners[idx])
if not timelapse:
    result=None
    result_mask=None
    result,result_mask = blender.blend(result,result_mask)
    cv.imwrite(result_name,result)
