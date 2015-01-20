from sklearn import datasets
import numpy as np
from skimage.feature import hog
import cv2
import getopt
import sys
import math
import time

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
def scan_rotate_line(img_scale=0.2, rot_angle=30):
    def scan_rotate_line_in(src, func) : 
        _feat = []
        x = rotate_about_center(src, 0, img_scale)
        for ang in range(0, 180, rot_angle) :
            rot_x = rotate_about_center(x, ang)
            for row in rot_x :
                _feat.append(func(row))
        return _feat
    return scan_rotate_line_in
def scan_rotate_segment(img_scale=0.2, rot_angle=30, seg_gap=3, seg_len=9, col_gap=3, col_siz=5, col_thres=2) : 
    def scan_rotate_segment_in(src, func):
        _feat = []
        for ang in range(0, 180, rot_angle) :
            rot_src = rotate_about_center(src, ang)
            pr = pack_rows(rot_src, col_gap, col_siz, col_thres)
#            cv2.imshow('im', pr)
#            cv2.waitKey()
#            cv2.destroyWindow('im')

            for row in pr :
                for i in range(0, len(row)-seg_len+seg_gap, seg_gap):
                    _feat.append(func(row[i:i+seg_len]))
        return _feat
    return scan_rotate_segment_in

def pack_rows(src, gap, siz, thres):
    src = np.transpose(src)
    new_sy = (src.shape[0]-siz)/gap
    new_sx = src.shape[1]
    dst = np.zeros((new_sy, new_sx))
    st = src
    for i in range(1,src.shape[0]):
        st[i] = np.add(st[i], st[i-1])
    for i in range(0, src.shape[0]-siz-1, gap):
        dst[i/gap] = st[i+siz]-st[i]-thres
    #    dst[i/gap] = min(max(dst[i/gap][j], 0),1)
    #for i in range(0, src.shape[0]-2*siz, gap):
    #    for j in range(new_sx):
    #        dst[i/gap][j] = st[i+siz][j]-st[i][j] - thres
    #        dst[i/gap][j] = min(max(dst[i/gap][j], 0),1)
    return dst

def conti_sq_func(gray_thres=0.1, res_thres=0.2, res_ratio=1) : 
    def conti_sq_func_in(line):
        res = 0
        nw_len = 0
        for ent in line :
            if ent>gray_thres :
                nw_len += 1
            else :
                res += nw_len*nw_len
                nw_len = 0
        res += nw_len*nw_len
        res = math.sqrt(res) / len(line) * res_ratio
        res = (res-res_thres) / (1-res_thres)
        return max(min(res, 1.0), 0.0)
    return conti_sq_func_in
def disct_pnt_func(gray_thres=0.2, res_thres=0.2, res_ratio=0.05) : 
    def disct_pnt_func_in(line) :
        res = 0
        nw_len = 0
        for ent in line+[0] :
            if ent>gray_thres :
                nw_len += 1
            elif nw_len>0 :
                res += 1.0/nw_len
                nw_len = 0
        res = math.sqrt(res) * res_ratio
        res = (res-res_thres) / (1-res_thres)
        return max(min(res, 1.0), 0.0)
    return disct_pnt_func_in
def sum_seg_func(gray_thres=0.1, res_thres=0.2, res_ratio=0.05):
    def sum_seg_func(seg):
        res = 0
        for x in seg:
            if x>gray_threshold:
                res += 1.0
        res /= len(seg)
        res = math.sqrt(res) * res_ratio
        res = (res-res_thres) / (1-res_thres)
        return max(min(res, 1.0), 0.0)
    return sum_seg_func_in




def cut_mat_border(src, tol, bdr, rat):
    sy = src.shape[0]
    sx = src.shape[1]
    srct = np.transpose(src)
    s_sum = get_mat_sum(src)
    t_sum = 0
    _res = []
    for m, rng in [(src, range(sy)), (src,range(sy-1,-1,-1)), (srct,range(sx)), (srct,range(sx-1,-1,-1))]:
        t_sum = 0
        for i in rng:
            t_sum += sum(m[i])
            if t_sum>s_sum*tol:
                break
        _res.append(i)
    (u, b, l, r) = (_res[0], _res[1], _res[2], _res[3])
    #u = max(_res[0]-bdr,0)
    u = u/(1+rat)
    #b = min(_res[1]+bdr,sy)
    b = (b+sy*rat)/(1+rat)
    #l = max(_res[2]-bdr,0)
    l = l/(1+rat)
    #r = min(_res[3]+bdr,sx)
    r = (r+sx*rat)/(1+rat)
    return src[u:b, l:r]
        
def get_mat_sum(src):
    return np.sum(np.sum(src))
def get_mat_mean(src, thres):
    _res = 0
    _resx = 0
    _resy = 0
    for i,r in enumerate(src):
        for j,e in enumerate(r):
            if e>thres:
                _res += 1.0
                _resx += j
                _resy += i
    return (_resy/_res, _resx/_res)
def enhance(src,thres):
    dst = src.copy()
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i][j]>thres:
                dst[i][j] = 1
            else:
                dst[i][j] = 0
    return dst

def refine(x, siz=(70,70), sum_thres=50, bdr_rat=0.3, siz_thres=25):
    #x = enhance(x, 0.01)
    x = cut_mat_border(x, 0.03, 5, bdr_rat)
    if x.shape[0]*x.shape[1] < siz_thres:
        return None
    x = cv2.resize(x, siz)
    #x = cv2.blur(x, (3,3))
    x = enhance(x, 0.01)
    return x


def show_usage():
    print "python scanRotateLine.py [OPTIONS...] datName"
    print "     -n              number of x"
    print "     -o <output>     save output data in <output>"
    print "     -v              verbal"
    print "project: numFeat=12810"

def msg(s, newline=True):
    if verbal:
        sys.stdout.write(s)
        if newline:
            sys.stdout.write('\n')
        sys.stdout.flush()
        
if __name__ == "__main__" : 

    num_read = 10000000

    opts, args = getopt.getopt(sys.argv[1:], "hvo:", [])
    if args==[] :
        show_usage()
        exit()

    dat_name = args[0]
    out_name = dat_name[:-4]+"_srl.dat"
    verbal = False

    for o, a in opts :
        if o=="-h": 
            show_usage()
            exit()
        elif o=="-o":
            out_name = a
        elif o=="-v":
            verbal = True


    msg("will try to load data.")
    x_train, y_train = datasets.load_svmlight_file(dat_name, n_features=12810)
    msg("load features done.")
    x_train = x_train.toarray()
    msg("feature to array done.")
    res_x = []
    res_y = []
    num_feat = 0
    msg("dealing with " + str(len(x_train)) + " data...")
    start_time = time.time()


    scan_rot_line = scan_rotate_line(img_scale=0.5, rot_angle=30)
    conti_sq = conti_sq_func(gray_thres=0.05, res_thres=0.15, res_ratio=1.0)
    scan_rot_seg = scan_rotate_segment(img_scale=1, seg_gap=3, seg_len=9, rot_angle=30, col_gap=3, col_siz=5, col_thres=1)
    conti_sq = conti_sq_func(gray_thres=0.05, res_thres=0.35, res_ratio=1.0)
    for (i, x) in enumerate(x_train) :
        x = x.reshape(122, 105)
        x = refine(x, (50,50), bdr_rat=.2)
        if x==None:
            continue
        feat = [0]
        feat = feat + scan_rot_seg(x, conti_sq)
        num_feat = len(feat)
        res_x.append(feat)
        res_y.append(y_train[i])
        if i!=0 : # show dots.
            if i%50==0 :
                msg(str("."), newline=False)
            if i%1000==0 :
                msg(" done:" + str(i) + "  " + str(time.time()-start_time) + " till now.")
        if i>num_read : break
    msg("\nall data done.")
    msg("used " + str(time.time()-start_time) + " seconds.")
    res_x = np.asarray(res_x)
    msg("list to array done.")
    datasets.dump_svmlight_file(res_x, res_y, out_name, )
    print "write to " + out_name
    print "there are total " + str(num_feat-1) + " features per datum."



