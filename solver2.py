import cv2
import numpy as np

def dtw(a,b,ca,cb, aid,bid,check_deltas=True):
    best_res = 1e9
    best_delta = None
    min_d, max_d = -2, 3
    min_d, max_d = 0, 1
    location_mult = 0
    if not check_deltas:
        min_d, max_d = 1, 2
    for delta in range(min_d, max_d):
        n = len(a)
        m = len(b)
        dp = [[0 for _ in range(m)] for _ in range(n)]
        pixcola = contour_inner[aid][contour_maps[aid][a]]
        pixcolb = contour_inner[bid][contour_maps[bid][b]]
        #dist = lambda a,b: location_mult * abs(a[0]-b[0]-ca[0]+cb[0] + delta) + location_mult * abs(a[1]-b[1]-ca[1]+cb[1]) + abs(int(im[a[1]][a[0]][0]) - int(im[b[1]][b[0]][0])) + abs(int(im[a[1]][a[0]][1]) - int(im[b[1]][b[0]][1])) + abs(int(im[a[1]][a[0]][2]) - int(im[b[1]][b[0]][2]))
        dist = lambda a,b: location_mult * abs(a[0]-b[0]-ca[0]+cb[0] + delta) + location_mult * abs(a[1]-b[1]-ca[1]+cb[1]) + abs(int(im[pixcola[1]][pixcola[0]][0]) - int(im[pixcolb[1]][pixcolb[0]][0])) + abs(int(im[pixcola[1]][pixcola[0]][1]) - int(im[pixcolb[1]][pixcolb[0]][1])) + abs(int(im[pixcola[1]][pixcola[0]][2]) - int(im[pixcolb[1]][pixcolb[0]][2]))
        for i in range(1,n):
            dp[i][0] = dist(a[i][0],b[0][0])+dp[i-1][0]
        for j in range(1,m):
            dp[0][j] = dist(a[0][0],b[j][0])+dp[0][j-1]
        for i in range(1,n):
            for j in range(1,m):
                d = dist(a[i][0],b[j][0])
                if dp[i-1][j] <= dp[i-1][j-1] and dp[i-1][j] <= dp[i][j-1]:
                    dp[i][j] = d + dp[i-1][j]
                elif dp[i-1][j-1] <= dp[i][j-1]:
                    dp[i][j] = d + dp[i-1][j-1]
                else:
                    dp[i][j] = d + dp[i][j-1]
        if dp[n - 1][m - 1] < best_res:
            best_res = min(best_res, dp[n - 1][m - 1])
            best_delta = delta
    if not check_deltas:
        return best_res
    return best_res, best_delta


def find_pieces():
    corners = []
    corners_coord = []
    corner_pieces = [-1,-1,-1,-1]
    # top, right, bot, left
    edge_pieces = [[],[],[],[]]
    for cii,c in enumerate(contours):
        # find corners
        box = cv2.boundingRect(c)
        rect_corns = [(box[0],box[1]),(box[0]+box[2], box[1]), (box[0]+box[2],box[1]+box[3]), (box[0],box[1]+box[3])]
        corns = [-1,-1,-1,-1]
        corns_d = [1e9,1e9,1e9,1e9]
        corns_id = [-1,-1,-1,-1]
        for i, pix in enumerate(c):
            for ci in range(4):
                d = abs(rect_corns[ci][0]-pix[0][0])+abs(rect_corns[ci][1]-pix[0][1])
                if d<corns_d[ci]:
                    corns_d[ci] = d
                    corns[ci] = pix[0]
                    corns_id[ci] = i
        corners.append(corns_id)
        corners_coord.append(corns)
        for corn in corns:
            cv2.circle(im, corn,2, (0,0,255),2)
        # find starting piece (top left piece)
        max_hor_delta = 0
        for pix in range_wrap(c,corns_id[0],corns_id[3]):
            max_hor_delta = max(max_hor_delta, abs(pix[0][0]-corns[0][0]))
        max_ver_delta = 0
        for pix in range_wrap(c,corns_id[1],corns_id[0]):
            max_ver_delta = max(max_ver_delta, abs(pix[0][1]-corns[0][1]))
        if max_hor_delta < 3 and max_ver_delta < 3:
            corner_pieces[0] = cii
        elif max_ver_delta < 3:
            edge_pieces[0].append(cii)
        elif max_hor_delta < 3:
            edge_pieces[3].append(cii)
        # find line end pieces
        max_hor_delta = 0
        for pix in range_wrap(c,corns_id[2],corns_id[1]):
            max_hor_delta = max(max_hor_delta, abs(pix[0][0]-corns[1][0]))
        if max_hor_delta < 3:
            if cii in edge_pieces[0]:
                edge_pieces[0].remove(cii)
                corner_pieces[1] = cii
            else:
                edge_pieces[1].append(cii)
        # find last line pieces
        max_ver_delta = 0
        for pix in range_wrap(c,corns_id[3],corns_id[2]):
            max_ver_delta = max(max_ver_delta, abs(pix[0][1]-corns[2][1]))
        if max_ver_delta < 3:
            if cii in edge_pieces[1]:
                corner_pieces[2] = cii
                edge_pieces[1].remove(cii)
            elif cii in edge_pieces[3]:
                corner_pieces[3] = cii
                edge_pieces[3].remove(cii)
            else:
                edge_pieces[2].append(cii)
    return corners, corners_coord, corner_pieces, edge_pieces

def find_res():
    row_w = len(edge_pieces[0]) + 2
    col_h = len(edge_pieces[1]) + 2
    # create mapping to fit pieces in
    res = [[(corner_pieces[0], 0)]]
    q = set(i for i in range(len(contours)) if i not in corner_pieces and max(i in edge_pieces[j] for j in range(4)) == 0)
    for i in range(row_w * col_h - 1):
        if len(res) == 1 and len(res[0]) == row_w-1:
            res[-1].append((corner_pieces[1], 0))
            continue
        elif len(res) == col_h-1 and len(res[-1]) == row_w:
            res.append([(corner_pieces[3], 0)])
            continue
        elif len(res) == col_h and len(res[-1]) == row_w-1:
            res[-1].append((corner_pieces[2], 0))
            continue
        if len(res[-1]) == row_w:
            id, _ = res[-1][0]
            edge1 = range_wrap(contours[id],corners[id][3],corners[id][2])
            best_diff = 1e9
            best_id = -1
            best_delta = 0
            bag = edge_pieces[3]
            for id2 in bag:
                edge2 = range_wrap(contours[id2],corners[id2][1],corners[id2][0])
                #diff = sum(abs(edge1[i][0][1]-corners_coord[id][3][1]-edge2[-i-1][0][1]+corners_coord[id2][0][1]) for i in range(min(len(edge1),len(edge2))))
                diff, delta = dtw(edge1,edge2[::-1],corners_coord[id][3],corners_coord[id2][0],id,id2)
                print(f"type1: {id} vs {id2}: {diff}")
                if diff<best_diff:
                    best_diff = diff
                    best_id = id2
                    best_delta = delta
            res.append([(best_id, best_delta)])
        else:
            id, _ = res[-1][-1]
            edge1 = range_wrap(contours[id],corners[id][2],corners[id][1])
            if len(res)>1:
                idb, _ = res[-2][len(res[-1])]
                edge1b = range_wrap(contours[idb],corners[idb][3],corners[idb][2])
            best_diff = 1e9
            best_id = -1
            if len(res)==1:
                bag = edge_pieces[0]
            elif len(res[-1])==row_w-1:
                bag = edge_pieces[1]
            elif len(q)==0:
                bag = edge_pieces[2]
            else:
                bag = q
            for id2 in bag:
                edge2 = range_wrap(contours[id2],corners[id2][0],corners[id2][3])
                #diff = sum(abs(edge1[i][0][0]-corners_coord[id][1][0]-edge2[-i-1][0][0]+corners_coord[id2][0][0]) for i in range(min(len(edge1),len(edge2))))
                diff, delta = dtw(edge1,edge2[::-1],corners_coord[id][1],corners_coord[id2][0],id,id2)
                #if len(res)>1:
                #    edge2b = range_wrap(contours[id2],corners[id2][3],corners[id2][2])
                #    diff_mod = dtw(edge1b,edge2b[::-1],corners_coord[idb][3],corners_coord[id2][0], False)
                #    diff += diff_mod
                print(f"type2: {id} vs {id2}: {diff}")
                if diff<best_diff:
                    best_diff = diff
                    best_id = id2
                    best_delta = delta
            res[-1].append((best_id, best_delta))
        print(i, best_id)
        if best_id == -1:
            print(res)
            print(len(res))
            print(len(res[-1]))
        bag.remove(best_id)
    return res


def calc_res_img():
    res_im = np.zeros((2000,2000,3),dtype=np.uint8)
    ctr = (10,10)
    next_line_start = (10,10)
    for l in res:
        ctr = next_line_start
        next_line_start = (ctr[0],ctr[1]+corners_coord[l[0][0]][3][1]-corners_coord[l[0][0]][0][1])
        for id, delta in l:
            # dfs over pixels in contour
            q = [(corners_coord[id][0][0]+(corners_coord[id][2][0]-corners_coord[id][0][0])//2,corners_coord[id][0][1]+(corners_coord[id][2][1]-corners_coord[id][0][1])//2)]
            seen = set(q[0])
            while q:
                node = q.pop()
                res_im[ctr[1]+node[1]-corners_coord[id][0][1],ctr[0]+node[0]-corners_coord[id][0][0]]=im[node[1],node[0]]
                for delta in [(1,0),(-1,0),(0,1),(0,-1)]:
                    next = (node[0]+delta[0],node[1]+delta[1])
                    if next not in contour_sets[id] and next not in seen:
                        seen.add(next)
                        q.append(next)
            ctr = (ctr[0]+corners_coord[l[0][0]][1][0]-corners_coord[l[0][0]][0][0],ctr[1])
    return res_im

range_wrap = lambda l, a, b: l[a:b] if a < b else np.concatenate((l[a:], l[:b]))
manh_d = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])

if True:
    im = cv2.imread("input_shuffled.png")
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 35, 255, 0)
else:
    im = cv2.imread("input_shuffled_small.png")
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 135, 255, cv2.THRESH_BINARY_INV)
contours, hierarch = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# calculate areas
contours = [(cv2.contourArea(c),c) for c in contours]
# filter out very small contours
contours = [c for c in contours if c[0]>10]
to_remove = set()
for ci1 in range(len(contours)):
    for ci2 in range(ci1+1,len(contours)):
        # box is tuple x,y,w,h
        box1 = cv2.boundingRect(contours[ci1][1])
        box2 = cv2.boundingRect(contours[ci2][1])
        if not(box2[0]>box1[0]+box1[2] or box2[0]+box2[2]<box1[0] or box2[1]>box1[1]+box1[3] or box2[1]+box2[3]<box1[1]):
            if box1[0]<=box2[0] and box2[0]+box2[2]<=box1[0]+box1[2] and box1[1]<=box2[1] and box2[1]+box2[3]<=box1[1]+box1[3] or \
               box2[0]<=box1[0] and box1[0]+box1[2]<=box2[0]+box2[2] and box2[1]<=box1[1] and box1[1]+box1[3]<=box2[1]+box2[3]:
               overlap = 1
            else:
                overlap = 0
            for pix1 in contours[ci1][1]:
                if overlap:
                    break
                for pix2 in contours[ci2][1]:
                    if pix1[0][0] == pix2[0][0] and pix1[0][1] == pix2[0][1]:
                        overlap = 1
                        break
            if overlap:
                to_remove.add(ci1 if contours[ci1][0] < contours[ci2][0] else ci2)
contours = [c[1] for i,c in enumerate(contours) if i not in to_remove]


print(contours[0])
print(contours[0].shape)
c = np.array([[im[x[0][1]][x[0][0]] for x in contours[54]] for i in range(100)])
cv2.imshow("asd", c)
cv2.waitKey(0)
print(c)
#exit()
contour_sets = [set(tuple(cc[0]) for cc in c) for c in contours]
contour_maps = [{cc:i for i,cc in enumerate(c)} for c in contours]



print(f"{len(contours)} contours")
# finding corners
corners, corners_coord, corner_pieces, edge_pieces = find_pieces()
res = find_res()
print(res) 

#dfs vol 2
contours_inner = []
for id,c in enumerate(contours):
    q = [(corners_coord[id][0][0]+(corners_coord[id][2][0]-corners_coord[id][0][0])//2,corners_coord[id][0][1]+(corners_coord[id][2][1]-corners_coord[id][0][1])//2)]
    contours_inner.append({})
    seen = set(q[0])
    while q:
        node = q.pop()
        for delta in [(1,0),(-1,0),(0,1),(0,-1)]:
            next = (node[0]+delta[0],node[1]+delta[1])
            if next in contour_sets[id]:
                contours_inner[contour_maps[id][next]] = node
            elif next not in seen:
                seen.add(next)
                q.append(next)


# copy pieces to result image
res_im = calc_res_img()

img = im.copy()
#cv2.drawContours(img, contours, -1, (0,255,0), 2)
cv2.imshow("1",img)
cv2.imshow("2",res_im)
k = cv2.waitKey(0)
