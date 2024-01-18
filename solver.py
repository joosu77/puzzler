import cv2
import numpy as np

def dtw(a,b):
    n = len(a)
    m = len(b)
    dp = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(1,n):
        dp[i][0] = abs(a[i][0][0]-b[0][0][0])+abs(a[i][0][1]-b[0][0][1])+dp[i-1][0]
    for j in range(1,m):
        dp[0][j] = abs(a[0][0][0]-b[j][0][0])+abs(a[0][0][1]-b[j][0][1])+dp[0][j-1]
    for i in range(1,n):
        for j in range(1,m):
            d = abs(a[i][0][0]-b[j][0][0])+abs(a[i][0][1]-b[j][0][1])
            if dp[i-1][j] <= dp[i-1][j-1] and dp[i-1][j] <= dp[i][j-1]:
                dp[i][j] = d + dp[i-1][j]
            elif dp[i-1][j-1] <= dp[i][j-1]:
                dp[i][j] = d + dp[i-1][j-1]
            else:
                dp[i][j] = d + dp[i][j-1]
    return dp[n-1][m-1]

range_wrap = lambda l, a,b: l[a:b] if a<b else np.concatenate((l[a:],l[:b]))
im = cv2.imread("input_shuffled.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 35, 255, 0)
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
contour_sets = [set(tuple(cc[0]) for cc in c) for c in contours]
print(len(contours))
# finding corners
corners = []
corners_coord = []
starting_piece_id = -1
edge_pieces_right = []
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
        starting_piece_id = cii
    max_hor_delta = 0
    # find line end pieces
    for pix in range_wrap(c,corns_id[2],corns_id[1]):
        max_hor_delta = max(max_hor_delta, abs(pix[0][0]-corns[1][0]))
    if max_hor_delta < 3:
        edge_pieces_right.append(cii)

# create mapping to fit pieces in
res = [[starting_piece_id]]
q = set(i for i in range(len(contours)) if i != starting_piece_id)
line_end = 0
while q:
    if line_end:
        id = res[-1][0]
        edge1 = range_wrap(contours[id],corners[id][3],corners[id][2])
        best_diff = 1e9
        best_id = -1
        for id2 in q:
            edge2 = range_wrap(contours[id2],corners[id2][1],corners[id2][0])
            # TODO: for two matching pieces one might have a longer contour in curves due
            # to being in the outer curve so indexes can go out of sync, might be better
            # to compare pixels that are in the same line instead of just ith pixel of the contour
            #diff = sum(abs(edge1[i][0][1]-corners_coord[id][3][1]-edge2[-i-1][0][1]+corners_coord[id2][0][1]) for i in range(min(len(edge1),len(edge2))))
            diff = dtw(edge1,edge2[::-1])
            if diff<best_diff:
                best_diff = diff
                best_id = id2
        res.append([best_id])
    else:
        id = res[-1][-1]
        edge1 = range_wrap(contours[id],corners[id][2],corners[id][1])
        best_diff = 1e9
        best_id = -1
        for id2 in q:
            edge2 = range_wrap(contours[id2],corners[id2][0],corners[id2][3])
            #TODO same here
            #diff = sum(abs(edge1[i][0][0]-corners_coord[id][1][0]-edge2[-i-1][0][0]+corners_coord[id2][0][0]) for i in range(min(len(edge1),len(edge2))))
            diff = dtw(edge1,edge2[::-1])
            if diff<best_diff:
                best_diff = diff
                best_id = id2
        res[-1].append(best_id)
    q.remove(best_id)
    line_end = best_id in edge_pieces_right or len(res[-1]) > 20

print(res)    
# copy pieces to result image
res_im = np.zeros((2000,2000,3),dtype=np.uint8)
ctr = (10,10)
next_line_start = (10,10)
for l in res:
    ctr = next_line_start
    next_line_start = (ctr[0],ctr[1]+corners_coord[l[0]][3][1]-corners_coord[l[0]][0][1])
    for id in l:
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
        ctr = (ctr[0]+corners_coord[l[0]][1][0]-corners_coord[l[0]][0][0],ctr[1])

img = im.copy()
#cv2.drawContours(img, contours, -1, (0,255,0), 2)
cv2.imshow("1",img)
cv2.imshow("2",res_im)
k = cv2.waitKey(0)
