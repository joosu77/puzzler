import cv2
import numpy as np
import time

def dtw(a, b, ca, cb, aid, bid, contour_maps, contours_inner, contours, im):
    location_mult = 20
    n = len(a)
    m = len(b)
    dp = [[0 for _ in range(m)] for _ in range(n)]
    def dist(a, b):
        ida = contour_maps[aid][tuple(a)]
        idb = contour_maps[bid][tuple(b)]
        while ida not in contours_inner[aid]:
            ida = (ida + 1) % len(contours[aid])
        while idb not in contours_inner[bid]:
            idb = (idb + 1) % len(contours[bid])
        pixcola = contours_inner[aid][ida]
        pixcolb = contours_inner[bid][idb]
        return location_mult * abs(a[0] - b[0] - ca[0] + cb[0]) + location_mult * abs(a[1] - b[1] - ca[1] + cb[1]) + abs(int(im[pixcola[1]][pixcola[0]][0]) - int(im[pixcolb[1]][pixcolb[0]][0])) + abs(int(im[pixcola[1]][pixcola[0]][1]) - int(im[pixcolb[1]][pixcolb[0]][1])) + abs(int(im[pixcola[1]][pixcola[0]][2]) - int(im[pixcolb[1]][pixcolb[0]][2]))
    for i in range(1, n):
        dp[i][0] = dist(a[i][0], b[0][0]) + dp[i - 1][0]
    for j in range(1, m):
        dp[0][j] = dist(a[0][0], b[j][0]) + dp[0][j - 1]
    for i in range(1, n):
        for j in range(1, m):
            d = dist(a[i][0], b[j][0])
            if dp[i - 1][j] <= dp[i - 1][j - 1] and dp[i - 1][j] <= dp[i][j - 1]:
                dp[i][j] = d + dp[i - 1][j]
            elif dp[i - 1][j - 1] <= dp[i][j - 1]:
                dp[i][j] = d + dp[i - 1][j - 1]
            else:
                dp[i][j] = d + dp[i][j - 1]
    return dp[n - 1][m - 1]


def find_pieces(contours, im):
    corners = []
    corners_coord = []
    corner_pieces = [-1, -1, -1, -1]
    # top, right, bot, left
    edge_pieces = [[], [], [], []]
    for cii, c in enumerate(contours):
        # find corners
        box = cv2.boundingRect(c)
        rect_corns = [(box[0], box[1]), (box[0] + box[2], box[1]), (box[0] + box[2], box[1] + box[3]), (box[0], box[1] + box[3])]
        corns = [-1, -1, -1, -1]
        corns_d = [1e9, 1e9, 1e9, 1e9]
        corns_id = [-1, -1, -1, -1]
        for i, pix in enumerate(c):
            for ci in range(4):
                d = abs(rect_corns[ci][0] - pix[0][0]) + abs(rect_corns[ci][1] - pix[0][1])
                if d < corns_d[ci]:
                    corns_d[ci] = d
                    corns[ci] = pix[0]
                    corns_id[ci] = i
        corners.append(corns_id)
        corners_coord.append(corns)
        for corn in corns:
            cv2.circle(im, corn, 2, (0, 0, 255), 2)

        # find starting piece (top left piece)
        max_hor_delta = 0
        for pix in range_wrap(c, corns_id[0], corns_id[3]):
            max_hor_delta = max(max_hor_delta, abs(pix[0][0] - corns[0][0]))
        max_ver_delta = 0
        for pix in range_wrap(c, corns_id[1], corns_id[0]):
            max_ver_delta = max(max_ver_delta, abs(pix[0][1] - corns[0][1]))
        if max_hor_delta < 3 and max_ver_delta < 3:
            corner_pieces[0] = cii
        elif max_ver_delta < 3:
            edge_pieces[0].append(cii)
        elif max_hor_delta < 3:
            edge_pieces[3].append(cii)

        # find line end pieces
        max_hor_delta = 0
        for pix in range_wrap(c, corns_id[2], corns_id[1]):
            max_hor_delta = max(max_hor_delta, abs(pix[0][0] - corns[1][0]))
        if max_hor_delta < 3:
            if cii in edge_pieces[0]:
                edge_pieces[0].remove(cii)
                corner_pieces[1] = cii
            else:
                edge_pieces[1].append(cii)
        # find last line pieces
        max_ver_delta = 0
        for pix in range_wrap(c, corns_id[3], corns_id[2]):
            max_ver_delta = max(max_ver_delta, abs(pix[0][1] - corns[2][1]))
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

def find_res(edge_pieces, corner_pieces, contours, corners, corners_coord, contour_maps, contours_inner, im):
    row_w = len(edge_pieces[0]) + 2
    col_h = len(edge_pieces[1]) + 2
    # create mapping to fit pieces in
    res = [[corner_pieces[0]]]
    q = set(i for i in range(len(contours)) if i not in corner_pieces and max(i in edge_pieces[j] for j in range(4)) == 0)
    for i in range(row_w * col_h - 1):
        if len(res) == 1 and len(res[0]) == row_w-1:
            res[-1].append(corner_pieces[1])
            continue
        if len(res) == col_h - 1 and len(res[-1]) == row_w:
            res.append([corner_pieces[3]])
            continue
        if len(res) == col_h and len(res[-1]) == row_w-1:
            res[-1].append(corner_pieces[2])
            continue
        if len(res[-1]) == row_w:
            id = res[-1][0]
            edge1 = range_wrap(contours[id], corners[id][3], corners[id][2])
            best_diff = 1e9
            best_id = -1
            bag = edge_pieces[3]
            for id2 in bag:
                edge2 = range_wrap(contours[id2], corners[id2][1], corners[id2][0])
                diff = dtw(edge1, edge2[::-1], corners_coord[id][3], corners_coord[id2][0], id, id2, contour_maps, contours_inner, contours, im)
                print(f"type1: {id} vs {id2}: {diff}")
                if diff < best_diff:
                    best_diff = diff
                    best_id = id2
            res.append([best_id])
        else:
            id = res[-1][-1]
            edge1 = range_wrap(contours[id], corners[id][2], corners[id][1])
            best_diff = 1e9
            best_id = -1
            if len(res) == 1:
                bag = edge_pieces[0]
            elif len(res[-1]) == row_w - 1:
                bag = edge_pieces[1]
            elif len(q) == 0:
                bag = edge_pieces[2]
            else:
                bag = q
            for id2 in bag:
                edge2 = range_wrap(contours[id2], corners[id2][0], corners[id2][3])
                diff = dtw(edge1, edge2[::-1], corners_coord[id][1], corners_coord[id2][0], id, id2, contour_maps, contours_inner, contours, im)
                print(f"type2: {id} vs {id2}: {diff}")
                if diff < best_diff:
                    best_diff = diff
                    best_id = id2
            res[-1].append(best_id)
        print(i, best_id)
        if best_id == -1:
            print(res)
            print(len(res))
            print(len(res[-1]))
        bag.remove(best_id)
    return res


def calc_res_img(res, corners_coord, contour_sets, im):
    res_im = np.zeros((2000, 2000, 3), dtype=np.uint8)
    ctr = (10, 10)
    next_line_start = (10, 10)
    for l in res:
        ctr = next_line_start
        next_line_start = (ctr[0], ctr[1] + corners_coord[l[0]][3][1] - corners_coord[l[0]][0][1])
        for id in l:
            # dfs over pixels in contour
            q = [(corners_coord[id][0][0] + (corners_coord[id][2][0] - corners_coord[id][0][0]) // 2, corners_coord[id][0][1] + (corners_coord[id][2][1] - corners_coord[id][0][1]) // 2)]
            seen = set(q[0])
            while q:
                node = q.pop()
                res_im[ctr[1] + node[1] - corners_coord[id][0][1], ctr[0] + node[0] - corners_coord[id][0][0]] = im[node[1], node[0]]
                for delta in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    next = (node[0] + delta[0], node[1] + delta[1])
                    if next not in contour_sets[id] and next not in seen:
                        seen.add(next)
                        q.append(next)
            ctr = (ctr[0] + corners_coord[l[0]][1][0] - corners_coord[l[0]][0][0], ctr[1])
    return res_im

def find_contours_inner(corners_coord, contour_sets, contour_maps):
    contours_inner = []
    for id, c in enumerate(corners_coord):
        print(c)
        q = [(c[0][0] + (c[2][0] - c[0][0]) // 2, c[0][1] + (c[2][1] - c[0][1]) // 2)]
        contours_inner.append({})
        seen = set(q[0])
        while q:
            node = q.pop()
            for delta in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                next = (node[0] + delta[0], node[1] + delta[1])
                if next in contour_sets[id]:
                    contours_inner[id][contour_maps[id][next]] = node
                elif next not in seen:
                    seen.add(next)
                    q.append(next)
    return contours_inner

def is_contained(box1, box2):
    xl1 = box1[0]
    xl2 = box2[0]
    xr1 = box1[0] + box1[2]
    xr2 = box2[0] + box2[2]
    yu1 = box1[1]
    yu2 = box2[1]
    yd1 = box1[1] + box1[3]
    yd2 = box2[1] + box2[3]
    return xl1 <= xl2 and xr2 <= xr1 and yu1 <= yu2 and yd2 <= yd1 or xl2 <= xl1 and xr1 <= xr2 and yu2 <= yu1 and yd1 <= yd2

def find_contours_to_remove(contours):
    to_remove = set()
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            # box is tuple x,y,w,h
            box1 = cv2.boundingRect(contours[i].points)
            box2 = cv2.boundingRect(contours[j].points)
            if not(box2[0] > box1[0] + box1[2] or box2[0] + box2[2] < box1[0] or box2[1] > box1[1] + box1[3] or box2[1] + box2[3] < box1[1]):
                contains = is_contained(box1, box2)
                for pix1 in contours[i].points:
                    if contains:
                        break
                    for pix2 in contours[j].points:
                        if pix1[0][0] == pix2[0][0] and pix1[0][1] == pix2[0][1]:
                            contains = True
                            break
                if contains:
                    to_remove.add(i if contours[i].area < contours[j].area else j)
    return to_remove

def range_wrap(l, a, b):
    if a < b:
        return l[a:b]
    return np.concatenate((l[a:], l[:b]))

def find_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [Contour(c) for c in contours]
    contours = [c for c in contours if c.area > 10]
    to_remove = find_contours_to_remove(contours)
    contours = [c.points for i, c in enumerate(contours) if i not in to_remove]
    return contours

class Contour:
    def __init__(self, contour):
        self.area = cv2.contourArea(contour)
        self.points = contour

def main(imgname, threshold_value, threshold_mode):
    im = cv2.imread(imgname)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, threshold_value, 255, threshold_mode)

    contours = find_contours(thresh)

    contour_sets = [set(tuple(cc[0]) for cc in c) for c in contours]
    contour_maps = [{tuple(cc[0]): i for i, cc in enumerate(c)} for c in contours]

    # finding corners
    corners, corners_coord, corner_pieces, edge_pieces = find_pieces(contours, im)

    #dfs vol 2
    contours_inner = find_contours_inner(corners_coord, contour_sets, contour_maps)

    res = find_res(edge_pieces, corner_pieces, contours, corners, corners_coord, contour_maps, contours_inner, im)
    #print(res) 

    # copy pieces to result image
    res_im = calc_res_img(res, corners_coord, contour_sets, im)

    img = im.copy()
    #cv2.drawContours(img, contours, -1, (0,255,0), 2)
    cv2.imshow("1",img)
    cv2.imshow("2",res_im)
    while True:
        k = cv2.waitKey(0)
        if k == 27:
            break

if __name__ == '__main__':
    #main("input_shuffled.png", 35, 0)
    main("input_shuffled_small.png", 135, cv2.THRESH_BINARY_INV)
