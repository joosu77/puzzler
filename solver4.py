import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

PRINT_CORNERS = False

def dtw(a, b, ca, cb, aid, bid, contours, im):
    location_mult = 20
    #location_mult = 10
    location_mult = 13
    #location_mult = 5
    #location_mult = 8
    n = len(a)
    m = len(b)
    dp = [[0 for _ in range(m)] for _ in range(n)]
    
    def dist(a, b):
        ida = contours[aid].map[tuple(a)]
        idb = contours[bid].map[tuple(b)]
        while ida not in contours[aid].inner:
            ida = (ida + 1) % len(contours[aid])
        while idb not in contours[bid].inner:
            idb = (idb + 1) % len(contours[bid])
        pixcola = contours[aid].inner[ida]
        pixcolb = contours[bid].inner[idb]
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


class Piece:
    def __init__(self, contour, ids):
        self.contour = contour
        self.corner_ids = ids


class Pieces:
    def __init__(self):
        self.top_left = []
        self.top_right = []
        self.bottom_left = []
        self.bottom_right = []

        self.top = []
        self.right = []
        self.bottom = []
        self.left = []

        self.middle = []

def std_dev(xs):
    avg = sum(xs)/len(xs)
    return (sum((x-avg)**2 for x in xs)/len(xs))**0.5

def find_pieces(contours, im):
    pieces = []
    corners = []
    corners_coord = []
    corner_pieces = [-1, -1, -1, -1]
    # top, right, bot, left
    edge_pieces = [[], [], [], []]
    std_devs = [[],[],[],[]]
    for cii, c in enumerate(contours):
        # find corners
        box = cv2.boundingRect(c.points)
        rect_corns = [(box[0], box[1]), (box[0] + box[2], box[1]), (box[0] + box[2], box[1] + box[3]), (box[0], box[1] + box[3])]
        corns = [-1, -1, -1, -1]
        corns_d = [1e9, 1e9, 1e9, 1e9]
        corns_id = [-1, -1, -1, -1]


        colours = [(0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        dirs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for ci in range(4):
            ds = []
            for i, pix in enumerate(c.points):
                d = abs(rect_corns[ci][0] - pix[0][0])**1 + abs(rect_corns[ci][1] - pix[0][1])**1
                ds.append(d)
                if d < corns_d[ci]:
                    corns_d[ci] = d
                    corns[ci] = pix[0]
                    corns_id[ci] = i
            
            minima = [i for i in range(len(ds)) if ds[(i - 3) % len(ds)] > ds[i] and ds[(i + 3) % len(ds)] > ds[i]]
            minimaa = []
            i = 0
            while i < len(minima):
                d = 1
                while i < len(minima) - d and minima[i + d] == minima[i] + d:
                    d += 1
                minimaa.append(sum(minima[i:i + d]) / d)
                i += d
            minima = minimaa
            minima = sorted([(ds[int(m)], int(m)) for m in minima])[:3]
            minima = [(c.points[m[1]][0], m[1]) for m in minima]
            minima = sorted([(((loc[0] - c.mass_centre[0]) * dirs[ci][0] + (loc[1] - c.mass_centre[1]) * dirs[ci][1]) / abs(((loc[0] - c.mass_centre[0])**2 + (loc[1] - c.mass_centre[1])**2)**0.5), (loc[0], loc[1]), id) for loc, id in minima], reverse=True)
            #print(minima)
            for dot, m, id in minima:
                #cv2.circle(im, m, 2, colours[ci], 2)
                corns[ci] = m
                corns_id[ci] = id
                #break
            for dot, m, id in minima:
                #cv2.circle(im, m, 2, colours[ci], 2)
                corns[ci] = m
                corns_id[ci] = id
                break

            #plt.plot(ds)
            #for m in minima:
            #    plt.axvline(x = m, color="b")
            #plt.show()

        corners.append(corns_id)
        corners_coord.append(corns)
        pieces.append(Piece(c.points, corns_id))
        c.corners = corns_id
        print(corns_id)
        if PRINT_CORNERS:
            for corn in corns:
                cv2.circle(im, corn, 2, (0, 0, 0), 2)
            m = c.mass_centre
            cv2.circle(im, (int(m[0]), int(m[1])), 2, (255, 0, 0), 2)

        # find starting piece (top left piece)
        vals = []
        for pix in get_edge(c.points, corns_id, "L"):
            vals.append(pix[0][0])
        std_devs[3].append((std_dev(vals),cii))
        vals = []
        for pix in get_edge(c.points, corns_id, "T"):
            vals.append(pix[0][1])
        std_devs[0].append((std_dev(vals),cii))
        vals = []
        for pix in get_edge(c.points, corns_id, "R"):
            vals.append(pix[0][0])
        std_devs[1].append((std_dev(vals),cii))
        vals = []
        for pix in get_edge(c.points, corns_id, "B"):
            vals.append(pix[0][1])
        std_devs[2].append((std_dev(vals),cii))
    
    std_devs = [sorted(xs) for xs in std_devs]
    for i in range(4):
        max_dist = 0
        max_dist_num = -1
        last = std_devs[i][0][0]
        for num, v in enumerate(std_devs[i]):
            diff = v[0]-last
            if diff>max_dist:
                max_dist = diff
                max_dist_num = num
            last = v[0]
        for id in range(max_dist_num):
            edge_pieces[i].append(std_devs[i][id][1])
    
    for i in range(4):
        for j in range(len(contours)):
            if j in edge_pieces[i] and j in edge_pieces[(i-1+4)%4]:
                edge_pieces[i].remove(j)
                edge_pieces[(i-1+4)%4].remove(j)
                corner_pieces[i] = j
    pieces_ob = Pieces()
    pieces_ob.top_left.append(corner_pieces[0])
    pieces_ob.top_right.append(corner_pieces[1])
    pieces_ob.bottom_right.append(corner_pieces[2])
    pieces_ob.bottom_left.append(corner_pieces[3])
    used = set(corner_pieces)
    # top, right, bot, left
    for idd in edge_pieces[0]:
        pieces_ob.top.append(idd)
        used.add(idd)
    for idd in edge_pieces[1]:
        pieces_ob.right.append(idd)
        used.add(idd)
    for idd in edge_pieces[2]:
        pieces_ob.bottom.append(idd)
        used.add(idd)
    for idd in edge_pieces[3]:
        pieces_ob.left.append(idd)
        used.add(idd)
    for i in range(len(pieces)):
        if i in used:
            continue
        pieces_ob.middle.append(i)
    pieces_ob.data = pieces
        
    return corners, corners_coord, pieces_ob

def find_res(corners, contours, corners_coord, im, pieces):
    row_w = len(pieces.top) + 2
    col_h = len(pieces.right) + 2
    
    res = [[-1 for _ in range(row_w)] for _ in range(col_h)]
    res[0][0] = pieces.top_left[0]
    res[0][-1] = pieces.top_right[0]
    res[-1][0] = pieces.bottom_left[0]
    res[-1][-1] = pieces.bottom_right[0]
    
    for y in range(col_h):
        for x in range(row_w):
            if res[y][x] > -1:
                continue
            best_diff = 1e9
            best_id = -1

            if x == 0:
                id = res[y - 1][0]
                edge1 = get_edge(contours[id].points, corners[id], "B")
                bag = pieces.left
                edge2_type = "T"
                previous_start = corners_coord[id][3]
            else:
                id = res[y][x - 1]
                edge1 = get_edge(contours[id].points, corners[id], "R")
                if y == 0:
                    bag = pieces.top
                elif x == row_w - 1:
                    bag = pieces.right
                elif y == col_h - 1:
                    bag = pieces.bottom
                else:
                    bag = pieces.middle
                edge2_type = "L"
                previous_start = corners_coord[id][1]

            for id2 in bag:
                edge2 = get_edge(contours[id2].points, corners[id2], edge2_type)
                diff = dtw(edge1, edge2[::-1], previous_start, corners_coord[id2][0], id, id2, contours, im)
                if diff < best_diff:
                    best_diff = diff
                    best_id = id2
            res[y][x] = best_id

            print(y, x, best_id)
            bag.remove(best_id)
    return res


def calc_res_img2(res, corners_coord, im, contours):
    res_im = np.zeros((2000, 2000, 3), dtype=np.uint8)
    offsets = [[(10, 10) for x in range(len(res[y]))] for y in range(len(res))]
    for y in range(len(res)):
        for x in range(len(res[0])):
            if y + x == 0:
                continue
            id = res[y][x]
            this_offsets = []
            
            if x > 0:
                left_id = res[y][x - 1]
                left_offset = offsets[y][x - 1]
                left_top_right = (left_offset[0] + corners_coord[left_id][1][0] - corners_coord[left_id][0][0], left_offset[1] + corners_coord[left_id][1][1] - corners_coord[left_id][0][1])
                this_offsets.append(left_top_right)
                left_bottom_right = (left_offset[0] + corners_coord[left_id][2][0] - corners_coord[left_id][0][0], left_offset[1] + corners_coord[left_id][2][1] - corners_coord[left_id][0][1])
                this_offsets.append((left_bottom_right[0] + corners_coord[id][0][0] - corners_coord[id][3][0], left_bottom_right[1] + corners_coord[id][0][1] - corners_coord[id][3][1]))
            
            if y > 0:
                top_id = res[y - 1][x]
                top_offset = offsets[y - 1][x]
                top_bottom_left = (top_offset[0] + corners_coord[top_id][3][0] - corners_coord[top_id][0][0], top_offset[1] + corners_coord[top_id][3][1] - corners_coord[top_id][0][1])
                this_offsets.append(top_bottom_left)
                top_bottom_right = (top_offset[0] + corners_coord[top_id][2][0] - corners_coord[top_id][0][0], top_offset[1] + corners_coord[top_id][2][1] - corners_coord[top_id][0][1])
                this_offsets.append((top_bottom_right[0] + corners_coord[id][0][0] - corners_coord[id][1][0], top_bottom_right[1] + corners_coord[id][0][1] - corners_coord[id][1][1]))
            #else:
            #    left_offset = offsets[y][x - 1]
            #    this_offsets.append((left_offset[0] + 150, left_offset[1]))

            this_x = sum([a[0] for a in this_offsets])
            this_y = sum([a[1] for a in this_offsets])
            offsets[y][x] = (this_x / len(this_offsets), this_y / len(this_offsets))

    for y, line in enumerate(res):
        for x, id in enumerate(line):
            offset = offsets[y][x]
            offset = (int(offset[0]), int(offset[1]))
            middle = (corners_coord[id][0][0] + (corners_coord[id][2][0] - corners_coord[id][0][0]) // 2, corners_coord[id][0][1] + (corners_coord[id][2][1] - corners_coord[id][0][1]) // 2)
            q = [middle]
            seen = {middle}
            #vals = set(contours[id].inner.values())
            vals = set()
            while q:
                node = q.pop()
                res_im[offset[1] + node[1] - corners_coord[id][0][1], offset[0] + node[0] - corners_coord[id][0][0]] = im[node[1], node[0]]
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    next = (node[0] + dx, node[1] + dy)
                    if next not in contours[id].set and next not in vals and next not in seen:
                        seen.add(next)
                        q.append(next)
    return res_im

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

edge_indices = {
    "T": (1, 0),
    "R": (2, 1),
    "B": (3, 2),
    "L": (0, 3)
}

def get_edge(points, corners, edge):
    a, b = edge_indices[edge]
    a, b = corners[a], corners[b]
    if a < b:
        return points[a:b]
    return np.concatenate((points[a:], points[:b]))


def find_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [Contour(c) for c in contours]
    contours = [c for c in contours if c.area > 10]
    to_remove = find_contours_to_remove(contours)
    contours = [c for i, c in enumerate(contours) if i not in to_remove]
    return contours

class Contour:
    def __init__(self, contour):
        self.area = cv2.contourArea(contour)
        self.points = contour
        self.set = set(tuple(cc[0]) for cc in contour)
        self.map = {tuple(cc[0]): i for i, cc in enumerate(contour)}
        self.corners = []
        self.inner = None
        self.mass_centre = None
        self.calculate_mass_centre()
    
    def calculate_mass_centre(self):
        amount = self.points.shape[0]
        tx = 0
        ty = 0
        for p in self.points:
            tx += p[0][0]
            ty += p[0][1]
        self.mass_centre = (tx / amount, ty / amount)
    
    def __len__(self):
        return len(self.points)

    def find_inner(self):
        self.inner = {}

        tl = self.points[self.corners[0]][0]
        br = self.points[self.corners[2]][0]
        middle = (tl[0] + (br[0] - tl[0]) // 2, tl[1] + (br[1] - tl[1]) // 2)

        q = [middle]
        seen = {middle}
        while q:
            node = q.pop()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                next = (node[0] + dx, node[1] + dy)
                if next in self.set:
                    self.inner[self.map[next]] = node
                elif next not in seen:
                    seen.add(next)
                    q.append(next)

def main(imgname, threshold_value, threshold_mode):
    im = cv2.imread(imgname)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, threshold_value, 255, threshold_mode)

    contours = find_contours(thresh)

    # finding corners
    corners, corners_coord, pieces_ob = find_pieces(contours, im)

    #dfs vol 2
    [c.find_inner() for c in contours]
    for c in contours:
        #print(c.points)
        #print(c.inner)
        max_key = max(c.inner.keys())
        #print()
        #c.points = c.inner
    #corners, corners_coord, pieces_ob = find_pieces(contours, im)

    print(f"Nr of contours: {len(contours)}, top: {len(pieces_ob.top)}, left: {len(pieces_ob.left)}, bot: {len(pieces_ob.bottom)}, right: {len(pieces_ob.right)}")
    img = im.copy()
    cv2.drawContours(img, [c.points for c in contours], -1, (0,255,0), 1)
    cv2.imshow("1",img)
    cv2.waitKey(0)
    
    res = find_res(corners, contours, corners_coord, im, pieces_ob)

    # copy pieces to result image
    res_im = calc_res_img2(res, corners_coord, im, contours)

    cv2.imshow("2",res_im)
    while True:
        k = cv2.waitKey(0)
        if k == 27:
            break

if __name__ == '__main__':
    #main("input_shuffled.png", 35, 0)
    #main("tartu_shuffled.png", 135, cv2.THRESH_BINARY_INV)
    #main("inp2.png", 135, cv2.THRESH_BINARY_INV)
    #main("inp3.png", 135, cv2.THRESH_BINARY_INV)
    main("inp8.png", 135, cv2.THRESH_BINARY_INV)
