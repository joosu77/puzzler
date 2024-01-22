#import os; os.environ["NUMBA_ENABLE_CUDASIM"] = "1"#; os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"

from numba import cuda, jit
import numba
import cv2
import numpy as np
import time
import json
import random

PRINT_CORNERS = True
CALCULATE_DISTS = True
A=1
B=64

@cuda.jit('void(int16[:,:,:], int16[:,:,:], int16,int16, int16, int16,int16,int16, int16, int16, int16[:,:], int16[:,:], int16[:,:,:],int32[:,:],int16[:,:,:],int16[:,:,:])', device=True)
def dtw_cuda(a, b, n, m, cax, cay, cbx, cby, aid, bid, points, inner, im, dp, a2,b2):
    
    location_mult = numba.int16(13)
    color_mult = numba.int16(1)
    for i in range(m):
        for j in range(n):
            dp[i][j] = 0
    cc0 = cbx-cax
    cc1 = cby-cay

    for i in range(1, n):
        pixcola = a2[n-1-i][0]
        pixcolb = b2[0][0]
        dist = location_mult * (abs(a[n-1-i][0][0] - b[0][0][0] + cc0) + abs(a[n-1-i][0][1] - b[0][0][1] + cc1))
        for j in range(3):
            dist += color_mult*abs(im[pixcola[1]][pixcola[0]][j] - im[pixcolb[1]][pixcolb[0]][j])
        dp[i][0] = dist + dp[i - 1][0]
    for j in range(1, m):
        pixcola = a2[n-1-0][0]
        pixcolb = b2[j][0]
        dist = location_mult * (abs(a[n-1-0][0][0] - b[j][0][0] + cc0) + abs(a[n-1-0][0][1] - b[j][0][1] + cc1))
        for o in range(3):
            dist += color_mult*abs(im[pixcola[1]][pixcola[0]][o] - im[pixcolb[1]][pixcolb[0]][o])
        dp[0][j] = dist + dp[0][j - 1]
    for i in range(1, n):
        for j in range(1, m):
            pixcola = a2[n-1-i][0]
            pixcolb = b2[j][0]
            dist = location_mult * (abs(a[n-1-i][0][0] - b[j][0][0] + cc0) + abs(a[n-1-i][0][1] - b[j][0][1] + cc1))
            for o in range(3):
                dist += color_mult*abs(im[pixcola[1]][pixcola[0]][o] - im[pixcolb[1]][pixcolb[0]][o])
            d = dist
            if dp[i - 1][j] <= dp[i - 1][j - 1] and dp[i - 1][j] <= dp[i][j - 1]:
                dp[i][j] = d + dp[i - 1][j]
            elif dp[i - 1][j - 1] <= dp[i][j - 1]:
                dp[i][j] = d + dp[i - 1][j - 1]
            else:
                dp[i][j] = d + dp[i][j - 1]


@cuda.jit
def cache_dtws_cuda(dists_hor, dists_ver, edges, lens, corners_coord, pieces_ob, inner, points, im, dp,num, edges2):
    start = cuda.threadIdx.x
    print(start)
    step = gridX = cuda.gridDim.x * cuda.blockDim.x
    for id1 in range(start, num, step):
        for id2 in range(id1+1,num):
            if not(pieces_ob[id1] & 2 or pieces_ob[id2] & 8):
                dtw_cuda(edges[id1][1],edges[id2][3],lens[id1][1],lens[id2][3],corners_coord[id1][1][0],corners_coord[id1][1][1],corners_coord[id2][0][0],corners_coord[id2][0][1],id1,id2,points,inner,im,dp[start],edges2[id1][1],edges2[id2][3])
                dists_hor[id1][id2] = dp[start][lens[id1][1]-1][lens[id2][3]-1]
            if not(pieces_ob[id2] & 2 or pieces_ob[id1] & 8):
                dtw_cuda(edges[id1][3],edges[id2][1],lens[id1][3],lens[id2][1],corners_coord[id1][0][0],corners_coord[id1][0][1],corners_coord[id2][1][0],corners_coord[id2][1][1],id1,id2,points,inner,im,dp[start],edges2[id1][3],edges2[id2][1])
                dists_hor[id2][id1] = dp[start][lens[id1][3]-1][lens[id2][1]-1]
            if not(pieces_ob[id1] & 4 or pieces_ob[id2] & 1):
                dtw_cuda(edges[id1][2],edges[id2][0],lens[id1][2],lens[id2][0],corners_coord[id1][3][0],corners_coord[id1][3][1],corners_coord[id2][0][0],corners_coord[id2][0][1],id1,id2,points,inner,im,dp[start],edges2[id1][2],edges2[id2][0])
                dists_ver[id1][id2] = dp[start][lens[id1][2]-1][lens[id2][0]-1]
            if not(pieces_ob[id2] & 4 or pieces_ob[id1] & 1):
                dtw_cuda(edges[id1][0],edges[id2][2],lens[id1][0],lens[id2][2],corners_coord[id1][0][0],corners_coord[id1][0][1],corners_coord[id2][3][0],corners_coord[id2][3][1],id1,id2,points,inner,im,dp[start],edges2[id1][0],edges2[id2][2])
                dists_ver[id2][id1] = dp[start][lens[id1][0]-1][lens[id2][2]-1]
            
def cache_dtws(corners, contours, corners_coord, pieces_ob, im):
    dists_hor = np.zeros((len(contours),len(contours)), dtype=np.int16)
    dists_ver = np.zeros((len(contours),len(contours)), dtype=np.int16)
    edges = []
    edges2 = []
    lens = []
    maxlen = 0
    for id in range(len(contours)):
        edges.append([get_edge(contours[id].points, corners[id], "TRBL"[i]) for i in range(4)])
        edges2.append([])
        for i in range(4):
            edges2[-1].append([[contours[id].inner[contours[id].map[tuple(p[0])]]] for p in edges[-1][i]])
        lens.append(np.array([len(edges[-1][i]) for i in range(4)]))
        maxlen = max(maxlen, max(lens[-1]))
    for id in range(len(contours)):
        for i in range(4):
            edges[id][i] = np.append(edges[id][i], np.zeros((maxlen-lens[id][i],edges[id][i].shape[1],edges[id][i].shape[2])),axis=0)
            edges2[id][i] = np.append(edges2[id][i], np.zeros((maxlen-lens[id][i],edges[id][i].shape[1],edges[id][i].shape[2])),axis=0)
        
    edges = np.array(edges, dtype=np.int16)
    edges2 = np.array(edges2, dtype=np.int16)
    lens = np.array(lens, dtype=np.int16)
    pieces_ob_l = np.zeros(len(contours), dtype=np.int16)
    for id in pieces_ob.top:
        pieces_ob_l[id] = 1
    for id in pieces_ob.right:
        pieces_ob_l[id] = 2
    for id in pieces_ob.bottom:
        pieces_ob_l[id] = 4
    for id in pieces_ob.left:
        pieces_ob_l[id] = 8
    maxlen2 = max(len(c.points) for c in contours)
    inner_np = np.array([np.append(c.inner,np.zeros((maxlen2-len(c.points),2))) for c in contours], dtype=np.int16)
    points_np = np.array([np.append(c.points,np.zeros((maxlen2-len(c.points),1,2))) for c in contours], dtype=np.int16)
    pieces_ob_l[pieces_ob.top_left] = 0b1001
    pieces_ob_l[pieces_ob.top_right] = 0b0011
    pieces_ob_l[pieces_ob.bottom_right] = 0b0110
    pieces_ob_l[pieces_ob.bottom_left] = 0b1100
    d_dists_hor = cuda.to_device(dists_hor)
    d_dists_ver = cuda.to_device(dists_ver)
    d_edges = cuda.to_device(edges)
    d_edges2 = cuda.to_device(edges2)
    d_lens = cuda.to_device(lens)
    d_im = cuda.to_device(im)
    corners_np = np.array(corners_coord, dtype=np.int16)
    d_corners_np = cuda.to_device(corners_np)
    d_pieces = cuda.to_device(pieces_ob_l)
    d_inner = cuda.to_device(inner_np)
    d_points = cuda.to_device(points_np)
    dp = np.empty((64,maxlen,maxlen), dtype=np.int32)
    d_dp = cuda.to_device(dp)
    cache_dtws_cuda[A,B](d_dists_hor, d_dists_ver, d_edges, d_lens, d_corners_np, d_pieces, d_inner, d_points, d_im, d_dp, len(contours), d_edges2)
    dists_hor = d_dists_hor.copy_to_host()
    dists_ver = d_dists_ver.copy_to_host()
    return dists_hor.tolist(), dists_ver.tolist()

def dtw_old(a, b, ca, cb, aid, bid, contours, im):
    location_mult = 20
    #location_mult = 10
    location_mult = 13
    #location_mult = 5
    #location_mult = 8
    n = len(a)
    m = len(b)
    dp = [[0 for _ in range(m)] for _ in range(n)]

    cc0 = cb[0] - ca[0]
    cc1 = cb[1] - ca[1]
    ac = contours[aid]
    bc = contours[bid]
    
    def dist(a, b):
        pixcola = ac.inner[ac.map[tuple(a)]]
        pixcolb = bc.inner[bc.map[tuple(b)]]
        #colour_part = sum(abs(int(im[pixcola[1]][pixcola[0]][i]) - int(im[pixcolb[1]][pixcolb[0]][i])) for i in range(3))
        #colour_part = np.sum(np.absolute(im[pixcola[1]][pixcola[0]] - im[pixcolb[1]][pixcolb[0]]))
        return location_mult * (abs(a[0] - b[0] + cc0) + abs(a[1] - b[1] + cc1)) + np.sum(np.absolute(im[pixcola[1]][pixcola[0]] - im[pixcolb[1]][pixcolb[0]]))
    
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
    return int(dp[n - 1][m - 1])


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
    display_im = im.copy()
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
                #cv2.circle(display_im, m, 2, colours[ci], 2)
                corns[ci] = m
                corns_id[ci] = id
                #break
            #cv2.line(display_im, (int(c.mass_centre[0]), int(c.mass_centre[1])), (int(c.mass_centre[0] + 25 * dirs[ci][0]), int(c.mass_centre[1] + 25 * dirs[ci][1])), colours[ci], 1)
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
                cv2.circle(display_im, corn, 2, (0, 0, 0), 2)
            m = c.mass_centre
            cv2.circle(display_im, (int(m[0]), int(m[1])), 2, (255, 0, 0), 2)

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
        
    return corners, corners_coord, pieces_ob, display_im


class Dists:
    def __init__(self, data):
        self.hor = data[0]
        self.ver = data[1]



def cache_dtws_old(corners, contours, corners_coord, im):
    # map: hor[id1][id2] means distance when id1 piece is to the left of id2
    #      ver[id1][id2] means distance when id1 piece is up from id2
    dists_hor = [[-1 for _ in range(len(contours))] for i in range(len(contours))]
    dists_ver = [[-1 for _ in range(len(contours))] for i in range(len(contours))]
    for id1 in range(len(contours)):
        edges1 = [get_edge(contours[id1].points, corners[id1], "TRBL"[i])[::-1] for i in range(4)]
        for id2 in range(id1+1,len(contours)):
            edges2 = [get_edge(contours[id2].points, corners[id2], "TRBL"[i]) for i in range(4)]
            dists_hor[id1][id2] = dtw(edges1[1],edges2[3],corners_coord[id1][1],corners_coord[id2][0],id1,id2,contours,im)
            dists_hor[id2][id1] = dtw(edges1[3],edges2[1],corners_coord[id1][0],corners_coord[id2][1],id1,id2,contours,im)
            dists_ver[id1][id2] = dtw(edges1[2],edges2[0],corners_coord[id1][3],corners_coord[id2][0],id1,id2,contours,im)
            dists_ver[id2][id1] = dtw(edges1[0],edges2[2],corners_coord[id1][0],corners_coord[id2][3],id1,id2,contours,im)
            print(f"Done {id1} vs {id2}")
    return dists_hor, dists_ver


def find_res_with_dists(corners, contours, corners_coord, im, pieces, dists):
    row_w = len(pieces.top) + 2
    col_h = len(pieces.right) + 2
    all_pieces = pieces.top_left + pieces.top_right + pieces.bottom_left + pieces.bottom_right + pieces.top + pieces.bottom + pieces.left + pieces.right + pieces.middle
    res = [[-1 for _ in range(row_w)] for _ in range(col_h)]
    
    res[0][0] = pieces.top_left[0]
    res[0][-1] = pieces.top_right[0]
    res[-1][0] = pieces.bottom_left[0]
    res[-1][-1] = pieces.bottom_right[0]
    to_explore = set()  # Has tuples of form (y, x)
    to_explore.add((0, 1))
    to_explore.add((1, 0))
    to_explore.add((0, row_w - 2))
    to_explore.add((1, row_w - 1))
    to_explore.add((col_h - 1, 1))
    to_explore.add((col_h - 2, 0))
    to_explore.add((col_h - 1, row_w - 2))
    to_explore.add((col_h - 2, row_w - 1))
    history = []
    history.append([[a for a in row] for row in res])
    while to_explore:
        print(to_explore)
        best_location = None
        best_id = None
        best_bag = None
        best_error = 1e12
        #print(f"res: {res}")
        for y, x in to_explore:
            if y == 0:
                bag = pieces.top
            elif y == col_h - 1:
                bag = pieces.bottom
            elif x == 0:
                bag = pieces.left
            elif x == row_w - 1:
                bag = pieces.right
            else:
                bag = pieces.middle
            #print(f"bag: {bag}")
            for id in bag:
                #print(f"id: {id}")
                error = 0
                error_term_amount = 0
                if y > 0 and res[y - 1][x] != -1:
                    error += dists.ver[res[y - 1][x]][id]
                    error_term_amount += 1
                if y < col_h - 1 and res[y + 1][x] != -1:
                    error += dists.ver[id][res[y + 1][x]]
                    error_term_amount += 1
                if x > 0 and res[y][x - 1] != -1:
                    error += dists.hor[res[y][x - 1]][id]
                    error_term_amount += 1
                if x < row_w - 1 and res[y][x + 1] != -1:
                    error += dists.hor[id][res[y][x + 1]]
                    error_term_amount += 1
                error /= error_term_amount
                if error < best_error:
                    best_error = error
                    best_location = (y, x)
                    best_id = id
                    best_bag = bag
        y, x = best_location
        res[y][x] = best_id
        best_bag.remove(best_id)
        to_explore.remove(best_location)
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            tx, ty = x + dx, y + dy
            if tx >= 0 and ty >= 0 and tx < row_w and ty < col_h and res[ty][tx] == -1:
                to_explore.add((ty, tx))
        history.append([[a for a in row] for row in res])
    return res, history


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


def show_history(res, corners_coord, im, contours, history, offsets):
    i = 0
    show_inner = False
    while True:
        cur = history[i]
        res_im = np.zeros((2000, 2000, 3), dtype=np.uint8)
        for y, line in enumerate(cur):
            for x, id in enumerate(line):
                if id == -1:
                    continue
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
        if show_inner:
            for y, line in enumerate(cur):
                for x, id in enumerate(line):
                    if id == -1:
                        continue
                    offset = offsets[y][x]
                    offset = (int(offset[0]), int(offset[1]))
                    for a, b in contours[id].inner:
                        res_im[offset[1] + b - corners_coord[id][0][1], offset[0] + a - corners_coord[id][0][0]] = (255, 255, 255)
                        #res_im[a][b] = (255, 255, 255)
                    #print(contours[id].inner)
        cv2.imshow("Result", res_im)
        while True:
            key = cv2.waitKey(0)
            if key == 109:
                i += 1
                i = min(i, len(history) - 1)
                break
            if key == 110:
                i -= 1
                i = max(i, 0)
                break
            if key == 105:
                show_inner = not show_inner
                break
            print(key)
            if key == 27:
                return




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
    return res_im, offsets

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

def find_inners(contours, im):
    im_temp = np.zeros(im.shape[:2], dtype=np.uint8)
    cv2.drawContours(im_temp, [c.points for c in contours], -1, 255, 1)
    im_temp = cv2.dilate(im_temp, np.ones((7,7),dtype=np.uint8))
    contours2, hier = cv2.findContours(im_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours2 = [c for i,c in enumerate(contours2) if hier[0][i][2] == -1]
    
    contours2 = [x[2] for x in sorted([(cv2.contourArea(c),i,c) for i,c in enumerate(contours2)])[-len(contours):]]
    #cv2.drawContours(im, contours2, -1, (255,255,255), 1)
    #cv2.drawContours(im, [c.points for c in contours], -1, (255,255,255), 1)
    #cv2.imshow("1",im)
    #cv2.waitKey(0)
    
    for c1 in contours:
        box1 = cv2.boundingRect(c1.points)
        for i,c2 in enumerate(contours2):
            box2 = cv2.boundingRect(c2)
            if box1[0] <= box2[0] <= box2[0]+box2[2] <= box1[0]+box1[2] and box1[1] <= box2[1] <= box2[1]+box2[3] <= box1[1]+box1[3]:
                c1.inner_full = c2
                break
        contours2.pop(i)
        c1.inner_full_set = set(tuple(cc[0]) for cc in c1.inner_full)

    dist_map = [x[1:] for x in sorted([(dx**2+dy**2,dx,dy) for dy in range(-7,8) for dx in range(-7,8)])]
    for c in contours:
        c.inner = [-1 for _ in range(len(c.points))]
        for i,p in enumerate(c.points):
            best_p = -1
            best_d = 1e9
            for dx,dy in dist_map:
                node = (p[0][0]+dx, p[0][1]+dy)
                if node in c.inner_full_set:
                    c.inner[i] = node
                    break

def main(imgname, threshold_value, threshold_mode):
    glob_start = time.time()
    im = cv2.imread(imgname)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, threshold_value, 255, threshold_mode)

    contours = find_contours(thresh)

    # finding corners
    corners, corners_coord, pieces_ob, display_im = find_pieces(contours, im)

    #dfs vol 2
    #[c.find_inner(im) for c in contours]
    find_inners(contours, im)

    print(f"Nr of contours: {len(contours)}, top: {len(pieces_ob.top)}, left: {len(pieces_ob.left)}, bot: {len(pieces_ob.bottom)}, right: {len(pieces_ob.right)}")
    #cv2.drawContours(display_im, [c.points for c in contours], -1, (0,255,0), 1)
    #cv2.imshow("1",display_im)
    #cv2.waitKey(0)
    
    if CALCULATE_DISTS:
        start = time.time()
        dists = cache_dtws(corners, contours, corners_coord, pieces_ob, im.astype(np.int16))
        taken = time.time() - start
        print(f"Cacheing dtw took {round(taken, 2)} seconds")
        # 123.79 initially
        # 117.31 after doing colour comparison with numpy
        file = open("cache.json", "w")
        file.write(json.dumps(dists))
        dists = Dists(dists)
    else:
        file = open("cache.json", "r")
        dists = Dists(json.loads(file.read()))
    file.close()
    #exit()

    res, history = find_res_with_dists(corners, contours, corners_coord, im, pieces_ob, dists)
    #res = find_res(corners, contours, corners_coord, im, pieces_ob)

    # copy pieces to result image
    res_im, offsets = calc_res_img2(res, corners_coord, im, contours)
    print(f"time: {time.time()-glob_start}")
    show_history(res, corners_coord, im, contours, history, offsets)

    #cv2.imshow("2",res_im)
    #while True:
    #    k = cv2.waitKey(0)
    #    if k == 27:
    #        break

if __name__ == '__main__':
    #main("input_shuffled.png", 35, 0)
    #main("tartu_shuffled.png", 135, cv2.THRESH_BINARY_INV)
    #main("inp2.png", 135, cv2.THRESH_BINARY_INV)
    #main("inp3.png", 135, cv2.THRESH_BINARY_INV)
    #main("inp10.png", 135, cv2.THRESH_BINARY_INV)
    main("tartu_shuffled.png", 135, cv2.THRESH_BINARY_INV)
