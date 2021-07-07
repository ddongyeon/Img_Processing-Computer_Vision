import cv2
import numpy as np
np.seterr(over='ignore')

# img 불러오기
frame1_img = cv2.resize(cv2.imread("Frame1.jpg", cv2.IMREAD_COLOR), dsize=(510, 288), interpolation=cv2.INTER_AREA)
interpolation_img = cv2.resize(cv2.imread("Frame1.jpg", cv2.IMREAD_COLOR), dsize=(510, 288), interpolation=cv2.INTER_AREA)
frame1_copy_img = cv2.resize(cv2.imread("Frame1.jpg", cv2.IMREAD_COLOR), dsize=(510, 288), interpolation=cv2.INTER_AREA)
frame2_img = cv2.resize(cv2.imread("Frame2.jpg", cv2.IMREAD_COLOR), dsize=(510, 288), interpolation=cv2.INTER_AREA)
height, width, channel = frame1_img.shape

'''Optical Flow'''
# rgb to gray
frame1 = cv2.cvtColor(frame1_img, cv2.COLOR_BGR2GRAY)
frame2 = cv2.cvtColor(frame2_img, cv2.COLOR_BGR2GRAY)

# numpy 형변환
frame1 = np.array(frame1, dtype='float32')
frame2 = np.array(frame2, dtype='float32')

# x,y,t 변화량 저장할 배열 생성
dx = np.zeros((height, width), dtype='float64')
dy = np.zeros((height, width), dtype='float64')
dt = np.zeros((height, width), dtype='float64')

# dx, dy, dt 축 변화량 계산
for x in range(width):
    for y in range(height):
        if x + 1 == width and y + 1 == height:
            dx[y, x] = 0 - frame1[y, x]
            dy[y, x] = 0 - frame1[y, x]
        elif x + 1 == width:
            dx[y, x] = 0 - frame1[y, x]
            dy[y, x] = frame1[y + 1, x] - frame1[y, x]
        elif y + 1 == height:
            dx[y, x] = frame1[y, x + 1] - frame1[y, x]
            dy[y, x] = 0 - frame1[y, x]
        else:
            dx[y, x] = frame1[y, x + 1] - frame1[y, x]
            dy[y, x] = frame1[y + 1, x] - frame1[y, x]
        dt[y, x] = frame2[y, x] - frame1[y, x]

# 모든 픽셀에 대하여, motion vector 계산
window_size = 13
border = int(window_size / 2)
A = np.zeros((window_size ** 2, 2), dtype='float64')
b = np.zeros((window_size ** 2, 1), dtype='float64')
motion = np.zeros((height, width, 2), dtype='float64')
for x in range(border, width-border):
    for y in range(border, height-border):
        # window 구성
        # 행렬 A, b 생성
        idx = 0
        for i in range(-border, border+1):
            for j in range(-border, border+1):
                A[idx, 0] = dy[y+i, x+j]
                A[idx, 1] = dx[y+i, x+j]
                b[idx] = -dt[y+i, x+j]
                idx = idx + 1

        # motion계산 - Normal Equation
        result = A.T
        result = result.dot(A)
        det = result[0, 0] * result[1, 1] - result[0, 1] * result[1, 0]
        if det != 0:
            result = np.linalg.inv(result)
        else:
            result = np.linalg.pinv(result)
        result = result.dot(A.T)
        result = result.dot(b)
        motion[y, x] = result.T

# motion vector 시각화
for x in range(3, width - 3):
    for y in range(3, height - 3):
        if x % 7 == 3 and y % 7 == 3:
            vx = 0
            vy = 0
            # 7*7 윈도우 내 벡터의 합
            for i in range(-3, 4):
                for j in range(-3, 4):
                    vy = vy + v[y + i, x + j, 0]
                    vx = vx + v[y + i, x + j, 1]

            if vy != 0 or vx != 0:
                # normalization
                arrow_size = 5
                norm = np.sqrt(vy ** 2 + vx ** 2)
                ny = int(round(arrow_size * (vy / norm)))
                nx = int(round(arrow_size * (vx / norm)))
                if nx > 0 or ny > 0:
                    cv2.arrowedLine(frame1_img, (x, y), (x + nx, y + ny), (255, 0, 0), 1, tipLength=0.5)

                # resize
                resize = 15
                iy = int(round((vy / 49) * resize))
                ix = int(round((vx / 49) * resize))
                if ix > 0 or iy > 0:
                    cv2.arrowedLine(interpolation_img, (x, y), (x + ix, y + iy), (255, 0, 0), 1, tipLength=0.3)

cv2.imshow("normalization_3", frame1_img)
cv2.imshow("resize_3", interpolation_img)
cv2.waitKey()
cv2.destroyAllWindows()

'''Mean Shift Segmentation'''
# img numpy 형변환
frame1 = np.array(frame1_img, dtype='float32')

# 5차원 벡터 생성
x_vec = np.zeros(5, dtype='float64')
y_vec = np.zeros(5, dtype='float64')
shift_vec = np.zeros(5, dtype='float64')
v = np.zeros((height, width, 5), dtype='float64')

# Scale of Kernel hs, hr
hs = 40
hr = 30

# parameter min_shift
epsilon = 2

# Model Seeking
for x in range(0, width):
    for y in range(0, height):
        # 현재 pixel의 vector x 값 초기화
        x_vec[0] = y
        x_vec[1] = x
        x_vec[2] = frame1[y, x, 0]
        x_vec[3] = frame1[y, x, 1]
        x_vec[4] = frame1[y, x, 2]

        # initial y 설정
        y_vec = x_vec

        # density가 가장 높은곳 찾기
        curr_vec = np.zeros(5, dtype='float64')
        while True:
            numerator = np.zeros(5, dtype='float64')
            denominator = 0
            for i in range(0, width):
                for j in range(0, height):
                    # calculate distance s, r -> kernel 내 pixel일 경우 연산
                    s_dist = np.sqrt((j - y_vec[0]) ** 2 + (i - y_vec[1]) ** 2)
                    if s_dist < hs:
                        r_dist = np.sqrt((frame1[j, i, 0] - y_vec[2]) ** 2 +
                                         (frame1[j, i, 1] - y_vec[3]) ** 2 +
                                         (frame1[j, i, 2] - y_vec[4]) ** 2)
                        if r_dist < hr:
                            # current pixel값 설정
                            curr_vec[0] = j
                            curr_vec[1] = i
                            curr_vec[2] = frame1[j, i, 0]
                            curr_vec[3] = frame1[j, i, 1]
                            curr_vec[4] = frame1[j, i, 2]

                            # ks 값 구하기
                            ks_param = (curr_vec - y_vec) / hs
                            l2_norm = np.sqrt(ks_param[0] ** 2 +
                                              ks_param[1] ** 2)
                            if l2_norm <= 1:
                                ks = np.exp(-(l2_norm**2))
                            else:
                                ks = 0

                            # kr 값 구하기
                            kr_param = (curr_vec - y_vec) / hr
                            l2_norm = np.sqrt(kr_param[2] ** 2 +
                                              kr_param[3] ** 2 +
                                              kr_param[4] ** 2)
                            if l2_norm <= 1:
                                kr = np.exp(-(l2_norm**2))
                            else:
                                kr = 0

                            # k 곱하기
                            k = ks * kr
                            numerator = numerator + k * curr_vec
                            denominator = denominator + k

            # check convergence
            next_y_vec = numerator / denominator
            shift_vec = next_y_vec - y_vec
            l2_norm = np.linalg.norm(shift_vec)
            if l2_norm < epsilon:
                v[y, x] = next_y_vec
                break;
            else:
                y_vec = next_y_vec

# 저장된 v 값 load
v = np.load("./v40_30_2.npy")
hs = 40
hr = 38

# clustering vector v
img_size = width * height
assigned_cluster = np.full((height, width), -1, dtype='int64')
c_centroid = []
c_points_num = []
c_num = -1
for i in range(img_size):
    w = int(i / height)
    h = int(i % height)
    # 클러스터링 안된 pixel 찾기 -> 새로운 cluster 생성
    if assigned_cluster[h, w] == -1:
        c_num = c_num + 1
        assigned_cluster[h, w] = c_num
        c_points_num.append(1)
        c_centroid.append(v[h, w])
        # 새로운 cluster에 포함되는 pixel 찾기
        for j in range(i + 1, img_size):
            target_w = int(j / height)
            target_h = int(j % height)
            if assigned_cluster[target_h, target_w] == -1:
                # s, r 거리 계산 -> 인접 pixel clustering
                s_dist = np.sqrt((v[target_h, target_w, 0] - v[h, w, 0]) ** 2 +
                                 (v[target_h, target_w, 1] - v[h, w, 1]) ** 2)
                if s_dist < hs:
                    r_dist = np.sqrt((v[target_h, target_w, 2] - v[h, w, 2]) ** 2 +
                                     (v[target_h, target_w, 3] - v[h, w, 3]) ** 2 +
                                     (v[target_h, target_w, 4] - v[h, w, 4]) ** 2)
                    if r_dist < hr:
                        assigned_cluster[target_h, target_w] = c_num
                        c_points_num[c_num] = c_points_num[c_num] + 1
                        c_centroid[c_num] = c_centroid[c_num] + v[target_h, target_w]

        # cluster centroid 계산
        c_centroid[c_num] = c_centroid[c_num] / c_points_num[c_num]

# Merge Cluster
assigned = np.full(len(c_centroid), -1, dtype='int64')
cluster = []
cluster_num = []
c_num = -1
ds = 600
dr = 38
for i in range(len(c_centroid)):
    # 기존 cluster를 인접 cluster와 merge
    if assigned[i] == -1:
        c_num = c_num + 1
        assigned[i] = c_num
        cluster_num.append(1)
        cluster.append(c_centroid[i])
        # 새로운 cluster에 포함되는 cluster 찾기
        for j in range(i + 1, len(c_centroid)):
            if assigned[j] == -1:
                # s, r 거리 계산 -> 인접 cluster끼리 clustering
                s_dist = np.sqrt((c_centroid[j][0] - c_centroid[i][0]) ** 2 +
                                 (c_centroid[j][1] - c_centroid[i][1]) ** 2)
                if s_dist < ds:
                    r_dist = np.sqrt((c_centroid[j][2] - c_centroid[i][2]) ** 2 +
                                     (c_centroid[j][3] - c_centroid[i][3]) ** 2 +
                                     (c_centroid[j][4] - c_centroid[i][4]) ** 2)
                    if r_dist < dr:
                        assigned[j] = c_num
                        cluster_num[c_num] = cluster_num[c_num] + 1
                        cluster[c_num] = cluster[c_num] + c_centroid[j]

        # cluster centroid 계산
        cluster[c_num] = cluster[c_num] / cluster_num[c_num]

# 클러스터 centroid값으로 세그멘테이션 표현
for i in range(width):
    for j in range(height):
        for k in range(3):
            frame1_img[j, i, k] = c_centroid[assigned_cluster[j, i]][k + 2]
            frame1_copy_img[j, i, k] = cluster[assigned[assigned_cluster[j,i]]][k+2]

# edge로 세그멘테이션 표현
for i in range (1,width-1):
    for j in range (1,height-1):
        if (assigned_cluster[j,i] != assigned_cluster[j-1,i]) or (assigned_cluster[j,i] != assigned_cluster[j,i-1]):
            frame1_img[j, i, 0] = 0
            frame1_img[j, i, 1] = 0
            frame1_img[j, i, 2] = 0
        if (assigned[assigned_cluster[j, i]] != assigned[assigned_cluster[j - 1, i]]) or (assigned[assigned_cluster[j, i]] != assigned[assigned_cluster[j, i - 1]]):
            frame1_copy_img[j, i, 0] = 0
            frame1_copy_img[j, i, 1] = 0
            frame1_copy_img[j, i, 2] = 0

# segment별 motion vector 표현
length = len(cluster)
segment_motion = np.zeros((length,2), dtype='float32')
pixel_num = np.zeros(length, dtype='float32')
for x in range(width):
    for y in range(height):
        number = assigned[assigned_cluster[y,x]]
        segment_motion[number] = segment_motion[number] + motion[y, x]
        pixel_num[number] = pixel_num[number] + 1

for i in range(length):
    center_y = int(cluster[i][0])
    center_x = int(cluster[i][1])
    segment_motion[i] = segment_motion[i] / pixel_num[i]
    dy = int(segment_motion[i][0] * 20)
    dx = int(segment_motion[i][1] * 20)
    cv2.arrowedLine(frame1_copy_img, (center_x, center_y), (center_x + dx, center_y + dy), (255, 0, 0), 1, tipLength=0.5)

cv2.imshow("clusterd_img", frame1_img) # 모션벡터 포함 클러스터링
cv2.imshow("clusterd2_img", frame1_copy_img) # merge cluster
cv2.waitKey()
cv2.destroyAllWindows()
