import cv2
import numpy as np

np.seterr(over='ignore')
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# img 불러오기
frame1_img = cv2.resize(cv2.imread("plane1.jpg", cv2.IMREAD_COLOR), dsize=(510, 288), interpolation=cv2.INTER_AREA)
frame2_img = cv2.resize(cv2.imread("plane1.jpg", cv2.IMREAD_COLOR), dsize=(510, 288), interpolation=cv2.INTER_AREA)
v_img = cv2.resize(cv2.imread("plane1.jpg", cv2.IMREAD_COLOR), dsize=(510, 288), interpolation=cv2.INTER_AREA)
frame1_gray = cv2.cvtColor(frame1_img, cv2.COLOR_BGR2GRAY)
frame2_gray = cv2.cvtColor(frame1_img, cv2.COLOR_BGR2GRAY)
height, width, channel = frame1_img.shape

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
'''
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
                            l2_norm = np.sqrt(ks_param[0] ** 2 + ks_param[1] ** 2)
                            if l2_norm <= 1:
                                ks = np.exp(-(l2_norm**2))
                            else:
                                ks = 0

                            # kr 값 구하기
                            kr_param = (curr_vec - y_vec) / hr
                            l2_norm = np.sqrt(kr_param[2] ** 2 + kr_param[3] ** 2 + kr_param[4] ** 2)
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
#--------------------------------------------------------------------------
'''

# 저장된 v 값 load
v = np.load("./v40_30_2.npy")
hs = 40
hr = 30

# v 이미지 확인
for x in range(width):
    for y in range(height):
        for i in range(2,5):
            v_img[y,x,i-2] = v[y,x,i]

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
ds = 30
dr = 30
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

cv2.imshow("v_img", v_img)
cv2.imshow("original_img", frame1_img)


# 클러스터 centroid값으로 세그멘테이션 표현
for i in range(width):
    for j in range(height):
        for k in range(3):
            frame1_img[j, i, k] = c_centroid[assigned_cluster[j, i]][k + 2]
            frame2_img[j,i,k] = cluster[assigned[assigned_cluster[j,i]]][k+2]

#edge표현
for i in range (1,width-1):
    for j in range (1,height-1):
        frame1_gray[j,i]=0
        frame2_gray[j,i]=0
        if (assigned_cluster[j,i] != assigned_cluster[j-1,i]) or (assigned_cluster[j,i] != assigned_cluster[j,i-1]):
                frame1_gray[j, i] = 255
                frame1_img[j, i, 0] = 0
                frame1_img[j, i, 1] = 0
                frame1_img[j, i, 2] = 0
        if (assigned[assigned_cluster[j, i]] != assigned[assigned_cluster[j - 1, i]]) or (assigned[assigned_cluster[j, i]] != assigned[assigned_cluster[j, i - 1]]):
            frame2_gray[j, i] = 255
            frame2_img[j, i, 0] = 0
            frame2_img[j, i, 1] = 0
            frame2_img[j, i, 2] = 0


cv2.imshow("clusterd_img", frame1_img)
cv2.imshow("Frame_gray", frame1_gray)

cv2.imshow("clusterd2_img", frame2_img)
cv2.imshow("Frame2_gray", frame2_gray)
#cv2.imwrite("./c_40_40.png", frame1_img)
#cv2.imwrite("./c_g_40_40.png", frame1_gray)
cv2.waitKey()
cv2.destroyAllWindows()