from ultralytics import YOLO
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from deep_sort_realtime.deepsort_tracker import DeepSort
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="__floordiv__ is deprecated")
plt.rcParams['font.family'] = 'Arial'  # Arial 글씨체 설정

# A* 알고리즘 관련 함수 정의
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan 거리

def astar(start, goal, grid):
    rows, cols = grid.shape
    open_set = PriorityQueue()
    open_set.put((0, start))  # (f_score, point)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    # 대각선 이동을 위한 추가 방향 정의
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            # 경로 재구성
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # 역순으로 반환

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 1:  # 장애물인 경우
                    continue

                # 대각선 이동 비용 조정
                tentative_g_score = g_score[current] + (1.4 if dx != 0 and dy != 0 else 1)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # 경로 업데이트
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    if neighbor not in [i[1] for i in open_set.queue]:
                        open_set.put((f_score[neighbor], neighbor))

    return []  # 경로가 없는 경우

# YOLO 모델 로드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_path = "content/test/images/val_image (251).png"
yolo_model = YOLO("yolov10x.pt")

# Deep SORT 초기화
tracker = DeepSort()

# 도로 이미지와 마스크 이미지 불러오기
image = cv2.imread(img_path)
image_height, image_width = image.shape[:2]

# 이진 마스크 불러오기 (도로와 장애물 구분)
binary_mask = cv2.imread('data/predicted_mask_0.png', cv2.IMREAD_GRAYSCALE)
binary_mask = (binary_mask == 0).astype(np.uint8)  # 도로 부분은 0, 장애물 부분은 1

# 마스크를 컬러로 변환
mask_color = np.zeros_like(image)  # 마스크와 같은 크기의 빈 배열 생성
mask_color[binary_mask == 0] = [0, 255, 0]  # 도로 부분을 초록색으로 설정

# 원본 이미지와 마스크 이미지를 오버레이
overlay_image = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)

# YOLO 모델로 객체 탐지 (신뢰도 조정)
results = yolo_model.predict(img_path, imgsz=640, conf=0.5, save=False, show=False)

# 감지된 객체의 경계 상자를 오버레이 이미지에 추가 및 방해물 설정
obstacle_mask = np.copy(binary_mask)  # 방해물 마스크를 복사하여 업데이트
detections = []  # 감지된 객체 저장
padding = 50  # 경계 상자 주변에 추가할 여유 공간 (픽셀)

# 클래스 이름 목록 가져오기
class_names = yolo_model.names  # YOLO 모델에서 클래스 이름을 가져옴

# 시작점과 목표점 정의
start = (786, 1200)  # 시작점 (y, x)
pre_length = 150
stop_area_y_threshold = start[0] - pre_length - 50  # Stop 기준 y좌표
stop_area_x_threshold_left = start[1] - 100  # Stop 영역의 x 좌표 기준
stop_area_x_threshold_right = start[1] + 100

# 목표 설정: y가 450 이상이고 x가 start와 비슷한 가장 큰 도로 좌표 찾기
road_coords = [(y, x) for y in range(500, binary_mask.shape[0]) 
               for x in range(binary_mask.shape[1]) 
               if obstacle_mask[y, x] == 0 and abs(x - start[1]) <= 200]

if road_coords:
    # y 값이 가장 작은 좌표들 중 x 값이 start에 가장 가까운 좌표 선택
    goal = min(road_coords, key=lambda coord: coord[0])
else:
    print("No valid road coordinates found.")
    goal = start  # 도로가 없을 경우 시작점으로 goal 설정

# 시작점과 목표점의 좌표 출력
print(f"Start Point: {start}")
print(f"Goal Point: {goal}")

# A* 알고리즘 실행
path = astar(start, goal, obstacle_mask)

left_bound = start[1] - 100
right_bound = start[1] + 100

# Stop 메시지 추가
stop_message_displayed = False  # Stop 메시지가 표시되었는지 확인하는 플래그
left_exceeded = False
right_exceeded = False



# 방향 지정
#cv2.arrowedLine(overlay_image, (start[1], start[0]), (start[1], start[0] - pre_length),
#                (255,0,0), thickness=7)    
cv2.line(overlay_image, (start[1]+100, start[0] - pre_length-50), (start[1]+100, start[0] - pre_length+150),  # left
         (255,0,0), thickness=4)
cv2.line(overlay_image, (start[1]-100, start[0] - pre_length-50), (start[1]-100, start[0] - pre_length+150),  # right
         (255,0,0), thickness=4)
cv2.line(overlay_image, (start[1]+100, start[0] - pre_length-50), (start[1]-100, start[0] - pre_length-50),  # right
         (255,0,0), thickness=4)

# stop 영역 시각화
#cv2.rectangle(overlay_image, 
#              (stop_area_x_threshold_left, stop_area_y_threshold), 
#              (stop_area_x_threshold_right, image_height),  # 하단은 이미지의 높이
#              (0, 0, 255),  # 사각형 색상 (BGR 형식, 빨간색)
 #             thickness=2)  # 두께

# Stop 메시지 예시 추가
#cv2.putText(overlay_image, "Stop Area", (stop_area_x_threshold_left + 10, stop_area_y_threshold + 30), 
#            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# 주황색 BGR 값
#orange_color = (0, 165, 255)

# 알파값 설정 (0.0은 완전 투명, 1.0은 완전 불투명)
alpha = 0.5

# stop_area를 그릴 이미지 생성 (투명 배경)
overlay = np.copy(overlay_image)

# 사각형의 좌측 상단과 우측 하단 좌표
top_left = (stop_area_x_threshold_left, stop_area_y_threshold)
bottom_right = (stop_area_x_threshold_right, image_height)

# 주황색 사각형을 그려줍니다 (알파값을 적용하려면 이 후 addWeighted 함수 사용)
#cv2.rectangle(overlay, top_left, bottom_right, orange_color, thickness=-1)  # -1로 채우기

# 기존 이미지를 오버레이 이미지와 합성하여 알파값 적용
cv2.addWeighted(overlay, alpha, overlay_image, 1 - alpha, 0, overlay_image)


# 결과 이미지 출력
plt.figure(figsize=(10, 5))
plt.title('Baseline', fontsize=24)
plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # 축 숨기기
plt.show()
