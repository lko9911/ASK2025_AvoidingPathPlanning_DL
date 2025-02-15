from ultralytics import YOLO
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from deep_sort_realtime.deepsort_tracker import DeepSort
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="__floordiv__ is deprecated")
import matplotlib.pyplot as plt

# 폰트 설정을 위한 matplotlib의 rcParams 사용
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
img_path = "content/test/images/val_image (258).png"
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
padding = 0  # 경계 상자 주변에 추가할 여유 공간 (픽셀)

# 클래스 이름 목록 가져오기
class_names = yolo_model.names  # YOLO 모델에서 클래스 이름을 가져옴

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 경계 상자 좌표
        conf = float(box.conf[0])  # 신뢰도
        class_id = int(box.cls[0])  # 클래스 ID
        if class_id >= 0 and class_id <= 7:
            class_name = class_names[class_id]  # 클래스 이름 가져오기
            
            # 패딩 적용
            x1_pad = max(x1 - padding, 0)
            y1_pad = max(y1 - padding, 0)
            x2_pad = min(x2 + padding, image_width)
            y2_pad = min(y2 + padding, image_height)

            # 감지된 객체의 경계 상자를 obstacle_mask에 업데이트
            obstacle_mask[y1_pad:y2_pad, x1_pad:x2_pad] = 1  # 패딩 영역을 장애물로 설정
            
            cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (235, 181, 179), 3)  # YOLO 경계 상자 그리기
            #cv2.rectangle(overlay_image, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 255, 255), 1)  # 위험 경계 상자 그리기

            # 클래스 이름과 신뢰도 표시
            label_text = f'{class_name}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_color = (255, 255, 255)  # 흰색 텍스트
            text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
            text_width, text_height = text_size

            # 텍스트 배경 그리기
            #cv2.rectangle(overlay_image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (235, 181, 179), -1)  # 배경
            #cv2.putText(overlay_image, label_text, (x1 + 5, y1 - 5), font, font_scale, text_color, font_thickness)  # 텍스트 그리기
            
            # detections 리스트에 추가하여 트래킹 처리에 사용
            detections.append([[x1, y1, x2, y2], conf, class_id])  # 클래스 ID 포함
            center = ((x1 + x2)//2, (y1 + y2)//2)

# 원본 detections의 복사본을 만들기
detections_copy = detections.copy()

# DeepSort로 객체 업데이트 및 트래킹
tracked_objects = tracker.update_tracks(detections_copy, frame=overlay_image)  # 복사본을 사용

# 트래킹된 객체를 오버레이 이미지에 표시
for track in tracked_objects:
    bbox = track.to_tlbr()  # 경계 상자 (top-left, bottom-right)
    track_id = track.track_id

    # 현재 객체의 중심 위치
    current_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    # 현재 중심 위치를 정수로 변환
    current_center = (int(current_center[0]), int(current_center[1]))

    # 예측된 중심 위치 계산 (DeepSORT가 예측한 다음 위치 사용)
    predict_center = (int(track.mean[0]), int(track.mean[1]))  # DeepSORT 예측 위치

    # 클래스 ID 찾기 (detections에서 해당 ID와 매칭되는 객체 찾기)
    class_id = None
    for det in detections:
        bbox_det, _, det_class_id = det
        x1, y1, x2, y2 = bbox_det
        if abs(x1 - bbox[0]) < 5 and abs(y1 - bbox[1]) < 5:  # 근접한 객체 찾기
            class_id = det_class_id
            break

    if class_id is not None:
        class_name = class_names[class_id]  # 클래스 이름 가져오기
    else:
        class_name = "Unknown"  # 기본값 설정

    font_scale = 1.5 
    font_thickness = 3  
    # ID 텍스트 설정
    id_text = f'{class_name} : {track_id}'
    id_text_size = cv2.getTextSize(id_text, font, font_scale, font_thickness)[0]
    id_text_width, id_text_height = id_text_size

    # 바운딩 박스의 좌상단 좌표
    x1, y1 = int(bbox[0]), int(bbox[1])

    # 텍스트 배경 위치 조정 (박스 위쪽에 배치)
    text_background_top = max(y1 - id_text_height - 10, 0)  # 화면 위쪽 경계를 초과하지 않도록
    text_background_bottom = y1

    # ID 배경 그리기
    cv2.rectangle(overlay_image, 
                (x1, text_background_top), 
                (x1 + id_text_width + 15, text_background_bottom),  # 여백 추가
                (235, 181, 179), -1)
    
    # ID 배경 그리기 (검은색 테두리 먼저 그림)
    cv2.rectangle(overlay_image, 
              (x1 - 2, text_background_top - 2),  # 박스 크기 약간 확장
              (x1 + id_text_width + 17, text_background_bottom + 2),  # 테두리 두께 고려
              (0, 0, 0), 3)  # 검은색 테두리 (두께 3)
    
    # 텍스트 그리기 (바운딩 박스 위쪽에 배치)
    cv2.putText(overlay_image, id_text, 
                (x1 + 8, text_background_bottom - 5),  # 여백 추가
                font, font_scale, text_color, font_thickness)  # 크기 증가된 텍스트 적용

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

if path:
    for i in range(len(path) - 1):
        # 경로를 빨간색 선으로 표시
        cv2.line(overlay_image, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), 
                 (0, 0, 255), thickness=5)

        # 객체가 Stop 영역에 들어온 경우 Stop 메시지 표시
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1_pad = max(x1 - padding, 0)
                y1_pad = max(y1 - padding, 0)
                x2_pad = min(x2 + padding, image_width)
                y2_pad = min(y2 + padding, image_height)

    # Goal 지점에 'Predicted Path' 텍스트 추가
    cv2.putText(overlay_image, 'Predicted Path', (goal[1] - 50, goal[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4) 
else:
    print("No path found!")
    cv2.putText(overlay_image, "Stop!", (start[1], start[0]), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 4)

# 결과 이미지 출력
plt.figure(figsize=(10, 5))
plt.title('YOLOv10x-DeepSORT and Predicted Path', fontsize=24)
plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # 축 숨기기
plt.show()
