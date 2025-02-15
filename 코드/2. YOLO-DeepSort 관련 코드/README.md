# 🥇 YOLO-DeepSort 코드

### yolov10x 모델은 Ultralytics 사용함<br><br>
- 검출 대상 설정 :  ‘사람’, ‘자전거’, ‘차량’, ‘오토바이’, ‘버스’, ‘기차’, ‘트럭’ <br><br>
### DeepSort는 검출 대상을 트래킹(넘버링) 용도로 사용함<br><br>
### 경로 시각화는 검출대상을 피하는(통과하지 않는) 방식으로 A*알고리즘과 OpenCV를 사용하여 시각화 <br><br>

<pre><code># A* 알고리즘 관련 함수 정의
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
</code></pre>


### YOLO 출처 : [https://www.cityscapes-dataset.com/citation/](https://docs.ultralytics.com/ko) <br><br>
### A* 알고리즘 인용 
<pre> PETER E. HART, NILS J. NILSSON, BERTRAM RAPHAEL. "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." 
  IEEE Transactions on Systems Science and Cybernetics, vol. 4, no. 2, 1968, pp. 100-107.</pre>
<br>

## ⭐ 결과

### 1차 경로 계획

![스크린샷 2025-02-01 192024](https://github.com/user-attachments/assets/f8da1c75-dc0d-4711-b62f-32b2d8fe68d7)
![스크린샷 2025-02-06 125451](https://github.com/user-attachments/assets/c7965857-affd-440b-98d7-15b20ec6c30d)
![스크린샷 2025-02-13 111244](https://github.com/user-attachments/assets/766ccc81-feac-4928-814b-5e9daf07aec0)
![스크린샷 2025-02-13 113518](https://github.com/user-attachments/assets/fc7ce728-9fc1-41b5-8902-d3998a6b98af)

