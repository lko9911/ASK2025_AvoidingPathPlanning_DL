# 🥇 ResU-Net 관련 코드

### 재학습에 사용한 데이터셋은 Cityscapes Dataset를 사용함<br>
### 출처 : https://www.cityscapes-dataset.com/citation/
<pre> 인용 : M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” 
  in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. </pre>
<br>

- 재학습 코드는 kaggle의 Cityscapes Dataset 프로젝트 중 U-net 학습 코드 다수를 참고하였고, ResU-Net 논문에서 잔차 학습 및 지름길 연결 개념에 대한 아이디어를 얻음, 일부 (코드 오류 수정 및 변수 이름 지정) 에서는 ChatGPT 3.0 을 사용함

## ⭐ 결과

### 1. U-Net 모델<br>
![다운로드](https://github.com/user-attachments/assets/578116bc-39f2-42f3-b3d4-b59633d9f6bc)

### 2. ResU-Net 모델<br>
![스크린샷 2025-02-06 120044](https://github.com/user-attachments/assets/a00be2e3-e512-4076-8e24-c64bab341926)

### 3. 성능 측정 (ResU-Net)

- 계산식

![스크린샷 2025-02-07 104534](https://github.com/user-attachments/assets/3a4e0279-53a4-410d-aa42-b25f44a72041)
![스크린샷 2025-02-07 104451](https://github.com/user-attachments/assets/b0fa1883-8059-4009-858d-a6296d268d73)

- 결과

![스크린샷 2024-11-07 223204](https://github.com/user-attachments/assets/aa588485-4acc-4fab-9c30-100b35691e0c)
