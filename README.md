# dsr2-calibration

Doosan A0509 로봇의 eye-in-hand 카메라 변환행렬(T_cam2gripper)을 ChArUco 보드로 자동 추정합니다.

호스트에서 실행되며, Docker 컨테이너 내 DSR_ROBOT2와는 자동 배포되는 JSON-RPC bridge로 통신합니다.
별도 clone 없이 `uvx`로 바로 사용 가능합니다.

## 설치

```bash
# 별도 설치 없이 바로 실행
uvx --from git+https://github.com/c0sogi/dsr2-calibration dsr2-calibration --help

# 또는 로컬 설치
pip install git+https://github.com/c0sogi/dsr2-calibration
```

## 워크플로

```
generate-charuco → preview → dry-run → calibrate
```

### 1. 보드 인쇄

```bash
dsr2-calibration generate-charuco
# -> charuco_board.png (100%로 인쇄하면 칸이 정확히 40mm)
```

A4에 **100% 배율**로 인쇄 후 딱딱한 평면(아크릴판 등)에 부착합니다.

### 2. 카메라 확인

```bash
dsr2-calibration preview
```

실시간 카메라 피드에서 보드 감지 여부를 확인합니다 (초록색 = 감지됨, 빨간색 = 미감지). `q`로 종료.

### 3. 안전 확인 (dry-run)

```bash
dsr2-calibration dry-run
```

로봇을 보드가 잘 보이는 위치로 수동 이동한 뒤 실행합니다.
모든 캘리브레이션 포즈를 **저속(10 deg/s)**으로 순회하며:
- 각 포즈에서 보드 감지 여부 표시
- 충돌 위험 확인 가능
- `q`로 즉시 중단

### 4. 캘리브레이션

```bash
dsr2-calibration calibrate
```

로봇이 20개 포즈를 자동 이동하며 데이터를 수집하고, 카메라 렌즈 파라미터 + 카메라-그리퍼 변환행렬을 한 번에 계산합니다.

출력:
- `calibration_result_intrinsics.npz` - 카메라 렌즈 파라미터 (K, D)
- `calibration_result.npz` - 카메라-그리퍼 변환행렬 (T_cam2gripper)

## center pose 지정

로봇의 캘리브레이션 중심 위치를 지정하는 방법:

| 방식 | 예시 | 설명 |
|------|------|------|
| 생략 | `calibrate` | 현재 로봇 위치 사용 |
| `-j` | `-j 0,0,90,0,90,0` | 절대 관절각 (도) |
| `-x` | `-x 367,0,440,45,180,45` | 절대 Cartesian (mm/도, ZYZ Euler) |
| `-j d:` | `-j d:5,0,-5,0,0,0` | 현재 관절각 기준 delta |
| `-x d:` | `-x d:100,0,0,0,0,0` | 현재 Cartesian 기준 delta |

> **참고**: `-x`의 orientation(w,p,r)은 Doosan ZYZ Euler 컨벤션입니다.

## 단계별 실행

전체 `calibrate` 대신 개별 단계를 따로 실행할 수 있습니다:

```bash
# 카메라 렌즈 파라미터만 (이미 촬영한 이미지로)
dsr2-calibration calibrate-camera --images-dir ./board_images

# 카메라 렌즈 파라미터만 (로봇 자동 이동)
dsr2-calibration calibrate-camera -j 0,0,90,0,90,0

# 카메라-그리퍼 변환행렬만 (카메라 렌즈 파라미터 필요)
dsr2-calibration calibrate-transform -j 0,0,90,0,90,0
```

## 결과 사용

```python
from dsr2_calibration import CalibrationResult

result = CalibrationResult.load("calibration_result.npz")
T = result.T_cam2gripper  # 4x4 카메라->그리퍼 변환행렬 (미터 단위)
```

## 커스텀 보드

기본 보드(5x7, 40mm)가 아닌 다른 크기를 쓸 경우:

```bash
dsr2-calibration generate-charuco --cols 6 --rows 8 --square-length 0.035 --marker-length 0.025
dsr2-calibration calibrate --cols 6 --rows 8 --square-length 0.035 --marker-length 0.025
```

기본 보드를 인쇄했는데 실측이 다를 경우 (예: 38mm):

```bash
dsr2-calibration calibrate --square-length 0.038 --marker-length 0.028
```

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-j` / `--joints` | (현재 위치) | center pose 관절각 |
| `-x` / `--posx` | (현재 위치) | center pose Cartesian (ZYZ Euler) |
| `-n` / `--n-poses` | `20` | 캘리브레이션 포즈 수 |
| `-o` / `--output` | `calibration_result.npz` | 출력 파일 경로 |
| `--container` | `ros-control` | Docker 컨테이너 이름 |
| `--camera` | `0` | OpenCV 카메라 ID |
| `--vel` | `30` | 관절 속도 (deg/s) |
| `--acc` | `30` | 관절 가속도 (deg/s^2) |
| `--settle-time` | `1.0` | 포즈 도달 후 대기 시간 (초) |
| `--wrist-range` | `20` | wrist 관절 섭동 범위 (도) |
| `--arm-range` | `8` | arm 관절 섭동 범위 (도) |
| `--cols` | `5` | 보드 열 수 |
| `--rows` | `7` | 보드 행 수 |
| `--square-length` | `0.040` | 체커보드 칸 한 변 (미터) |
| `--marker-length` | `0.030` | ArUco 마커 한 변 (미터) |

## 라이브러리로 사용

```python
from dsr2_calibration import (
    BoardDetector,
    HandEyeCalibrator,
    DSR2Robot,
    calibrate_camera,
    auto_calibrate,
    generate_calibration_poses,
    posx_to_matrix,
    CalibrationResult,
)

# 로봇 연결
with DSR2Robot(container="ros-control") as robot:
    posx = robot.get_posx()  # [x, y, z, w, p, r]
    T = robot.get_pose_matrix()  # 4x4 변환행렬
    robot.move_to_joints([0, 0, 90, 0, 90, 0])
```

## 구조

```
src/dsr2_calibration/
├── cli.py          # CLI entry point
├── detector.py     # ChArUco 보드 감지 + 카메라 렌즈 캘리브레이션
├── calibration.py  # Hand-eye 캘리브레이션 솔버 + 자동화 파이프라인
├── robot.py        # Docker bridge 클라이언트 (DSR_ROBOT2 의존성 없음)
└── bridge.py       # 컨테이너 내 JSON-RPC 서버 (자동 배포됨)
```

`bridge.py`는 `DSR2Robot()` 생성 시 `docker cp`로 컨테이너에 주입되고
`docker exec -i`로 기동됩니다. 별도 볼륨 마운트나 포트 매핑이 필요 없습니다.
