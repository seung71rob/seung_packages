# Baseplate 검사·이송 및 뚜껑 조립 로봇팔 인수인계

작성 기준: 2026-09-23에 전달된 Python 코드 및 모델 파일. B팔 코드는 최신 전달본 `B_arm_v2_tolerance_1(10).py`를 기준으로 한다. 작성자가 Baseplate 공정의 실기 동작 성공을 확인했다. 뚜껑 조립 공정의 실기 성공 여부, 설치 패키지 전체 구성과 최종 장치 포트는 이 자료만으로 확인하지 못했다.

## 1. 시스템 구성과 읽는 순서

- **A팔**: 5축 로봇팔. q1~q4로 TCP 위치를 만들고 q5로 그리퍼를 회전한다. Dynamixel ID 1~7을 사용한다.
- **B팔**: 6관절 보조 로봇팔. Baseplate 다면 검사를 위해 카메라 시점을 맞춘다. Dynamixel ID 9~16을 사용한다.
- **레일·흡착**: Arduino와 시리얼로 통신한다. 레일 원점 및 위치, 흡착 ON/OFF를 제어한다.
- **비전**: 상부 Baseplate 판정, 다면 결함 판정, 뚜껑 조립용 구멍 위치 검출을 각각 별도 노드가 담당한다.
- **두 공정은 별도 실행 시퀀스**다. 전달된 메인 코드 사이에는 Baseplate 완료 후 뚜껑 공정을 자동으로 시작하는 직접 호출이 없다. 한 A팔·레일에 연결된 두 메인 노드를 동시에 실기 모드로 실행하지 않는다.

권장 읽는 순서: ① 이 문서 → ② `process_sequence_5axis.py` / `suction_sequence_5axis.py` → ③ 관련 비전 코드 → ④ 운동학·경로·Dynamixel·레일 공용 모듈.

## 2. 파일 목록과 배치

원본 업로드명에는 날짜 또는 `(번호)`가 붙어 있다. 아래 **배치 파일명은 메인 코드의 상대 import와 모델 기본 경로에 맞춘 제안**이다. 원본 파일은 보존하고, 설치할 사본을 필요한 이름으로 배치한다. 이미 실기에서 동작한 PC가 있다면 그 PC의 실제 폴더 구조와 실행 명령을 먼저 복사한다.

| 받은 파일 | 배치 위치·이름 예시 | 역할 |
|---|---|---|
| `process_sequence_5axis(20260923-115936).py` | `robot_description/process_sequence_5axis.py` | Baseplate 픽, OK/NG 분기, A팔·레일·B팔 동기화 |
| `B_arm_v2_tolerance_1(10).py` | 별도 실행 파일 `B_arm_v2_tolerance_1.py` | B팔 자세 이동 및 다면 검사 핸드셰이크; `(9)`가 아닌 최신본 사용 |
| `Baseplate_top(20260923-115917).py` | 별도 비전 실행 파일 `Baseplate_top.py` | 상부 검사와 `/detected_object` 발행 |
| `Baseplate_side_wonseok_2(20260923-115926).py` | 별도 비전 실행 파일 `Baseplate_side_wonseok_2.py` | bottom/back/flash 검사 결과 발행 |
| `top(4).pt` | 상부 코드의 절대경로가 가리키는 `top.pt` | 상부 YOLO 모델 |
| `bottom(5).pt`, `side(4).pt`, `flash(5).pt` | 측면 코드와 같은 위치의 `bottom.pt`, `side.pt`, `flash.pt`, 또는 실행 인자로 각 경로 지정 | 다면 검사 시점별 YOLO 모델 |
| `suction_sequence_5axis(20260923-120225).py` | `robot_description/suction_sequence_5axis.py` | 뚜껑 픽업·조립·복귀 |
| `baseplate_hole_wonseok(8).py` | 별도 비전 실행 파일 `baseplate_hole_wonseok.py` | 두 구멍 좌표·각도 계산과 `/detected_holes` 발행 |
| `best_Baseplate_hole(3).pt` | `best_Baseplate_hole.pt`로 배치하거나 `model_path`로 원본 경로 지정 | 구멍 검출 YOLO 모델 |
| `kinematics_ee_5axis(10).py` | `robot_description/kinematics_ee_5axis.py` | modified DH 기반 FK, IK, TCP 위치 |
| `trajectory_5axis(20260923-120438).py` | `robot_description/trajectory_5axis.py` | 관절 S-curve 이동 경로 |
| `vertical_path_5axis(20260923-120439).py` | `robot_description/vertical_path_5axis.py` | 수직 자세 하강, IK 완화, 속도 재배분 |
| `path_collision_guard_ee_5axis(20260923-120434).py` | `robot_description/path_collision_guard_ee_5axis.py` | PyBullet 바닥·자체 충돌 검사 |
| `dxl_bridge_suction_5axis(9).py` | `robot_description/dxl_bridge_suction_5axis.py` | A팔 ID 1~7 제어, 영점·기어비·방향 |
| `rail_serial_5axis(6).py` | `robot_description/rail_serial_5axis.py` | 레일 Arduino 시리얼 통신·흡착 제어 |

**추가로 필요한 파일:** `robot_arm_5axis_rail_suction.urdf`, `robot_arm_5axis_suction.urdf`, URDF가 참조하는 메시 전체, `robot_description`의 `package.xml`·`setup.py`·entry point, 레일 Arduino 펌웨어, 카메라 실행 설정. 이들은 이번 전달 파일에 없으므로 **실제로 동작했던 워크스페이스에서 가져와야 한다.** PyBullet 충돌 검사 코드는 `robot_arm_5axis_suction.urdf`의 visual geometry를 collision geometry로 복사한다.

기본 `pkg_dir`는 `~/robot_sim/src/robot_description`, `mesh_dir`는 그 아래 `meshes_robot_5axis`다. 실제 경로가 다르면 두 메인 노드에 `-p pkg_dir:=... -p mesh_dir:=...`를 전달한다. Python 상대 import가 동작하도록 여섯 공용 모듈은 ROS2 Python 패키지 `robot_description` 안에 놓는다.

## 3. Baseplate 검사·이송 공정

### 입력과 실행 순서

1. 상부 비전 노드가 A3 용지 네 모서리(좌상→우상→우하→좌하)를 화면에서 클릭해 보정한다. Baseplate 중심이 안정되면 `baseplate,x_mm,y_mm,0.0,0,OK|NG` 형식의 `std_msgs/String`을 `/detected_object`에 **한 번** 발행한다. `/inspection_state`에는 프레임별 판정을 JSON으로 발행한다.
2. `process_sequence_5axis`가 A3 좌표를 로봇 좌표로 바꾼다. 1차 접근 → 잠시 대기 → 2차점 수직 하강 → 추가 5 mm 압착 → 도착 확인 → 흡착 ON → 복귀점으로 이동한다.
3. **상부 NG**: A팔 이동자세 `[0, 50, -50, -95]°` → 레일 절대 700 mm → 배출 좌표 `[-0.35, 0, 0.05] m` → 기본 2초 대기 → 흡착 OFF → 레일 0 mm 복귀. 상부 NG에서는 B팔 다면 검사를 시작하지 않는다.
4. **상부 OK**: B팔의 `CHECK_READY` → bottom 검사 1회 → back 검사 8회 → flash 검사 8회. A팔 q5는 back과 flash에서 각각 `[0, 45, 90, 135, 180, -45, -90, -135]°`를 따른다. A팔은 back→flash 사이에 경유 자세를 사용한다.
5. 다면 검사에서 NG가 하나라도 확정되면 A팔이 B팔에 `ABORT(return_home=True)`를 보내고 불량 배출로 전환한다. 최신 B팔 `(10)`도 남은 검사를 진행하지 않도록 바뀌었다. 17회 모두 OK이면 B팔이 `SEQUENCE_COMPLETE`를 보낸 뒤 A팔은 정상 배출 좌표 `[-0.35, 0, 0.05] m`에서 흡착을 해제한다.

**다면 검사 기준:** `Baseplate_side_wonseok_2.py`는 검사 START마다 bottom에는 `bottom.pt`, back에는 `side.pt`, flash에는 `flash.pt`를 사용한다. 기본 검사 창은 3초이고, 결함이 **연속 2초** 나타나면 NG를 발행한다. 이미지 프레임 누락이나 중단 시 `INCOMPLETE`를 보낼 수 있다. B팔의 메뉴 **12번**이 반복 사이클 동기 검사 엔진을 실행한다.

### 주요 ROS2 인터페이스

| 토픽 | 형식 | 발신 → 수신 | 의미 |
|---|---|---|---|
| `/detected_object` | `std_msgs/String`, CSV | 상부 비전 → A팔 메인 | Baseplate 좌표·상부 OK/NG |
| `/inspection_state` | `std_msgs/String`, JSON | 상부 비전 → 관찰용 | 프레임별 상부 판정 |
| `/multi_arm/inspection_command` | `std_msgs/String`, JSON | A팔 메인 → B팔 | `CHECK_READY`, `GO_*`, `INSPECT_*`, `FINISH`, `ABORT` |
| `/multi_arm/b_arm_status` | `std_msgs/String`, JSON | B팔 → A팔 메인 | `READY`, `ARRIVED`, `INSPECTION_COMPLETE` 등 |
| `/baseplate_side/inspection_control` | `std_msgs/String`, JSON | B팔 → 측면 비전 | 시점별 `START`/`STOP` |
| `/baseplate_side/view_result` | `std_msgs/String`, JSON | 측면 비전 → B팔·A팔 메인 | view별 OK/NG/INCOMPLETE; `cycle_id`, `request_id` 포함 |
| `/process_state` | `std_msgs/String` | A팔 메인 → 외부 | `PHASE|cycle_id` |
| `/process_cmd` | `std_msgs/String` | 사용자 → A팔 메인 | `stop`, `reset`, `auto on/off`, `home`, `rail zero`, `rail <mm>` |
| `/rail_position` | `std_msgs/Float64` | A팔 메인 → 외부 | 레일 위치 m |

**현재 코드의 중요한 수치:** 접근 높이 80 mm, 압착 5 mm, `reject_rail_mm=700`, `reject_rail_pause=0.0초`, 배출 대기 2초, `auto_run=false`, `require_barm=true`, `barm_timeout=30초`, 레일 원점 요구. 파일 맨 위의 불량 레일 정지 “1초” 설명과 실제 기본값 `0.0초`가 다르므로 설정값을 기준으로 한다.

## 4. 뚜껑 조립 공정

1. `baseplate_hole_wonseok.py`가 상부 영상에서 Baseplate와 구멍 둘을 찾고, A3 네 점 보정 후 안정된 좌표를 `/detected_holes` (`Float64MultiArray`)로 **한 번** 발행한다. 값 순서는 `[mid_x_mm, mid_y_mm, side_x_mm, side_y_mm, angle_deg, distance_mm]`다. `r` 키는 비전 보정과 추적 상태를 초기화한다.
2. `suction_sequence_5axis`는 기본으로 **mid 구멍**을 목표로 선택한다(`hole_point=mid`). 레일 원점 설정 여부를 확인한 뒤 A팔 이동자세 `[0, 50, -50, -95]°` → 레일 0 mm → **630 mm** → 픽 자세 `[0, -45, -65, -65]°` → 흡착 ON → 이동자세 → 레일 0 mm → 팔 홈 순서로 진행한다.
3. 두 구멍이 이루는 각도가 `-90°, 150°, 30°` 중 하나에서 ±10° 안이면 q5를 **-20°**로 회전하고, 아니면 0°를 유지한다. 이후 측정한 구멍 XY로 이동한다. 목표 높이는 1차 80 mm, 2차 50 mm다.
4. 2차점부터 최대 **40 mm** 추가 하강하며 A팔 모터 **ID 5 또는 6**의 부하 절댓값을 감시한다. 기본 임계값은 **30%**, 탐침 속도는 **0.02 m/s**, 임계값 감지 후 정지는 **1초**다. 접촉 뒤 이동자세 → 레일 630 mm → 픽 자세 → 흡착 OFF → 레일 원점·팔 홈으로 돌아간다.
5. 레일 이동 전에는 이동자세 도달 확인과 기본 1초 대기를 거치도록 시퀀스가 구성되어 있다. 아두이노 레일의 `zeroSet`이 참이어야 실기 공정을 시작한다.

| 토픽 | 형식 | 의미 |
|---|---|---|
| `/detected_holes` | `std_msgs/Float64MultiArray` | 두 구멍 좌표·각도·거리 입력 |
| `/detected_hole_mid`, `/detected_hole_side` | `std_msgs/String` | 구멍별 별도 좌표 출력 |
| `/suction_state` | `std_msgs/String` | `PHASE|cycle_id` |
| `/suction_cmd` | `std_msgs/String` | `stop`, `reset`, `auto on/off`, `home` |
| `/rail_cmd` | `std_msgs/String` | `z` 원점, `p <mm>` 절대이동, `m <mm>` 상대이동, `s` 감속정지, `0` 즉시정지, `?` 상태조회 |

튜닝은 ROS2 인자가 환경변수보다 우선한다. 환경변수: `SUCTION_PROBE_LOAD`, `SUCTION_PROBE_EXTRA_MM`, `SUCTION_PROBE_SPEED`, `SUCTION_PROBE_HOLD`. 원본 코드의 머리말에는 픽 위치를 640 mm, 픽 마지막 관절을 -70°로 적었지만 **실제 기본값은 630 mm, -65°**다.

## 5. 공용 코드와 장치 설정

| 파일 | 변경 시 함께 확인할 내용 |
|---|---|
| `kinematics_ee_5axis.py` | 링크 길이 `L2=0.195 m`, `L3=0.165 m`, TCP 오프셋 `0.2115 m`, 수정 DH, 역기구학; URDF 치수와 일치해야 한다. |
| `trajectory_5axis.py` | 관절 S-curve 경로. |
| `vertical_path_5axis.py` | `q2+q3+q4=π` 수직 조건, 직선 하강·관절공간 대체 경로, 프로파일 속도 기준 시간 재배분. |
| `path_collision_guard_ee_5axis.py` | 별도 PyBullet DIRECT 인스턴스로 바닥/자체 충돌 확인; q5는 0에 두고 검사한다. |
| `dxl_bridge_suction_5axis.py` | A팔 ID 1=q1, 2=q2, 3·4=q3, 5·6=q4, 7=q5. 2,000,000 bps, Extended Position Mode. 실행 시 현재 자세를 영점으로 저장한다. |
| `rail_serial_5axis.py` | 기본 115,200 bps, 160 pulse/mm, 소프트리미트 0~900 mm. Arduino에 `z`, `p`, `m`, `von`, `voff` 등 문자열 명령을 보낸다. |

B팔은 `B_arm_v2_tolerance_1.py`에 자체 모터 매핑과 영점·저장 자세가 있다. B팔도 시작 시 **현재 물리 자세를 영점으로 캡처**하므로, 기존 성공한 실기 세팅과 같은 안전 자세에 둔 뒤 실행해야 한다.


