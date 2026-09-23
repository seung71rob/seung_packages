1. 전체적인 공정 과정

&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
1.1, 1.2는 전체적인 공정 순서이고
2.1, 2.2, 2.3 는 코드 실행 명령어 및 코드 이름 소개
3.1, 3.2는 gpt 이용한 코드 관계 설명입니다.

혹시라도 모르시는 부분은 언제든 연락부탁드립니다.
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

-> carpart는 개별 공정
-> baseplate 검사 공정 -> baseplate 조립 공정 -> 뚜껑 조립 공정 
이 워크스페이스에 담겨져있는 코드는 (baseplate 검사 공정, 뚜껑 조립 공정입니다.)

로봇팔의 이동자세는 다음과 같습니다
(0, 50, -50, -95), q1~q4 순서대로 명시
   
   1.1 Baseplate 공정
   Baseplate의 불량을 검출하는 과정
   
   먼저 topview 검사를 진행합니다.
   top_view 검사 실패 → 로봇팔 이동자세 이동 후 레일 이동(700mm), 불량 배출

   top_view 검사 성공 → 다면검사 공정 시작 → bottom view → back view → side view 순으로 검사가 진행됩니다

   다면검사 성공 → B_arm 초기위치 복귀, A_arm 정상품 배출 (레일 이동 X)

   다면검사에서 로봇팔의 자세는 다음과 같음 (수정필요)
   
   A_arm (process_sequence_5axis.py 코드 123번째 줄 참고)
   d('bottom_view_deg', [0.0, 15.0, -25.0, -80.0])
   d('back_view_deg', [0.0, 0.0, -85.0, 85.0])
   d('flash_view_deg', [0.0, 5.0, -60.0, -105.0])
   → q1, q2, q3, q4 순서대로 관절각이 명시되어있음

   B_arm (B_arm_v2_tolerance_1.py 코드 참고)
   B_BOTTOM_VIEW: Dict[str, float] = field(default_factory=lambda: {
   "q1": 0.0, "q2": 0.0, "q3": 15.0, "q4": 0.0, "q5": -85.0, "q6": 180.0})

   B_FLASH_VIEW: Dict[str, float] = field(default_factory=lambda: {
   "q1": 0.0, "q2": 0.0, "q3": 50.0, "q4": 0.0, "q5": -82.0, "q6": 180.0})

   B_BACK_VIEW: Dict[str, float] = field(default_factory=lambda: {
   "q1": 0.0, "q2": 0.0, "q3": 25.0, "q4": 0.0, "q5": -60.0, "q6": 180.0})

   1.2 뚜껑 공정
   baseplate 좌표 인식 -> 로봇팔 이동자세 -> 로봇팔 레일 이동(630mm) -> 로봇팔이 뚜껑 흡착 -> 이동자세 -> 
   로봇팔 레일 이동 (0mm) → baseplate 각도에 따라 로봇팔 회전 → 뚜껑 결합 작업 (부하값을 받을때까지 수직으로 내림) → 
   로봇팔 이동자세 → 로봇팔 이동 (630mm) → 뚜껑 지정된 위치에 두기 → 로봇팔 이동자세 → 레일 이동(0mm) → 로봇팔 초기자세


   2. 코드 분류
      !! 뚜껑공정과 baseplate에서 공통으로 사용되는 파일들
      dxl_bridge_suction_5axis.py
      rail_serial_5axis.py
      path_collision_guard_ee_5axis.py
      vertical_path_5axis.py
      kinematics_ee_5axis.py
      위치 -> rotbo_sim/src/robot_description/robot_description

      2.1 top_view 검사 코드 (python3로 구동), 첫 화면에 파일 있음(robot_sim)
      Basepalte_top.py (pt 파일 실행 코드, robot_sim에 위치)
      top.pt (top_view 검사의 pt 파일, robot_sim에 위치)

      실행 명령어
      → top 카메라 퍼블리셔
      ros2 run v4l2_camera v4l2_camera_node --ros-args -r
      __ns:=/top -p video_device:="/dev/video4" -p image_size:="[640,480]"

      → top 검사
      python3 Baseplate_top.py --ros-args -r image_raw:=/top/image_raw

      !! 다면 검사 코드 (bottom view → back view → side view)
      B_arm_v2_tolerance_1.py (카메라 로봇팔 실행파일, 실행 후 12번 입력해야 함, robot_sim에 위치)
      side.pt, bottom.pt, flash.pt (다면검사의 pt 파일, robot_sim에 위치)
      Baseplate_side_wonseok_2.py (robot_sim에 위치, 다면검사 pt파일 실행 코드)
      process_sequence_5axis.py (다면검사 공정 실행 코드, rotbo_sim/src/robot_description/robot_description에 위치)

      실행 명령어
      → 다면검사 카메라 (B_arm) 퍼블리셔
      ros2 run v4l2_camera v4l2_camera_node --ros-args -r __ns:=/side
      -p video_device:="/dev/video6" -p image_size:="[640,480]"

      → 다면검사 (B_arm) 키는 명령어 (파일 실행 후 12번 누르기)
      cd robot_sim
      python3 B_arm_v2_tolerance_1.py

      → 다면 검사 실행 명령어
      python3 Baseplate_side_wonseok_2.py --ros-args -r image_raw:=/side/image_raw
      -p max_sample_gap_sec:=1.0 -p inspection_duration_sec:=6.0

      -> 공정 실행 명령어
      ros2 run robot_description process_sequence_5axis --ros-args -p use_robot:=true
      -p dxl_port:=/dev/ttyUSB0 -p use_rail:=true -p rail_port:=/dev/ttyACM0
      -p auto_run:=true -p barm_timeout:=40.0

      -> 공정시작전 다른 터미널에서 원점을 설정해주어야 합니다ㅏ
      ros2 topic pub --once /process_cmd std_msgs/msg/String "{data: 'rail zero'}"

      2.2 뚜껑 공정 코드
      baseplate_hole_wonseok.py (pt 파일 실행 코드, robot_sim에 위치)
      best_Baseplate_hole.pt (뚜껑 공정 좌표, 각도 추출 pt 파일, robot_sim에 위치)
      suction_sequence_hole.py (뚜껑 공정 코드, rotbo_sim/src/robot_description/robot_description에 위치)

      카메라 퍼블리셔
      -> ros2 run v4l2_camera v4l2_camera_node --ros-args -r __ns:=/top
      -p video_device:="/dev/video4" -p image_size:="[640,480]"

      -> 좌표검출 코드 실행 명령어
      python3 Baseplate_hole_wonseok.py --ros-args -r image_raw:=/top/image_raw

      -> 뚜껑 공정 실행 명령어
      ros2 run robot_description suction_sequence_5axis --ros-args -p use_robot:=true
      -p dxl_port:=/dev/ttyUSB0 -p use_rail:=true -p rail_port:=/dev/ttyACM0 -p auto_run:=true

      -> 리니어 레일 원점 설정
      ros2 topic pub --once /rail_cmd std_msgs/msg/String "{data: 'z'}"

      2.3 아두이노 코드
      스텝모터를 제어하는 코드는 robot_sim/src/robot_description 에 위치한 rail_stepmotor_mega_scurve.ino 입니다.
      
      
      

      
      
