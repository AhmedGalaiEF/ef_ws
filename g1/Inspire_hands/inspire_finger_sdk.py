%!PS-Adobe-3.0
%%Pages: (atend)
/Courier findfont 10 scalefont setfont
%%Page: 1 1
72 780 moveto (Inspire Hands Scripts Explanation) show
72 768 moveto () show
72 756 moveto (Python) show
72 744 moveto () show
72 732 moveto (python/三代手控制程序/demo_485.py) show
72 720 moveto (Controls a 3rd-generation hand over RS-485 serial. It opens /dev/ttyUSB0, writes) show
72 708 moveto (speed, force, and angle registers, reads back temperature and actual angle values,) show
72 696 moveto (and triggers an action sequence. This is the simplest serial demo.) show
72 684 moveto () show
72 672 moveto (python/三代手控制程序/demo_can.py) show
72 660 moveto (Controls a 3rd-generation hand using the CAN-style framed protocol. It builds) show
72 648 moveto (custom frames with 0xAA ... 0x55, splits 6 values across two frames when needed,) show
72 636 moveto (and reads values back in two parts.) show
72 624 moveto () show
72 612 moveto (python/四代手控制程序/demo_485.py) show
72 600 moveto (Same structure as the 3rd-generation demo_485.py, but intended for the) show
72 588 moveto (4th-generation hand. Good starting point if the device is connected by USB serial.) show
72 576 moveto () show
72 564 moveto (python/四代手控制程序/demo_can.py) show
72 552 moveto (CAN-style demo for the 4th-generation hand. It encodes addresses into extended) show
72 540 moveto (identifiers, sends write and read frames, and reconstructs the returned values.) show
72 528 moveto () show
72 516 moveto (python/四代手控制程序/demo_modbus.py) show
72 504 moveto (Controls one hand over Modbus TCP, defaulting to 192.168.11.210:6000. It writes) show
72 492 moveto (angle, force, and speed registers, reads status, error, and temperature values,) show
72 480 moveto (and runs an action sequence.) show
72 468 moveto () show
72 456 moveto (python/四代手控制程序/demo_modbus_multi-device.py) show
72 444 moveto (Same Modbus approach, but for multiple hands at once. It tries to connect to) show
72 432 moveto (several IPs, then starts one thread per device and runs the same control) show
72 420 moveto (sequence in parallel.) show
72 408 moveto () show
72 396 moveto (python/四代手控制程序/touch_data.py) show
72 384 moveto (Reads full tactile sensor register blocks over Modbus TCP. It formats raw data) show
72 372 moveto (into matrices for pinky, ring, middle, index, thumb, and palm, then prints the) show
72 360 moveto (structured tactile arrays.) show
72 348 moveto () show
72 336 moveto (python/四代手控制程序/touch_data_ts.py) show
72 324 moveto (Reads higher-level tactile force values instead of full matrices. It extracts) show
72 312 moveto (floating-point normal force and tangential force for each finger and prints them) show
72 300 moveto (continuously.) show
72 288 moveto () show
72 276 moveto (C) show
72 264 moveto () show
72 252 moveto (C/control_485_C/hand_api.h) show
72 240 moveto (Header for the low-level RS-485 API. It defines protocol constants, data) show
72 228 moveto (structures for finger state, and function declarations for reading and writing) show
72 216 moveto (registers and controlling the hand.) show
72 204 moveto () show
72 192 moveto (C/control_485_C/hand_api.c) show
72 180 moveto (Implementation of the low-level RS-485 API. It builds serial frames, parses) show
72 168 moveto (incoming frames, manages serial settings, and exposes functions like setting) show
72 156 moveto (angle, speed, force, clearing errors, and reading status.) show
72 144 moveto () show
72 132 moveto (C/control_485_C/test.cpp) show
72 120 moveto (Small C++ test program that uses hand_api. It opens /dev/ttyUSB0, starts) show
72 108 moveto (receive threads, and repeatedly runs Action\(0\) as a demo loop.) show
72 96 moveto () show
72 84 moveto (C/control_485_C/CMakeLists.txt) show
72 72 moveto (Build file for the C API and test program. It creates a static library) show
showpage
%%Page: 2 2
72 780 moveto (named hand_api and the test executable.) show
72 768 moveto () show
72 756 moveto (C/control_485_C/1.txt) show
72 744 moveto (A short usage note. It tells you to change the serial device and hand ID before) show
72 732 moveto (building and running.) show
72 720 moveto () show
72 708 moveto (C++) show
72 696 moveto () show
72 684 moveto (C++/三、四代手通用控制程序/control_485.cpp) show
72 672 moveto (Standalone C++ serial demo using Boost.Asio. It writes speed, force, and angle) show
72 660 moveto (registers, reads back values, and prints raw frames and parsed results.) show
72 648 moveto () show
72 636 moveto (C++/三、四代手通用控制程序/control_can.cpp) show
72 624 moveto (Standalone C++ CAN-style demo using Boost.Asio. It builds the custom frame) show
72 612 moveto (format, handles timeouts, writes six-value register groups in two chunks, and) show
72 600 moveto (reads them back.) show
72 588 moveto () show
72 576 moveto (Recommended starting points) show
72 564 moveto () show
72 552 moveto (Use python/.../demo_485.py for USB serial control.) show
72 540 moveto (Use python/.../demo_modbus.py for Ethernet or Modbus control.) show
72 528 moveto (Use touch_data.py or touch_data_ts.py if tactile sensor data is needed.) show
showpage
%%Trailer
%%Pages: 2
