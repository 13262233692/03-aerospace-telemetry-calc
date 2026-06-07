import sys
import os
import struct
import base64
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.python.telemetry.telemetry_processor import TelemetryProcessor
from src.python.orbit.orbital_elements import OrbitalElements, orbital_elements_to_rv
from src.python.orbit.orbit_propagator import OrbitPropagator
from src.python.orbit.ekf_filter import ExtendedKalmanFilter
from src.python.telemetry.shared_memory import TelemetrySharedMemory


def generate_telemetry_frame(timestamp_ns: int, r: np.ndarray, v: np.ndarray, attitude_wxyz: np.ndarray) -> bytes:
    payload = bytearray()

    payload += struct.pack('>Q', timestamp_ns)

    payload += struct.pack('>Q', timestamp_ns)
    for val in attitude_wxyz:
        payload += struct.pack('>d', float(val))

    gps_count = 4
    payload.append(gps_count)

    for i in range(gps_count):
        angle = np.radians(i * 90)
        sat_pos = np.array([
            26559700.0 * np.cos(angle),
            26559700.0 * np.sin(angle),
            26559700.0 * np.sin(angle * 0.5)
        ])

        pseudorange = np.linalg.norm(sat_pos - r) + np.random.normal(0, 5.0)
        carrier_phase = pseudorange / 0.19029
        doppler = -np.dot(sat_pos - r, v) / np.linalg.norm(sat_pos - r) * 1575.42e6 / 299792458

        payload += struct.pack('>I', i + 1)
        payload += struct.pack('>d', pseudorange)
        payload += struct.pack('>d', carrier_phase)
        payload += struct.pack('>d', doppler)
        payload += struct.pack('>Q', timestamp_ns)

    hk_count = 3
    payload.append(hk_count)
    payload += struct.pack('>d', 25.0 + np.random.normal(0, 0.1))
    payload += struct.pack('>d', 4.5 + np.random.normal(0, 0.01))
    payload += struct.pack('>d', 3.3)

    asm = struct.pack('>I', 0x1ACFFC1D)
    header = bytearray(6)
    spacecraft_id = 42
    virtual_channel_id = 1
    frame_version = 0

    header[0] = (frame_version << 5) | ((spacecraft_id >> 8) & 0x1F)
    header[1] = spacecraft_id & 0xFF
    header[2] = (virtual_channel_id << 2)
    frame_length = 10 + len(payload) - 1
    header[2] |= (frame_length >> 8) & 0x03
    header[3] = frame_length & 0xFF
    header[4] = 0
    header[5] = 0

    return asm + bytes(header) + bytes(payload)


def demo_telemetry_parsing():
    print("=" * 60)
    print("演示 1: 遥测数据帧解析")
    print("=" * 60)

    processor = TelemetryProcessor(use_rs_decode=False)

    initial_elements = OrbitalElements(
        semi_major_axis=6778137.0,
        eccentricity=0.001,
        inclination=np.radians(97.5),
        raan=np.radians(0),
        arg_of_perigee=np.radians(90),
        true_anomaly=np.radians(0)
    )

    r, v = orbital_elements_to_rv(initial_elements)
    attitude = np.array([0.70710678, 0.0, 0.0, 0.70710678])

    print(f"初始轨道状态:")
    print(f"  位置: {r}")
    print(f"  速度: {v}")
    print(f"  姿态四元数: {attitude}")
    print()

    all_frames = b''
    timestamps = []
    for i in range(5):
        ts = int(time.time_ns() + i * 1000000000)
        timestamps.append(ts)
        frame = generate_telemetry_frame(ts, r + np.random.normal(0, 10, 3), v, attitude)
        all_frames += frame
        print(f"  生成帧 {i+1}, 长度: {len(frame)} 字节")

    print()
    print("解析遥测数据流...")
    packets = processor.process_data_stream(all_frames)

    print(f"\n解析结果:")
    for i, packet in enumerate(packets[:3]):
        print(f"\n  数据包 {i+1}:")
        print(f"    有效: {packet.is_valid}")
        if packet.attitude:
            print(f"    姿态: w={packet.attitude.w:.6f}, x={packet.attitude.x:.6f}, "
                  f"y={packet.attitude.y:.6f}, z={packet.attitude.z:.6f}")
        print(f"    GPS 测量数: {len(packet.gps_measurements)}")
        for gps in packet.gps_measurements[:2]:
            print(f"      PRN{gps.prn}: {gps.pseudorange:.2f} m")
        print(f"    遥测参数数: {len(packet.housekeeping)}")

    stats = processor.get_stats()
    print(f"\n统计信息:")
    print(f"  总帧数: {stats['total_frames']}")
    print(f"  有效帧: {stats['valid_frames']}")
    print(f"  处理字节数: {stats['bytes_processed']}")


def demo_orbit_propagation():
    print("\n" + "=" * 60)
    print("演示 2: 轨道传播 (RK45 + J2摄动 + 大气阻力)")
    print("=" * 60)

    propagator = OrbitPropagator(use_j2=True, use_drag=True)

    initial_elements = OrbitalElements(
        semi_major_axis=6778137.0,
        eccentricity=0.001,
        inclination=np.radians(97.5),
        raan=np.radians(45),
        arg_of_perigee=np.radians(90),
        true_anomaly=np.radians(0)
    )

    print(f"初始轨道根数:")
    print(f"  半长轴: {initial_elements.semi_major_axis/1000:.2f} km")
    print(f"  偏心率: {initial_elements.eccentricity:.6f}")
    print(f"  倾角: {np.degrees(initial_elements.inclination):.2f}°")
    print(f"  升交点赤经: {np.degrees(initial_elements.raan):.2f}°")
    print()

    print("传播 600 秒 (10 分钟)...")
    t0 = time.time()
    result = propagator.propagate_elements(
        initial_elements,
        t_span=(0, 600),
        dt=60
    )
    t1 = time.time()

    print(f"传播完成，耗时: {t1-t0:.3f} 秒")
    print(f"时间步数: {len(result.time)}")
    print()

    print("轨道根数演变:")
    for i, elem in enumerate(result.elements):
        if elem and i % 2 == 0:
            print(f"  t={result.time[i]:4.0f}s: a={elem.semi_major_axis/1000:.2f}km, "
                  f"e={elem.eccentricity:.6f}, i={np.degrees(elem.inclination):.2f}°, "
                  f"Ω={np.degrees(elem.raan):.2f}°")

    if result.event_times is not None and len(result.event_times) > 0:
        print(f"\n事件时间: {result.event_times}")


def demo_ekf_filtering():
    print("\n" + "=" * 60)
    print("演示 3: 扩展卡尔曼滤波 (EKF) 数据融合")
    print("=" * 60)

    propagator = OrbitPropagator(use_j2=True, use_drag=False)
    ekf = ExtendedKalmanFilter(propagator)

    initial_elements = OrbitalElements(
        semi_major_axis=6778137.0,
        eccentricity=0.001,
        inclination=np.radians(97.5),
        raan=np.radians(45),
        arg_of_perigee=np.radians(90),
        true_anomaly=np.radians(0)
    )

    r_true, v_true = orbital_elements_to_rv(initial_elements)
    r_init = r_true + np.array([5000.0, 3000.0, 1000.0])
    v_init = v_true + np.array([10.0, 5.0, 2.0])

    print(f"真实位置误差: {np.linalg.norm(r_init - r_true):.1f} m")
    print(f"真实速度误差: {np.linalg.norm(v_init - v_true):.3f} m/s")
    print()

    ekf.initialize_from_rv(
        r_init, v_init,
        position_uncertainty=10000.0,
        velocity_uncertainty=20.0,
        timestamp=0.0
    )

    print("EKF 初始化完成")
    print(f"初始位置协方差迹: {np.trace(ekf.get_position_covariance()):.1f}")
    print()

    print("开始滤波处理...")
    dt = 10.0
    n_steps = 30

    for step in range(n_steps):
        t = (step + 1) * dt

        r_true, v_true = propagator.step(r_true, v_true, dt)

        z = r_true + np.random.normal(0, 100.0, 3)
        R = np.eye(3) * (50.0 ** 2)

        state, innovation = ekf.update(z, R, timestamp=t)

        pos_error = np.linalg.norm(state.position - r_true)
        vel_error = np.linalg.norm(state.velocity - v_true)

        if step % 5 == 0:
            print(f"  t={t:4.0f}s: 位置误差={pos_error:8.2f}m, 速度误差={vel_error:6.3f}m/s, "
                  f"位置协方差迹={np.trace(ekf.get_position_covariance()):8.1f}")

    print()
    print(f"滤波完成!")
    print(f"最终位置误差: {np.linalg.norm(ekf.state.position - r_true):.2f} m")
    print(f"最终速度误差: {np.linalg.norm(ekf.state.velocity - v_true):.3f} m/s")

    elements = ekf.get_orbital_elements()
    if elements:
        print(f"\n最终轨道根数:")
        print(f"  半长轴: {elements.semi_major_axis/1000:.2f} km")
        print(f"  偏心率: {elements.eccentricity:.6f}")
        print(f"  倾角: {np.degrees(elements.inclination):.4f}°")


def demo_shared_memory():
    print("\n" + "=" * 60)
    print("演示 4: 跨语言共享内存")
    print("=" * 60)

    shm = TelemetrySharedMemory("demo_shm")
    print("创建共享内存...")
    success = shm.initialize(create=True)
    print(f"创建成功: {success}")

    if success:
        r = np.array([7000000.0, 1000000.0, 500000.0])
        v = np.array([1000.0, 7500.0, 200.0])
        ts = 1234567890.0

        print(f"\n写入状态向量:")
        print(f"  位置: {r}")
        print(f"  速度: {v}")
        print(f"  时间戳: {ts}")

        success_write = shm.write_state(r, v, ts)
        print(f"写入成功: {success_write}")
        print(f"写入计数: {shm.state_buffer.write_count}")

        print("\n从共享内存读取...")
        read_result = shm.read_state()

        if read_result:
            r_read, v_read, ts_read = read_result
            print(f"  读取位置: {r_read}")
            print(f"  读取速度: {v_read}")
            print(f"  读取时间戳: {ts_read}")
            print(f"  位置误差: {np.linalg.norm(r_read - r):.6f} m")
            print(f"  读取计数: {shm.state_buffer.read_count}")

        telemetry_data = {
            "timestamp": int(time.time_ns()),
            "attitude": {"w": 0.707, "x": 0, "y": 0, "z": 0.707},
            "gps_count": 4
        }
        shm.write_telemetry(telemetry_data)
        read_back = shm.read_telemetry()
        print(f"\n遥测数据回读: {read_back}")

        shm.close()
        print("\n共享内存已关闭")


def main():
    print("微小卫星入轨阶段遥测数据融合与轨道摄动科学计算引擎")
    print("演示程序")
    print()

    try:
        demo_telemetry_parsing()
        demo_orbit_propagation()
        demo_ekf_filtering()
        demo_shared_memory()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
