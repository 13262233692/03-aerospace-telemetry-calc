import sys
import os
import base64
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.python.telemetry.telemetry_processor import TelemetryProcessor
from src.python.orbit.orbit_propagator import OrbitPropagator
from src.python.orbit.ekf_filter import ExtendedKalmanFilter
from src.python.orbit.orbital_elements import (
    OrbitalElements, rv_to_orbital_elements, orbital_elements_to_rv
)
from src.python.telemetry.shared_memory import TelemetrySharedMemory

from .models import (
    TelemetryRequest, TelemetryResponse, AttitudeQuaternion, GPSMeasurement,
    OrbitalElementsRequest, OrbitalElementsResponse,
    PropagationRequest, PropagationResponse,
    StateVector, EKFMeasurement, EKFResponse, SystemStatus
)

app = FastAPI(
    title="Aerospace Telemetry Calculation Engine",
    description="微小卫星入轨阶段遥测数据融合与轨道摄动科学计算引擎",
    version="1.0.0"
)

telemetry_processor = TelemetryProcessor()
orbit_propagator = OrbitPropagator(use_j2=True, use_drag=True)
ekf_filter = ExtendedKalmanFilter(orbit_propagator)
shm_manager = TelemetrySharedMemory()

connected_clients: Set[WebSocket] = set()

try:
    shm_manager.initialize(create=True)
except Exception:
    pass


def broadcast_sync(data: Dict[str, Any]):
    if connected_clients:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            message = json.dumps(data)
            for client in connected_clients:
                try:
                    loop.run_until_complete(client.send_text(message))
                except Exception:
                    pass
        finally:
            loop.close()


@app.get("/")
async def root():
    return {
        "name": "Aerospace Telemetry Calculation Engine",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    stats = telemetry_processor.get_stats()
    return SystemStatus(
        telemetry_frames_processed=stats['total_frames'],
        valid_frames=stats['valid_frames'],
        total_corrected_errors=stats['total_corrected_errors'],
        bytes_processed=stats['bytes_processed'],
        ekf_initialized=ekf_filter.state.timestamp > 0,
        last_update_time=ekf_filter.state.timestamp if ekf_filter.state.timestamp > 0 else None,
        connected_clients=len(connected_clients)
    )


@app.post("/api/telemetry/process", response_model=List[TelemetryResponse])
async def process_telemetry(request: TelemetryRequest):
    try:
        data = base64.b64decode(request.data)
        packets = telemetry_processor.process_data_stream(data)

        responses = []
        for packet in packets:
            attitude = None
            if packet.attitude:
                attitude = AttitudeQuaternion(
                    w=packet.attitude.w,
                    x=packet.attitude.x,
                    y=packet.attitude.y,
                    z=packet.attitude.z,
                    timestamp=packet.attitude.timestamp
                )

            gps_list = [
                GPSMeasurement(
                    prn=gps.prn,
                    pseudorange=gps.pseudorange,
                    carrier_phase=gps.carrier_phase,
                    doppler=gps.doppler,
                    timestamp=gps.timestamp
                )
                for gps in packet.gps_measurements
            ]

            response = TelemetryResponse(
                timestamp=packet.timestamp,
                attitude=attitude,
                gps_measurements=gps_list,
                housekeeping=packet.housekeeping,
                is_valid=packet.is_valid,
                corrected_errors=packet.corrected_errors
            )
            responses.append(response)

            if packet.is_valid and packet.gps_measurements:
                try:
                    if ekf_filter.state.timestamp == 0:
                        approx_pos = np.array([7000000.0, 0, 0])
                        approx_vel = np.array([0, 7500.0, 0])
                        ekf_filter.initialize_from_rv(approx_pos, approx_vel, timestamp=packet.timestamp / 1e9)

                    ts_seconds = packet.timestamp / 1e9
                    gps_dicts = [g.model_dump() for g in gps_list]
                    sat_positions = {}
                    for gps in gps_list:
                        angle = np.radians(gps.prn * 30)
                        sat_positions[gps.prn] = np.array([
                            26559700.0 * np.cos(angle),
                            26559700.0 * np.sin(angle),
                            0
                        ])
                    ekf_filter.process_gps_pseudorange(
                        gps_dicts, sat_positions, ts_seconds
                    )

                    elements = ekf_filter.get_orbital_elements()
                    if elements:
                        broadcast_data = {
                            "type": "orbital_elements",
                            "data": elements.to_dict(),
                            "timestamp": datetime.now().timestamp()
                        }
                        asyncio.create_task(
                            asyncio.gather(*[
                                client.send_json(broadcast_data)
                                for client in connected_clients
                            ])
                        )

                        shm_manager.write_state(
                            ekf_filter.state.position,
                            ekf_filter.state.velocity,
                            ts_seconds
                        )
                except Exception:
                    pass

        return responses
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/orbit/elements/from_rv", response_model=OrbitalElementsResponse)
async def rv_to_elements(state: StateVector):
    try:
        r = np.array(state.position)
        v = np.array(state.velocity)
        elements = rv_to_orbital_elements(r, v)
        elements.epoch = state.timestamp

        return OrbitalElementsResponse(
            semi_major_axis=elements.semi_major_axis,
            eccentricity=elements.eccentricity,
            inclination=elements.inclination,
            raan=elements.raan,
            arg_of_perigee=elements.arg_of_perigee,
            true_anomaly=elements.true_anomaly,
            epoch=elements.epoch
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/orbit/rv/from_elements", response_model=StateVector)
async def elements_to_rv(elements: OrbitalElementsRequest):
    try:
        elem = OrbitalElements(
            semi_major_axis=elements.semi_major_axis,
            eccentricity=elements.eccentricity,
            inclination=elements.inclination,
            raan=elements.raan,
            arg_of_perigee=elements.arg_of_perigee,
            true_anomaly=elements.true_anomaly,
            epoch=elements.epoch
        )
        r, v = orbital_elements_to_rv(elem)

        return StateVector(
            position=r.tolist(),
            velocity=v.tolist(),
            timestamp=elements.epoch
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/orbit/propagate", response_model=PropagationResponse)
async def propagate_orbit(request: PropagationRequest):
    try:
        elem = OrbitalElements(
            semi_major_axis=request.initial_elements.semi_major_axis,
            eccentricity=request.initial_elements.eccentricity,
            inclination=request.initial_elements.inclination,
            raan=request.initial_elements.raan,
            arg_of_perigee=request.initial_elements.arg_of_perigee,
            true_anomaly=request.initial_elements.true_anomaly,
            epoch=request.initial_elements.epoch
        )

        propagator = OrbitPropagator(use_j2=request.use_j2, use_drag=request.use_drag)
        result = propagator.propagate_elements(
            elem,
            t_span=(elem.epoch, elem.epoch + request.duration),
            dt=request.step
        )

        elements_list = []
        for el in result.elements:
            if el:
                elements_list.append(OrbitalElementsResponse(
                    semi_major_axis=el.semi_major_axis,
                    eccentricity=el.eccentricity,
                    inclination=el.inclination,
                    raan=el.raan,
                    arg_of_perigee=el.arg_of_perigee,
                    true_anomaly=el.true_anomaly,
                    epoch=el.epoch
                ))

        return PropagationResponse(
            time=result.time.tolist(),
            positions=result.position.tolist(),
            velocities=result.velocity.tolist(),
            elements=elements_list,
            event_times=result.event_times.tolist() if result.event_times is not None else None
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ekf/initialize")
async def initialize_ekf(state: StateVector, position_uncertainty: float = 100.0, velocity_uncertainty: float = 1.0):
    try:
        r = np.array(state.position)
        v = np.array(state.velocity)
        ekf_filter.initialize_from_rv(r, v, position_uncertainty, velocity_uncertainty, state.timestamp)
        return {"status": "success", "message": "EKF initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ekf/update", response_model=EKFResponse)
async def update_ekf(measurement: EKFMeasurement):
    try:
        if ekf_filter.state.timestamp == 0:
            raise HTTPException(status_code=400, detail="EKF not initialized")

        if measurement.position:
            pos = np.array(measurement.position)
            noise = np.eye(3) * (measurement.position_noise ** 2 if measurement.position_noise else 1.0)
            ekf_filter.update(pos, noise, measurement.timestamp)
        elif measurement.gps_pseudoranges:
            sat_positions = {}
            for gps in measurement.gps_pseudoranges:
                angle = np.radians(gps.prn * 30)
                sat_positions[gps.prn] = np.array([
                    26559700.0 * np.cos(angle),
                    26559700.0 * np.sin(angle),
                    0
                ])
            gps_dicts = [g.model_dump() for g in measurement.gps_pseudoranges]
            ekf_filter.process_gps_pseudorange(gps_dicts, sat_positions, measurement.timestamp)

        elements = ekf_filter.get_orbital_elements()
        elements_resp = None
        if elements:
            elements_resp = OrbitalElementsResponse(
                semi_major_axis=elements.semi_major_axis,
                eccentricity=elements.eccentricity,
                inclination=elements.inclination,
                raan=elements.raan,
                arg_of_perigee=elements.arg_of_perigee,
                true_anomaly=elements.true_anomaly,
                epoch=elements.epoch
            )

        pos_cov = ekf_filter.get_position_covariance()

        return EKFResponse(
            position=ekf_filter.state.position.tolist(),
            velocity=ekf_filter.state.velocity.tolist(),
            timestamp=ekf_filter.state.timestamp,
            elements=elements_resp,
            position_covariance=pos_cov.tolist()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/ekf/state", response_model=EKFResponse)
async def get_ekf_state():
    try:
        if ekf_filter.state.timestamp == 0:
            raise HTTPException(status_code=400, detail="EKF not initialized")

        elements = ekf_filter.get_orbital_elements()
        elements_resp = None
        if elements:
            elements_resp = OrbitalElementsResponse(
                semi_major_axis=elements.semi_major_axis,
                eccentricity=elements.eccentricity,
                inclination=elements.inclination,
                raan=elements.raan,
                arg_of_perigee=elements.arg_of_perigee,
                true_anomaly=elements.true_anomaly,
                epoch=elements.epoch
            )

        pos_cov = ekf_filter.get_position_covariance()

        return EKFResponse(
            position=ekf_filter.state.position.tolist(),
            velocity=ekf_filter.state.velocity.tolist(),
            timestamp=ekf_filter.state.timestamp,
            elements=elements_resp,
            position_covariance=pos_cov.tolist()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws/orbital_elements")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().timestamp()
                    })
                elif message.get("type") == "request_state":
                    if ekf_filter.state.timestamp > 0:
                        elements = ekf_filter.get_orbital_elements()
                        if elements:
                            await websocket.send_json({
                                "type": "orbital_elements",
                                "data": elements.to_dict(),
                                "timestamp": datetime.now().timestamp()
                            })
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except Exception:
        connected_clients.discard(websocket)


@app.post("/api/telemetry/stats/reset")
async def reset_telemetry_stats():
    telemetry_processor.reset_stats()
    return {"status": "success", "message": "Statistics reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
