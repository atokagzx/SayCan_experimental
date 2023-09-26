#! /usr/binenv python3

import socket
from enum import Enum
import struct
from threading import Thread, Lock
from typing import Literal, Union, Tuple, Callable, Any

class Status(Enum):
    E_SUCCESS = 0
    E_NOT_AVAILABLE = 1
    E_NO_SENSOR = 2
    E_NOT_INITIALIZED = 3
    E_ALREADY_RUNNING = 4
    E_FEATURE_NOT_SUPPORTED = 5
    E_INCONSISTENT_DATA = 6
    E_TIMEOUT = 7
    E_READ_ERROR = 8
    E_WRITE_ERROR = 9
    E_INSUFFICIENT_RESOURCES = 10
    E_CHECKSUM_ERROR = 11
    E_NO_PARAM_EXPECTED = 12
    E_NOT_ENOUGH_PARAMS = 13
    E_CMD_UNKNOWN = 14
    E_CMD_FORMAT_ERROR = 15
    E_ACCESS_DENIED = 16
    E_ALREADY_OPEN = 17
    E_CMD_FAILED = 18
    E_CMD_ABORTED = 19
    E_INVALID_HANDLE = 20
    E_NOT_FOUND = 21
    E_NOT_OPEN = 22
    E_IO_ERROR = 23
    E_INVALID_PARAMETER = 24
    E_INDEX_OUT_OF_BOUNDS = 25
    E_CMD_PENDING = 26
    E_OVERRUN = 27
    E_RANGE_ERROR = 28
    E_AXIS_BLOCKED = 29
    E_FILE_EXISTS = 30

class Command(Enum):
    LOOP = 0x06
    DISCONNECT = 0x07
    HOMING = 0x20
    PREPOSITION = 0x21
    STOP = 0x22
    FAST_STOP = 0x23
    ACKN_FAST_STOP = 0x24
    GRIP = 0x25
    RELEASE = 0x26
    SET_ACC = 0x30
    GET_ACC = 0x31
    SET_FORCE_LIMIT = 0x32
    GET_FORCE_LIMIT = 0x33
    SET_SOFT_LIMIT = 0x34
    GET_SOFT_LIMIT = 0x35
    CLEAR_SOFT_LIMIT = 0x36
    OVERDRIVE_MODE = 0x37
    TARE_FORCE_SENSOR = 0x38
    GET_SYSTEM_STATE = 0x40
    GET_GRIP_STATE = 0x41
    GET_GRIPPING_STAT = 0x42
    GET_OPENING_WIDTH = 0x43
    GET_SPEED = 0x44
    GET_FORCE = 0x45
    GET_TEMP = 0x46
    GET_SYSTEM_INFO = 0x50
    SET_DEVICE_TAG = 0x51
    GET_DEVICE_TAG = 0x52
    GET_SYSTEM_LIMITS = 0x53
    GET_FINGER1_INFO = 0x60
    GET_FINGER1_FLAGS = 0x61
    FINGER1_POWER_CTRL = 0x62
    GET_FINGER1_DATA = 0x63
    GET_FINGER2_INFO = 0x70
    GET_FINGER2_FLAGS = 0x71
    FINGER2_POWER_CTRL = 0x72
    GET_FINGER2_DATA = 0x73

def float_to_bytes(f):
        assert isinstance(f, float)
        return struct.pack('f', f)

class WSG50:
    def __init__(self, host, port, timeout=5):
        self._host = host
        self._port = port
        self._socket = None
        self._timeout = timeout
        self._receive_thread = None
        self._connected = False
        self._lock = Lock()
        self._lock_reason = None
        # self._result_callback = None
    
    def _generate_cmd(self, cmd_id, payload, payload_size, checksum=False) -> bytearray:
        request = bytearray()
        request.extend(b'\xaa\xaa\xaa')
        request.append(cmd_id)

        payload_size_bytes = payload_size.to_bytes(2, byteorder='little')
        request.extend(payload_size_bytes)
        if isinstance(payload, int):
            payload = payload.to_bytes(payload_size, byteorder='big')
        elif isinstance(payload, (bytearray, bytes)):
            pass
        else:
            raise Exception('payload type not supported, must be int or bytearray')
        request.extend(payload)
        if checksum:
            raise NotImplementedError()
        else:
            request.extend(b'\x00\x00')
        return request
    
    
    def _receive(self) -> (int, int, bytearray, int):
        preamble = self._socket.recv(3)
        if not preamble == b'\xaa\xaa\xaa':
            raise Exception('preamble not ok')
        command_id = self._socket.recv(1)
        command_id = hex(int.from_bytes(command_id, byteorder='little'))
        payload_size = self._socket.recv(2)
        payload_size = int.from_bytes(payload_size, byteorder='little')
        payload = self._socket.recv(payload_size)
        checksum = self._socket.recv(2)
        return command_id, payload_size, payload, checksum
    
    @property
    def is_locked(self):
        return self._lock.locked(), self._lock_reason
    
    @property
    def is_connected(self):
        return self._connected
    
    @property
    def timeout(self):
        if self._socket:
            return self._socket.gettimeout()
        else:
            return self._timeout
        
    @timeout.setter
    def timeout(self, value):
        if self._socket:
            self._socket.settimeout(value)
        self._timeout = value

    def homing(self, direction=0):
        """
        direction: 0 = default from system config, 1 = open, 2 = close
        """
        message = self._generate_cmd(Command.HOMING.value, direction, 1)
        self._lock.acquire()
        self._lock_reason = Command.HOMING
        self._send(message)

    def grip(self, width=109, speed=420):
        width = float(width)
        speed = float(speed)
        width_bytes = float_to_bytes(width)
        speed_bytes = float_to_bytes(speed)
        message = self._generate_cmd(Command.GRIP.value, width_bytes + speed_bytes, 8)
        self._lock.acquire()
        self._lock_reason = Command.GRIP
        self._send(message)

    def release(self, width=110, speed=420):
        width = float(width)
        speed = float(speed)
        width_bytes = float_to_bytes(width)
        speed_bytes = float_to_bytes(speed)
        message = self._generate_cmd(Command.RELEASE.value, width_bytes + speed_bytes, 8)
        self._lock.acquire()
        self._lock_reason = Command.RELEASE
        self._send(message)

    def preposition(self, width, speed, stop_on_block=False, relative=False):
        '''
        stop_on_block: false - clamp, true - stop on block
        '''
        assert 0 <= width <= 110, "width must be between 0 and 110"
        assert 0 <= speed <= 420, "speed must be between 0 and 420"
        stop_on_block_vector = 0b00000001 if stop_on_block else 0b00000000
        relative_vector = 0b00000010 if relative else 0b00000000
        bit_vector = stop_on_block_vector | relative_vector
        vector_bytes = bit_vector.to_bytes(1, byteorder='little')
        width = float(width)
        speed = float(speed)
        width_bytes = float_to_bytes(width)
        speed_bytes = float_to_bytes(speed)
        message = self._generate_cmd(Command.PREPOSITION.value, vector_bytes + width_bytes + speed_bytes, 9)
        self._lock.acquire()
        self._lock_reason = Command.PREPOSITION
        self._send(message)

    def acknowledge(self):
        message = self._generate_cmd(Command.ACKN_FAST_STOP.value, b'\x61\x63\x6B', 3)
        self._send(message)

    def stop(self):
        message = self._generate_cmd(Command.STOP.value, 0, 0)
        # self._lock.acquire()
        # self._lock_reason = Command.STOP
        self._send(message)

    def set_force_limit(self, force):
        force = float(force)
        force_bytes = float_to_bytes(force)
        message = self._generate_cmd(Command.SET_FORCE_LIMIT.value, force_bytes, 4)
        self._send(message)

    def set_acc(self, acc):
        acc = float(acc)
        acc_bytes = float_to_bytes(acc)
        message = self._generate_cmd(Command.SET_ACC.value, acc_bytes, 4)
        self._send(message)
    
    def set_soft_limit(self, limit_minus, limit_plus):
        limit_minus = float(limit_minus)
        limit_plus = float(limit_plus)
        limit_minus_bytes = float_to_bytes(limit_minus)
        limit_plus_bytes = float_to_bytes(limit_plus)
        message = self._generate_cmd(Command.SET_SOFT_LIMIT.value, limit_minus_bytes + limit_plus_bytes, 8)
        self._send(message)

    def position_monitoring(self, mode: Literal['periodic', 'on_change', 'off'], period=100):
        """
        mode: 'periodic', 'on_change' or 'off'
        period: in ms, only used if mode is 'periodic'
        """
        assert mode in ('periodic', 'on_change', 'off'), "mode must be 'periodic', 'on_change' or 'off'"
        assert period >= 10, "period should not be less than 10 ms"
        if mode == 'periodic':
            vector = 0b000000001
        elif mode == 'on_change':
            vector = 0b000000011
        elif mode == 'off':
            vector = 0b00000000
        vector_bytes = vector.to_bytes(1, byteorder='little')
        period_bytes = period.to_bytes(2, byteorder='little')
        message = self._generate_cmd(Command.GET_OPENING_WIDTH.value, vector_bytes + period_bytes, 3)
        self._send(message)

            

    def connect(self):
        # Create a TCP/IP socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout)
        # Connect the socket to the port where the server is listening
        server_address = (self._host, self._port)
        self._socket.connect(server_address)
        self._connected = True
        self._receive_thread = Thread(target=self._receive_loop, daemon=True, name='wsg50_receive_thread')
        self._receive_thread.start()
    
    def _receive_loop(self):
        while True:
            try:
                received = self._receive()
            except socket.timeout:
                print("timeout")
                self._connected = False
                self._lock.locked() and self._lock.release()
                break
            else:
                print(f"lock reason: {self._lock_reason}")
                self._process_feedback(*received)
    
    def _send(self, message):
        if not self._connected:
            raise ConnectionAbortedError('not connected')
        self._socket.send(message)

    def _process_feedback(self, command_id, payload_size, payload, checksum):
        cmd = Command(int(command_id, 16))
        if cmd in (Command.HOMING, 
                Command.GRIP,
                Command.PREPOSITION,
                Command.RELEASE,
                Command.STOP):
            status_code = payload[:2]
            status_code = int.from_bytes(status_code, byteorder='little')
            status = Status(status_code)
            if status == Status.E_SUCCESS:
                params = payload[2:]
                print(f"{cmd} successful, params: {params}")
                if self._lock_reason == cmd:
                    self._lock.locked() and self._lock.release()
                    self._lock_reason = None
            elif status == Status.E_CMD_PENDING:
                print(f"{cmd} pending")
            else:
                print(f"{cmd} failed, status: {status}")
                if self._lock_reason == cmd:
                    self._lock.locked() and self._lock.release()
                    self._lock_reason = None

        elif cmd in (Command.ACKN_FAST_STOP, 
                    Command.SET_FORCE_LIMIT, 
                    Command.SET_SOFT_LIMIT, 
                    Command.SET_ACC):
            status_code = payload[:2]
            status_code = int.from_bytes(status_code, byteorder='little')
            status = Status(status_code)
            if status == Status.E_SUCCESS:
                params = payload[2:]
                print(f"{cmd} successful, params: {params}")
            else:
                print(f"{cmd} failed, status: {status}")
        elif cmd == Command.GET_OPENING_WIDTH:
            status_code = payload[:2]
            status_code = int.from_bytes(status_code, byteorder='little')
            status = Status(status_code)
            if status == Status.E_SUCCESS:
                params = payload[2:]
                width = struct.unpack('f', params[:4])[0]
                print(f"{cmd} successful, width: {width}")
        else:
            print(f"{cmd} not implemented")

if __name__ == "__main__":
    from time import sleep
    import traceback
    wsg = WSG50('192.168.0.111', 1000, timeout=10)
    wsg.connect()
    wsg.set_force_limit(20)
    wsg.position_monitoring('periodic')
    # wsg.set_acc(1)
    try:
        wsg.acknowledge()
        wsg.stop()
        wsg.homing()
        for i in range(20):
            wsg.grip(90, 280)
            # wsg.release(110, 280)
            wsg.grip(110, 100)
        # while wsg.is_connected:
        #     sleep(0.1)
    except ConnectionAbortedError:
        traceback.print_exc()
        print("connection aborted")