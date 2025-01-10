import re
import socket
import sys
import threading
import time


class SchunkGripper:
    # num of commands after which clear_interpreter() command will be invoked.
    # If interpreted statements are not cleared periodically then "runtime too much behind" error may
    # be shown when leaving interpreter mode
    CLEARBUFFER_LIMIT = (
        20  # DONOT USE IT NOW --- clear the buffer for interpreter mode (the number of command you want to perform)
    )
    # EGUEGK_rpc_ip = "127.0.0.1"
    EGUEGK_rpc_ip = "10.42.0.162"
    local_ip = "10.42.0.1"
    local_port = 47000
    rpc_port = 55050
    UR_INTERPRETER_SOCKET = 30020
    Enable_Interpreter_Socket = 30003
    STATE_REPLY_PATTERN = re.compile(r"(\w+):\W+(\d+)?")  # DONOT USE IT NOW
    schunk_socket_name = "socket_grasp_sensor"
    gripper_index = 0
    ENCODING = "UTF-8"
    EGUEGK_socket_uid = 0
    uid_th = 500
    timeout = 1000
    rpc_socket_name = "rpc_socket"

    def __init__(self, local_ip="10.42.0.1", local_port=46995):
        self.enable_interpreter_socket = None  # the script port socket, for enabling interpreter mode
        self.socket = None  # interpreter mode socket, for sending gripper command
        self.schunk_socket = None  # schunk msg socket
        self.localhost, self.localport = local_ip, local_port

    def recv_schunk(self):
        rcv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        rcv_socket.settimeout(5)
        rcv_socket.bind((self.localhost, self.localport))
        rcv_socket.listen()
        # connect = True
        while True:
            try:
                self.schunk_client, addr = rcv_socket.accept()
                print("socket accepted for recv_shcunk")
                # connect = False
            except:
                print("socket timeout for recv_schunk")
                continue

    def connect(self, remote_function: bool = False, socket_timeout: float = 2.0) -> None:
        # connect to the gripper's address
        # hostname: robot's ip
        # port: the local free port created for Schunk (not 30003 and 30020)
        print("connect interpreter mode and schunk")
        # self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.enable_interpreter_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.EGUEGK_rpc_ip, self.UR_INTERPRETER_SOCKET))
            self.socket.settimeout(socket_timeout)
            self.enable_interpreter_socket.connect((self.EGUEGK_rpc_ip, self.Enable_Interpreter_Socket))
            self.enable_interpreter_socket.settimeout(socket_timeout)
        except socket.error as exc:
            raise exc
        # enable interpreter mode
        try:
            self.scriptport_command("interpreter_mode()")
            time.sleep(3)  # waiting the polyscope enable interpreter mode
        except:
            print("[ERROR] open interpreter mode failed")
            exit(0)

        # send funtion to remote interpreter port
        if remote_function is True:
            self._send_funtions()
            print("send funtion to interpreter port")
        else:
            print("NO funtion be sent to interpreter port")

    def _send_funtions(self):
        EGUEGK_abs = f"def EGUEGK_abs(value): if (value < 0): return -value end return value end"
        EGUEGK_socket_uid = f"global EGUEGK_socket_uid = 0"
        EGUEGK_getNextId = f"def EGUEGK_getNextId(): enter_critical EGUEGK_socket_uid = (EGUEGK_socket_uid + 1) % 100 uid = EGUEGK_socket_uid exit_critical return uid end"
        EGUEGK_rpcCall = f"def EGUEGK_rpcCall(socket_name, socket_address, socket_port, command, timeout = 2): socket_open(socket_address, socket_port, socket_name) socket_send_line(command, socket_name) sync() response = socket_read_line(socket_name, timeout) socket_close(socket_name) return response end"
        EGUEGK_executeCommand = f"def EGUEGK_executeCommand(socket_name, command, timeout = 1000): response = EGUEGK_rpcCall(socket_name, “{self.EGUEGK_rpc_ip}”, {self.rpc_port}, command, timeout) return response end"
        EGUEGK_moveAbsolute = f'def EGUEGK_moveAbsolute(socket_name, gripperIndex, position, speed): command = "absolute(" + to_str(gripperIndex) + ", " + to_str(position) + ", " + to_str(speed) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_moveRelative = f'def EGUEGK_moveRelative(socket_name, gripperIndex, position, speed): command = "relative(" + to_str(gripperIndex) + ", " + to_str(position) + ", " + to_str(speed) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_grip = f'def EGUEGK_grip(socket_name, gripperIndex, isDirectionOuter, position, force, speed): command = "grip(" + to_str(gripperIndex) + ", " + to_str(isDirectionOuter) + ", " + to_str(position) + ", " + to_str(force) + ", " + to_str(speed) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_release = f'def EGUEGK_release(socket_name, gripperIndex): command = "release(" + to_str(gripperIndex) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_simpleGrip = f'def EGUEGK_simpleGrip(socket_name, gripperIndex, isDirectionOuter, force, speed): command = "simpleGrip(" + to_str(gripperIndex) + ", " + to_str(isDirectionOuter) + ", " + to_str(force) + ", " + to_str(speed) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_fastStop = f'def EGUEGK_fastStop(socket_name, gripperIndex): command = "fastStop(" + to_str(gripperIndex) EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_stop = f'def EGUEGK_stop(socket_name, gripperIndex): command = "stop(" + to_str(gripperIndex) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_waitForComplete = f'def EGUEGK_waitForComplete(socket_name, gripperIndex, timeout = 10000): command = "waitForComplete(" + to_str(gripperIndex) + ", " + to_str(timeout) + ")" EGUEGK_executeCommand(socket_name + "waitForComplete", command, timeout + 1000) end'
        EGUEGK_setBrakingEnabled = f'def EGUEGK_setBrakingEnabled(socket_name, gripperIndex, braking): command = "setBrakingEnabled(" + to_str(gripperIndex) + ", " + to_str(braking) + ")" EGUEGK_executeCommand(socket_name + "braking", command) end'
        EGUEGK_brakeTest = f'def EGUEGK_brakeTest(socket_name, gripperIndex): command = "brakeTest(" + to_str(gripperIndex) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_acknowledge = f'def EGUEGK_acknowledge(socket_name, gripperIndex): command = "acknowledge(" + to_str(gripperIndex) + ")" EGUEGK_executeCommand(socket_name, command) end'
        EGUEGK_getPosition = f'def EGUEGK_getPosition(gripperNumber = 1): gripperIndex = gripperNumber - 1  command = "getPosition(" + to_str(gripperIndex) + ")" response = EGUEGK_executeCommand(to_str("socket_status_position_")), command) socket_send_line(response, "{self.schunk_socket_name}") return response end'
        all_list = [
            EGUEGK_abs,
            EGUEGK_socket_uid,
            EGUEGK_getNextId,
            EGUEGK_rpcCall,
            EGUEGK_executeCommand,
            EGUEGK_moveAbsolute,
            EGUEGK_moveRelative,
            EGUEGK_grip,
            EGUEGK_simpleGrip,
            EGUEGK_acknowledge,
            EGUEGK_fastStop,
            EGUEGK_stop,
            EGUEGK_waitForComplete,
            EGUEGK_setBrakingEnabled,
            EGUEGK_brakeTest,
            EGUEGK_getPosition,
        ]
        for i in range(len(all_list)):
            self.execute_command(all_list[i])

    def disconnect(self) -> None:
        time.sleep(1)  # waiting all of the command complete
        self.scriptport_command("end_interpreter()")
        self.socket.close()
        self.enable_interpreter_socket.close()

    def clear(self):
        self.scriptport_command("clear_interpreter()")

    def connect_server_socket(self):
        # self.rcv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # # self.rcv_socket.settimeout(5)
        # print(self.localhost, self.localport)
        # self.rcv_socket.bind((self.localhost, self.localport))
        # self.rcv_socket.listen()
        # command = f'socket_open("{self.localhost}", {self.localport}, "{self.schunk_socket_name}")'
        # self.execute_command(command)
        # self.schunk_listener, self.schunk_addr = self.rcv_socket.accept()
        # print(self.schunk_listener)

        # !threading method
        self.t_schunk = threading.Thread(target=self._socket_server)
        self.t_schunk.setDaemon(True)
        self.t_schunk.start()

    def _socket_server(self):
        rcv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        rcv_socket.bind((self.localhost, self.localport))
        rcv_socket.listen()
        command = f'socket_open("{self.localhost}", {self.localport}, "{self.schunk_socket_name}")'
        self.execute_command(command)
        try:
            schunk_listener, schunk_addr = rcv_socket.accept()
            # print('get schunk listener:', schunk_listener)
        except:
            print("accept the listener failed")
            sys.exit(0)
        while True:
            self.data = schunk_listener.recv(1024)
            # print('recv data:', self.data)

    def schunk_rpcCall(self, socket_name, command):
        # open another socket name for Schunk rpc_ip and port
        try:
            self.execute_command(f'socket_open("{self.EGUEGK_rpc_ip}", {self.rpc_port}, "{self.rpc_socket_name}")')
            # time.sleep(2) # waiting the socket opened in robotic local network
        except:
            print("[ERROR] open robotic local socket failed")
            exit(0)
        self.execute_command(f'socket_send_line("{command}", "{self.rpc_socket_name}")\nsync()\n')
        self.execute_command(f'socket_close("{self.rpc_socket_name}")')

    def schunk_rpcCallRe(self, socket_name, command):
        try:
            self.execute_command(f'socket_open("{self.EGUEGK_rpc_ip}", {self.rpc_port}, "{socket_name}")')
            # command = f'socket_open("{self.localhost}", {self.localport}, "{self.schunk_socket_name}")'
            # self.execute_command(command)
        except:
            print("[ERROR] open robotic local socket failed")
            exit(0)
        self.execute_command(f'socket_send_line("{command}", "{socket_name}")')
        self.execute_command(f'response=socket_read_line("{socket_name}", 2)')
        self.execute_command(f'socket_send_line(response, "{self.schunk_socket_name}")')
        # self.execute_command(f'popup(response)\n')
        self.execute_command(f'socket_close("{socket_name}")')
        # print(self.schunk_listener)
        # self.data = self.schunk_listener.recv(1024)
        return None

    def scriptport_command(self, command) -> None:
        # the port 30003 for urscript to communicate with UR robot
        if not command.endswith("\n"):
            command += "\n"
        # print(command.encode(self.ENCODING))
        self.enable_interpreter_socket.sendall(command.encode(self.ENCODING))

    def execute_command(self, command):
        # the port 30020 for interpreter mode to communicate with the binding port for Schunk
        if not command.endswith("\n"):
            command += "\n"
        # print('sending command:', command.encode(self.ENCODING))
        self.socket.sendall(command.encode(self.ENCODING))
        # data = self.socket.recv(1024)
        # return data

    def get_reply(self):
        collected = b""
        while True:
            part = self.socket.recv(1)
            if part != b"\n":
                collected += part
            elif part == b"\n":
                break
        return collected.decode(self.ENCODING)

    def EGUEGK_getNextId(self):
        self.EGUEGK_socket_uid = (self.EGUEGK_socket_uid + 1) % self.uid_th
        # uid = self.EGUEGK_socket_uid
        return self.EGUEGK_socket_uid

    # Control API Commands ----------------------------------------
    def moveAbsolute(self, gripperIndex, position, speed):
        command = "absolute(" + str(gripperIndex) + ", " + str(position) + ", " + str(speed) + ")"
        self.schunkHelperFunc(self.schunk_socket_name, command)

    def moveRelative(self, gripperIndex, position, speed):
        command = "relative(" + str(gripperIndex) + ", " + str(position) + ", " + str(speed) + ")"
        self.schunkHelperFunc(self.schunk_socket_name, command)

    def grip(self, gripperIndex, isDirectionOuter, position, force, speed):
        command = (
            "grip("
            + str(gripperIndex)
            + ", "
            + str(isDirectionOuter)
            + ", "
            + str(position)
            + ", "
            + str(force)
            + ", "
            + str(speed)
            + ")"
        )
        self.schunkHelperFunc(self.schunk_socket_name, command)

    def simpleGrip(self, gripperIndex, isDirectionOuter, force, speed):
        command = (
            "simpleGrip("
            + str(gripperIndex)
            + ", "
            + str(isDirectionOuter)
            + ", "
            + str(force)
            + ", "
            + str(speed)
            + ")"
        )
        self.schunkHelperFunc(self.schunk_socket_name, command)

    def release(self, gripperIndex):
        command = "release(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(self.schunk_socket_name, command)

    def fastStop(self, gripperIndex):
        command = "fastStop(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(self.schunk_socket_name, command)

    def stop(self, gripperIndex):
        command = "stop(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(self.schunk_socket_name, command)

    def acknowledge(self, gripperIndex):
        command = "acknowledge(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(self.schunk_socket_name, command)
        time.sleep(0.5)

    def waitForComplete(self, gripperIndex, timeout=10000):
        command = "waitForComplete(" + str(gripperIndex) + ", " + str(timeout) + ")"
        self.schunkHelperFunc(self.schunk_socket_name, command)

    def setBrakingEnabled(self, gripperIndex, braking):
        command = "setBrakingEnabled(" + str(gripperIndex) + ", " + str(braking) + ")"
        self.schunkHelperFunc(self.schunk_socket_name, command)

    def brakeTest(self, gripperIndex):
        command = "brakeTest(" + str(gripperIndex) + ")"
        self.schunkHelperFunc(self.schunk_socket_name, command)

    # Status Commands ----------------------------------------
    def getPosition(self, gripperNunber=1):  # TODO: use other socket name, see egk_contribution.script
        gripperIndex = gripperNunber - 1
        command = "getPosition(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_position_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        # self.status_rpcCall(socket_name, command)
        # response = self.schunkHelperFunc(str("socket_status_position_" + str(EGUEGK_getNextId())), command)
        return response

    def isCommandProcessed(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isCommandProcessed(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_cmd_processed_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def isCommandReceived(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isNotFeasible(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_cmd_received_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def notFeasible(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isNotFeasible(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_not_feasible_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def isReadyForOp(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isReadyForOperation(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_ready_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def isPositionReached(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isPositionReached(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_position_reached_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def isSWLimitReached(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "isSoftwareLimitReached(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_software_limit_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def getError(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "getError(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_error_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    def getWarning(self, gripperNumber=1):
        gripperIndex = gripperNumber - 1
        command = "getWarning(" + str(gripperIndex) + ")"
        socket_name = str("socket_status_warning_" + str(self.EGUEGK_getNextId()))
        response = self.schunkHelperFuncRe(socket_name, command)
        return response

    # Schunk Helper Functions --------------------------------
    def schunkHelperFunc(self, socket_name, command):
        # send command
        # command += "\n\tsync()\n"
        self.schunk_rpcCall(socket_name, command)

    def schunkHelperFuncRe(self, socket_name, command):
        # send command
        # command += "\n\tsync()\n"
        response = self.schunk_rpcCallRe(socket_name, command)
        # self.execute_command(f'socket_send_line("{command}", "{self.schunk_socket_name}")\n')
        return response
