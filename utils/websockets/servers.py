import json
import logging
import queue
import threading

from websockets.sync.server import serve

from . import exceptions

logger = logging.getLogger(__name__)

class HollowGymServer:
    def __init__(self, server_ip: str, server_port: int, frame_skip: int, game_speed: float, boss_name: str,
                 boss_scene_name: str):
        self.server_ip = server_ip
        self.server_port = server_port

        self.mod_client = None
        self.mod_client_connected = threading.Event()
        self.mod_client_ready = threading.Event()
        self.incoming_messages = queue.Queue()

        self.frame_skip = frame_skip
        self.game_speed = game_speed
        self.boss_name = boss_name
        self.boss_scene_name = boss_scene_name

        self.server_thread = threading.Thread(target=self._start_server, daemon=True)

    def start(self):
        logger.info("Starting websocket server thread")
        self.server_thread.start()

    def _start_server(self):
        logger.info(f"Websocket server listening on {self.server_ip}:{self.server_port}")
        with serve(self._handler, self.server_ip, self.server_port) as server:
            server.serve_forever()

    def _consume(self):
        for message in self.mod_client:
            try:
                json_message = json.loads(message)
            except json.decoder.JSONDecodeError:
                logging.error("response from mod client can't be json decoded, ignoring the message")

            if json_message["Cmd"] == 0:
                logging.info("The mod client has sent a closing message")
                self.reset_connection()
            elif json_message["Cmd"] == 4:
                logging.info("The mod client is ready to receive commands")
                self.send_message(4, None, {
                    "BossSceneName": self.boss_scene_name,
                    "BossName": self.boss_name,
                    "FrameSkip": self.frame_skip,
                    "GameSpeed": self.game_speed,
                })
                response = self.mod_client.recv()
                try:
                    json_response = json.loads(response)
                except json.decoder.JSONDecodeError:
                    logging.error("response from mod client can't be json decoded, ignoring the message")

                if (json_response["Cmd"] == 4):
                    self.mod_client_ready.set()
            else:
                self.incoming_messages.put_nowait(json_message)

    def _handler(self, websocket):
        if self.mod_client is not None: raise exceptions.ModClientAlreadyConnected()
        self.mod_client = websocket
        self.mod_client_connected.set()

        self._consume()

        self.reset_connection()

    def reset_connection(self):
        self.mod_client_connected.clear()
        self.mod_client_ready.clear()
        self.mod_client = None

        while not self.incoming_messages.empty():
            self.incoming_messages.get_nowait()

    def send_message(self, cmd, action=None, settings=None):
        if self.mod_client is None: raise exceptions.ModClientNotConnected()
        message = {
            "Cmd" : cmd,
            "Data": {}
        }
        if action is not None: message["Data"]["Action"] = action
        if settings is not None: message["Data"]["Settings"] = settings
        json_message = json.dumps(message)
        try:
            self.mod_client.send(json_message)
        except websockets.exceptions.ConnectionClosed:
            logging.error("Message sending aborted due to connection closed")
            self.reset_connection()

    def message_exchange(self, cmd, action=None, settings=None):
        if cmd == 0: raise ValueError(f"command {cmd} doesn't not involve any message exchange")

        self.send_message(cmd, action, settings)
        response = self.incoming_messages.get()

        return response
