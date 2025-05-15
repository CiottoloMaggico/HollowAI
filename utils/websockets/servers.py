import json
import logging
import queue
import threading

import websockets
from websockets.sync.server import serve, ServerConnection

logger = logging.getLogger(__name__)

class HollowGymServer:
    def __init__(
            self, server_ip: str, server_port: int, client_settings : dict, n_clients: int
    ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_settings = client_settings
        self.n_clients = n_clients

        self._max_connections_sema = threading.Semaphore(self.n_clients)
        self.all_connected = threading.Event()
        self.ready = threading.Event()
        self.clients = []

        self._server_thread = threading.Thread(target=self._start_server, daemon=True)
        self._initialization_thread = threading.Thread(target=self._listen_until_ready, daemon=True)
        self._server_thread.start()
        self._initialization_thread.start()


    def _listen_until_ready(self):
        while not self.all_connected.is_set():
            if len(self.clients) == self.n_clients: self.all_connected.set()

        self.clients[0].master = True
        logger.info("All the client are connected, waiting for ready signal")
        while not self.ready.is_set():
            if all([client.ready.is_set() for client in self.clients]): self.ready.set()

        logger.info("The server is ready for the environment")
        return

    def _start_server(self):
        logger.info(f"Websocket server listening on '{self.server_ip}:{self.server_port}'")
        with serve(self._handler, self.server_ip, self.server_port) as server:
            server.serve_forever()

    # No need for thread pool, websocket library spawn a new thread for each incoming connection
    def _handler(self, websocket):
        with self._max_connections_sema:
            new_client = HollowClient(self, websocket, self.client_settings)
            self.clients.append(new_client)
            new_client.consume()
            self.clients.remove(new_client)


class HollowClient():
    def __init__(self, server : HollowGymServer, websocket: ServerConnection, master : bool = False):
        self.master = master
        self.server = server
        self.websocket = websocket
        self.incoming_messages = queue.SimpleQueue()
        self.ready = threading.Event()
        self.closed = threading.Event()


    def consume(self):
        self.server.all_connected.wait()
        for message in self.websocket:
            try:
                json_message = json.loads(message)
            except json.decoder.JSONDecodeError:
                logger.error("response from mod client can't be json decoded, ignoring the message")
                continue

            if json_message["Cmd"] == 0:
                self.exit()
                return
            elif json_message["Cmd"] == 4:
                self.handshake()
            elif json_message["Cmd"] == 5:
                self.handle_error(json_message["Data"]["ErrorMessage"])
            else:
                self.incoming_messages.put_nowait(json_message)

    def handshake(self):
        if self.ready.is_set():
            logger.warning(f"Client: is already ready")
            return
        logger.info(f"The client has started the ready handshake")
        self.send_message(4, None, self.server.client_settings)

        response = self.websocket.recv()
        try:
            json_response = json.loads(response)

            if not json_response["Cmd"] == 4:
                logger.warning("wrong response cmd from client, handshake interrupted")
                return
            if self.master: self.server.client_settings["ObservationSize"] = json_response["Data"]["Settings"]["ObservationSize"]
        except json.decoder.JSONDecodeError:
            logger.error("response from mod client can't be json decoded, handshake interrupted")
            return


        self.ready.set()
        logger.info("Client ready")
        return

    def handle_error(self, error_msg):
        logger.error(f"Client: has encountered an error\nError msg: {error_msg}")

    def exit(self):
        logger.info("The mod client has sent a closing message")
        self.closed.set()
        self.websocket.close()

    def send_message(self, cmd, action=None, settings=None):
        message = {"Cmd": cmd, "Data": {}}
        if action is not None: message["Data"]["Action"] = action
        if settings is not None: message["Data"]["Settings"] = settings

        json_message = json.dumps(message)
        try:
            self.websocket.send(json_message)
        except websockets.exceptions.ConnectionClosed:
            logger.error(f"Client: message sending aborted due to connection closed")

    def message_exchange(self, cmd, action=None, settings=None):
        if cmd == 0:
            logger.error(f"Client: command {cmd} doesn't involve any message exchange")
            raise AttributeError(f"Client: command {cmd} doesn't involve any message exchange")

        self.send_message(cmd, action, settings)
        response = self.incoming_messages.get()

        return response