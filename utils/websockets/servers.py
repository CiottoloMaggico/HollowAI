import asyncio
import json
import logging

import websockets
from websockets.asyncio.server import serve
from . import exceptions

logger = logging.getLogger(__name__)

class HollowGymServer:
    def __init__(self, server_ip: str, server_port: int):
        self.server_ip = server_ip
        self.server_port = server_port
        self.server_task = asyncio.create_task(self.start_server())

        self.mod_client = None
        self.mod_client_connected = asyncio.Event()
        self.mod_client_ready = asyncio.Event()
        self.incoming_messages = asyncio.Queue()

    def reset_connection(self):
        self.mod_client_connected.clear()
        self.mod_client_ready.clear()
        self.mod_client = None

        while not self.incoming_messages.empty():
            self.incoming_messages.get_nowait()

    async def consumer_handler(self, ws):
        async for message in ws:
            try:
                json_message = json.loads(message)
            except json.decoder.JSONDecodeError:
                logging.error("response from mod client can't be json decoded, ignoring the message")

            if json_message["Cmd"] == 0:
                logging.info("The mod client has sent a closing message")
                self.reset_connection()
            elif json_message["Cmd"] == 4:
                logging.info("The mod client is ready to receive commands")
                self.mod_client_ready.set()
            else:
                self.incoming_messages.put_nowait(json_message)

    async def handler(self, ws):
        if self.mod_client is not None: raise exceptions.ModClientAlreadyConnected()
        self.mod_client = ws
        consumer_task = asyncio.create_task(self.consumer_handler(ws))
        self.mod_client_connected.set()
        await consumer_task
        self.reset_connection()

    async def start_server(self):
        async with serve(self.handler, self.server_ip, self.server_port) as server:
            await server.serve_forever()

    async def send_message(self, cmd, action = None):
        if self.mod_client is None: raise exceptions.ModClientNotConnected()
        message = {
            "Cmd" : cmd,
            "Data": {}
        }
        if action is not None: message["Data"]["Action"] = action
        json_message = json.dumps(message)
        try:
            await self.mod_client.send(json_message)
        except websockets.exceptions.ConnectionClosed:
            logging.error("Message sending aborted due to connection closed")
            self.reset_connection()

    async def message_exchange(self, cmd, action=None):
        if cmd == 0: raise ValueError(f"command {cmd} doesn't not involve any message exchange")

        await self.send_message(cmd, action)
        try:
            response = await asyncio.wait_for(self.incoming_messages.get(), timeout=10)
        except asyncio.TimeoutError:
            if self.mod_client is not None:
                logging.info("Message exchange timed out but connection still on, retrying")
                return await self.message_exchange(cmd, action)
            raise exceptions.ModClientNotConnected()

        return response
