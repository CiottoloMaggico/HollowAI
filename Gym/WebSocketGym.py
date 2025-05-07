import asyncio
import json

import websockets
from websockets.asyncio.server import serve


class WebSocketGym():
    def __init__(self, server_ip: str, server_port: int):
        self.server_ip = server_ip
        self.server_port = server_port

        self.mod_client = None
        self.server_task = asyncio.create_task(self.start_server())

    async def handler(self, ws):
        print("Mod client connected")
        self.mod_client = ws
        await asyncio.Future()

    async def start_server(self):
        async with serve(self.handler, self.server_ip, self.server_port) as server:
            await server.serve_forever()

    async def _perform_action(self, type, action=None):
        if self.mod_client is None: return None
        query = {
            "Cmd": type,
            "Data": {}
        }
        if action is not None: query["Data"]["Action"] = action

        try:
            await self.mod_client.send(json.dumps(query))
        except websockets.exceptions.ConnectionClosed:
            self.mod_client = None

    async def _receive_message(self):
        if self.mod_client is None: return None
        try:
            message = await self.mod_client.recv()
            return json.loads(message)
        except websockets.exceptions.ConnectionClosed:
            self.mod_client = None

    async def _message_exchange(self, type, action=None):
        print("Starting message exchange")
        if self.mod_client is None or type == 0: return None
        query = {
            "Cmd": type,
            "Data": {}
        }
        if action is not None: query["Data"]["Action"] = action

        print("Sending message")
        await self.mod_client.send(json.dumps(query))
        try:
            print("trying to receive message")
            res = await asyncio.wait_for(self.mod_client.recv(), timeout=10)
        except asyncio.TimeoutError:
            print("timeout")
            return await self._message_exchange(type, action)

        res = json.loads(res)

        print(res)
        return res
