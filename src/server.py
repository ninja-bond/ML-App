from fastapi import FastAPI, WebSocket
from typing import List

app = FastAPI()
connected_clients: List[WebSocket] = []

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            #broadcast changes to all clients
            for client in connected_clients:
                if client != websocket:
                    await client.send_text(data)
                
    except:
        connected_clients.remove(websocket)
                