from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading

from src.packet_capture import capture_packets
from src.detection import detect

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

events = []


@app.get("/events")
def get_events():
    return events[-50:]


def packet_callback(packet):

    result = detect(packet)

    if result:

        events.append(result)


@app.on_event("startup")
def start_sniffer():

    thread = threading.Thread(
        target=capture_packets,
        args=(packet_callback,)
    )

    thread.daemon = True
    thread.start()