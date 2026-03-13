import numpy as np
from collections import defaultdict

flows = defaultdict(list)

WINDOW_SIZE = 10

def extract_features(packet):

    try:
        src = packet["IP"].src
        dst = packet["IP"].dst
    except:
        return None

    key = (src, dst)

    flows[key].append(packet)

    if len(flows[key]) < WINDOW_SIZE:
        return None

    packets = flows[key]

    start_time = packets[0].time
    end_time = packets[-1].time

    duration = end_time - start_time

    total_packets = len(packets)

    total_bytes = sum(len(p) for p in packets)

    bytes_per_sec = total_bytes / duration if duration > 0 else 0
    packets_per_sec = total_packets / duration if duration > 0 else 0

    features = [
        duration,
        total_packets,
        total_bytes,
        bytes_per_sec,
        packets_per_sec
    ]

    while len(features) < 78:
        features.append(0)

    flows[key] = []

    return np.array(features)