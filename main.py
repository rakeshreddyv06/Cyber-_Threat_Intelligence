from src.packet_capture import capture_packets
from src.detection import detect

print("Starting ConvLSTM Cyber Threat Detection System...")

capture_packets(detect)