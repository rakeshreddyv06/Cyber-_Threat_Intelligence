from scapy.all import sniff

def capture_packets(callback):

    sniff(
        prn=callback,
        store=False
    )