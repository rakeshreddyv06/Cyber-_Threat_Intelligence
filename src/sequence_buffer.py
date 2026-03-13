import numpy as np
from collections import deque

sequence_buffer = deque(maxlen=10)

def build_sequence(features):

    if features is None:
        return None

    sequence_buffer.append(features)

    if len(sequence_buffer) < 10:
        return None

    seq = np.array(sequence_buffer)

    seq = seq.reshape(1, 10, 1, seq.shape[1], 1)

    return seq