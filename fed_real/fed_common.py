"""Shared pieces for the real-hardware federation experiment."""
import pickle
import struct

import torch


def model():
    return torch.nn.Sequential(
        torch.nn.Linear(400, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 400), torch.nn.Sigmoid(),
    )


def weighted_loss(prediction, target):
    return (((prediction - target) ** 2) * (1.0 + 4.0 * target)).mean()


def pack_state(state, wire):
    """Encode a float32 state dict for the wire: fp32 | fp16 | int8."""
    import numpy as np
    out = {}
    for k, v in state.items():
        a = v.detach().cpu().numpy().astype("float32")
        if wire == "fp32":
            out[k] = a
        elif wire == "fp16":
            out[k] = a.astype("float16")
        elif wire == "int8":
            scale = float(np.abs(a).max()) / 127.0 or 1e-12
            out[k] = (np.round(a / scale).astype("int8"), scale)
        else:
            raise ValueError(wire)
    return out


def unpack_state(packed, wire):
    import numpy as np
    import torch as _t
    out = {}
    for k, v in packed.items():
        if wire == "int8":
            q, scale = v
            a = q.astype("float32") * scale
        else:
            a = np.asarray(v, dtype="float32")
        out[k] = _t.from_numpy(a.copy())
    return out


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">Q", len(data)) + data)
    return len(data) + 8


def recv_msg(sock):
    hdr = b""
    while len(hdr) < 8:
        chunk = sock.recv(8 - len(hdr))
        if not chunk:
            raise ConnectionError("peer closed")
        hdr += chunk
    n = struct.unpack(">Q", hdr)[0]
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        buf += chunk
    return pickle.loads(buf), n + 8
