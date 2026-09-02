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
