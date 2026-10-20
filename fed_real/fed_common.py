"""Shared pieces for the real-hardware federation experiment."""
import pickle
import struct

import torch


def _numpy2_pickle_compat():
    """Let a numpy < 1.26 host unpickle what numpy 2.x hosts send.

    numpy 2 pickles reference numpy._core.*, which older numpy lacks
    (the 2026-09 Pi 4 dropout); 1.26 ships its own alias, older needs
    this one. The reverse direction already works: numpy 2 still loads
    numpy.core.* pickles. Lets the JetPack 4 Nano (Python 3.6, numpy
    1.19, NVIDIA torch 1.10) join without leaving its CUDA stack."""
    import sys
    import numpy
    if tuple(int(x) for x in numpy.__version__.split(".")[:2]) >= (1, 26):
        return
    import numpy.core.multiarray
    import numpy.core.numeric
    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)


_numpy2_pickle_compat()


def model():
    return torch.nn.Sequential(
        torch.nn.Linear(400, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 400), torch.nn.Sigmoid(),
    )


def weighted_loss(prediction, target):
    return (((prediction - target) ** 2) * (1.0 + 4.0 * target)).mean()


class Int8EF:
    """Per-endpoint error-feedback int8 compressor (1-bit-SGD / EF-SGD
    style): the residual of each quantization is carried into the next
    one, so rounding error becomes a correction instead of a loss."""

    def __init__(self):
        self.e = {}

    def pack(self, state):
        import numpy as np
        out = {}
        for k, v in state.items():
            a = (v.detach().cpu().numpy().astype("float32")
                 + self.e.get(k, 0.0))
            scale = float(np.abs(a).max()) / 127.0 or 1e-12
            q = np.round(a / scale).astype("int8")
            self.e[k] = a - q.astype("float32") * scale
            out[k] = (q, scale)
        return out


class TopKEF:
    """Uplink top-k sparsification with error feedback (Aji & Heafield
    2017; Stich et al. 2018): keep only the k largest-magnitude entries
    of the residual-corrected tensor (values as fp16 + int32 indices),
    accumulate everything else into the residual."""

    def __init__(self, frac=0.10):
        self.frac = frac
        self.e = {}

    def pack(self, state):
        import numpy as np
        out = {}
        for k, v in state.items():
            a = (v.detach().cpu().numpy().astype("float32").ravel()
                 + self.e.get(k, 0.0))
            kk = max(1, int(round(len(a) * self.frac)))
            idx = np.argpartition(np.abs(a), -kk)[-kk:].astype("int32")
            vals = a[idx].astype("float16")
            e = a.copy()
            e[idx] -= vals.astype("float32")
            self.e[k] = e
            out[k] = (idx, vals, tuple(v.shape))
        return out


def unpack_topk(packed):
    import numpy as np
    import torch as _t
    out = {}
    for k, (idx, vals, shape) in packed.items():
        a = np.zeros(int(np.prod(shape)), dtype="float32")
        a[idx] = vals.astype("float32")
        out[k] = _t.from_numpy(a.reshape(shape))
    return out


# application-layer link emulation profiles: (down_bps, up_bps, rtt_s).
# Applied CLIENT-side to both directions, so no root/tc is needed and the
# emulation is identical on every board.
LINK_PROFILES = {
    "lte": (10e6, 5e6, 0.07),
    "ltem": (1e6, 375e3, 0.15),
    "nbiot": (60e3, 30e3, 1.0),
}


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
        if wire in ("int8", "int8ef", "int8_delta_ef", "int8_delta2_ef",
                    "topk_ef"):
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


def eval_sets(z):
    """Pooled held-out sets from a partition file of any size: returns
    (n_clients, test_all, val_all); val_all is None for files built
    without --val-frac (all pre-2026-10 campaigns)."""
    import torch as _t
    n = sum(1 for k in z.files if k.startswith("test_"))

    def pooled(prefix):
        parts = []
        for i in range(n):
            a = z[f"{prefix}_{i}"].astype("float32") / 255.0
            parts.append(_t.tensor(a.reshape(len(a), -1)))
        return _t.cat(parts)
    val = pooled("val") if f"val_0" in z.files else None
    return n, pooled("test"), val


def tcp_path_stats(sock):
    """Kernel-measured smoothed RTT, RTT variance and total retransmits of
    a connection (Linux TCP_INFO; tcpi_rtt/rttvar in us at offsets 68/72,
    tcpi_total_retrans at 100). Under WireGuard this is the full tunnelled
    WAN path. Empty dict where TCP_INFO is unavailable."""
    import socket as _s
    if not hasattr(_s, "TCP_INFO"):
        return {}
    try:
        raw = sock.getsockopt(_s.IPPROTO_TCP, _s.TCP_INFO, 104)
    except OSError:
        return {}
    rtt, rttvar = struct.unpack_from("<II", raw, 68)
    retrans = struct.unpack_from("<I", raw, 100)[0]
    return {"tcp_rtt_ms": round(rtt / 1000, 2),
            "tcp_rttvar_ms": round(rttvar / 1000, 2),
            "tcp_total_retrans": retrans}
