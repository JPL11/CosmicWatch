"""Path-capacity probe for the WAN federation: is the bottleneck each host's
link or the shared home uplink they all sit behind?

Server (on the Azure VM, bound to the WireGuard address only):
  python net_probe.py serve --bind 10.8.0.1
Client (on a host, over the tunnel):
  python net_probe.py run --server 10.8.0.1 --host pi5 --mb 20

The client uploads N MB, then downloads N MB, over one TCP connection, and
prints one JSON line with both rates measured at the client. Run hosts one
at a time for per-host capacity, then all at once for the shared ceiling.
Python 3.6+ (the JetPack 4 Nano), standard library only.
"""
import argparse
import json
import socket
import struct
import threading
import time

PORT = 29600
CHUNK = 1 << 16


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(min(CHUNK, n - len(buf)))
        if not part:
            raise ConnectionError("peer closed")
        buf += part
    return bytes(buf)


def drain(sock, n):
    left = n
    while left:
        part = sock.recv(min(CHUNK, left))
        if not part:
            raise ConnectionError("peer closed")
        left -= len(part)


def handle(conn):
    with conn:
        n = struct.unpack(">Q", recv_exact(conn, 8))[0]
        drain(conn, n)                      # client's upload
        conn.sendall(b"\x01")               # upload done marker
        blob = b"\x00" * CHUNK
        left = n
        while left:                         # client's download
            k = min(CHUNK, left)
            conn.sendall(blob[:k])
            left -= k


def serve(bind):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind, PORT))
    srv.listen(32)
    print("net_probe listening on %s:%d" % (bind, PORT), flush=True)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def run(server, host, mb):
    n = int(mb * 1e6)
    blob = b"\x00" * CHUNK
    s = socket.create_connection((server, PORT), timeout=300)
    s.sendall(struct.pack(">Q", n))
    t0 = time.time()
    left = n
    while left:
        k = min(CHUNK, left)
        s.sendall(blob[:k])
        left -= k
    recv_exact(s, 1)                        # server has received everything
    t_up = time.time() - t0
    t1 = time.time()
    drain(s, n)
    t_down = time.time() - t1
    s.close()
    print(json.dumps({"host": host, "mb": mb, "start_unix": round(t0, 3),
                      "up_mbps": round(n * 8 / t_up / 1e6, 2),
                      "down_mbps": round(n * 8 / t_down / 1e6, 2),
                      "up_s": round(t_up, 2), "down_s": round(t_down, 2)}),
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("serve")
    a.add_argument("--bind", default="10.8.0.1")
    b = sub.add_parser("run")
    b.add_argument("--server", default="10.8.0.1")
    b.add_argument("--host", required=True)
    b.add_argument("--mb", type=float, default=20)
    args = ap.parse_args()
    if args.cmd == "serve":
        serve(args.bind)
    elif args.cmd == "run":
        run(args.server, args.host, args.mb)
    else:
        ap.print_help()
