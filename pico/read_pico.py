"""Capture the Pico benchmark JSON (and optionally the power phases).

Usage (Pico running pico_bench.uf2, plugged in over USB):
  pip install pyserial
  python pico/read_pico.py --out pico_benchmark_rp2350.json
  python pico/read_pico.py --out ... --phases 4   # also log 4 phase lines

Opening the port is what starts the benchmark (the firmware waits for a
USB connection), so start this BEFORE or right after plugging in.
"""
import argparse
import glob
import json
import sys
import time


def find_port():
    ports = (glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/ttyACM*"))
    if not ports:
        sys.exit("no Pico serial port found: is pico_bench.uf2 flashed "
                 "and the board plugged in (not in BOOTSEL mode)?")
    return ports[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="")
    ap.add_argument("--out", default="pico_benchmark_rp2350.json")
    ap.add_argument("--phases", type=int, default=0,
                    help="also record this many PHASE lines (each 10 s)")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    import serial
    port = args.port or find_port()
    s = serial.Serial(port, 115200, timeout=1)
    s.dtr = True
    result, phases, t0 = None, [], time.time()
    while time.time() - t0 < args.timeout + 10 * args.phases:
        line = s.readline().decode(errors="replace").strip()
        if not line:
            continue
        print(line)
        if line.startswith('{"pico_bench"'):
            result = json.loads(line)["pico_bench"]
        elif line.startswith("PHASE") and result is not None:
            phases.append({"host_t_unix": round(time.time(), 2),
                           "line": line})
        if result is not None and len(phases) >= args.phases:
            break
    if result is None:
        sys.exit(f"no benchmark line from {port} within {args.timeout}s")
    result["port"] = port
    result["captured_unix"] = round(time.time(), 1)
    if phases:
        result["phases"] = phases
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")
    if not result["verify_pass"]:
        sys.exit("VERIFY FAILED: on-device output differs from numpy")


if __name__ == "__main__":
    main()
