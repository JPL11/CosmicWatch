"""Bring up the Waveshare SIM7080G HAT (Cat-M / NB-IoT) with a Hologram
SIM and record the radio state, so a federation client can run over a
REAL cellular link instead of the emulated LINK_PROFILES.

Subcommands (run on the Pi that carries the HAT):
  setup   configure the modem (LTE-only, Cat-M or NB-IoT, APN
          "hologram"), wait for network registration, save status JSON
  status  read-only snapshot of registration + signal; run it right
          before and after each federation run so the radio conditions
          are in the record next to the results
  ppp     write the pppd peer + chat script (needs sudo), so that
          `sudo pon hologram` gives a ppp0 interface ordinary sockets
          (fed_async_client.py) can use

Usage:
  pip install pyserial
  python cell_bringup.py setup --mode catm --out cell_status_catm.json
  python cell_bringup.py status --out cell_pre_run.json
  sudo python cell_bringup.py ppp
  sudo pon hologram            # modem port now belongs to pppd
  sudo ip route add <server public ip>/32 dev ppp0

Only the route to the federation server goes over ppp0; Wi-Fi keeps
the default route, so SSH and everything else are unaffected and only
federation traffic is billed. The server must be reachable from the
internet (router port-forward of --port 29500 to the server box), since
the Hologram client sits behind carrier NAT and can only dial out.

On the server, give this host a name that is NOT in --link-hosts /
--link-map (e.g. --host zero2w_cell): its link is real, so it must not
also be shaped.
"""
import argparse
import json
import os
import re
import sys
import time

MODES = {"catm": 1, "nbiot": 2, "both": 3}     # AT+CMNB values
APN = "hologram"
# registration states from +CEREG / +CGREG: 1 = home, 5 = roaming
# (Hologram SIMs always register as roaming)
REG_OK = {1, 5}


class Modem:
    def __init__(self, port, baud, verbose):
        import serial
        self.s = serial.Serial(port, baud, timeout=0.2)
        self.verbose = verbose

    def at(self, cmd, timeout=5.0):
        """Send one AT command; return its response lines (without the
        echo and final OK). Raises on ERROR or timeout."""
        self.s.reset_input_buffer()
        self.s.write((cmd + "\r").encode())
        lines, t0 = [], time.time()
        while time.time() - t0 < timeout:
            raw = self.s.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if not line or line == cmd:
                continue
            if self.verbose:
                print(f"  {cmd!r} <- {line!r}", file=sys.stderr)
            if line == "OK":
                return lines
            if "ERROR" in line:
                raise RuntimeError(f"{cmd}: {line}")
            lines.append(line)
        raise TimeoutError(f"{cmd}: no OK within {timeout}s")

    def query(self, cmd, prefix, timeout=5.0):
        """Value part of the first '<prefix>: ...' response line."""
        for line in self.at(cmd, timeout):
            if line.startswith(prefix + ":"):
                return line.split(":", 1)[1].strip()
        return ""


def parse_reg(value):
    """'+CEREG: 0,5' -> 5 (the stat field); -1 if unparseable."""
    parts = value.split(",")
    try:
        return int(parts[1] if len(parts) > 1 else parts[0])
    except ValueError:
        return -1


def parse_csq(value):
    """'+CSQ: 18,99' -> approx RSSI dBm (-113 + 2*rssi); None if 99."""
    try:
        rssi = int(value.split(",")[0])
    except ValueError:
        return None
    return None if rssi == 99 else -113 + 2 * rssi


def parse_cpsi(value):
    """Best-effort split of the SIM7080 +CPSI LTE line:
    mode,op_state,MCC-MNC,TAC,SCellID,PCellID,band,EARFCN,dlbw,ulbw,
    RSRQ,RSRP,RSSI,RSSNR. Raw string is always kept alongside."""
    f = [x.strip() for x in value.split(",")]
    out = {"system_mode": f[0] if f else ""}
    if len(f) >= 14:
        names = ["op_state", "mcc_mnc", "tac", "scell_id", "pcell_id",
                 "band", "earfcn", "dl_bw", "ul_bw", "rsrq", "rsrp",
                 "rssi", "rssnr"]
        out.update(dict(zip(names, f[1:14])))
        for k in ("rsrq", "rsrp", "rssi", "rssnr"):
            try:
                out[k] = float(out[k])
            except ValueError:
                pass
    return out


def snapshot(m):
    """Registration, operator, and signal in one dict."""
    st = {"t_unix": round(time.time(), 1)}
    st["cereg"] = parse_reg(m.query("AT+CEREG?", "+CEREG"))
    st["registered"] = st["cereg"] in REG_OK
    st["operator"] = m.query("AT+COPS?", "+COPS")
    st["csq_dbm"] = parse_csq(m.query("AT+CSQ", "+CSQ"))
    cpsi = m.query("AT+CPSI?", "+CPSI")
    st["cpsi_raw"] = cpsi
    st["cpsi"] = parse_cpsi(cpsi)
    return st


def cmd_setup(m, args):
    m.at("ATE0")
    if "READY" not in m.query("AT+CPIN?", "+CPIN"):
        sys.exit("SIM not ready: check it is seated and the card is "
                 "activated in the Hologram dashboard")
    lines = m.at("AT+CCID")
    iccid = next((x for x in lines if re.fullmatch(r"\d{18,22}F?", x)), "")
    print(f"ICCID {iccid}  (must match the SIM activated on the dashboard)")
    m.at("AT+CFUN=0", timeout=10)                # radio off while configuring
    m.at("AT+CNMP=38")                           # LTE only
    m.at(f"AT+CMNB={MODES[args.mode]}")          # Cat-M / NB-IoT / both
    m.at(f'AT+CGDCONT=1,"IP","{APN}"')
    m.at("AT+CFUN=1", timeout=10)
    print(f"waiting up to {args.wait:.0f}s for {args.mode} registration...")
    t0 = time.time()
    while time.time() - t0 < args.wait:
        stat = parse_reg(m.query("AT+CEREG?", "+CEREG"))
        if stat in REG_OK:
            break
        time.sleep(3)
    st = snapshot(m)
    st.update({"iccid": iccid, "requested_mode": args.mode,
               "register_s": round(time.time() - t0, 1)})
    return st


def cmd_ppp(args):
    if os.geteuid() != 0:
        sys.exit("ppp: run with sudo (writes /etc/ppp and /etc/chatscripts)")
    chat = (
        'ABORT "BUSY"\nABORT "NO CARRIER"\nABORT "ERROR"\n'
        "TIMEOUT 30\n"
        '"" AT\n'
        "OK ATE0\n"
        f'OK AT+CGDCONT=1,"IP","{APN}"\n'
        "OK ATD*99#\n"
        'CONNECT ""\n')
    # nodefaultroute: only the federation server is routed over ppp0
    peer = (
        f"{args.port}\n{args.baud}\n"
        'connect "/usr/sbin/chat -v -f /etc/chatscripts/hologram"\n'
        "noauth\nnodefaultroute\nusepeerdns\nnoipdefault\n"
        "nocrtscts\nlocal\nlock\npersist\nmaxfail 0\nholdoff 10\n")
    os.makedirs("/etc/chatscripts", exist_ok=True)
    with open("/etc/chatscripts/hologram", "w") as f:
        f.write(chat)
    with open("/etc/ppp/peers/hologram", "w") as f:
        f.write(peer)
    print("wrote /etc/chatscripts/hologram and /etc/ppp/peers/hologram\n"
          "next: sudo pon hologram; ip addr show ppp0;\n"
          "      sudo ip route add <server public ip>/32 dev ppp0\n"
          "stop: sudo poff hologram")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["setup", "status", "ppp"])
    ap.add_argument("--port", default="/dev/serial0",
                    help="modem AT port (GPIO UART: /dev/serial0; USB: "
                         "usually /dev/ttyUSB2)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mode", choices=sorted(MODES), default="catm")
    ap.add_argument("--wait", type=float, default=180.0,
                    help="seconds to wait for registration")
    ap.add_argument("--out", default="", help="write status JSON here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.cmd == "ppp":
        cmd_ppp(args)
        return
    m = Modem(args.port, args.baud, args.verbose)
    try:
        m.at("AT", timeout=2)
    except (TimeoutError, RuntimeError):
        sys.exit(f"no AT response on {args.port}: is the HAT powered on "
                 "(press its PWR key; status LED lit), is the serial "
                 "console disabled in raspi-config, and is pppd not "
                 "holding the port?")
    st = cmd_setup(m, args) if args.cmd == "setup" else snapshot(m)
    print(json.dumps(st, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(st, f, indent=2)
    if not st["registered"]:
        sys.exit("not registered (CEREG stat "
                 f"{st['cereg']}); try a window, --mode both, or --wait 600")


if __name__ == "__main__":
    main()
