#!/usr/bin/env bash
# Azure VM as the federation server, reachable ONLY over WireGuard.
#
# fed_common.recv_msg unpickles whatever arrives, so port 29500 must
# never face the internet: the VM's firewall (NSG) opens only SSH and
# WireGuard UDP 51820, and every edge device joins a private 10.8.0.0/24
# network with the server at 10.8.0.1.
#
# Run from the Mac (needs: az, wg, ssh, rsync), in this order:
#   ./azure_setup.sh create                  # VM + NSG rule + nightly auto-shutdown
#   ./azure_setup.sh keys pi5 pi4 jetson zero2w_cell
#                                            # keys/configs in wg/ (re-run to ADD hosts)
#   ./azure_setup.sh install                 # WireGuard + Python env + code on the VM
#   ./azure_setup.sh push                    # (after adding hosts) reload VM peers
#   ./azure_setup.sh client pi5 pi@pi5.local # install wg0.conf on a device
#   ./azure_setup.sh start | stop | status   # stop = deallocate (no compute billing)
#
# Then on the VM:  ssh <user>@<ip>; cd fed; ~/fl/bin/python fed_async_server.py ...
# On each device:  python fed_async_client.py --server 10.8.0.1 --host pi5 ...
#
# Settings (env overrides): AZ_RG AZ_VM AZ_LOC AZ_SIZE AZ_USER AZ_SHUTDOWN_UTC
set -euo pipefail

RG=${AZ_RG:-cosmicwatch}
VM=${AZ_VM:-fedsrv}
LOC=${AZ_LOC:-westus}            # student policy: westus, centralus, canadacentral, mexicocentral, francecentral
# Azure for Students (2026-10-20): D v3-v6 are quota 0 or capacity-restricted
# in every allowed region; Bpsv2 (ARM64, burstable) is the only usable family.
# Record CPU credits around each run (status) so no result is from a throttled VM.
SIZE=${AZ_SIZE:-Standard_B4ps_v2}
IMAGE=${AZ_IMAGE:-Canonical:ubuntu-24_04-lts:server-arm64:latest}
ADMIN=${AZ_USER:-$USER}
SSH_PUB=${AZ_SSH_PUB:-$HOME/.ssh/id_ed25519.pub}  # reuse the fleet key
SHUTDOWN_UTC=${AZ_SHUTDOWN_UTC:-0700} # 00:00 PDT / 23:00 PST
NUMPY=2.4.6                           # fleet-wide pin (see FED_REAL_NOTES)

HERE=$(cd "$(dirname "$0")" && pwd)
WG="$HERE/wg"                         # private keys live here: gitignored
NET=10.8.0
PORT=51820

die() { echo "error: $*" >&2; exit 1; }
# Azure CLI calls can hang indefinitely on an API stall (one did for 1.5 h
# mid-campaign on 2026-10-20); macOS has no timeout(1), so use perl alarm.
az_t() { perl -e 'alarm shift; exec @ARGV' "${AZ_TIMEOUT:-60}" az "$@"; }
need() { command -v "$1" >/dev/null || die "$1 not found ($2)"; }

server_ip() {
    [[ -s "$WG/server_ip" ]] || die "no $WG/server_ip: run create first"
    cat "$WG/server_ip"
}

# hosts.txt lines: "<name> <last octet>", server is .1
host_octet() { awk -v h="$1" '$1 == h {print $2}' "$WG/hosts.txt" 2>/dev/null; }

genkey() {  # genkey <name>: create <name>.key/.pub once, never overwrite
    [[ -s "$WG/$1.key" ]] && return
    (umask 077; wg genkey > "$WG/$1.key")
    wg pubkey < "$WG/$1.key" > "$WG/$1.pub"
}

write_server_conf() {
    {
        echo "[Interface]"
        echo "Address = $NET.1/24"
        echo "ListenPort = $PORT"
        echo "PrivateKey = $(cat "$WG/server.key")"
        while read -r h n; do
            echo
            echo "# $h"
            echo "[Peer]"
            echo "PublicKey = $(cat "$WG/$h.pub")"
            echo "AllowedIPs = $NET.$n/32"
        done < "$WG/hosts.txt"
    } > "$WG/server.conf"
    chmod 600 "$WG/server.conf"
}

write_client_conf() {  # write_client_conf <name> <octet> <server ip>
    {
        echo "[Interface]"
        echo "Address = $NET.$2/24"
        echo "PrivateKey = $(cat "$WG/$1.key")"
        echo
        echo "[Peer]"
        echo "PublicKey = $(cat "$WG/server.pub")"
        echo "Endpoint = $3:$PORT"
        echo "AllowedIPs = $NET.0/24"   # only federation traffic uses the tunnel
        echo "PersistentKeepalive = 25" # keeps home-router / carrier NAT open
    } > "$WG/$1.conf"
    chmod 600 "$WG/$1.conf"
}

cmd_create() {
    need az "brew install azure-cli; az login"
    [[ -s "$SSH_PUB" ]] || die "no public key at $SSH_PUB (set AZ_SSH_PUB)"
    az group create -n "$RG" -l "$LOC" -o none
    az vm create -g "$RG" -n "$VM" --image "$IMAGE" --size "$SIZE" \
        --admin-username "$ADMIN" --ssh-key-values "$SSH_PUB" \
        --public-ip-sku Standard -o none   # Standard SKU IP is static
    az vm open-port -g "$RG" -n "$VM" --port "$PORT" --priority 900 -o none
    # NSG rule above is TCP+UDP "*" protocol by default; 29500 stays closed
    az vm auto-shutdown -g "$RG" -n "$VM" --time "$SHUTDOWN_UTC" -o none
    mkdir -p "$WG"
    az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv > "$WG/server_ip"
    echo "VM $VM up at $(server_ip); auto-shutdown daily $SHUTDOWN_UTC UTC"
}

cmd_keys() {
    [[ $# -gt 0 ]] || die "usage: keys <host> [host...]"
    need wg "brew install wireguard-tools"
    mkdir -p "$WG"
    touch "$WG/hosts.txt"
    genkey server
    for h in "$@"; do
        [[ $h =~ ^[A-Za-z0-9_-]+$ ]] || die "bad host name: $h"
        if [[ -z $(host_octet "$h") ]]; then
            last=$(awk 'BEGIN{m=1} $2>m{m=$2} END{print m}' "$WG/hosts.txt")
            echo "$h $((last + 1))" >> "$WG/hosts.txt"
        fi
        genkey "$h"
    done
    write_server_conf
    local ip; ip=$(server_ip)
    while read -r h n; do write_client_conf "$h" "$n" "$ip"; done < "$WG/hosts.txt"
    echo "configs in $WG:"
    awk -v net="$NET" '{printf "  %s.%s -> %s\n", net, $2, $1}' "$WG/hosts.txt"
}

cmd_install() {
    local ip; ip=$(server_ip)
    [[ -s "$WG/server.conf" ]] || die "run keys first"
    rsync -az "$HERE"/*.py "$HERE/legacy_fl_hardware.npz" "$ADMIN@$ip:fed/"
    scp -q "$WG/server.conf" "$ADMIN@$ip:/tmp/wg0.conf"
    ssh "$ADMIN@$ip" "set -e
        sudo apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
            wireguard python3-venv chrony
        sudo install -m 600 /tmp/wg0.conf /etc/wireguard/wg0.conf && rm /tmp/wg0.conf
        sudo systemctl enable --now wg-quick@wg0
        python3 -m venv ~/fl
        ~/fl/bin/pip install -q numpy==$NUMPY
        ~/fl/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
        ~/fl/bin/python -c 'import numpy, torch; print(\"numpy\", numpy.__version__, \"torch\", torch.__version__)'
        sudo wg show wg0 | head -4"
}

cmd_push() {  # reload peers on the VM without dropping existing tunnels
    local ip; ip=$(server_ip)
    scp -q "$WG/server.conf" "$ADMIN@$ip:/tmp/wg0.conf"
    rsync -az "$HERE"/*.py "$ADMIN@$ip:fed/"
    ssh "$ADMIN@$ip" "set -e
        sudo install -m 600 /tmp/wg0.conf /etc/wireguard/wg0.conf && rm /tmp/wg0.conf
        sudo bash -c 'wg syncconf wg0 <(wg-quick strip wg0)'
        sudo wg show wg0 peers | wc -l | xargs echo peers:"
}

cmd_client() {  # client <host> <ssh target>
    [[ $# -eq 2 ]] || die "usage: client <host> <user@device>"
    [[ -s "$WG/$1.conf" ]] || die "no config for $1: run keys $1"
    scp -q "$WG/$1.conf" "$2:/tmp/wg0.conf"
    ssh -t "$2" "set -e
        command -v wg-quick >/dev/null || sudo apt-get install -y wireguard-tools
        # Jetson kernels ship without CONFIG_WIREGUARD: wg-quick then falls
        # back to the userspace wireguard-go, which must be installed
        if ! modinfo wireguard >/dev/null 2>&1 && [ ! -e /sys/module/wireguard ]; then
            command -v wireguard-go >/dev/null || sudo apt-get install -y wireguard-go
        fi
        sudo install -m 600 /tmp/wg0.conf /etc/wireguard/wg0.conf && rm /tmp/wg0.conf
        sudo systemctl enable --now wg-quick@wg0
        sudo systemctl restart wg-quick@wg0
        sleep 2; ping -c 3 $NET.1"
}

cmd_start() {
    az vm start -g "$RG" -n "$VM" -o none
    echo "started; ip $(server_ip) (static)"
}

cmd_stop() {
    az vm deallocate -g "$RG" -n "$VM" -o none
    echo "deallocated: compute billing stopped (disk still billed, ~cents/day)"
}

cmd_status() {
    az_t vm get-instance-view -g "$RG" -n "$VM" \
        --query "instanceView.statuses[?starts_with(code,'PowerState')].displayStatus" -o tsv
    # B-series: a run is only clean if credits never hit zero during it
    az_t monitor metrics list --resource "$(az_t vm show -g "$RG" -n "$VM" --query id -o tsv)" \
        --metric "CPU Credits Remaining" --interval PT5M --offset 1h \
        --query "value[0].timeseries[0].data[-1].{time:timeStamp,credits_remaining:average}" \
        -o tsv 2>/dev/null | sed 's/^/  cpu credits: /'
    ssh -o ConnectTimeout=5 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 "$ADMIN@$(server_ip)" \
        "sudo wg show wg0 latest-handshakes" 2>/dev/null \
        | while read -r pub ts; do
            h=$(grep -lF -- "$pub" "$WG"/*.pub | xargs -n1 basename | sed 's/.pub$//')
            if [[ $ts -eq 0 ]]; then echo "  $h: never"; else
                echo "  $h: last handshake $(( $(date +%s) - ts ))s ago"; fi
        done
}

case "${1:-}" in
    create) cmd_create ;;
    keys) shift; cmd_keys "$@" ;;
    install) cmd_install ;;
    push) cmd_push ;;
    client) shift; cmd_client "$@" ;;
    start) cmd_start ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    *) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
