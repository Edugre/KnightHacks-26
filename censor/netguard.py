"""Egress guard: with CENSOR_OFFLINE=1, refuse and log every outbound network connection.

This is how we show success condition 4 (IMPLEMENTATION.md section 6): inference never leaves the machine.
Loopback (the app <-> model server hop) and Unix sockets still work. Hostname lookups other than localhost
are refused too, since a DNS query already leaks what we were about to contact. Every refused attempt goes
to stderr and to logs/egress.log, so an empty log after the demo is the proof.

Covers Python-level sockets (urllib, httpx, requests, asyncio, huggingface_hub). Native code that opens
its own sockets bypasses it, which is why run.sh also puts the libraries themselves in offline mode.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import sys
import time
from pathlib import Path


class EgressBlocked(ConnectionRefusedError):
    pass


_real: dict = {}


def _log(what: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} blocked {what}"
    print(f"[netguard] {line}", file=sys.stderr)
    path = Path(os.environ.get("CENSOR_EGRESS_LOG", "logs/egress.log"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass  # logging must never be the thing that breaks inference


def _is_local_host(host) -> bool:
    if host in (None, "", "localhost"):
        return True
    try:
        ip = ipaddress.ip_address(str(host).split("%")[0])
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified  # unspecified = binding a server on 0.0.0.0 / ::


def _check_address(address) -> None:
    if isinstance(address, tuple) and not _is_local_host(address[0]):
        _log(f"connect to {address[0]}:{address[1]}")
        raise EgressBlocked(f"outbound connection to {address[0]} blocked (CENSOR_OFFLINE=1)")


def _connect(self, address):
    _check_address(address)
    return _real["connect"](self, address)


def _connect_ex(self, address):
    _check_address(address)
    return _real["connect_ex"](self, address)


def _getaddrinfo(host, *args, **kwargs):
    if isinstance(host, bytes):
        host = host.decode()
    if not _is_local_host(host):
        try:
            ipaddress.ip_address(str(host).split("%")[0])  # an IP literal resolves without the network
        except ValueError:
            _log(f"DNS lookup of {host}")
            raise socket.gaierror(socket.EAI_NONAME, f"lookup of {host} blocked (CENSOR_OFFLINE=1)")
    return _real["getaddrinfo"](host, *args, **kwargs)


def install() -> None:
    if _real:
        return
    _real.update(connect=socket.socket.connect, connect_ex=socket.socket.connect_ex, getaddrinfo=socket.getaddrinfo)
    socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = _connect, _connect_ex, _getaddrinfo


def uninstall() -> None:
    if _real:
        socket.socket.connect, socket.socket.connect_ex = _real["connect"], _real["connect_ex"]
        socket.getaddrinfo = _real["getaddrinfo"]
        _real.clear()


def active() -> bool:
    return bool(_real)


def install_from_env() -> bool:
    if os.environ.get("CENSOR_OFFLINE") == "1":
        install()
        return True
    return False
