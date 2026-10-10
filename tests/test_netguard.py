"""Egress guard (censor/netguard.py): outbound connections and lookups are refused and logged; loopback still works."""
import socket
import urllib.error
import urllib.request

import pytest

from censor import netguard


@pytest.fixture()
def guard(tmp_path, monkeypatch):
    log = tmp_path / "egress.log"
    monkeypatch.setenv("CENSOR_EGRESS_LOG", str(log))
    netguard.install()
    yield log
    netguard.uninstall()


def test_outbound_connect_is_refused_and_logged(guard):
    s = socket.socket()
    with pytest.raises(netguard.EgressBlocked):
        s.connect(("93.184.216.34", 443))  # refused before any packet leaves
    s.close()
    assert "connect to 93.184.216.34:443" in guard.read_text()


def test_hostname_lookups_are_refused(guard):
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("huggingface.co", 443)
    with pytest.raises(urllib.error.URLError):  # what a model hub client would hit
        urllib.request.urlopen("https://huggingface.co/api/models", timeout=2)
    assert guard.read_text().count("DNS lookup of huggingface.co") == 2


def test_loopback_still_works(guard):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = socket.create_connection(("localhost", srv.getsockname()[1]), timeout=2)  # like the app -> model server hop
    conn, _ = srv.accept()
    c.sendall(b"ok")
    assert conn.recv(2) == b"ok"
    for s in (c, conn, srv):
        s.close()
    assert not guard.exists()  # nothing was blocked


def test_off_unless_asked(monkeypatch):
    monkeypatch.delenv("CENSOR_OFFLINE", raising=False)
    assert not netguard.install_from_env() and not netguard.active()
