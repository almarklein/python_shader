import os
import io
import tempfile

from spirv import bytes2spirv, file2spirv


minimal = [
    b"\x03\x02#\x07\x00\x05\x01\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00"
    b"\x00\x11\x00\x02\x00\x00\x00\x00\x00\x11\x00\x02\x00\x01\x00\x00\x00\x11"
    b"\x00\x02\x00\n\x00\x00\x00\x11\x00\x02\x00\r\x00\x00\x00\x0e\x00\x03\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x05\x00\x00\x00\x00\x00\x01\x00"
    b"\x00\x00main\x00\x00\x00\x00\x13\x00\x02\x00\x02\x00\x00\x00!\x00\x03\x00"
    b"\x03\x00\x00\x00\x02\x00\x00\x006\x00\x05\x00\x02\x00\x00\x00\x01\x00\x00"
    b"\x00\x00\x00\x00\x00\x03\x00\x00\x00\xf8\x00\x02\x00\x04\x00\x00\x00\xfd"
    b"\x00\x01\x008\x00\x01\x00"
]
minimal = b"".join(minimal)


def test_bytes2spirv():
    m = bytes2spirv(minimal)
    assert m.input == minimal
    assert "from bytes" in repr(m)
    assert m.to_bytes() == minimal

    m.validate()
    x = m.disassble()
    assert "SPIR-V" in x
    assert "Version" in x
    assert "OpTypeVoid" in x


def test_file2spirv1():
    f = io.BytesIO(minimal)
    m = file2spirv(f)
    assert m.input == minimal  # not f
    assert "from file object" in repr(m)
    assert m.to_bytes() == minimal

    m.validate()
    x = m.disassble()
    assert "SPIR-V" in x
    assert "Version" in x
    assert "OpTypeVoid" in x


def test_file2spirv2():
    fname = "test.spv"
    filename = os.path.join(tempfile.gettempdir(), fname)
    with open(filename, "wb") as f:
        f.write(minimal)

    m = file2spirv(filename)
    assert m.input == minimal  # not f
    assert f"from {fname}" in repr(m)
    assert m.to_bytes() == minimal

    m.validate()
    x = m.disassble()
    assert "SPIR-V" in x
    assert "Version" in x
    assert "OpTypeVoid" in x
