#!/usr/bin/env python3
import socket
import struct
import sys
import threading

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} /path/to/vsock.sock <guest_cid> <port>", file=sys.stderr)
    sys.exit(1)

sock_path = sys.argv[1]
guest_cid = int(sys.argv[2])
port = int(sys.argv[3])

# Connect to the Firecracker vsock backend UDS
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock_path)

# Send 8-byte header: little-endian u32 CID, little-endian u32 port
s.sendall(struct.pack("<II", guest_cid, port))

# Now just proxy stdin ↔ socket ↔ stdout

def reader():
    try:
        while True:
            data = s.recv(4096)
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
    except KeyboardInterrupt:
        pass

t = threading.Thread(target=reader, daemon=True)
t.start()

try:
    for chunk in sys.stdin.buffer:
        s.sendall(chunk)
except KeyboardInterrupt:
    pass
finally:
    s.close()
