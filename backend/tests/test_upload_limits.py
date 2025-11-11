import os
import io

MAX_UPLOAD_SIZE_IN_BYTE = int(os.getenv("MAX_UPLOAD_MB")) * 1024 * 1024

def test_upload_file_size_limit(client, auth_headers):
    data = {"file": (io.BytesIO(b"0" * (MAX_UPLOAD_SIZE_IN_BYTE + 10)), "big.pdf")}
    res = client.post("/upload", headers=auth_headers, data=data, content_type="multipart/form-data")
    assert res.status_code == 413, f"Expected 413 Payload Too Large, got {res.status_code}"

def test_upload_file_within_limit(client, auth_headers):
    data = {"file": (io.BytesIO(b"0" * (MAX_UPLOAD_SIZE_IN_BYTE - 1000)), "small.pdf")}
    res = client.post("/upload", headers=auth_headers, data=data, content_type="multipart/form-data")
    assert res.status_code == 200, f"Expected 200 OK, got {res.status_code}"