def test_chat_rate_limit(client, auth_headers):
    payload = {
        "messages": [
            {"role": "user", "content": "Hello, this is a test message."}
        ]
    }
    status_codes = []
    for _ in range(40):
        res = client.post("/chat", headers=auth_headers, json=payload)
        status_codes.append(res.status_code)

    assert any(code == 429 for code in status_codes), "Expected at least one 429 Too Many Requests response"