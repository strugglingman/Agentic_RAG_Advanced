from safety import coverage_ok

def test_coverage_ok_thresholds():
    assert coverage_ok([0.6,0.5,0.4], topk=3, score_avg=0.28, score_min=0.38) is True
    assert coverage_ok([0.3,0.2], topk=3, score_avg=0.35, score_min=0.5) is False

def test_noanswer_when_low_confidence(client, auth_headers, monkeypatch):
    # Monkeypatch retrieve to simulate low scores
    import app as appmod
    def fake_retrieve(query, dept_id=None, user_id=None, top_k=5, where=None, use_reranker=False, use_hybrid=False):
        return [], None  # or return [{"score":0.05,"text":"noise"}]
    monkeypatch.setattr(appmod, "retrieve", fake_retrieve)
    payload = {
        "messages": [
            {"role": "user", "content": "Hello, this is a test message."}
        ]
    }
    rv = client.post("/chat", json=payload, headers=auth_headers)
    assert rv.status_code == 200
    assert "no_answer" in rv.get_data(as_text=True).lower() or \
    "couldn’t find" in rv.get_data(as_text=True).lower() or \
    "I don't" in rv.get_data(as_text=True)
