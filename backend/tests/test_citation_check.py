from safety import enforce_citations

def test_enforce_citations_drops_unsupported():
    answer = "Alpha fact [1]. Beta claim [2]. Hallucinated line without cite."
    cita_ids = [1,2]
    clean, all_supported = enforce_citations(answer, valid_ids=cita_ids)
    assert "Hallucinated" not in clean
    assert all_supported is False

def test_enforce_citations_all_supported():
    answer = "Alpha fact [1]. Beta claim [2]."
    cita_ids = [1,2]
    clean, all_supported = enforce_citations(answer, valid_ids=cita_ids)
    assert clean == answer
    assert all_supported is True