import json

from app.agent.kb import FAQKnowledgeBase


def test_search_faq(tmp_path):
    faq_file = tmp_path / "faq.json"
    faq_file.write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "question": "مواعيد العمل",
                    "answer": "من التاسعة إلى الخامسة",
                    "keywords": ["مواعيد", "العمل"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    kb = FAQKnowledgeBase(faq_file)
    results = kb.search("مواعيد العمل")

    assert len(results) == 1
    assert results[0]["id"] == "1"


def test_empty_query():
    assert FAQKnowledgeBase().search("") == []


def test_default_faq_matches_egyptian_dialect():
    results = FAQKnowledgeBase().search("عايز أكلم موظف")
    assert results
    assert results[0]["id"] == "faq_004"