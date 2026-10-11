import json

import pytest

from app.agents import technical_agent as technical
from app.agents.intake_agent import intake_agent
from app.tools.technical import search_technical_logs


@pytest.mark.parametrize('query', ['router internet', '', 'login checkout error'])
def test_log_lookup_never_returns_other_devices(query):
    result = json.loads(search_technical_logs.invoke({'search_term': query, 'device_id': 'router-01'}))
    assert all(record['device_id'] == 'router-01' for record in result['matches'])
    if query in ('', 'router internet'):
        assert {record['code'] for record in result['matches']} == {'WAN_DHCP_TIMEOUT', 'WAN_RECONNECTED'}


@pytest.mark.parametrize('device_id', ['', 'router-unknown'])
def test_missing_or_unknown_device_returns_no_records(device_id):
    result = json.loads(search_technical_logs.invoke({'search_term': 'router', 'device_id': device_id}))
    assert result['matches'] == []


def test_missing_device_asks_for_identifier_without_model_or_log_lookup(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Technical model or lookup called without device ID')

    monkeypatch.setattr(technical, 'run_tool_agent', forbidden)
    monkeypatch.setattr(technical, 'lookup_logs', forbidden)
    result = technical.technical_agent(intake_agent({'customer_message': 'My router internet is down'}))
    assert 'provide your device ID' in result['findings']
    assert result['current_tool_calls'] == []
    assert result['technical_context'] == {'device_id': None, 'lookups': []}


def test_lookup_device_is_bound_by_application_and_evidence_is_returned(monkeypatch):
    def fake_agent(state, *, specialist, tools):
        assert specialist == 'technical'
        # Even a model-supplied override cannot change the application-bound ID.
        response = json.loads(tools[0].invoke({'search_term': '', 'device_id': 'web-client'}))
        assert response['device_id'] == 'router-01'
        assert all(record['device_id'] == 'router-01' for record in response['matches'])
        return {'specialist': 'technical', 'findings': 'Historical records for router-01.'}

    monkeypatch.setattr(technical, 'run_tool_agent', fake_agent)
    result = technical.technical_agent(intake_agent({'customer_message': 'My router-01 internet is down'}))
    evidence = result['technical_context']
    assert evidence['device_id'] == 'router-01'
    assert len(evidence['lookups'][0]['matches']) == 2


def test_device_id_follow_up_and_device_switch():
    initial = intake_agent({'customer_message': 'My router internet is down'})
    identified = intake_agent({**initial, 'customer_message': 'My device ID is router-01'})
    assert identified['intent'] == 'technical'
    assert identified['entities']['device_id'] == 'router-01'
    follow_up = intake_agent({**identified, 'customer_message': 'It is still down'})
    assert follow_up['entities']['device_id'] == 'router-01'
    changed = intake_agent({**follow_up, 'customer_message': 'Try device ID router-99'})
    assert changed['entities']['device_id'] == 'router-99'
    new_topic = intake_agent({**changed, 'customer_message': 'What are your delivery address change policies for shipping?'})
    assert 'device_id' not in new_topic['entities']
    assert new_topic['technical_context'] == {}


def test_ambiguous_device_follow_up_does_not_reuse_previous_id(monkeypatch):
    previous = intake_agent({'customer_message': 'My router-01 internet is down'})
    ambiguous = intake_agent({**previous, 'customer_message': 'router-02 or router-03'})
    assert 'device_id' not in ambiguous['entities']
    assert 'provide your device ID' in technical.technical_agent(ambiguous)['findings']


def test_qa_retry_preserves_lookup_evidence_only_for_same_device(monkeypatch):
    monkeypatch.setattr(technical, 'run_tool_agent', lambda *args, **kwargs: {'findings': 'Revised reply'})
    previous = {'device_id': 'router-01', 'lookups': [{'device_id': 'router-01', 'matches': []}]}
    state = {'customer_message': 'My router is down', 'entities': {'device_id': 'router-01'}, 'technical_context': previous}
    assert technical.technical_agent(state)['technical_context'] == previous
    state['entities'] = {'device_id': 'router-99'}
    assert technical.technical_agent(state)['technical_context'] == {'device_id': 'router-99', 'lookups': []}
