from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app import api
from app.graph import workflow


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv('SUPPORT_TRACE_FILE', str(tmp_path / 'api-traces.jsonl'))
    monkeypatch.setenv('SUPPORT_APPROVAL_KEY', 'reviewer-test-key')

    def specialist(state):
        updates = {'specialist': state['intent'], 'reply': 'Verified support response.'}
        if 'refund' in state['customer_message'].lower():
            updates.update(proposed_action={
                'action': 'refund', 'proposal_id': 'test-proposal', 'order_id': '12345',
                'amount_cents': 500, 'currency': 'INR',
            })
        return updates

    def passing_qa(state):
        return {'qa_result': {'passed': True, 'attempt': (state.get('qa_result') or {}).get('attempt', 0) + 1}}

    def execute(state):
        assert state['approval_decision']['approved'] is True
        return {'action_result': {'status': 'issued'}, 'reply': 'Approved demo refund recorded.'}

    for name in ('faq_agent', 'billing_agent', 'technical_agent'):
        monkeypatch.setattr(workflow, name, specialist)
    monkeypatch.setattr(workflow, 'qa_reviewer_agent', passing_qa)
    monkeypatch.setattr(workflow, 'execute_approved_action', execute)
    graph = workflow.build_graph(InMemorySaver())

    @contextmanager
    def open_graph():
        yield graph

    monkeypatch.setattr(api, 'open_support_graph', open_graph)
    with TestClient(api.app) as test_client:
        yield test_client


def create(client, message='How can I track my parcel?'):
    response = client.post('/api/conversations', json={'message': message})
    assert response.status_code == 201, response.text
    return response.json()


def test_create_follow_up_and_retrieve_history(client):
    view = create(client)
    path = '/api/conversations/' + view['thread_id']
    assert view['status'] == 'ready'
    assert [entry['role'] for entry in view['history']] == ['user', 'assistant']
    assert view['qa_passed'] is True
    assert [step['node'] for step in view['routing_path']] == ['intake', 'faq', 'qa', 'finalize_turn']
    assert view['routing_path'][2]['status'] == 'passed'
    assert 'technical_context' not in view and 'entities' not in view
    response = client.post(path + '/messages', json={'message': 'Can you explain that?'})
    assert response.status_code == 200
    assert len(response.json()['history']) == 4
    assert response.json()['routing_path'] == view['routing_path']  # Latest turn only.
    assert client.get(path).json() == response.json()


@pytest.mark.parametrize('message', ['', '   ', 'x' * 8001])
def test_invalid_messages_rejected(client, message):
    assert client.post('/api/conversations', json={'message': message}).status_code == 422


def test_missing_and_invalid_conversation_ids(client):
    assert client.get('/api/conversations/' + '0' * 32).status_code == 404
    assert client.get('/api/conversations/not-a-uuid').status_code == 422


@pytest.mark.parametrize('approved', [False, True])
def test_approval_requires_key_and_pending_action(client, approved):
    view = create(client, 'Please refund the duplicate charge for order 12345')
    path = '/api/conversations/' + view['thread_id']
    assert view['status'] == 'awaiting_approval'
    assert view['pending_approval']['amount_cents'] == 500
    assert view['pending_approval']['draft_reply'] == 'Verified support response.'
    assert view['routing_path'][-1] == {'node': 'approval_gate', 'status': 'awaiting_approval'}
    assert len(view['history']) == 1  # Pending draft has not been sent as a final reply.
    assert client.post(path + '/messages', json={'message': 'Hello'}).status_code == 409
    assert client.post(path + '/retry').status_code == 409
    assert client.post(path + '/approval', json={'approved': approved}).status_code == 401
    assert client.post(path + '/approval', json={'approved': approved}, headers={'Authorization': 'Bearer wrong'}).status_code == 401
    headers = {'Authorization': 'Bearer reviewer-test-key'}
    response = client.post(path + '/approval', json={'approved': approved, 'note': 'Reviewed'}, headers=headers)
    assert response.status_code == 200
    assert response.json()['status'] == 'ready'
    assert response.json()['pending_approval'] is None
    route = response.json()['routing_path']
    assert route[-3] == {'node': 'approval_gate', 'status': 'approved' if approved else 'rejected'}
    assert route[-2]['node'] == ('execute_action' if approved else 'escalate')
    assert route[-1]['node'] == 'finalize_turn'
    assert ('Approved demo refund recorded.' if approved else 'not approved') in response.json()['history'][-1]['content']
    assert client.post(path + '/approval', json={'approved': approved}, headers=headers).status_code == 409


def test_unconfigured_support_access_is_disabled(client, monkeypatch):
    monkeypatch.delenv('SUPPORT_APPROVAL_KEY')
    response = client.post('/api/conversations/' + '0' * 32 + '/approval', json={'approved': True})
    assert response.status_code == 503


def test_rate_limit_error_can_resume_without_duplicate_user_turn(client, monkeypatch):
    attempts = []

    class RateLimitError(Exception):
        status_code = 429

    def flaky_qa(state):
        attempts.append(1)
        if len(attempts) == 1:
            raise RateLimitError('secret-provider-key must not appear')
        return {'qa_result': {'passed': True, 'attempt': 1}}

    monkeypatch.setattr(workflow, 'qa_reviewer_agent', flaky_qa)
    graph = workflow.build_graph(InMemorySaver())

    @contextmanager
    def open_graph():
        yield graph

    monkeypatch.setattr(api, 'open_support_graph', open_graph)
    failed = client.post('/api/conversations', json={'message': 'Where is my parcel?'})
    assert failed.status_code == 429
    assert failed.headers['Retry-After'] == '60'
    assert 'secret-provider-key' not in failed.text
    path = '/api/conversations/' + failed.json()['detail']['thread_id']
    assert client.get(path).json()['status'] == 'interrupted'
    assert client.get(path).json()['routing_path'][-1] == {'node': 'qa', 'status': 'interrupted'}
    assert client.post(path + '/messages', json={'message': 'Again'}).status_code == 409
    resumed = client.post(path + '/retry')
    assert resumed.status_code == 200
    assert resumed.json()['status'] == 'ready'
    assert [step['node'] for step in resumed.json()['routing_path']] == ['intake', 'faq', 'qa', 'finalize_turn']
    assert [entry['role'] for entry in resumed.json()['history']] == ['user', 'assistant']
    assert client.post(path + '/retry').status_code == 409


def test_simultaneous_mutation_is_rejected(client):
    view = create(client)
    with api.conversation_lock(view['thread_id']):
        response = client.post('/api/conversations/' + view['thread_id'] + '/messages', json={'message': 'Another message'})
    assert response.status_code == 409


def test_local_ui_and_origin_checks(client):
    assert client.get('/').status_code == 200
    assert 'Support Desk' in client.get('/').text
    assert client.get('/assets/app.js').status_code == 200
    assert client.get('/assets/styles.css').status_code == 200
    assert client.get('/api/health').json() == {'status': 'ok'}
    assert client.get('/openapi.json').status_code == 200
    assert client.post('/api/conversations', json={'message': 'Hello'}, headers={'Origin': 'https://other.example'}).status_code == 403
    assert client.get('/', headers={'Host': 'other.example'}).status_code == 400


@pytest.mark.parametrize('failures', [1, 3])
def test_routing_records_qa_loops_and_fallback_without_private_state(client, monkeypatch, failures):
    def qa(state):
        attempt = (state.get('qa_result') or {}).get('attempt', 0) + 1
        return {'qa_result': {'passed': attempt > failures, 'attempt': attempt}}

    monkeypatch.setattr(workflow, 'qa_reviewer_agent', qa)
    graph = workflow.build_graph(InMemorySaver())

    @contextmanager
    def open_graph():
        yield graph

    monkeypatch.setattr(api, 'open_support_graph', open_graph)
    result = create(client, 'Where is my parcel for order 98765?')
    route = result['routing_path']
    reviews = [step['status'] for step in route if step['node'] == 'qa']
    assert reviews == (['failed', 'passed'] if failures == 1 else ['failed'] * 3)
    assert sum(step['node'] == 'faq' for step in route) == len(reviews)
    assert ('qa_fallback' in [step['node'] for step in route]) == (failures == 3)
    assert route[-1]['node'] == 'finalize_turn'
    assert all(set(step) == {'node', 'status'} for step in route)
    assert '98765' not in str(route)
