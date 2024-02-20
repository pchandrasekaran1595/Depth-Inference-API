import pytest

from main import app
from sanic_testing.testing import SanicTestClient

test_client = SanicTestClient(app)


def test_get_depth():
    _, response = test_client.get("/depth")
    assert response.json == {
        "statusText": "Depth Inference Endpoint",
    }
    assert response.status_code == 200