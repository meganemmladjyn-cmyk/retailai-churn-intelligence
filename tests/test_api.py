import pytest


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_list_customers_empty(client):
    response = await client.get("/api/v1/customers")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["page"] == 1


@pytest.mark.asyncio
async def test_create_customer(client):
    payload = {
        "customer_id": "RET-TEST-001",
        "age": 32,
        "gender": "F",
        "city": "Paris",
        "country": "France",
        "registration_date": "2023-01-15",
        "last_purchase_date": "2024-03-01",
        "total_orders": 12,
        "total_spent": 1250.50,
        "avg_order_value": 104.21,
        "days_since_last_purchase": 45,
        "purchase_frequency": 0.8,
        "return_rate": 0.05,
        "email_open_rate": 0.42,
        "support_tickets_count": 1,
        "preferred_category": "Fashion",
        "is_churned": False,
    }
    response = await client.post("/api/v1/customers", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["customer_id"] == "RET-TEST-001"
    assert data["churn_score"] is None


@pytest.mark.asyncio
async def test_get_customer(client):
    response = await client.get("/api/v1/customers/RET-TEST-001")
    assert response.status_code == 200
    assert response.json()["customer_id"] == "RET-TEST-001"


@pytest.mark.asyncio
async def test_get_nonexistent_customer(client):
    response = await client.get("/api/v1/customers/NONEXISTENT-000")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_customer(client):
    response = await client.patch(
        "/api/v1/customers/RET-TEST-001",
        json={"days_since_last_purchase": 10, "is_churned": False},
    )
    assert response.status_code == 200
    assert response.json()["days_since_last_purchase"] == 10
