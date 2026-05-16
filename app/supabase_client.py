import os
import requests


def get_supabase_config():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise RuntimeError("SUPABASE_URL atau SUPABASE_SERVICE_ROLE_KEY belum tersedia di environment.")

    return url.rstrip("/"), key


def update_expense_confirmation(expense_id: str, category: str):
    supabase_url, service_key = get_supabase_config()

    endpoint = f"{supabase_url}/rest/v1/expenses"

    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    params = {
        "id": f"eq.{expense_id}",
        "select": "id,subject,amount,category,confirmed_category,is_confirmed",
    }

    payload = {
        "category": category,
        "confirmed_category": category,
        "is_confirmed": True,
    }

    response = requests.patch(
        endpoint,
        headers=headers,
        params=params,
        json=payload,
        timeout=10,
    )

    if not response.ok:
        raise RuntimeError(f"Supabase update gagal: {response.status_code} {response.text}")

    data = response.json()

    if not data:
        raise RuntimeError("Supabase update tidak mengubah row apa pun. Cek expense_id.")

    return data[0]
