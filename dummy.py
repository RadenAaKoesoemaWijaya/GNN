import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import json

# Inisialisasi Faker
fake = Faker()

# Jumlah data yang akan dibuat
num_records = 800

# Daftar endpoint API yang mungkin
endpoints = [
    '/api/users',
    '/api/products',
    '/api/orders',
    '/api/auth/login',
    '/api/auth/logout',
    '/api/auth/register',
    '/api/payments',
    '/api/shipping',
    '/api/categories',
    '/api/search'
]

# Daftar metode HTTP
http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']

# Daftar user agent yang mungkin
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36',
    'PostmanRuntime/7.28.0',
    'curl/7.64.1',
    'python-requests/2.25.1'
]

# Daftar status code HTTP yang mungkin dengan distribusi yang realistis
status_codes = [200] * 75 + [201] * 10 + [204] * 5 + [400] * 5 + [401] * 2 + [403] * 1 + [404] * 2 + [500] * 1

# Fungsi untuk menghasilkan parameter berdasarkan endpoint
def generate_params(endpoint, method):
    if method == 'GET':
        if endpoint == '/api/users':
            return json.dumps({'page': random.randint(1, 10), 'limit': random.choice([10, 20, 50, 100])})
        elif endpoint == '/api/products':
            return json.dumps({'category': random.choice(['electronics', 'clothing', 'books', 'home']), 'sort': random.choice(['price_asc', 'price_desc', 'newest'])})
        elif endpoint == '/api/search':
            return json.dumps({'q': fake.word(), 'filter': random.choice(['all', 'products', 'categories'])})
        else:
            return json.dumps({'id': random.randint(1, 1000)})
    elif method == 'POST':
        if endpoint == '/api/auth/login':
            return json.dumps({'username': fake.user_name(), 'password': 'password123'})
        elif endpoint == '/api/auth/register':
            return json.dumps({'username': fake.user_name(), 'email': fake.email(), 'password': 'password123'})
        elif endpoint == '/api/orders':
            return json.dumps({'product_id': random.randint(1, 100), 'quantity': random.randint(1, 5), 'payment_method': random.choice(['credit_card', 'paypal', 'bank_transfer'])})
        else:
            return json.dumps({'name': fake.name(), 'value': fake.word()})
    else:
        return json.dumps({'id': random.randint(1, 1000), 'update_field': fake.word()})

# Tanggal mulai untuk timestamp (30 hari yang lalu)
start_date = datetime.now() - timedelta(days=30)

# Membuat data
data = []
for _ in range(num_records):
    # Pilih endpoint dan metode secara acak
    endpoint = random.choice(endpoints)
    method = random.choice(http_methods)
    
    # Hasilkan timestamp acak dalam 30 hari terakhir
    timestamp = start_date + timedelta(seconds=random.randint(0, 30*24*60*60))
    
    # Hasilkan response time dengan distribusi yang realistis (lebih banyak respons cepat)
    response_time = max(10, int(np.random.exponential(100)))
    
    # Hasilkan status code berdasarkan distribusi yang telah ditentukan
    status_code = random.choice(status_codes)
    
    # Buat record
    record = {
        'ip_address': fake.ipv4(),
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'user_agent': random.choice(user_agents),
        'endpoint': endpoint,
        'method': method,
        'parameters': generate_params(endpoint, method),
        'status_code': status_code,
        'response_time_ms': response_time
    }
    
    data.append(record)

# Buat DataFrame
df = pd.DataFrame(data)

# Tambahkan beberapa anomali (sekitar 1% dari data)
num_anomalies = int(num_records * 0.01)
for _ in range(num_anomalies):
    idx = random.randint(0, num_records-1)
    # Anomali bisa berupa response time yang sangat tinggi
    if random.random() < 0.5:
        df.at[idx, 'response_time_ms'] = random.randint(5000, 20000)
    # Atau status code error
    else:
        df.at[idx, 'status_code'] = random.choice([500, 502, 503, 504])

# Simpan ke CSV
df.to_csv('api_log_data.csv', index=False)

print(f"Data dummy berhasil dibuat dengan {num_records} baris dan disimpan ke 'api_log_data.csv'")