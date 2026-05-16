# BotFinanceku AI

Layanan AI kecil untuk BotFinanceku, berfungsi memprediksi kategori pengeluaran berdasarkan subject transaksi. Service ini kini diselaraskan dengan bot WhatsApp utama (asisten-farros) dan dashboard web (web-botfinanceku).

## Kategori Tersedia
- makan
- jajan
- kebutuhan_kos
- tagihan
- laundry
- transportasi
- kesehatan
- hiburan
- sosial
- belanja_pribadi
- edukasi
- lain_lain

*Catatan: Kategori lama ("makan & minum", "operasional", dll) akan otomatis dinormalisasi menjadi kategori baru.*

## Cara Menjalankan (Local)

1. Buat dan aktifkan virtual environment (opsional namun disarankan):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variable untuk Supabase (khusus jika ingin tes `/confirm`):
   ```bash
   set SUPABASE_URL=...
   set SUPABASE_SERVICE_ROLE_KEY=...
   ```
4. Jalankan server:
   ```bash
   uvicorn app.main:app --reload
   ```

## Endpoints

### `GET /health`
Mengecek status service dan mengambil daftar kategori yang tersedia.
```bash
curl http://localhost:8000/health
```

### `POST /predict`
Memprediksi kategori berdasarkan nama subject pengeluaran.
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d "{\"subject\":\"beli kopi\",\"amount\":15000}"
```

### `POST /feedback`
Menambahkan feedback manual jika AI salah memprediksi. Model akan otomatis dilatih ulang.
```bash
curl -X POST http://localhost:8000/feedback \
     -H "Content-Type: application/json" \
     -d "{\"subject\":\"servis motor\",\"amount\":50000,\"correct_category\":\"transportasi\"}"
```

### `POST /confirm`
Mengkonfirmasi prediksi AI, mengupdate data di Supabase, dan melatih ulang model.
```bash
curl -X POST http://localhost:8000/confirm \
     -H "Content-Type: application/json" \
     -d "{\"expense_id\":\"uuid-expense-123\",\"subject\":\"nonton bioskop\",\"amount\":50000,\"correct_category\":\"hiburan\"}"
```

### `POST /train`
Melatih ulang model secara manual berdasarkan data training terbaru.
```bash
curl -X POST http://localhost:8000/train
```
