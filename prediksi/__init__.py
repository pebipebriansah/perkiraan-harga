import logging
import azure.functions as func
from .run_model import load_data_from_blob, train_model, predict, predict_next_month

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('⚡ Azure Function triggered.')

    try:
        bulan = int(req.params.get('bulan', 5))  # default: Mei
        curah = float(req.params.get('curah', 120.0))  # default: 120 mm

        df = load_data_from_blob()
        model = train_model(df)

        harga_now = predict(model, bulan, curah)
        bulan_next, harga_next = predict_next_month(model, bulan, curah)

        result = {
            "bulan_ini": bulan,
            "harga_sekarang": round(harga_now, 2),
            "bulan_depan": bulan_next,
            "harga_bulan_depan": round(harga_next, 2)
        }

        return func.HttpResponse(str(result), mimetype="application/json")
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return func.HttpResponse(f"Terjadi kesalahan: {e}", status_code=500)
