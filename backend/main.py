import uvicorn
import logging
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from app.services.result import ResultService
from app.models.result import ResultModel
from app.services.upload import UploadService
from app.ai.text_helpers import TextHelpers
from app.services.preprocess import PreprocessService
from app.services.process import ProcessService
from app.models.image import ImageModelType
from app.models.request import PredictReq

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = Path("db/results.db")
UPLOAD_ROOT_DIR = Path("uploads")

NOT_USE_AUDIO = {
    'Chợ nổi Cái Răng',
    'Đua bò Bảy Núi',
    'Nghề đan tre',
    'Nghề dệt chiếu',
}

result_service = ResultService(DB_PATH)
upload_service = UploadService(UPLOAD_ROOT_DIR)
preprocess_service = PreprocessService()
process_service = ProcessService()


@app.post("/api/upload")
async def upload(
    files: List[UploadFile] = File(...),
    text: str = Form(...)
):
    try:
        last_idx = result_service.find_last_id()
        upload_dir = f"{last_idx + 1}" if last_idx else "1"
        image_urls, extracted_text, audio_urls = upload_service.upload(files, upload_dir)
        full_text = (extracted_text or "") + (text or "")
        words = TextHelpers.get_words(full_text)
        
        result_model = ResultModel(
            image_urls=image_urls,
            audio_urls=audio_urls,
            text=full_text,
            processed_texts=words,
            img_label_idx=None,
            img_label_name=None,
            img_prob=None,
            img_txt_label_idx=None,
            img_txt_label_name=None,
            img_txt_prob=None,
            img_txt_weights=[],
            full_label_idx=None,
            full_label_name=None,
            full_prob=None,
            full_weights=[]
        )
        
        result_model = result_service.insert(result_model)        
        return JSONResponse({"id": result_model.id})
        
    except Exception as e:
        upload_service.clean_all(upload_dir)
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/predict/img")
async def predict_img(req: PredictReq):
    try:
        result_model = result_service.find(req.id)
        if not result_model:
            return JSONResponse({"error": f"Result ID={req.id} not found"}, status_code=404)
        

        images_subgraph = preprocess_service.preprocess_images(result_model.image_urls, req.model)
        graph = process_service.build_graph(images_subgraph, [], [])
        pred = process_service.predict(graph, predict_type="image", image_model_name=req.model)
        
        result_model.img_label_idx = pred["label_idx"]
        result_model.img_label_name = pred["label_name"]
        result_model.img_prob = pred["prob"]
        
        result_service.update(result_model)
        
        return JSONResponse(result_model.model_dump())
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/predict/img-txt")
async def predict_img_txt(req: PredictReq):
    try:
        result_model = result_service.find(req.id)
        if not result_model:
            return JSONResponse({"error": f"Result ID={req.id} not found"}, status_code=404)
        
        images_subgraph = preprocess_service.preprocess_images(result_model.image_urls, req.model)
        texts_subgraph = preprocess_service.preprocess_texts(result_model.processed_texts)
        
        graph = process_service.build_graph(images_subgraph, texts_subgraph, [])
        pred = process_service.predict(graph, predict_type="image_text", image_model_name=req.model)
        
        result_model.img_txt_label_idx = pred["label_idx"]
        result_model.img_txt_label_name = pred["label_name"]
        result_model.img_txt_prob = pred["prob"]
        result_model.img_txt_weights = pred["weights"]
        
        result_service.update(result_model)
        return JSONResponse(result_model.model_dump())
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/predict/full")
async def predict_full(req: PredictReq):
    try:
        result_model = result_service.find(req.id)
        if not result_model:
            return JSONResponse({"error": f"Result ID={req.id} not found"}, status_code=404)
        
        images_subgraph = preprocess_service.preprocess_images(result_model.image_urls, req.model)
        texts_subgraph = preprocess_service.preprocess_texts(result_model.processed_texts)
        audios_subgraph = preprocess_service.preprocess_audios(result_model.audio_urls)
        
        img_txt_graph = process_service.build_graph(images_subgraph, texts_subgraph, [])
        coarse_pred_1 = process_service.predict(img_txt_graph, predict_type="image_text", image_model_name=req.model)
        coarse_pred_2 = process_service.predict(img_txt_graph, predict_type="full", image_model_name=req.model)

        if coarse_pred_1["label_name"] in NOT_USE_AUDIO and coarse_pred_2["label_name"] in NOT_USE_AUDIO: 
            pred = coarse_pred_2 if coarse_pred_2["prob"] >= coarse_pred_1["prob"] else coarse_pred_1
        else:
            graph = process_service.build_graph(images_subgraph, texts_subgraph, audios_subgraph)
            pred = process_service.predict(graph, predict_type="full", image_model_name=req.model)
        
        result_model.full_label_idx = pred["label_idx"]
        result_model.full_label_name = pred["label_name"]
        result_model.full_prob = pred["prob"]
        result_model.full_weights = pred["weights"]

        result_service.update(result_model)
        
        return JSONResponse(result_model.model_dump())
        
    except Exception as e:
        logger.info(e)
        return JSONResponse({"error": str(e)}, status_code=400)



@app.get("/api/results/{id}")
async def get_result(id: int):
    record = result_service.find(id)
    if not record:
        return JSONResponse({"error": f"Không tìm thấy kết quả với ID={id}"}, status_code=404)
    return JSONResponse(record.model_dump())


app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

