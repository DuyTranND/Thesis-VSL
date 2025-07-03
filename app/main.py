import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uuid
import uvicorn
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import hàm predict đã được sửa lỗi từ predictor.py
from app.inference.predictor import predict

app = FastAPI(
    title="Sign Language Recognition API",
    description="API để nhận diện ngôn ngữ ký hiệu từ video.",
    version="1.0.0"
)

# Tạo thư mục static nếu chưa tồn tại
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- MÔ HÌNH DỮ LIỆU TRẢ VỀ (RESPONSE MODEL) ---
# 1. Định nghĩa cấu trúc cho một mục dự đoán
class PredictionItem(BaseModel):
    rank: int
    word: str
    confidence: float

# 2. Xóa bỏ class VideoUploadResponse cũ vì không còn phù hợp


# --- ENDPOINT API ---
# 3. Cập nhật response_model để trả về một danh sách các PredictionItem
@app.post("/predict", response_model=List[PredictionItem])
async def predict_video(file: UploadFile = File(...)):
    """
    Nhận một file video, lưu lại, chạy nhận diện và trả về một
    danh sách các dự đoán có khả năng cao nhất.
    """
    video_path = None  # Khởi tạo để dùng trong finally
    try:
        # Lưu file video được tải lên
        file_extension = os.path.splitext(file.filename)[1]
        video_filename = f"{uuid.uuid4()}{file_extension}"
        video_path = os.path.join("static", video_filename)

        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())

        # 4. Gọi hàm predict và nhận về một danh sách
        predictions = predict(video_path, top_k=5)  # Giờ hàm này trả về list

        # Xử lý trường hợp predict không thành công
        if predictions is None:
            raise HTTPException(
                status_code=400,
                detail="Không thể xử lý hoặc trích xuất đặc trưng từ video."
            )

        # 5. Trả về trực tiếp danh sách predictions.
        # FastAPI sẽ tự động chuyển nó thành JSON.
        return predictions

    except Exception as e:
        # Sử dụng HTTPException để xử lý lỗi một cách chuẩn mực trong FastAPI
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Dọn dẹp file video đã lưu sau khi xử lý xong (tùy chọn)
        if video_path and os.path.exists(video_path):
            os.remove(video_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)