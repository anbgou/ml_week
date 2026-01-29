import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn


app = FastAPI()

class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:;"
        return response

app.add_middleware(CSPMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

CLASSES = ["healthy", "sick"]
DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def load_model():
    print("⏳ Завантажую ResNet18...")
    try:
        model = models.resnet18(weights=None)

        # Налаштовуємо під 1 вихід (як у тебе було)
        model.fc = nn.Linear(model.fc.in_features, 1)

        # 1. Завантажуємо файл повністю (це словник)
        checkpoint = torch.load("checkpoints/best_model_v2.pt", map_location=DEVICE)

        # 2. Дістаємо з нього саме ваги (вони лежать під ключем 'state_dict')
        # Якщо в файлі немає ключа 'state_dict', то беремо весь файл (на випадок, якщо я помилився)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # 3. Завантажуємо ці ваги в модель
        model.load_state_dict(state_dict)

        model.to(DEVICE)
        model.eval()

        print("✅ Модель успішно завантажена!")
        return model
    except Exception as e:
        print(f"❌ ПОМИЛКА: {e}")
        return None


model = load_model()



@app.get("/")
async def read_index():
    return FileResponse("index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не завантажена.")

    # Перевірка типу файлу
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Тільки файли JPG або PNG.")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

            sick_probability = torch.sigmoid(output).item()

            if sick_probability > 0.5:
                predicted_class = "sick"
                confidence = sick_probability
            else:
                predicted_class = "healthy"
                confidence = 1 - sick_probability

        return {
            "prediction": predicted_class,
            "confidence": f"{confidence * 100:.1f}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)