// script.js

const fileInput = document.getElementById('fileInput');
const fileNameLabel = document.getElementById('fileName');
const previewImg = document.getElementById('preview');
const resultDiv = document.getElementById('result');
const loader = document.getElementById('loader');
const btnAnalyze = document.getElementById('btnAnalyze');
const resTitle = document.getElementById('resTitle');
const resConf = document.getElementById('resConf');

// Показуємо прев'ю картинки
fileInput.addEventListener('change', function() {
    if (this.files && this.files[0]) {
        fileNameLabel.textContent = this.files[0].name;

        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            previewImg.style.display = "block";
            // Ховаємо попередній результат при виборі нового файлу
            resultDiv.style.display = "none";
        }
        reader.readAsDataURL(this.files[0]);
    }
});

async function analyzeImage() {
    const file = fileInput.files[0];
    if (!file) {
        alert("Будь ласка, спочатку виберіть файл!");
        return;
    }

    // UI: Показуємо завантаження
    loader.style.display = 'block';
    btnAnalyze.disabled = true;
    resultDiv.style.display = 'none';

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Помилка з'єднання з сервером");
        }

        const data = await response.json();

        // UI: Показуємо результат
        if (data.prediction === "healthy") {
            resultDiv.className = "healthy";
            resTitle.textContent = "🌱 Здорова рослина";
        } else {
            resultDiv.className = "sick";
            resTitle.textContent = "⚠️ Рослина хвора";
        }

        resultDiv.style.display = "block";

    } finally {
        loader.style.display = 'none';
        btnAnalyze.disabled = false;
    }
}