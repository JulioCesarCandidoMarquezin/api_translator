# API de Tradução e Extração de Texto de Imagens/Vídeos

Esta API permite traduzir texto e extrair texto de imagens e vídeos usando o Tesseract OCR e o Google Translate.

## Dependências

As seguintes bibliotecas são necessárias para executar a API:

- [FastAPI](https://fastapi.tiangolo.com/): Estrutura rápida para desenvolvimento de APIs com Python.
- [cv2](https://pypi.org/project/opencv-python/): OpenCV para processamento de imagens.
- [moviepy](https://zulko.github.io/moviepy/): Manipulação de vídeos usando a biblioteca MoviePy.
- [pytesseract](https://pypi.org/project/pytesseract/): Interface Python para o Tesseract OCR.
- [googletrans](https://pypi.org/project/googletrans/): Interface Python para o Google Translate.
- [numpy](https://numpy.org/): Biblioteca para manipulação de arrays em Python.

## Instalação

Para instalar as dependências, você pode usar o seguinte comando:

```bash
pip install fastapi opencv-python moviepy pytesseract googletrans numpy
```

## Endpoints
GET /api/supported-languages
Obtém a lista de idiomas suportados pelo Google Translate.

POST /api/detect-language
Detecta o idioma de um texto fornecido.

POST /api/translate-text
Traduz um texto de um idioma de origem para um idioma de destino.

POST /api/translate-multiple
Traduz uma lista de textos para um idioma de destino.

POST /api/extract-text-from-image
Extrai texto de uma imagem.

POST /api/extract-texts-from-images
Extrai texto de uma lista de imagens.

POST /api/extract-text-from-video
Extrai texto de um vídeo.
