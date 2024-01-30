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

### GET /api/supported-languages

**Descrição:**
Obtém a lista de idiomas suportados pelo Google Translate.

### POST /api/detect-language

**Descrição:**
Detecta o idioma de um texto fornecido.

**Parâmetros:**
- `text` (obrigatório): O texto para o qual você deseja detectar o idioma.

### POST /api/translate-text

**Descrição:**
Traduz um texto de um idioma de origem para um idioma de destino.

**Parâmetros:**
- `text` (obrigatório): O texto que você deseja traduzir.
- `src` (opcional, padrão: 'auto'): O idioma de origem do texto. Se não fornecido, será detectado automaticamente.
- `dest` (opcional, padrão: 'en'): O idioma de destino para a tradução.

### POST /api/translate-multiple

**Descrição:**
Traduz uma lista de textos para um idioma de destino.

**Parâmetros:**
- `texts` (obrigatório): Uma lista de textos que você deseja traduzir.
- `src` (opcional, padrão: 'auto'): O idioma de origem dos textos. Se não fornecido, será detectado automaticamente.
- `dest` (opcional, padrão: 'en'): O idioma de destino para a tradução.

### POST /api/extract-text-from-image

**Descrição:**
Extrai texto de uma imagem.

**Parâmetros:**
- `src` (obrigatório): O idioma esperado do texto na imagem.
- `image` (obrigatório): A imagem da qual você deseja extrair o texto. Pode ser fornecida como um arquivo de upload ou um link para a imagem.

**Retorno:**

- Dados estruturados do texto extraído, incluindo coordenadas e texto.


### POST /api/extract-texts-from-images

**Descrição:**
Extrai texto de uma lista de imagens.

**Parâmetros:**
- `src` (obrigatório): O idioma esperado do texto nas imagens.
- `images` (obrigatório): Uma lista de imagens das quais você deseja extrair o texto. Cada imagem pode ser fornecida como um arquivo de upload ou um link para a imagem.

**Retorno:**

- Lista de dados estruturados do texto extraído, incluindo coordenadas e texto para cada imagem.

### POST /api/translate-image-and-replace

**Descrição:**
Traduz um texto em uma imagem e substitui o texto original pela tradução.

**Parâmetros:**

- `src` (obrigatório): O idioma esperado do texto na imagem.
- `dest` (opcional, padrão: 'en'): O idioma de destino para a tradução.
- `image` (obrigatório): A imagem da qual você deseja traduzir e substituir o texto. Pode ser fornecida como um arquivo de upload ou um link para a imagem.

**Retorno:**

- Imagem com o texto traduzido substituindo o texto original.


### POST /api/extract-text-from-video

**Descrição:**
Extrai texto de um vídeo.

**Parâmetros:**
- `src` (obrigatório): O idioma esperado do texto no vídeo.
- `video` (obrigatório): O vídeo do qual você deseja extrair o texto. Deve ser fornecido como um arquivo de upload ou um link para o vídeo.

**Retorno:**

- Lista de dados estruturados do texto extraído, incluindo coordenadas e texto para cada frame do vídeo.
