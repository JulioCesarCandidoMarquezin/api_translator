# Translation and Text Extraction API for Images/Videos

This API enables text translation and extraction from images and videos using Tesseract OCR and Google Translate. 

## Dependencies

The following libraries are required to run the API:

- [FastAPI](https://fastapi.tiangolo.com/): Fast framework for building APIs with Python.
- [cv2](https://pypi.org/project/opencv-python/): OpenCV for image processing.
- [moviepy](https://zulko.github.io/moviepy/): Video manipulation using the MoviePy library.
- [pytesseract](https://pypi.org/project/pytesseract/): Python interface for Tesseract OCR.
- [googletrans](https://pypi.org/project/googletrans/): Python interface for Google Translate.
- [numpy](https://numpy.org/): Library for array manipulation in Python.
- [imageio](https://imageio.readthedocs.io/): Library for reading and writing images and videos in various formats.

## Installation

To install the dependencies, you can use the following command:

```bash
pip install fastapi opencv-python moviepy pytesseract googletrans numpy imageio python-magic-bin
```

## Initialization

To initialize the API, you can use the following command:

```bash
unicorn --host 0.0.0.0 --port 80
```

### Extracted Text Structure

Some API endpoints return a list of structured data representing the extracted text from the provided images. The structure includes paragraphs, lines, and words with their respective coordinates and text content.

#### Example Response:

```json
{
    "paragraphs": [
        {
            "x1": 10,
            "y1": 20,
            "x2": 200,
            "y2": 70,
            "text": "Welcome to multimodal OCR translator.",
            "lines": [
                {
                    "x1": 10,
                    "y1": 20,
                    "x2": 200,
                    "y2": 30,
                    "text": "Welcome to",
                    "words": [
                        {"x1": 10, "y1": 20, "x2": 50, "y2": 30, "text": "Welcome"},
                        {"x1": 60, "y1": 20, "x2": 100, "y2": 30, "text": "to"}
                    ]
                },
                {
                    "x1": 10,
                    "y1": 40,
                    "x2": 200,
                    "y2": 50,
                    "text": "multimodal",
                    "words": [
                        {"x1": 10, "y1": 40, "x2": 50, "y2": 50, "text": "multimodal"}
                    ]
                },
                {
                    "x1": 10,
                    "y1": 60,
                    "x2": 200,
                    "y2": 70,
                    "text": "OCR translator.",
                    "words": [
                        {"x1": 60, "y1": 60, "x2": 100, "y2": 70, "text": "OCR"},
                        {"x1": 110, "y1": 60, "x2": 150, "y2": 70, "text": "translator."}
                    ]
                }
            ]
        },
        {
            "x1": 10,
            "y1": 160,
            "x2": 200,
            "y2": 210,
            "text": "I'm glad to be here to assist you!",
            "lines": [
                {
                    "x1": 10,
                    "y1": 160,
                    "x2": 200,
                    "y2": 190,
                    "text": "I'm glad to be",
                    "words": [
                        {"x1": 10, "y1": 160, "x2": 50, "y2": 170, "text": "I'm"},
                        {"x1": 60, "y1": 160, "x2": 100, "y2": 170, "text": "glad"},
                        {"x1": 110, "y1": 160, "x2": 150, "y2": 170, "text": "to"},
                        {"x1": 160, "y1": 160, "x2": 200, "y2": 170, "text": "be"}
                    ]
                },
                {
                    "x1": 10,
                    "y1": 180,
                    "x2": 200,
                    "y2": 190,
                    "text": "here to assist",
                    "words": [
                        {"x1": 10, "y1": 180, "x2": 50, "y2": 190, "text": "here"},
                        {"x1": 60, "y1": 180, "x2": 100, "y2": 190, "text": "to"},
                        {"x1": 110, "y1": 180, "x2": 150, "y2": 190, "text": "assist"}
                    ]
                },
                {
                    "x1": 10,
                    "y1": 200,
                    "x2": 200,
                    "y2": 210,
                    "text": "you!",
                    "words": [
                        {"x1": 160, "y1": 200, "x2": 200, "y2": 210, "text": "you!"}
                    ]
                }
            ]
        }
    ]
}
```

The returned data includes:
- `x1`, `y1`, `x2`, `y2`: Coordinates of the bounding box for the paragraph.
- `text`: Text content of the paragraph.
- `lines`: List of lines within the paragraph, with similar structure including `id`, `x1`, `y1`, `x2`, `y2`, `text`, and `words` (nested structure).

Each line contains:
- `x1`, `y1`, `x2`, `y2`:
- `text`: Text content of the line.
- `words`: List of words within the line, with similar structure including `id`, `x1`, `y1`, `x2`, `y2`, and `text`.

Each word contains:
- `x1`, `y1`, `x2`, `y2`: Coordinates of the bounding box for the word.
- `text`: Text content of the word.

## Endpoints

### GET /api/supported-languages

**Description:**

Gets the list of languages supported by Google Translate.

**Usage:**

*Python:*

```python
import requests

url = "http://localhost:80/api/supported-languages"
    
response = requests.get(url)
    
if response.status_code == 200:
    supported_languages = response.json()["supported_languages"]
    print("Supported Languages:", supported_languages)
else:
    print(f"Error: {response.status_code}, {response.text}")
```

*JavaScript:*

```javascript
const axios = require('axios');

const apiUrl = "https://your-api-base-url/api/supported-languages";

try {
    const response = await axios.get(apiUrl);
    const supportedLanguages = response.data.supported_languages;
    console.log("Supported Languages:", supportedLanguages);
} catch (error) {
    if (error.response) {
        console.error(`Error from server: ${error.response.data.detail}`);
    } else if (error.request) {
        console.error("Error making the request:", error.request);
    } else {
        console.error("An unexpected error occurred:", error.message);
    }
}
```

**Possible Errors:**

- **500 Internal Server Error:**
  - If an unexpected error occurs on the server while retrieving the list of supported languages.

### POST /api/detect-language

**Description:**

Detects the language of a provided text.

**Parameters:**
- `text` (required): The text for which you want to detect the language.

**Usage:**

*Python:*

```python
import requests

url = "http://localhost:80/api/detect-language"
text_to_detect = "Your text goes here."

data = {"text": text_to_detect}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Detected language:", result["detected_language"])
else:
    print(f"Error: {response.status_code}, {response.text}")
```

*JavaScript:*

```javascript
const axios = require('axios');

const apiUrl = "https://your-api-base-url/api/detect-language";
const textToDetect = "Your text goes here.";

try {
    const response = await axios.post(apiUrl, { text: textToDetect });
    console.log("Detected language:", response.data.detected_language);
} catch (error) {
    if (error.response) {
        console.error(`Error from server: ${error.response.data.detail}`);
    } else if (error.request) {
        console.error("Error making the request:", error.request);
    } else {
        console.error("An unexpected error occurred:", error.message);
    }
}
```

**Possible Errors:**

- **400 Bad Request:**
  - If the `text` parameter is missing in the request.
  - If the provided `text` is empty.

- **500 Internal Server Error:**
  - If an unexpected error occurs on the server while detecting the language.

### POST /api/translate-text

**Description:**

Translates a text from a source language to a target language.

**Parameters:**

- `text` (required): The text you want to translate.
- `src` (optional, default: 'auto'): The source language of the text. If not provided, it will be detected automatically.
- `dest` (optional, default: 'en'): The target language for translation.

**Usage:**

*Python:*

```python
import requests

url = "http://localhost:80/api/translate-text"
text_to_translate = "Your text goes here."

data = {"text": text_to_translate, "src": 'en', "dest": 'es'}

response = requests.post(url, json=data)

if response.status_code == 200:
    translated_text = response.json()["translated_text"]
    print("Translated text:", translated_text)
else:
    print(f"Error: {response.status_code}, {response.text}")
```

*JavaScript:*

```javascript
const axios = require('axios');

const apiUrl = "https://your-api-base-url/api/translate-text";
const textToTranslate = "Your text goes here.";

try {
    const response = await axios.post(apiUrl, { text: textToTranslate });
    console.log("Translated text:", response.data.translated_text);
} catch (error) {
    if (error.response) {
        console.error(`Error from server: ${error.response.data.detail}`);
    } else if (error.request) {
        console.error("Error making the request:", error.request);
    } else {
        console.error("An unexpected error occurred:", error.message);
    }
}
```

**Possible Errors:**

- **400 Bad Request:**
  - If the `text` parameter is missing in the request.
  - If the provided `text` is empty.
  - If the `src` parameter contains an unsupported language.
  - If the `dest` parameter contains an unsupported language.

- **500 Internal Server Error:**
  - If an unexpected error occurs on the server while translating the text.

### POST /api/translate-multiple

**Description:**

Translates a list of texts to a target language.

**Parameters:**

- `texts` (required): A list of texts you want to translate.
- `src` (optional, default: 'auto'): The source language of the texts. If not provided, it will be detected automatically.
- `dest` (optional, default: 'en'): The target language for translation.

**Usage:**

*Python:*

```python
import requests

url = "http://localhost:80/api/translate-multiple"
texts_to_translate = ["Text 1", "Text 2", "Text 3"]
destination_language = "es"

data = {"texts": texts_to_translate, "dest": destination_language}

response = requests.post(url, json=data)

if response.status_code == 200:
    translated_texts = response.json()["translated_texts"]
    print("Translated texts:", translated_texts)
else:
    print(f"Error: {response.status_code}, {response.text}")
```

*JavaScript:*

```javascript
const axios = require('axios');

const apiUrl = "https://your-api-base-url/api/translate-multiple";
const textsToTranslate = ["Text 1", "Text 2", "Text 3"];
const destinationLanguage = "es";

try {
    const response = await axios.post(apiUrl, { texts: textsToTranslate, dest: destinationLanguage });
    console.log("Translated texts:", response.data.translated_texts);
} catch (error) {
    if (error.response) {
        console.error(`Error from server: ${error.response.data.detail}`);
    } else if (error.request) {
        console.error("Error making the request:", error.request);
    } else {
        console.error("An unexpected error occurred:", error.message);
    }
}
```

**Possible Errors:**

- **400 Bad Request:**
  - If the `texts` parameter is missing in the request.
  - If the `texts` parameter is an empty list.
  - If the `src` parameter contains an unsupported language.
  - If the `dest` parameter contains an unsupported language.

- **500 Internal Server Error:**
  - If an unexpected error occurs on the server while translating the texts.

### POST /api/translate-multiple-for-multiple

**Description:**

Translates multiple texts to multiple target languages.

**Parameters:**

- `texts` (required): A list of texts you want to translate.
- `dest_languages` (required): A list of destination languages corresponding to each text.

**Usage:**

*Python:*

```python
import requests

url = "http://localhost:80/api/translate-multiple-for-multiple"
texts_to_translate = ["Text 1", "Text 2", "Text 3"]
destination_languages = ["es", "fr", "pt"]

data = {"texts": texts_to_translate, "dest_languages": destination_languages}

response = requests.post(url, json=data)

if response.status_code == 200:
    translations = response.json()["translations"]
    print("Translations:", translations)
else:
    print(f"Error: {response.status_code}, {response.text}")
```

*JavaScript:*

```javascript
const axios = require('axios');

const apiUrl = "https://your-api-base-url/api/translate-multiple-for-multiple";
const textsToTranslate = ["Text 1", "Text 2", "Text 3"];
const destinationLanguages = ["es", "fr", "de"];

try {
    const response = await axios.post(apiUrl, { texts: textsToTranslate, dest_languages: destinationLanguages });
    console.log("Translations:", response.data.translations);
} catch (error) {
    if (error.response) {
        console.error(`Error from server: ${error.response.data.detail}`);
    } else if (error.request) {
        console.error("Error making the request:", error.request);
    } else {
        console.error("An unexpected error occurred:", error.message);
    }
}
```

**Possible Errors:**

- **400 Bad Request:**
  - If the `texts` parameter is missing in the request.
  - If the `dest_languages` parameter is missing in the request.
  - If the `texts` parameter or `dest_languages` parameter is an empty list.
  - If the number of texts and destination languages provided do not match.
  - If any destination language is not supported.

- **500 Internal Server Error:**
  - If an unexpected error occurs on the server while translating the texts.

### POST /api/extract-text-from-image

**Description:**

Extracts text from an image.

**Parameters:**

- `src` (required): The expected language of the text in the image.
- `image` (required): The image from which you want to extract text. Can be provided as an upload file or a link to the image.

**Return:**

- Structured data of the extracted text, including coordinates and text.

**Usage:**

*Python:*

```python
import requests

url = "http://localhost:80/api/extract-text-from-image"
src_language = "en"

data = {"src": src_language}
files = {
    "image": ("file", open("image.png", "rb")),
}
    
response = requests.post(url, data=data, files=files)

if response.status_code == 200:
    extracted_data = response.json()["paragraphs"]
    print("Extracted text data:", extracted_data)
else:
    print(f"Error: {response.status_code}, {response.text}")
```

*JavaScript:*

```javascript
const axios = require('axios');
const fs = require('fs');

const apiUrl = "https://your-api-base-url/api/extract-text-from-image";
const srcLanguage = "en";
const imagePath = "/path/to/your/image.jpg";

try {
    const imageFile = fs.createReadStream(imagePath);
    const formData = new FormData();
    formData.append('src', srcLanguage);
    formData.append('image', imageFile);

    const response = await axios.post(apiUrl, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    const extractedData = response.data.paragraphs;
    console.log("Extracted text data:", extractedData);
} catch (error) {
    if (error.response) {
        console.error(`Error from server: ${error.response.data.detail}`);
    } else if (error.request) {
        console.error("Error making the request:", error.request);
    } else {
        console.error("An unexpected error occurred:", error.message);
    }
}
```

**Possible Errors:**

- **400 Bad Request:**
  - If the `src` parameter is missing in the request.
  - If the `src` parameter contains an unsupported language.
  - If the `image` parameter is missing in the request.
  - If there is an error loading the image from the provided link.

- **500 Internal Server Error:**
  - If an unexpected error occurs on the server while extracting text from the image.

### WebSocket /api/extract-texts-from-images

**Description:**

Extracts text from a list of images, providing real-time updates through a WebSocket.

**Parameters:**

- `src` (required): The expected language of the text in the images.
- `images` (required): A list of images from which you want to extract text. Each image can be provided as an upload file or a link to the image.

**Return:**

- Real-time updates on the extraction process, including partial results and completion status.

**Python:**

```python
import json
import asyncio
import websockets
import base64


async def extract_text_from_images():
    url = "ws://localhost:8000/api/extract-texts-from-images"
    data = {
        "src": "en",
        "images": [
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQyaLAVlSTb5IReovWHw5a8j7u6PG7GRoAVIA&usqp=CAU",
            base64.b64encode(open("8967175dc7a974412b9ce1e3e28805c7.jpg", "rb").read()).decode('utf-8')
        ]
    }

    try:
        async with websockets.connect(url) as websocket:
            await websocket.send(json.dumps(data))

            while True:
                response = await websocket.recv()

                if not response:
                    break

                try:
                    result = json.loads(response)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {response}")
                    continue

                if result["status"] == "complete":
                    print("Extraction complete!")
                    break
                elif result["status"] == "partial":
                    print(f"Partial result: {result['percentage_complete']}% complete")
                    print(f"Result {result['result']}")

    except Exception as e:
        print(f"Error during extraction: {str(e)}")

asyncio.run(extract_text_from_images())
```

**JavaScript:**

```javascript
const WebSocket = require('ws');

const url = 'ws://localhost:80/api/extract-texts-from-images';
const data = {
    src: 'en',
    images: [
        {filename: 'image1.jpg', content: fs.readFileSync('image1.jpg')},
        {filename: 'image2.jpg', content: fs.readFileSync('image2.jpg')},
    ],
};

const ws = new WebSocket(url);

ws.on('open', () => {
    ws.send(JSON.stringify(data));
});

ws.on('message', (response) => {
    const result = JSON.parse(response);
    
    if (result.status === 'complete') {
        console.log('Extraction complete!');
        ws.close();
    } else if (result.status === 'partial') {
        console.log(`Partial result: ${result.percentage_complete}% complete`);
        // Process partial result as needed
    }
});
```

**Possible Errors:**

During the image processing, if an error occurs, a message will be sent through the WebSocket connection with details about the error. Here are some examples of error messages that may be sent:

- "No media provided.": If no image is provided.
- "Unsupported source language": If the source language (`src`) is not supported or is invalid.
- "Media file not found.": If there is an error loading a media file from the provided link.
- "Error processing media": If there is an error during the processing of the media.
- "Unexpected error": If an unexpected error occurs.

### POST /api/translate-image-and-replace

**Description:**

Translates and replaces text in an image with the specified alignment.

**Parameters:**

- `src` (required): The source language of the text in the image.
- `dest` (required): The destination language for translation.
- `align` (optional): Alignment of the translated text within the image, can be 'left', 'center', or 'right' (default is 'center').
- `image` (required): The image file or link containing the text to be translated and replaced.

**Return:**

- Translated image with replaced text.

**Python:**

```python
import requests
from io import BytesIO

url = "http://localhost:80/api/translate-image-and-replace"
data = {
    "src": "en",
    "dest": "es",
    "align": "center",
}
files = {
    "image": ("file", open("image.jpg", "rb")),
}
response = requests.post(url, data=data, files=files)

if response.status_code == 200:
    translated_image = BytesIO(response.content)
    # Process the translated image as needed
else:
    print(f"Error: {response.status_code}, {response.text}")
```

**JavaScript:**

```javascript
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

const url = 'http://localhost:80/api/translate-image-and-replace';
const formData = new FormData();

formData.append('src', 'en');
formData.append('dest', 'es');
formData.append('align', 'center');
formData.append('image', fs.createReadStream('image.jpg'));

axios.post(url, formData, {
    headers: {
        ...formData.getHeaders(),
    },
})
.then(response => {
    const translatedImage = Buffer.from(response.data, 'binary');
    // Process the translated image as needed
})
.catch(error => {
    console.error(`Error: ${error.response.status}, ${error.response.data}`);
});
```

**Possible Errors:**

- **400 Bad Request:**
  - If no image is provided.
  - If the source language (`src`) is not supported or invalid.
  - If the destination language (`dest`) is not supported.

- **400 Bad Request:**
  - If there is an error loading the image from the provided link.

- **500 Internal Server Error:**
  - If there is an error loading or processing the image.

Claro, aqui está a versão atualizada do seu README:

### WebSocket /api/extract-text-from-media

**Description:**

Extracts text from media files (e.g., videos) on a per-frame basis, providing real-time updates through a WebSocket.

**Parameters:**

- `src` (required): The source language of the text in the media.
- `media` (required): The media file or link from which you want to extract text.

**Return:**

- Real-time updates on the extraction process, including partial results and completion status. 
- ```json
  {
    "status": "String: 'partial' or 'complete'.",
    "frame":  0,
    "result": ["List with the specified structure in top of README."]
  }
  ```

**Python:**

```python
import json
import asyncio
import websockets
import cv2


# Send periodic frame to API
async def send_frames(websocket, cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send(buffer.tobytes())
        else:
            break
    await websocket.send("EOF")  # Indicate end of file


# Receive periodic information from status of the requisition
async def receive_messages(websocket):
    while True:
        response = await websocket.recv()
        if not response:
            break
        result = json.loads(response)

        if result["status"] == "complete":
            print("Extraction complete!")
            break
        elif result["status"] == "partial":
            print(f"Frame: {result['frame']}")
            print(f"Partial result: {result['result']}")
            # Process partial result as needed


async def send_video(video_file_path: str):
    url = "ws://localhost:8000/api/extract-text-from-media"
    cap = cv2.VideoCapture(video_file_path)

    # Send periodic ping to avoid timeout
    async def ping():
        while True:
            await websocket.ping()
            await asyncio.sleep(10)

    try:
        async with websockets.connect(url, timeout=10 * 60) as websocket:
            data = {
                "src": "en",
            }
            await websocket.send(json.dumps(data))

            asyncio.create_task(ping())

            send_frames_task = asyncio.create_task(send_frames(websocket, cap))
            receive_messages_task = asyncio.create_task(receive_messages(websocket))

            await asyncio.gather(send_frames_task, receive_messages_task)

    except Exception as e:
        print(f"Error during extraction: {str(e)}")

    finally:
        cap.release()

asyncio.get_event_loop().run_until_complete(send_video("video.mp4"))
```

**JavaScript:**

```javascript
const WebSocket = require('ws');
const fs = require('fs');

const url = 'ws://example.com/api/extract-text-from-media';
const data = {
    src: 'en',
    media: {filename: 'video.mp4', content: fs.readFileSync('video.mp4')},
};

const ws = new WebSocket(url);

ws.on('open', () => {
    ws.send(JSON.stringify(data));
});

ws.on('message', (response) => {
    const result = JSON.parse(response);
    
    if (result.status === 'complete') {
        console.log('Extraction complete!');
        ws.close();
    } else if (result.status === 'partial') {
        console.log(`Partial result: ${result.percentage_complete}% complete`);
        // Process partial result as needed
    }
});
```

**Possible Errors:**

- "No media provided.": If no media is provided.
- "Unsupported source language": If the source language (`src`) is not supported or is invalid.
- "Media file not found.": If there is an error loading a media file from the provided link.
- "Error processing media": If there is an error during the processing of the media.
- "Unexpected error": If an unexpected error occurs.

### POST /api/translate-and-replace-from-media

**Description:**

Translates and replaces text in each frame of a media file (e.g., videos) with the specified alignment.

**Parameters:**

- `src` (required): The source language of the text in the media.
- `dest` (required): The destination language for translation.
- `align` (optional): Alignment of the translated text within each frame, can be 'left', 'center', or 'right' (default is 'center').
- `media` (required): The media file or link from which you want to extract text.

**Return:**

- Real-time updates on the translation and replacement process, including partial results and completion status.

**Python:**

```python
import json
import asyncio
import websockets

url = "http://localhost:80/api/translate-and-replace-from-media"
data = {
    "src": "en",
    "dest": "es",
    "align": "center",
}
files = {
    "media": ("file", open("video.mp4", "rb")),
}

async def translate_and_replace_from_media():
    async with websockets.connect(url) as websocket:
        await websocket.send(json.dumps(data))

        while True:
            response = await websocket.recv()
            result = json.loads(response)
            
            if result["status"] == "complete":
                print("Translation and replacement complete!")
                break
            elif result["status"] == "partial":
                print(f"Partial result: {result['percentage_complete']}% complete")
                # Process partial result as needed

asyncio.run(translate_and_replace_from_media())
```

**JavaScript:**

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const url = 'http://localhost:80/api/translate-and-replace-from-media';
const formData = new FormData();

formData.append('src', 'en');
formData.append('dest', 'es');
formData.append('align', 'center');
formData.append('media', fs.createReadStream('video.mp4'));

const ws = new WebSocket(url);

ws.on('open', () => {
    ws.send(formData);
});

ws.on('message', (response) => {
    const result = JSON.parse(response);
    
    if (result.status === 'complete') {
        console.log('Translation and replacement complete!');
        ws.close();
    } else if (result.status === 'partial') {
        console.log(`Partial result: ${result.percentage_complete}% complete`);
        // Process partial result as needed
    }
});
```

**Possible Errors:**

- "No media provided.": If no media is provided.
- "Unsupported source language": If the source language (`src`) is not supported or is invalid.
- "Media file not found.": If there is an error loading a media file from the provided link.
- "Error processing media": If there is an error during the processing of the media.
- "Unexpected error": If an unexpected error occurs.

## Possible Alignments:

- Left (left)
- Center (center)
- Right (right)

## Supported Languages:

- Portuguese (pt)
- English (en)
- Spanish (es)
- Afrikaans (af)
- Albanian (sq)
- Amharic (am)
- Arabic (ar)
- Armenian (hy)
- Azerbaijani (az)
- Basque (eu)
- Belarusian (be)
- Bengali (bn)
- Bosnian (bs)
- Bulgarian (bg)
- Catalan (ca)
- Cebuano (ceb)
- Chichewa (ny)
- Chinese (Simplified) (zh-CN)
- Chinese (Traditional) (zh-TW)
- Corsican (co)
- Croatian (hr)
- Czech (cs)
- Danish (da)
- Dutch (nl)
- English (en)
- Esperanto (eo)
- Estonian (et)
- Filipino (tl)
- Finnish (fi)
- French (fr)
- Frisian (fy)
- Galician (gl)
- Georgian (ka)
- German (de)
- Greek (el)
- Gujarati (gu)
- Haitian Creole (ht)
- Hausa (ha)
- Hawaiian (haw)
- Hebrew (iw)
- Hindi (hi)
- Hmong (hmn)
- Hungarian (hu)
- Icelandic (is)
- Igbo (ig)
- Indonesian (id)
- Irish (ga)
- Italian (it)
- Japanese (ja)
- Javanese (jw)
- Kannada (kn)
- Kazakh (kk)
- Khmer (km)
- Kinyarwanda (rw)
- Korean (ko)
- Kurdish (ku)
- Kyrgyz (ky)
- Lao (lo)
- Latin (la)
- Latvian (lv)
- Lithuanian (lt)
- Luxembourgish (lb)
- Macedonian (mk)
- Malagasy (mg)
- Malay (ms)
- Malayalam (ml)
- Maltese (mt)
- Maori (mi)
- Marathi (mr)
- Mongolian (mn)
- Burmese (my)
- Nepali (ne)
- Norwegian (no)
- Pashto (ps)
- Persian (fa)
- Polish (pl)
- Portuguese (pt)
- Punjabi (pa)
- Romanian (ro)
- Russian (ru)
- Samoan (sm)
- Scottish Gaelic (gd)
- Serbian (sr)
- Sesotho (st)
- Shona (sn)
- Sindhi (sd)
- Sinhala (si)
- Slovak (sk)
- Slovenian (sl)
- Somali (so)
- Spanish (es)
- Sundanese (su)
- Swahili (sw)
- Swedish (sv)
- Tajik (tg)
- Tamil (ta)
- Telugu (te)
- Thai (th)
- Turkish (tr)
- Ukrainian (uk)
- Urdu (ur)
- Uzbek (uz)
- Vietnamese (vi)
- Welsh (cy)
- Xhosa (xh)
- Yiddish (yi)
- Yoruba (yo)
- Zulu (zu)