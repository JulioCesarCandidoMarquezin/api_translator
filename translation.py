from googletrans import Translator, LANGUAGES

urls = ['translate.googleapis.com', 'translate.google.com', 'translate.google.com.ar', 'translate.google.com.br',
        'translate.google.com']

user = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:83.0) Gecko/20100101 Firefox/83.0'

translator = Translator(service_urls=urls, user_agent=user, raise_exception=False)


async def detect_language(text: str) -> str:
    return translator.detect(text).lang


async def translate_text(text: str, src: str = 'auto', dest: str = 'en') -> str:
    return translator.translate(text=text, src=src, dest=dest).text