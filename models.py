from typing import List, Optional, Union, Tuple
import numpy as np


class Word:
    def __init__(self, x: int, y: int, w: int, h: int, text: str):
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h
        self.text: str = text

    def get_coordinates(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def __str__(self) -> str:
        return f"Word(x={self.x}, y={self.y}, w={self.w}, h={self.h}, text='{self.text}')"


class Line:
    def __init__(self, x: int, y: int, w: int, h: int, text: str, words: List[Word]):
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h
        self.text: str = text
        self.words: List[Word] = words

    def add_word(self, word: Word) -> None:
        self.words.append(word)

    def get_coordinates(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def update_attributes_from_words(self) -> None:
        if self.words:
            coordinates = np.array([word.get_coordinates() for word in self.words])
            self.text = ' '.join(word.text for word in self.words)
            self.x, self.y, self.w, self.h = np.min(coordinates[:, 0]), np.min(coordinates[:, 1]), np.max(coordinates[:, 2]), np.max(coordinates[:, 3])

    def __str__(self) -> str:
        return f"Line(words={self.words})"


class Paragraph:
    def __init__(self, x: int, y: int, w: int, h: int, text: str, lines: List[Line]):
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h
        self.text: str = text
        self.lines: List[Line] = lines

    def add_line(self, line: Line) -> None:
        self.lines.append(line)

    def get_coordinates(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def update_attributes_from_lines(self) -> None:
        if self.lines:
            coordinates = np.array((line.get_coordinates() for line in self.lines), dtype=int)
            self.text = '\n'.join(line.text for line in self.lines)
            self.x, self.y, self.w, self.h = np.min(coordinates[:, 0]), np.min(coordinates[:, 1]), np.max(coordinates[:, 2]), np.max(coordinates[:, 3])

    def get_compatible_position_paragraph(self, paragraphs: List['Paragraph'], width_margin_factor: float = 0.2, height_margin_factor: float = 0.2) -> Optional['Paragraph']:
        x, y, w, h = self.get_coordinates()

        for paragraph in paragraphs:
            para_x, para_y, para_w, para_h = paragraph.get_coordinates()

            width_margin = para_w * width_margin_factor
            height_margin = para_h * height_margin_factor

            x_between_margin: bool = (para_x - width_margin <= x + w <= para_x + para_w + width_margin)
            y_between_margin: bool = (para_y - height_margin <= y + h <= para_y + para_h + height_margin)

            if x_between_margin and y_between_margin:
                return paragraph

        return None

    @staticmethod
    def get_paragraphs_from_textdata(textdata: 'TextData') -> List['Paragraph']:
        paragraphs: List['Paragraph'] = []
        current_paragraph: Optional['Paragraph'] = None
        current_line: Optional['Line'] = None

        for i in range(len(textdata.conf)):
            x, y, w, h, text = textdata.left[i], textdata.top[i], textdata.width[i], textdata.height[i], textdata.text[i]

            word = Word(x, y, w, h, text.strip())
            compatible_paragraph = current_paragraph.get_compatible_position_paragraph(paragraphs) if current_paragraph else None

            if compatible_paragraph:
                paragraphs.append(current_paragraph)
                current_line = Line(x, y, w, h, text, [word])
                current_paragraph = compatible_paragraph
                current_paragraph.add_line(current_line)

            elif (y - current_line.words[-1].y > current_line.words[-1].h) or (x - current_line.words[-1].x > current_line.words[-1].w * 2):
                current_paragraph.add_line(current_line)
                current_line = Line(x, y, w, h, text, [word])
            else:
                current_line.add_word(word)

        if current_line.words:
            current_paragraph.add_line(current_line)

        if current_paragraph:
            paragraphs.append(current_paragraph)

        return paragraphs

    def __str__(self) -> str:
        return f"Paragraph(text='{self.text}', x={self.x}, y={self.y}, width={self.w}, height={self.h}, lines={self.lines})"


class TextData:
    def __init__(self, textdata: Optional[dict] = None):
        self.level: List[Union[int, None]] = textdata.get('level', [])
        self.page_num: List[Union[int, None]] = textdata.get('page_num', [])
        self.block_num: List[Union[int, None]] = textdata.get('block_num', [])
        self.par_num: List[Union[int, None]] = textdata.get('par_num', [])
        self.line_num: List[Union[int, None]] = textdata.get('line_num', [])
        self.word_num: List[Union[int, None]] = textdata.get('word_num', [])
        self.left: List[Union[int, None]] = textdata.get('left', [])
        self.top: List[Union[int, None]] = textdata.get('top', [])
        self.width: List[Union[int, None]] = textdata.get('width', [])
        self.height: List[Union[int, None]] = textdata.get('height', [])
        self.conf: List[Union[float, None]] = textdata.get('conf', [])
        self.text: List[Union[str, None]] = textdata.get('text', [])

    def add_entry(self, level: Union[int, None] = None, page_num: Union[int, None] = None, block_num: Union[int, None] = None, par_num: Union[int, None] = None, line_num: Union[int, None] = None, word_num: Union[int, None] = None, left: Union[int, None] = None, top: Union[int, None] = None, width: Union[int, None] = None, height: Union[int, None] = None, conf: Union[float, None] = None, text: Union[str, None] = None) -> None:
        self.level.append(level)
        self.page_num.append(page_num)
        self.block_num.append(block_num)
        self.par_num.append(par_num)
        self.line_num.append(line_num)
        self.word_num.append(word_num)
        self.left.append(left)
        self.top.append(top)
        self.width.append(width)
        self.height.append(height)
        self.conf.append(conf)
        self.text.append(text)

    def get_entries(self) -> List[Tuple[Union[int, None], ...]]:
        return list(zip(self.level, self.page_num, self.block_num, self.par_num, self.line_num, self.word_num, self.left, self.top, self.width, self.height, self.conf, self.text))

    def filter_textdata(self, confidence_threshold: float) -> 'TextData':
        num_entries: int = len(self.conf)

        filtered_self: 'TextData' = TextData()

        for i in range(num_entries):
            confidence: float = float(self.conf[i])
            text: str = str(self.text[i])

            if confidence >= confidence_threshold and self.text[i]:
                if any(c.isalnum() for c in text):
                    filtered_self.level.append(int(self.level[i]))
                    filtered_self.page_num.append(int(self.page_num[i]))
                    filtered_self.block_num.append(int(self.block_num[i]))
                    filtered_self.par_num.append(int(self.par_num[i]))
                    filtered_self.line_num.append(int(self.line_num[i]))
                    filtered_self.word_num.append(int(self.word_num[i]))
                    filtered_self.left.append(int(self.left[i]))
                    filtered_self.top.append(int(self.top[i]))
                    filtered_self.width.append(int(self.width[i]))
                    filtered_self.height.append(int(self.height[i]))
                    filtered_self.conf.append(float(self.conf[i]))
                    filtered_self.text.append(str(self.text[i]))

        return filtered_self

    @classmethod
    def merge_textdata(cls, textdata_list: List['TextData'], position_margin: float) -> 'TextData':
        merged_textdata: 'TextData' = TextData()

        word_positions: dict = {}

        for textdata in textdata_list:
            num_entries: int = len(textdata.conf)

            for i in range(num_entries):
                text: str = str(textdata.text[i])
                position: Tuple[int, int] = (int(textdata.left[i]), int(textdata.top[i]))

                if text not in word_positions or not any(abs(position[0] - pos[0]) < position_margin and abs(position[1] - pos[1]) < position_margin for pos in word_positions[text]):
                    word_positions.setdefault(text, []).append(position)
                    merged_textdata.level.append(int(textdata.level[i]))
                    merged_textdata.page_num.append(int(textdata.page_num[i]))
                    merged_textdata.block_num.append(int(textdata.block_num[i]))
                    merged_textdata.par_num.append(int(textdata.par_num[i]))
                    merged_textdata.line_num.append(int(textdata.line_num[i]))
                    merged_textdata.word_num.append(int(textdata.word_num[i]))
                    merged_textdata.left.append(int(textdata.left[i]))
                    merged_textdata.top.append(int(textdata.top[i]))
                    merged_textdata.width.append(int(textdata.width[i]))
                    merged_textdata.height.append(int(textdata.height[i]))
                    merged_textdata.conf.append(float(textdata.conf[i]))
                    merged_textdata.text.append(textdata.text[i])

        return merged_textdata

    def __str__(self) -> str:
        return f"TextData(\n" \
               f"  level: {self.level}\n" \
               f"  page_num: {self.page_num}\n" \
               f"  block_num: {self.block_num}\n" \
               f"  par_num: {self.par_num}\n" \
               f"  line_num: {self.line_num}\n" \
               f"  word_num: {self.word_num}\n" \
               f"  left: {self.left}\n" \
               f"  top: {self.top}\n" \
               f"  width: {self.width}\n" \
               f"  height: {self.height}\n" \
               f"  conf: {self.conf}\n" \
               f"  text: {self.text}\n" \
               f")"
