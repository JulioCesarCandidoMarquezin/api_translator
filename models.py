from typing import List, Optional, Union, Tuple, Dict
import numpy as np


class Word:
    id_counter: int = 0

    def __init__(self, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0, text: str = ""):
        Word.id_counter += 1
        self.id = Word.id_counter
        self.x1: int = x1
        self.y1: int = y1
        self.x2: int = x2
        self.y2: int = y2
        self.text: str = text

    def get_coordinates(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def is_object_around(self, other: 'Object', h_margin_percentage: float) -> 'Object':
        x1, y1, x2, y2 = self.get_coordinates()
        other_x1, other_y1, other_x2, other_y2 = other.get_coordinates()

        w_margin = int((x2 - x1) / (len(self.text) - 1 if len(self.text) > 1 else 1))
        h_margin = int((y2 - y1) * h_margin_percentage)

        around_x = (x1 - w_margin <= other_x2 <= x2 + w_margin) or (other_x1 - w_margin <= x2 <= other_x2 + w_margin)
        around_y = (y1 - h_margin <= other_y2 <= y2 + h_margin) or (other_y1 - h_margin <= y2 <= other_y2 + h_margin)

        if around_x and around_y:
            return other
        return None

    def __str__(self) -> str:
        return f"Word(id={self.id}, x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, text='{self.text}')"


class Line:
    id_counter: int = 0

    def __init__(self, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0, text: str = "", words: List[Word] = None):
        Line.id_counter += 1
        self.id: int = Line.id_counter
        self.x1: int = x1
        self.y1: int = y1
        self.x2: int = x2
        self.y2: int = y2
        self.text: str = text
        self.words: List[Word] = words

    def add_word(self, word: Word) -> None:
        word_x1, word_y1, word_x2, word_y2 = word.x1, word.y1, word.x2, word.y2

        self.x1 = min(self.x1, word_x1)
        self.y1 = min(self.y1, word_y1)
        self.x2 = max(self.x2, word_x2)
        self.y2 = max(self.y2, word_y2)

        self.text += " " + word.text

        self.words.append(word)

    def get_coordinates(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def organize_words_order(self):
        self.words.sort(key=lambda word: (word.x1, word.y1))
        self.text = ' '.join(word.text for word in self.words)

    def update_attributes_from_words(self) -> None:
        if self.words:
            coordinates = np.array([word.get_coordinates() for word in self.words])
            self.text = ' '.join(word.text for word in self.words)
            self.x1, self.y1, self.x2, self.y2 = np.min(coordinates[:, 0]), np.min(coordinates[:, 1]), np.max(coordinates[:, 2]), np.max(coordinates[:, 3])

    def __str__(self) -> str:
        return f"""
            Line(
                id='{self.id}',
                text='{self.text}',
                x={self.x1},
                y={self.y1},
                width={self.x2},
                height={self.y2},
                words=[
                    {' '.join(str(word) for word in self.words)}
                ]
            )
        """


class Paragraph:
    id_counter: int = 0

    def __init__(self, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0, text: str = "", lines: List[Line] = None):
        Paragraph.id_counter += 1
        self.id: int = Paragraph.id_counter
        self.x1: int = x1
        self.y1: int = y1
        self.x2: int = x2
        self.y2: int = y2
        self.text: str = text
        self.lines: List[Line] = lines

    def add_line(self, line: Line) -> None:
        line_x1, line_y1, line_x2, line_y2 = line.get_coordinates()

        self.x1 = min(self.x1, line_x1)
        self.y1 = min(self.y1, line_y1)
        self.x2 = max(self.x2, line_x2)
        self.y2 = max(self.y2, line_y2)

        self.text += "\n" + line.text

        self.lines.append(line)

    def get_coordinates(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def update_attributes_from_lines(self) -> None:
        if self.lines:
            coordinates = np.array([line.get_coordinates() for line in self.lines], dtype=int)
            self.text = '\n'.join(line.text for line in self.lines)
            self.x1, self.y1, self.x2, self.y2 = np.min(coordinates[:, 0]), np.min(coordinates[:, 1]), np.max(coordinates[:, 2]), np.max(coordinates[:, 3])

    def organize_lines_order(self):
        self.lines.sort(key=lambda line: (line.y1, line.x1))
        self.text = ' '.join(word.text for word in self.lines)

    @staticmethod
    def get_paragraphs_from_textdata(textdata: 'TextData') -> List['Paragraph']:
        paragraphs: List['Paragraph'] = []
        current_paragraph: Optional['Paragraph'] = None
        current_line: Optional['Line'] = None

        textdata.sort_entries(['top', 'left'])

        for x, y, w, h, text in zip(textdata.left, textdata.top, textdata.width, textdata.height, textdata.text):

            word = Word(x, y, x + w, y + h, text.strip())

            if current_paragraph is None:
                current_line = Line(x, y, x + w, y + h, text, [word])
                current_paragraph = Paragraph(x, y, x + w, y + h, text, [current_line])
                paragraphs.append(current_paragraph)
            else:
                compatible_paragraph = next((paragraph for paragraph in paragraphs if word.is_object_around(paragraph, 0.5)), None)

                if compatible_paragraph:
                    current_paragraph = compatible_paragraph
                    compatible_line = next((line for line in current_paragraph.lines if word.is_object_around(line, 0.25)), None)

                    if compatible_line:
                        current_line = compatible_line
                        current_line.add_word(word)
                    else:
                        current_line = Line(x, y, x + w, y + h, text, [word])
                        current_paragraph.add_line(current_line)
                else:
                    current_line = Line(x, y, x + w, y + h, text, [word])
                    current_paragraph = Paragraph(x, y, x + w, y + h, text, [current_line])
                    paragraphs.append(current_paragraph)

                for paragraph in paragraphs:
                    for line in paragraph.lines:
                        line.update_attributes_from_words()
                    paragraph.update_attributes_from_lines()

        for paragraph in paragraphs:
            for line in paragraph.lines:
                line.organize_words_order()
            paragraph.organize_lines_order()

        return paragraphs

    def __str__(self) -> str:
        lines_str = '\n'.join(str(line) for line in self.lines)
        return f"""
            Paragraph(
                id='{self.id}',
                text='{self.text}',
                x={self.x},
                y={self.y},
                width={self.w},
                height={self.h},
                lines=[
                    {lines_str}
                ]
            )
        """


class TextData:
    def __init__(self, textdata: Optional[dict] = None):
        if textdata is None:
            textdata = {}

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

    def get_entries(self) -> List[Dict[str, Union[int, float, str, None]]]:
        return [
            {
                'level': level,
                'page_num': page_num,
                'block_num': block_num,
                'par_num': par_num,
                'line_num': line_num,
                'word_num': word_num,
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'conf': conf,
                'text': text
            }
            for level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text in zip(
                self.level, self.page_num, self.block_num, self.par_num, self.line_num, self.word_num,
                self.left, self.top, self.width, self.height, self.conf, self.text
            )
        ]

    def sort_entries(self, sorting_keys: list) -> None:
        sorted_indices = sorted(range(len(self.top)), key=lambda k: tuple(getattr(self, key)[k] for key in sorting_keys))

        self.level = [self.level[i] for i in sorted_indices]
        self.page_num = [self.page_num[i] for i in sorted_indices]
        self.block_num = [self.block_num[i] for i in sorted_indices]
        self.par_num = [self.par_num[i] for i in sorted_indices]
        self.line_num = [self.line_num[i] for i in sorted_indices]
        self.word_num = [self.word_num[i] for i in sorted_indices]
        self.left = [self.left[i] for i in sorted_indices]
        self.top = [self.top[i] for i in sorted_indices]
        self.width = [self.width[i] for i in sorted_indices]
        self.height = [self.height[i] for i in sorted_indices]
        self.conf = [self.conf[i] for i in sorted_indices]
        self.text = [self.text[i] for i in sorted_indices]

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

                if text not in word_positions or not any(abs(position[0] - pos[0]) < position_margin * 2 and abs(position[1] - pos[1]) < position_margin * 2 for pos in word_positions[text]):
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
