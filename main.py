import ctypes
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import agent_sim
import cv2
import keyboard
import mss
import numpy as np
import pygame
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap

# ===== КОНСТАНТЫ =====
MODELS_DIR = "models"  # каталог моделей
MODEL_HASH = "1873d95a6db30af7b1bb69b0b8c44dfe8f78b000"  # хеш модели

DYNAMIC_STEP = 50  # шаг
NUM_AGENTS = 1000  # число агентов
NUM_ITERATIONS = 1000000  # макс. итераций
ALLOW_MULTIPLE_AGENTS_IN_CELL = True  # много агентов в 1 ячейке
CELL_SIZE = 10  # размер ячейки (пикс)
SCREEN_CAPTURE_INTERVAL = 1.0  # интервал захвата (с)
CAPTURE_INTERVAL_STEP = 0.1  # шаг изменения интервала
MIN_CAPTURE_INTERVAL = 0.05  # мин. интервал
MAX_CAPTURE_INTERVAL = 1.0  # макс. интервал
INITIAL_ALPHA = 128  # нач. прозрачность
PREDICT_ENABLED = True  # предсказание вкл
STEPS_PER_FRAME = 200  # шагов на кадр
TOPMOST_FORCE_INTERVAL = 0.05  # период поднятия окна
OUTPUT_DIR = "processed_videos"  # папка результатов

if sys.platform == 'win32':
    ctypes.windll.user32.SetProcessDPIAware()  # учёт DPI


# ===== ЦВЕТОВАЯ СХЕМА =====
def create_custom_colormap():
    # градиент: синий -> голубой -> жёлтый -> красный
    colors = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)]
    positions = [0.0, 0.33, 0.66, 1.0]
    return LinearSegmentedColormap.from_list('blue_to_red', list(zip(positions, colors)))


def apply_custom_heatmap(gaze_map):
    # преобразование карты значимости в цветное BGR
    if gaze_map.max() > gaze_map.min():
        gaze_map_norm = (gaze_map - gaze_map.min()) / (gaze_map.max() - gaze_map.min() + 1e-8)
    else:
        gaze_map_norm = np.zeros_like(gaze_map)
    cmap = create_custom_colormap()
    colored = cmap(gaze_map_norm)[:, :, :3]  # RGB 0..1
    return (colored[:, :, ::-1] * 255).astype(np.uint8)  # BGR


# ===== МОДЕЛЬ КАРТЫ ВЗГЛЯДА =====
class GazeMapModel:
    def __init__(self):
        self.model_fn = None
        self.load_model()

    def _find_model_path(self):
        # поиск директории с saved_model.pb
        script_dir = Path(__file__).parent.absolute()
        models_dir = script_dir / MODELS_DIR
        model_hash_dir = models_dir / MODEL_HASH
        if model_hash_dir.exists():
            return str(model_hash_dir)
        if models_dir.exists():
            for item in models_dir.iterdir():
                if item.is_dir() and (item / "saved_model.pb").exists():
                    return str(item)
        raise FileNotFoundError("Модель не найдена")

    def load_model(self):
        try:
            model_path = self._find_model_path()
            model = tf.saved_model.load(model_path)
            self.model_fn = model.signatures[list(model.signatures.keys())[0]]
            return True
        except Exception:
            return False

    def predict_gaze_map(self, frame):
        # возвращает карту значимости и тепловую карту
        if self.model_fn is None:
            return None, None

        original_shape = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame

        input_tensor = tf.expand_dims(frame_rgb.astype(np.float32), axis=0)
        input_tensor = tf.image.resize(input_tensor, (320, 320), preserve_aspect_ratio=True)

        h_pad = 320 - input_tensor.shape[1]
        w_pad = 320 - input_tensor.shape[2]
        input_tensor = tf.pad(input_tensor,
                              [[0, 0], [h_pad // 2, h_pad - h_pad // 2],
                               [w_pad // 2, w_pad - w_pad // 2], [0, 0]])

        outputs = self.model_fn(input_tensor)
        out = outputs[list(outputs.keys())[0]]

        # удаление паддинга
        out = out[:, h_pad // 2:out.shape[1] - (h_pad - h_pad // 2),
        w_pad // 2:out.shape[2] - (w_pad - w_pad // 2), :]
        out = tf.image.resize(out, original_shape)
        gaze_map = out.numpy().squeeze()
        heatmap = apply_custom_heatmap(gaze_map)
        return gaze_map, heatmap


# ===== ФУНКЦИИ РАБОТЫ С ОКНАМИ (WIN32) =====
def force_window_topmost(hwnd):
    # поднять окно поверх всех
    styles = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
    if not (styles & 0x00000008):
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)

    foreground = ctypes.windll.user32.GetForegroundWindow()
    if foreground != hwnd:
        curr_thread = ctypes.windll.kernel32.GetCurrentThreadId()
        fg_thread = ctypes.windll.user32.GetWindowThreadProcessId(foreground, None)
        if fg_thread != curr_thread:
            ctypes.windll.user32.AttachThreadInput(fg_thread, curr_thread, True)
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        if fg_thread != curr_thread:
            ctypes.windll.user32.AttachThreadInput(fg_thread, curr_thread, False)

    ctypes.windll.user32.BringWindowToTop(hwnd)
    ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0x0002 | 0x0001)


def set_window_click_through(hwnd):
    # пропускать клики сквозь окно
    styles = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
    ctypes.windll.user32.SetWindowLongW(hwnd, -20, styles | 0x00000020 | 0x00080000)
    ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0002 | 0x0001)


def set_window_alpha(hwnd, alpha):
    # установить прозрачность
    ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, alpha, 0x00000002)


# ===== ЗАХВАТ ЭКРАНА И СИМУЛЯЦИЯ =====
class ScreenCaptureMode:
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.reference_distribution = None  # эталонное распределение
        self.dynamic_distribution = None  # динамическое распределение
        self.agents = None  # позиции агентов
        self.sequences = None  # последовательности
        self.running = True  # флаг работы
        self.paused = False  # флаг паузы
        self.current_iteration = 0  # текущая итерация
        self.alpha = INITIAL_ALPHA  # прозрачность
        self.current_display = 1  # режим отображения
        self.minimized = False  # окно свёрнуто
        self.capture_interval = SCREEN_CAPTURE_INTERVAL  # интервал захвата

        self.alpha_lock = threading.Lock()
        self.screen_capture_lock = threading.Lock()
        self.capturing_in_progress = False

        self.command_queue = deque()
        self.command_lock = threading.Lock()

        self.model = GazeMapModel()
        self.predicted_heatmap = None
        self.predict_enabled = PREDICT_ENABLED
        self.model_loaded = self.model.model_fn is not None

        self.frame_queue = queue.Queue(maxsize=30)
        self.processing_thread = threading.Thread(target=self._process_frames_worker, daemon=True)
        self.processing_thread.start()

        self._initialize_simulation()
        self.screen_capture_thread = threading.Thread(target=self._continuous_screen_capture, daemon=True)
        self.screen_capture_thread.start()

        self.topmost_thread = threading.Thread(target=self._topmost_loop, daemon=True)
        self.topmost_thread.start()

        self.command_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.command_thread.start()

    def _topmost_loop(self):
        while self.running:
            if not self.minimized:
                force_window_topmost(self.hwnd)
            time.sleep(TOPMOST_FORCE_INTERVAL)

    def _process_commands(self):
        # обработка команд с приоритетом последней
        while self.running:
            if self.command_queue:
                with self.command_lock:
                    while len(self.command_queue) > 1:
                        self.command_queue.popleft()
                    command = self.command_queue[0] if self.command_queue else None

                if command:
                    cmd_type, value = command
                    if cmd_type == 'display':
                        self.current_display = value
                        force_window_topmost(self.hwnd)
                    elif cmd_type == 'alpha':
                        self.alpha = max(0, min(255, value))
                        self._set_window_alpha_safe(self.alpha)
                        force_window_topmost(self.hwnd)
                    elif cmd_type == 'capture_interval':
                        with self.screen_capture_lock:
                            self.capture_interval = max(MIN_CAPTURE_INTERVAL, min(MAX_CAPTURE_INTERVAL, value))
                    elif cmd_type == 'predict':
                        self.predict_enabled = value
                        if not self.predict_enabled:
                            self.predicted_heatmap = None
                        elif not self.frame_queue.empty():
                            self._generate_gaze_heatmap(self.frame_queue.queue[-1])
                    elif cmd_type == 'pause':
                        self.paused = value
                    elif cmd_type == 'minimize':
                        self._toggle_minimize_internal()

                if self.command_queue:
                    with self.command_lock:
                        if self.command_queue and self.command_queue[0] == command:
                            self.command_queue.popleft()
            time.sleep(0.01)

    def add_command(self, cmd_type, value):
        with self.command_lock:
            self.command_queue.append((cmd_type, value))

    def get_overlay(self, screen_shape):
        # формирование наложения
        if self.reference_distribution is None or self.dynamic_distribution is None:
            return np.zeros((screen_shape[1], screen_shape[0], 4), dtype=np.uint8)

        screen_w, screen_h = screen_shape

        if self.current_display == 1:
            if self.predicted_heatmap is not None and self.predict_enabled:
                img = self.predicted_heatmap
            else:
                img = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255
                cv2.putText(img, "Карта взгляда ВЫКЛ", (screen_w // 4, screen_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        elif self.current_display == 2:
            dyn_norm = np.clip(self.dynamic_distribution, 0, 255).astype(np.uint8)
            img = apply_custom_heatmap(dyn_norm / 255.0)
        else:  # 3 - эталон
            ref_norm = np.clip(self.reference_distribution, 0, 255).astype(np.uint8)
            img = apply_custom_heatmap(ref_norm / 255.0)

        if img.shape[:2] != (screen_h, screen_w):
            img = cv2.resize(img, (screen_w, screen_h))

        overlay = np.zeros((screen_h, screen_w, 4), dtype=np.uint8)
        overlay[:, :, :3] = img
        overlay[:, :, 3] = 255
        return overlay

    def _set_window_alpha_safe(self, alpha):
        with self.alpha_lock:
            if not self.capturing_in_progress:
                set_window_alpha(self.hwnd, alpha)

    def _capture_screen_without_overlay(self):
        try:
            with self.alpha_lock:
                self.capturing_in_progress = True

            current_alpha = self.alpha
            set_window_alpha(self.hwnd, 0)
            time.sleep(0.01)

            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                result = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

            set_window_alpha(self.hwnd, current_alpha)
            return result
        except Exception:
            return None
        finally:
            with self.alpha_lock:
                self.capturing_in_progress = False

    def _update_reference_in_cpp(self, frame):
        try:
            agent_sim.update_reference(frame.astype(np.uint8), CELL_SIZE)
            rows, cols = agent_sim.get_reference_dims()
            if rows > 0 and cols > 0:
                new_ref = np.zeros((rows, cols), dtype=np.float32)
                agent_sim.get_reference(new_ref)
                self.reference_distribution = new_ref
        except Exception:
            pass

    def _generate_gaze_heatmap(self, frame):
        if not self.predict_enabled or not self.model_loaded:
            self.predicted_heatmap = None
            return
        _, heatmap = self.model.predict_gaze_map(frame)
        self.predicted_heatmap = heatmap

    def _capture_and_queue_frame(self):
        frame = self._capture_screen_without_overlay()
        if frame is not None:
            try:
                self.frame_queue.put(frame, timeout=0.5)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put(frame)

    def _process_frames_worker(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._update_reference_in_cpp(frame)
            self._generate_gaze_heatmap(frame)
            self.frame_queue.task_done()

    def _continuous_screen_capture(self):
        while self.running:
            with self.screen_capture_lock:
                interval = self.capture_interval
            time.sleep(interval)
            self._capture_and_queue_frame()

    def _initialize_simulation(self):
        self._capture_and_queue_frame()
        time.sleep(0.5)
        rows, cols = 0, 0
        for _ in range(10):
            rows, cols = agent_sim.get_reference_dims()
            if rows > 0 and cols > 0:
                break
            time.sleep(0.5)
            self._capture_and_queue_frame()
        if rows == 0 or cols == 0:
            raise RuntimeError("Не удалось получить размеры")
        self.reference_distribution = np.zeros((rows, cols), dtype=np.float32)
        self.dynamic_distribution = np.zeros((rows, cols), dtype=np.float32)
        self.agents = np.array([(np.random.randint(rows), np.random.randint(cols)) for _ in range(NUM_AGENTS)],
                               dtype=np.int32)
        self.sequences = [np.random.permutation(8) for _ in range(1000)]

    def update_simulation(self):
        if not self.running or self.paused or self.current_iteration >= NUM_ITERATIONS:
            return False
        try:
            positions = self.agents.astype(np.int32)
            dyn_float = self.dynamic_distribution.astype(np.float32)
            seq = self.sequences[self.current_iteration % len(self.sequences)].astype(np.int32)
            rows, cols = self.dynamic_distribution.shape
            out_dyn = np.zeros_like(dyn_float)
            out_ref = np.zeros((rows, cols), dtype=np.float32)
            agent_sim.multi_step_agents(
                positions, dyn_float, seq,
                ALLOW_MULTIPLE_AGENTS_IN_CELL,
                STEPS_PER_FRAME, DYNAMIC_STEP,
                out_dyn, out_ref
            )
            self.agents = positions
            self.dynamic_distribution = out_dyn
            self.reference_distribution = out_ref
            self.current_iteration += STEPS_PER_FRAME
            return True
        except Exception:
            self.running = False
            return False

    # интерфейсные методы
    def toggle_predict(self):
        self.add_command('predict', not self.predict_enabled)
        return self.predict_enabled

    def increase_capture_interval(self):
        self.add_command('capture_interval', self.capture_interval + CAPTURE_INTERVAL_STEP)

    def decrease_capture_interval(self):
        self.add_command('capture_interval', self.capture_interval - CAPTURE_INTERVAL_STEP)

    def set_display_mode(self, mode):
        self.add_command('display', mode)

    def set_alpha(self, alpha):
        self.add_command('alpha', alpha)

    def set_pause(self, paused):
        self.add_command('pause', paused)

    def _toggle_minimize_internal(self):
        if not self.minimized:
            ctypes.windll.user32.ShowWindow(self.hwnd, 6)  # свернуть
            self.minimized = True
        else:
            ctypes.windll.user32.ShowWindow(self.hwnd, 9)  # восстановить
            self.minimized = False
            force_window_topmost(self.hwnd)
            set_window_click_through(self.hwnd)
            self._set_window_alpha_safe(self.alpha)

    def toggle_minimize(self):
        self.add_command('minimize', None)


# ===== ГРАФИЧЕСКИЙ ИНТЕРФЕЙС =====
class ModernButton:
    def __init__(self, x, y, w, h, text, color, hover_color, text_color=(0, 0, 0)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hover = False
        self.enabled = True

    def draw(self, screen, font):
        color = (180, 180, 180) if not self.enabled else (self.hover_color if self.hover else self.color)
        pygame.draw.rect(screen, color, self.rect, border_radius=12)
        pygame.draw.rect(screen, (150, 150, 170), self.rect, 3, border_radius=12)
        surf = font.render(self.text, True, self.text_color)
        rect = surf.get_rect(center=self.rect.center)
        screen.blit(surf, rect)

    def handle_event(self, event):
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False


class TextInput:
    def __init__(self, x, y, w, h, font):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = ""  # введённый текст
        self.cursor_pos = 0  # позиция курсора
        self.scroll_offset = 0  # смещение прокрутки
        self.active = False  # активность поля
        self.color = (230, 230, 250)
        self.active_color = (200, 220, 255)
        self.cursor_blink = 0
        self.cursor_visible = True
        self.backspace_timer = 0
        self.backspace_delay = 500
        self.backspace_interval = 50
        self.font = font

    def draw(self, screen):
        color = self.active_color if self.active else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, (150, 150, 170), self.rect, 3, border_radius=10)

        display_text = self.text if self.text else "Введите путь к видео на компьютере..."
        text_color = (80, 80, 80) if not self.text else (0, 0, 0)

        if self.active and self.cursor_pos < len(self.text):
            cx = self.font.size(self.text[:self.cursor_pos])[0]
            iw = self.rect.width - 20
            if cx - self.scroll_offset > iw:
                self.scroll_offset = cx - iw
            elif cx - self.scroll_offset < 0:
                self.scroll_offset = cx

        if self.scroll_offset > 0 and self.text:
            start = 0
            for i in range(len(self.text)):
                if self.font.size(self.text[:i])[0] >= self.scroll_offset:
                    start = i
                    break
            visible = self.text[start:]
        else:
            visible = self.text
            self.scroll_offset = 0

        surf = self.font.render(visible, True, text_color)
        ty = self.rect.y + self.rect.height // 2 - surf.get_height() // 2
        screen.blit(surf, (self.rect.x + 15, ty))

        if self.active and self.cursor_visible:
            cx = self.rect.x + 15 + self.font.size(self.text[:self.cursor_pos])[0] - self.scroll_offset
            if self.rect.x + 10 <= cx <= self.rect.x + self.rect.width - 10:
                pygame.draw.line(screen, (0, 0, 0), (cx, self.rect.y + 12),
                                 (cx, self.rect.y + self.rect.height - 12), 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            if self.active:
                self.cursor_blink = 0
                self.cursor_visible = True
                mx = event.pos[0] - self.rect.x - 15 + self.scroll_offset
                for i in range(len(self.text) + 1):
                    if self.font.size(self.text[:i])[0] >= mx:
                        self.cursor_pos = i
                        break
            return False

        if event.type == pygame.KEYDOWN and self.active:
            self.cursor_blink = 0
            self.cursor_visible = True

            if event.key == pygame.K_RETURN:
                self.active = False
                return True
            if event.key == pygame.K_BACKSPACE:
                if self.cursor_pos > 0:
                    self.text = self.text[:self.cursor_pos - 1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
                    self.backspace_timer = pygame.time.get_ticks()
            elif event.key == pygame.K_DELETE:
                if self.cursor_pos < len(self.text):
                    self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos + 1:]
            elif event.key == pygame.K_LEFT:
                self.cursor_pos = max(0, self.cursor_pos - 1)
            elif event.key == pygame.K_RIGHT:
                self.cursor_pos = min(len(self.text), self.cursor_pos + 1)
            elif event.key == pygame.K_HOME:
                self.cursor_pos = 0
                self.scroll_offset = 0
            elif event.key == pygame.K_END:
                self.cursor_pos = len(self.text)
            elif event.key == pygame.K_v and pygame.key.get_mods() & pygame.KMOD_CTRL:
                try:
                    import pyperclip
                    cb = pyperclip.paste()
                    if cb:
                        self.text = self.text[:self.cursor_pos] + cb + self.text[self.cursor_pos:]
                        self.cursor_pos += len(cb)
                except ImportError:
                    pass
            elif event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL:
                try:
                    import pyperclip
                    pyperclip.copy(self.text)
                except ImportError:
                    pass
            elif event.unicode and event.unicode.isprintable():
                self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                self.cursor_pos += 1
            return False

        if event.type == pygame.KEYUP and self.active and event.key == pygame.K_BACKSPACE:
            self.backspace_timer = 0
        return False

    def update(self):
        if self.active:
            self.cursor_blink += 1
            if self.cursor_blink >= 30:
                self.cursor_blink = 0
                self.cursor_visible = not self.cursor_visible

            if self.backspace_timer > 0:
                now = pygame.time.get_ticks()
                if now - self.backspace_timer > self.backspace_delay:
                    if now - self.backspace_timer > self.backspace_delay + self.backspace_interval:
                        self.backspace_timer = now - self.backspace_delay
                        if self.cursor_pos > 0:
                            self.text = self.text[:self.cursor_pos - 1] + self.text[self.cursor_pos:]
                            self.cursor_pos -= 1


class ProgressBar:
    def __init__(self, x, y, w, h, font):
        self.rect = pygame.Rect(x, y, w, h)
        self.progress = 0.0
        self.font = font

    def draw(self, screen):
        pygame.draw.rect(screen, (200, 200, 210), self.rect, border_radius=10)
        if self.progress > 0:
            pw = int(self.rect.width * self.progress)
            pr = pygame.Rect(self.rect.x, self.rect.y, pw, self.rect.height)
            pygame.draw.rect(screen, (80, 150, 200), pr, border_radius=10)
        text = self.font.render(f"{int(self.progress * 100)}%", True, (0, 0, 0))
        tr = text.get_rect(center=self.rect.center)
        screen.blit(text, tr)

    def set_progress(self, value):
        self.progress = max(0.0, min(1.0, value))


class VideoProcessor:
    def __init__(self, video_path, output_path, status_cb, progress_cb, complete_cb):
        self.video_path = video_path
        self.output_path = output_path
        self.status_cb = status_cb
        self.progress_cb = progress_cb
        self.complete_cb = complete_cb
        self.running = True
        self.thread = None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def process(self):
        self.thread = threading.Thread(target=self._process, daemon=True)
        self.thread.start()

    def _process(self):
        try:
            self.status_cb("Загрузка модели...")
            model = GazeMapModel()

            self.status_cb("Открытие видео...")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.status_cb("ОШИБКА: Не удалось открыть видео")
                self.complete_cb(False, None)
                return

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

            cnt = 0
            self.status_cb("Обработка видео...")
            while self.running and cnt < total:
                ret, frame = cap.read()
                if not ret:
                    break
                _, hm = model.predict_gaze_map(frame)
                blended = cv2.addWeighted(frame, 0.6, hm, 0.4, 0) if hm is not None else frame
                out.write(blended)
                cnt += 1
                self.progress_cb(cnt / total)

            cap.release()
            out.release()

            if self.running and cnt == total:
                self.complete_cb(True, self.output_path)
            else:
                self.complete_cb(False, None)
        except Exception:
            self.complete_cb(False, None)


class MenuGUI:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.WIDTH = int(1200 * 1.5)  # ширина окна
        self.HEIGHT = int(800 * 1.5)  # высота окна
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Карта Взгляда - Система Визуального Внимания")

        if sys.platform == 'win32':
            hwnd = pygame.display.get_wm_info()["window"]
            force_window_topmost(hwnd)

        self.BUTTON_COLOR = (200, 200, 220)
        self.BUTTON_HOVER = (160, 160, 190)
        self.TEXT_COLOR = (0, 0, 0)
        self.ACCENT_COLOR = (80, 150, 200)
        self.SUCCESS_COLOR = (60, 180, 60)

        self.title_font = pygame.font.SysFont('arial', int(48 * 1.5))
        self.button_font = pygame.font.SysFont('arial', int(32 * 1.5))
        self.small_font = pygame.font.SysFont('arial', int(24 * 1.5))
        self.status_font = pygame.font.SysFont('arial', int(22 * 1.5))

        self.current_screen = "main"  # текущий экран
        self.processing = False
        self.processor = None
        self.status_text = ""
        self.status_color = self.TEXT_COLOR
        self.progress_bar = None
        self.completed = False
        self.completed_path = ""

        self.create_main_buttons()
        self.create_video_buttons()
        self.clock = pygame.time.Clock()
        self.running = True

    def create_main_buttons(self):
        bw = int(350 * 1.5)
        bh = int(80 * 1.5)
        bx = (self.WIDTH - bw) // 2
        self.main_buttons = [
            ModernButton(bx, int(250 * 1.5), bw, bh, "Захват экрана",
                         self.BUTTON_COLOR, self.BUTTON_HOVER, self.TEXT_COLOR),
            ModernButton(bx, int(360 * 1.5), bw, bh, "Обработка видео",
                         self.BUTTON_COLOR, self.BUTTON_HOVER, self.TEXT_COLOR),
            ModernButton(bx, int(470 * 1.5), bw, bh, "Выход",
                         self.BUTTON_COLOR, self.BUTTON_HOVER, self.TEXT_COLOR)
        ]

    def create_video_buttons(self):
        bw = int(220 * 1.5)
        bh = int(65 * 1.5)
        self.video_input = TextInput(int(200 * 1.5), int(250 * 1.5), int(800 * 1.5), int(65 * 1.5), self.small_font)
        self.back_btn = ModernButton(int(200 * 1.5), int(650 * 1.5), bw, bh, "Назад",
                                     self.BUTTON_COLOR, self.BUTTON_HOVER, self.TEXT_COLOR)
        self.process_btn = ModernButton(self.WIDTH - int(200 * 1.5) - bw, int(650 * 1.5), bw, bh, "Обработать",
                                        self.ACCENT_COLOR, (100, 170, 220), self.TEXT_COLOR)
        self.stop_btn = ModernButton(self.WIDTH - int(200 * 1.5) - bw, int(650 * 1.5), bw, bh, "Остановить",
                                     (200, 100, 100), (220, 120, 120), self.TEXT_COLOR)
        self.stop_btn.enabled = False
        self.back_after_complete_btn = ModernButton(self.WIDTH // 2 - int(120 * 1.5), int(550 * 1.5), int(240 * 1.5),
                                                    int(65 * 1.5),
                                                    "Назад", self.ACCENT_COLOR, (100, 170, 220), self.TEXT_COLOR)

    def draw_gradient(self):
        for i in range(self.HEIGHT):
            val = 240 + int(i / self.HEIGHT * 15)
            pygame.draw.line(self.screen, (val, val, 250), (0, i), (self.WIDTH, i))

    def draw_main_screen(self):
        self.draw_gradient()
        title = self.title_font.render("Карта Взгляда", True, self.ACCENT_COLOR)
        tr = title.get_rect(center=(self.WIDTH // 2, int(120 * 1.5)))
        self.screen.blit(title, tr)
        sub = self.small_font.render("Система анализа визуального внимания", True, self.TEXT_COLOR)
        sr = sub.get_rect(center=(self.WIDTH // 2, int(180 * 1.5)))
        self.screen.blit(sub, sr)
        for btn in self.main_buttons:
            btn.draw(self.screen, self.button_font)

    def draw_video_screen(self):
        self.draw_gradient()
        title = self.title_font.render("Обработка видео", True, self.ACCENT_COLOR)
        tr = title.get_rect(center=(self.WIDTH // 2, int(80 * 1.5)))
        self.screen.blit(title, tr)

        if self.completed:
            comp = self.title_font.render("Видео обработано!", True, self.SUCCESS_COLOR)
            cr = comp.get_rect(center=(self.WIDTH // 2, int(280 * 1.5)))
            self.screen.blit(comp, cr)
            if self.completed_path:
                path_txt = self.small_font.render(f"Сохранено: {os.path.basename(self.completed_path)}", True,
                                                  self.TEXT_COLOR)
                pr = path_txt.get_rect(center=(self.WIDTH // 2, int(360 * 1.5)))
                self.screen.blit(path_txt, pr)
            self.back_after_complete_btn.draw(self.screen, self.button_font)
        elif self.processing:
            status = self.status_font.render(self.status_text, True, self.status_color)
            sr = status.get_rect(center=(self.WIDTH // 2, int(200 * 1.5)))
            self.screen.blit(status, sr)
            if self.progress_bar:
                self.progress_bar.draw(self.screen)
            self.stop_btn.draw(self.screen, self.button_font)
        else:
            label = self.small_font.render("Путь к видео на компьютере:", True, self.TEXT_COLOR)
            lr = label.get_rect(center=(self.WIDTH // 2, int(190 * 1.5)))
            self.screen.blit(label, lr)
            self.video_input.draw(self.screen)
            if self.status_text:
                st = self.small_font.render(self.status_text, True, self.status_color)
                sr = st.get_rect(center=(self.WIDTH // 2, int(400 * 1.5)))
                self.screen.blit(st, sr)
            self.back_btn.draw(self.screen, self.button_font)
            self.process_btn.draw(self.screen, self.button_font)

    @staticmethod
    def get_video_path(inp):
        return inp if os.path.exists(inp) else None

    def process_video(self, path):
        self.processing = True
        self.completed = False
        self.status_text = "Загрузка модели..."
        self.status_color = self.ACCENT_COLOR

        out_name = f"processed_{Path(path).stem}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        self.progress_bar = ProgressBar(int(250 * 1.5), int(300 * 1.5), int(700 * 1.5), int(40 * 1.5), self.button_font)

        def upd_status(txt):
            self.status_text = txt
            self.status_color = self.ACCENT_COLOR

        def upd_prog(v):
            if self.progress_bar:
                self.progress_bar.set_progress(v)

        def on_complete(success, fpath):
            self.processing = False
            if success:
                self.completed = True
                self.completed_path = fpath
            else:
                self.status_text = "Ошибка обработки видео"
                self.status_color = (200, 50, 50)

        self.processor = VideoProcessor(path, out_path, upd_status, upd_prog, on_complete)
        self.processor.process()

    def run(self):
        while self.running:
            mp = pygame.mouse.get_pos()
            if self.current_screen == "main":
                for btn in self.main_buttons:
                    btn.hover = btn.rect.collidepoint(mp)
            elif self.current_screen == "video":
                if not self.processing and not self.completed:
                    self.back_btn.hover = self.back_btn.rect.collidepoint(mp)
                    self.process_btn.hover = self.process_btn.rect.collidepoint(mp)
                elif self.processing:
                    self.stop_btn.hover = self.stop_btn.rect.collidepoint(mp)
                elif self.completed:
                    self.back_after_complete_btn.hover = self.back_after_complete_btn.rect.collidepoint(mp)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return "exit"

                if self.current_screen == "main":
                    for i, btn in enumerate(self.main_buttons):
                        if btn.handle_event(event):
                            if i == 0:
                                return "screen"
                            elif i == 1:
                                self.current_screen = "video"
                                self.status_text = ""
                                self.processing = False
                                self.completed = False
                                self.video_input.text = ""
                                self.video_input.cursor_pos = 0
                                self.video_input.scroll_offset = 0
                            elif i == 2:
                                return "exit"
                elif self.current_screen == "video":
                    if self.completed:
                        if self.back_after_complete_btn.handle_event(event):
                            self.current_screen = "video"
                            self.completed = False
                            self.status_text = ""
                            self.processing = False
                            self.video_input.text = ""
                            self.video_input.cursor_pos = 0
                            self.video_input.scroll_offset = 0
                    elif self.processing:
                        if self.stop_btn.handle_event(event) and self.processor:
                            self.processor.stop()
                            self.processing = False
                            self.status_text = "Обработка остановлена"
                            self.status_color = (200, 130, 0)
                            self.stop_btn.enabled = False
                            self.process_btn.enabled = True
                    else:
                        self.video_input.handle_event(event)
                        if self.back_btn.handle_event(event):
                            self.current_screen = "main"
                            self.status_text = ""
                            self.completed = False
                        if self.process_btn.handle_event(event):
                            if not self.video_input.text:
                                self.status_text = "Пожалуйста, введите путь к видео"
                                self.status_color = (200, 50, 50)
                            else:
                                vpath = self.get_video_path(self.video_input.text)
                                if vpath:
                                    self.process_video(vpath)
                                    self.stop_btn.enabled = True
                                    self.process_btn.enabled = False
                                else:
                                    self.status_text = "Видео не найдено! Проверьте путь"
                                    self.status_color = (200, 50, 50)

            if self.current_screen == "main":
                self.draw_main_screen()
            else:
                if not self.processing and not self.completed:
                    self.video_input.update()
                self.draw_video_screen()
            pygame.display.flip()
            self.clock.tick(60)
        return "exit"


# ===== РЕЖИМ ЗАХВАТА ЭКРАНА =====
def run_screen_capture():
    pygame.init()
    pygame.font.init()
    info = pygame.display.Info()
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.NOFRAME)
    pygame.display.set_caption("Карта Взгляда - Наложение")

    if sys.platform == 'win32':
        hwnd = pygame.display.get_wm_info()["window"]
        force_window_topmost(hwnd)
        set_window_click_through(hwnd)
        set_window_alpha(hwnd, INITIAL_ALPHA)

    clock = pygame.time.Clock()
    sim = ScreenCaptureMode(hwnd)

    def on_key_press(e):
        try:
            if e.name == '1' and keyboard.is_pressed('ctrl'):
                sim.set_display_mode(1)
            elif e.name == '2' and keyboard.is_pressed('ctrl'):
                sim.set_display_mode(2)
            elif e.name == '3' and keyboard.is_pressed('ctrl'):
                sim.set_display_mode(3)
            elif (e.name == '+' or e.name == '=') and keyboard.is_pressed('ctrl'):
                sim.set_alpha(min(255, sim.alpha + 10))
            elif e.name == '-' and keyboard.is_pressed('ctrl'):
                sim.set_alpha(max(0, sim.alpha - 10))
            elif e.name == '[' and keyboard.is_pressed('ctrl'):
                sim.decrease_capture_interval()
            elif e.name == ']' and keyboard.is_pressed('ctrl'):
                sim.increase_capture_interval()
            elif e.name == 'space':
                sim.set_pause(not sim.paused)
            elif e.name == 'h' and keyboard.is_pressed('ctrl'):
                sim.toggle_predict()
            elif e.name == 'enter' and keyboard.is_pressed('ctrl'):
                sim.toggle_minimize()
            elif e.name == 'esc' or e.name == 'q':
                sim.running = False
        except Exception:
            pass

    keyboard.on_press(on_key_press)

    while sim.running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                sim.running = False

        if not sim.paused:
            sim.update_simulation()

        if not sim.minimized:
            ov = sim.get_overlay((info.current_w, info.current_h))
            if ov is not None:
                surf = pygame.image.frombuffer(ov.tobytes(), ov.shape[1::-1], "RGBA")
                screen.fill((255, 255, 255, 0))
                screen.blit(surf, (0, 0))

                font = pygame.font.SysFont('arial', 24)
                mode_names = {1: "Предсказанная карта взгляда", 2: "Динамическое распределение",
                              3: "Эталонное распределение"}
                mode = mode_names.get(sim.current_display, "")
                model_stat = "Карта взгляда ВКЛ" if (sim.predict_enabled and sim.model_loaded) else "Карта взгляда ВЫКЛ"
                txt = f"Режим: {mode} | Прозрачность: {sim.alpha} | Интервал: {sim.capture_interval:.2f}с | {model_stat}"
                text = font.render(txt, True, (0, 0, 0))
                screen.blit(text, (10, 10))

                hint_font = pygame.font.SysFont('arial', 18)
                hint = hint_font.render(
                    "Ctrl+1/2/3 - смена | Ctrl+/- - прозрачность | Ctrl+[/] - частота | Ctrl+H - карта взгляда | Пробел - пауза",
                    True, (50, 50, 50))
                screen.blit(hint, (10, 45))

                force_window_topmost(hwnd)
                pygame.display.flip()

        clock.tick(60)

    keyboard.unhook_all()
    pygame.quit()


# ===== ГЛАВНАЯ ФУНКЦИЯ =====
def main():
    try:
        import pyperclip
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyperclip", "-q"])
        import pyperclip

    while True:
        menu = MenuGUI()
        choice = menu.run()
        pygame.quit()
        if choice == "screen":
            run_screen_capture()
        elif choice == "exit":
            break


if __name__ == "__main__":
    main()
