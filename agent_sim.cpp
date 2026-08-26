#include <pybind11/pybind11.h> // Связывание C++ с Python
#include <pybind11/numpy.h>    // Работа с NumPy-массивами
#include <cstdint>             // Использование uint8_t
#include <algorithm>           // Стандартные алгоритмы
#include <vector>              //динамический массив
#include <cmath>               // Математические функции
#include <mutex>               // для многопоточной работы
#include <stdexcept>   // Исключения (для обработки ошибок)

namespace py = pybind11;  // Создание псевдонима для удобства

// Глобальные переменные для хранения эталонной карты
static std::vector<float> g_reference;
static int g_rows = 0;
static int g_cols = 0;
static std::mutex g_ref_mutex;

// Вспомогательная функция для выполнения одного шага всех агентов
void single_step(
    int* pos_ptr,                 // входные/выходные позиции (N,2)
    float* ref_ptr,                // reference (rows, cols) - нормализовано 0-255
    float* dyn_ptr,                // dynamic (rows, cols) - будет изменён (ненормализован)
    int* seq_ptr,                  // последовательность направлений (8)
    bool allow_multiple,
    float coeff,
    int rows,
    int cols,
    int N,
    int dynamic_step
) {
    // Строим карту занятости на начало шага
    std::vector<uint8_t> occupied(rows * cols, 0);
    for (int i = 0; i < N; ++i) {
        int r = pos_ptr[i * 2];
        int c = pos_ptr[i * 2 + 1];
        if (r >= 0 && r < rows && c >= 0 && c < cols)
            occupied[r * cols + c] = 1;
    }

    // Направления (8 направлений)
    const int directions[8][2] = {
        {0, 1}, {1, 0}, {0, -1}, {-1, 0},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };

    // Временные массивы для новых позиций и обновлений
    std::vector<int> new_positions(N * 2);
    std::vector<int> updates(N);

    #pragma omp parallel for
    for (int idx = 0; idx < N; ++idx) {
        int r = pos_ptr[idx * 2];
        int c = pos_ptr[idx * 2 + 1];
        int self_linear = r * cols + c;

        int best_r = r;
        int best_c = c;
        float best_Kp = -1e9f;
        bool found = false;

        // Перебираем направления в заданном порядке
        for (int d = 0; d < 8; ++d) {
            int dir = seq_ptr[d];
            int nr = r + directions[dir][0];
            int nc = c + directions[dir][1];

            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                int target_linear = nr * cols + nc;
                // Проверка занятости целевой клетки
                if (!allow_multiple && occupied[target_linear])
                    continue;

                // Проверка соседей целевой клетки (8 соседей), исключая самого агента
                bool neighbor_occupied = false;
                for (int di = -1; di <= 1 && !neighbor_occupied; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        int ni = nr + di;
                        int nj = nc + dj;
                        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                            int neighbor_linear = ni * cols + nj;
                            if (neighbor_linear == self_linear) continue;
                            if (occupied[neighbor_linear]) {
                                neighbor_occupied = true;
                                break;
                            }
                        }
                    }
                }
                if (neighbor_occupied)
                    continue;

                float Kp = ref_ptr[target_linear] - coeff * dyn_ptr[target_linear];
                if (Kp > best_Kp) {
                    best_Kp = Kp;
                    best_r = nr;
                    best_c = nc;
                    found = true;
                }
            }
        }

        if (!found) {
            best_r = r;
            best_c = c;
        }

        new_positions[idx * 2] = best_r;
        new_positions[idx * 2 + 1] = best_c;
        updates[idx] = best_r * cols + best_c;
    }

    // Копируем новые позиции
    for (int i = 0; i < N * 2; ++i) {
        pos_ptr[i] = new_positions[i];
    }

    // Обновляем dynamic distribution (суммируем по уникальным клеткам)
    std::vector<int> counts(rows * cols, 0);
    for (int i = 0; i < N; ++i) {
        int idx = updates[i];
        counts[idx]++;
    }
    for (int lin = 0; lin < rows * cols; ++lin) {
        if (counts[lin] > 0) {
            dyn_ptr[lin] += counts[lin] * dynamic_step;
        }
    }
}

// Функция обновления эталонной карты (вызывается из потока захвата)
void update_reference(
    py::array_t<uint8_t> image,          // (H, W, 3) RGB изображение
    int cell_size
) {
    auto buf_img = image.request();
    if (buf_img.ndim != 3 || buf_img.shape[2] != 3) {
        throw std::runtime_error("Input image must be H x W x 3 RGB");
    }

    int H = buf_img.shape[0];
    int W = buf_img.shape[1];
    int out_h = H / cell_size;
    int out_w = W / cell_size;

    if (out_h == 0 || out_w == 0) {
        // Слишком маленькое изображение - не обновляем карту
        return;
    }

    uint8_t* img_ptr = static_cast<uint8_t*>(buf_img.ptr);

    // Вычисляем средние значения по ячейкам
    std::vector<float> cell_means(out_h * out_w);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < out_h; ++i) {
        for (int j = 0; j < out_w; ++j) {
            float sum = 0.0f;
            for (int dy = 0; dy < cell_size; ++dy) {
                for (int dx = 0; dx < cell_size; ++dx) {
                    int y = i * cell_size + dy;
                    int x = j * cell_size + dx;
                    if (y < H && x < W) {
                        uint8_t* pixel = img_ptr + (y * W + x) * 3;
                        float gray = 0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2];
                        sum += gray;
                    }
                }
            }
            cell_means[i * out_w + j] = sum / (cell_size * cell_size);
        }
    }

    // Находим максимальное значение для нормализации
    float max_val = 0.0f;
    for (int idx = 0; idx < out_h * out_w; ++idx) {
        if (cell_means[idx] > max_val) max_val = cell_means[idx];
    }

    // Нормализуем к 0-255 и сохраняем в глобальную переменную
    std::vector<float> new_ref(out_h * out_w);
    if (max_val > 0) {
        for (int idx = 0; idx < out_h * out_w; ++idx) {
            new_ref[idx] = (cell_means[idx] / max_val) * 255.0f;
        }
    } else {
        for (int idx = 0; idx < out_h * out_w; ++idx) {
            new_ref[idx] = 0.0f;
        }
    }

    // Заменяем глобальную карту под защитой мьютекса
    {
        std::lock_guard<std::mutex> lock(g_ref_mutex);
        g_reference = std::move(new_ref);
        g_rows = out_h;
        g_cols = out_w;
    }
}

// Функция для получения размеров эталонной карты
std::pair<int, int> get_reference_dims() {
    std::lock_guard<std::mutex> lock(g_ref_mutex);
    return {g_rows, g_cols};
}

// Функция для выполнения нескольких шагов подряд и возврата обоих распределений
void multi_step_agents(
    py::array_t<int> positions,          // (N, 2) входные/выходные позиции
    py::array_t<float> dynamic_in,       // (rows, cols) входное состояние dynamic (нормализовано 0-255)
    py::array_t<int> sequence,           // (8,) порядок направлений
    bool allow_multiple,
    int steps,
    int dynamic_step,
    py::array_t<float> dynamic_out,      // (rows, cols) выходное состояние dynamic (будет нормализовано)
    py::array_t<float> reference_out     // (rows, cols) выходная эталонная карта (копия)
) {
    auto buf_pos = positions.request();
    auto buf_dyn_in = dynamic_in.request();
    auto buf_seq = sequence.request();
    auto buf_dyn_out = dynamic_out.request();
    auto buf_ref_out = reference_out.request();

    int N = buf_pos.shape[0];

    // Получаем текущую эталонную карту из глобальной переменной
    std::vector<float> current_ref;
    int rows, cols;
    {
        std::lock_guard<std::mutex> lock(g_ref_mutex);
        if (g_rows == 0 || g_cols == 0) {
            throw std::runtime_error("Reference map not initialized. Call update_reference first.");
        }
        rows = g_rows;
        cols = g_cols;
        current_ref = g_reference; // копируем
    }

    // Проверяем размеры входных массивов
    if (buf_dyn_in.ndim != 2 || buf_dyn_in.shape[0] != rows || buf_dyn_in.shape[1] != cols) {
        throw std::runtime_error("dynamic_in dimensions do not match reference");
    }
    if (buf_dyn_out.ndim != 2 || buf_dyn_out.shape[0] != rows || buf_dyn_out.shape[1] != cols) {
        throw std::runtime_error("dynamic_out dimensions do not match reference");
    }
    if (buf_ref_out.ndim != 2 || buf_ref_out.shape[0] != rows || buf_ref_out.shape[1] != cols) {
        throw std::runtime_error("reference_out dimensions do not match reference");
    }

    int* pos_ptr = static_cast<int*>(buf_pos.ptr);
    float* dyn_in_ptr = static_cast<float*>(buf_dyn_in.ptr);
    int* seq_ptr = static_cast<int*>(buf_seq.ptr);
    float* dyn_out_ptr = static_cast<float*>(buf_dyn_out.ptr);
    float* ref_out_ptr = static_cast<float*>(buf_ref_out.ptr);

    // Копируем входной dynamic в выходной (будем работать с ним)
    for (int i = 0; i < rows * cols; ++i) {
        dyn_out_ptr[i] = dyn_in_ptr[i];
    }

    // Предвычисляем сумму reference (она постоянна)
    float total_reference = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        total_reference += current_ref[i];
    }

    for (int step = 0; step < steps; ++step) {
        // Вычисляем коэффициент на основе текущего dynamic_out
        float total_dynamic = 0.0f;
        for (int i = 0; i < rows * cols; ++i) {
            total_dynamic += dyn_out_ptr[i];
        }
        float coeff = total_reference / (total_dynamic + 1e-10f);

        // Выполняем один шаг
        single_step(pos_ptr, current_ref.data(), dyn_out_ptr, seq_ptr, allow_multiple,
                    coeff, rows, cols, N, dynamic_step);
    }

    // Нормализуем dynamic_out к 0-255
    float max_val = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        if (dyn_out_ptr[i] > max_val) max_val = dyn_out_ptr[i];
    }
    if (max_val > 0) {
        for (int i = 0; i < rows * cols; ++i) {
            dyn_out_ptr[i] = (dyn_out_ptr[i] / max_val) * 255.0f;
        }
    } else {
        // Если все нули, оставляем как есть (уже нули)
    }

    // Копируем текущую эталонную карту в выходной массив reference_out
    for (int i = 0; i < rows * cols; ++i) {
        ref_out_ptr[i] = current_ref[i];
    }
}

PYBIND11_MODULE(agent_sim, m) {
    m.def("update_reference", &update_reference, "Update internal reference map from screen image");
    m.def("get_reference_dims", &get_reference_dims, "Get dimensions of the stored reference map");
    m.def("multi_step_agents", &multi_step_agents, "Perform multiple simulation steps, return normalized dynamic and reference");
}