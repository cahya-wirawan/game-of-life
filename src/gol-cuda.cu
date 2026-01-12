/**
 * @file gol-cuda.cu
 * @brief Conway's Game of Life implementation using Bitboards and CUDA.
 *
 * This program implements Conway's Game of Life using a bitboard representation
 * and CUDA for GPU acceleration. It utilizes shared memory tiles to optimize
 * global memory access patterns.
 *
 * Usage:
 *   ./gol-cuda --load <file> --save <file> --generations <n> [--measure]
 *
 * Arguments:
 *   --load <file>       Input file containing the initial state.
 *   --save <file>       Output file to save the final state.
 *   --generations <n>   Number of generations to simulate.
 *   --measure           (Optional) Measure and print execution time.
 */
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <functional>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "Timing.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err_) << std::endl; \
        exit(1); \
    } \
}

/**
 * @brief Configuration structure for the Game of Life simulation.
 * 
 * Holds command-line arguments and settings.
 */
struct Config {
    std::string loadFile;     ///< Path to input file.
    std::string saveFile;     ///< Path to output file.
    int generations = 0;      ///< Number of generations to simulate.
    bool measure = false;     ///< Flag to enable performance measurement.
};

/**
 * @brief Represents the game board using a bitboard.
 * 
 * Each cell state is stored as a bit in a 64-bit integer (uint64_t).
 * Rows are padded to align with 64-bit word boundaries.
 */
struct Board {
    int rows = 0;           ///< Number of rows in the grid.
    int cols = 0;           ///< Number of columns in the grid.
    int words_per_row = 0;  ///< Number of 64-bit words required per row.
    std::vector<uint64_t> grid; ///< Flat vector storing the bitboard.

    /**
     * @brief Resizes the board to the specified dimensions.
     * 
     * @param r Number of rows.
     * @param c Number of columns.
     */
    void resize(int r, int c) {
        rows = r;
        cols = c;
        words_per_row = (c + 63) / 64;
        grid.assign(rows * words_per_row, 0);
    }

    /**
     * @brief Sets the state of a specific cell.
     * 
     * @param r Row index.
     * @param c Column index.
     * @param val The state to set (true for alive, false for dead).
     */
    void set(int r, int c, bool val) {
        if (!val) return;
        int word_idx = c / 64;
        int bit_idx = c % 64;
        grid[r * words_per_row + word_idx] |= (1ULL << bit_idx);
    }

    /**
     * @brief Gets the state of a specific cell.
     * 
     * @param r Row index.
     * @param c Column index.
     * @return true if the cell is alive, false otherwise.
     */
    bool get(int r, int c) const {
        int word_idx = c / 64;
        int bit_idx = c % 64;
        return (grid[r * words_per_row + word_idx] >> bit_idx) & 1ULL;
    }
};




/**
 * @brief Prints a duration in a formatted string (MM:SS.sss).
 * 
 * @param d The duration to print.
 */
void print_duration(std::chrono::duration<double> d) {
    auto total_seconds = d.count();
    int minutes = static_cast<int>(total_seconds / 60);
    double seconds = total_seconds - (minutes * 60);

    std::cout << std::setfill('0') << std::setw(2) << minutes << ":"
              << std::setfill('0') << std::setw(6) << std::fixed << std::setprecision(3) << seconds;
}


#define BLOCK_W 16
#define BLOCK_H 16

/**
 * @brief CUDA kernel to compute the next generation of the game board.
 * 
 * Uses shared memory tiling to load a block of the board plus its halo.
 * Computes the next state for the block using bitwise operations.
 * 
 * @param g_in Pointer to the input grid in global memory.
 * @param g_out Pointer to the output grid in global memory.
 * @param rows Number of rows in the grid.
 * @param cols Number of columns in the grid.
 * @param wpr Number of 64-bit words per row.
 */
__global__ void evolve_kernel(const uint64_t* __restrict__ g_in, uint64_t* __restrict__ g_out, int rows, int cols, int wpr) {

    __shared__ uint64_t tile[BLOCK_H + 2][BLOCK_W + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int r_base = blockIdx.y * BLOCK_H;
    const int w_base = blockIdx.x * BLOCK_W;

    // Each thread in the 16x16 block will load a point into the 18x18 shared tile.
    // To cover all 324 points of the tile, some threads will loop and load a second point.
    for (int i = ty * BLOCK_W + tx; i < (BLOCK_H + 2) * (BLOCK_W + 2); i += BLOCK_H * BLOCK_W) {
        int tile_y = i / (BLOCK_W + 2);
        int tile_x = i % (BLOCK_W + 2);

        // Calculate the source coordinates in the global grid, including the halo region
        int r = r_base + tile_y - 1;
        int w = w_base + tile_x - 1;

        // If the source row 'r' is outside the grid, it's a padding row (top or bottom halo)
        // and we don't need to load anything. But for toroidal wrapping, we calculate the wrapped row.
        // For the last block of rows, 'r' can be >= 'rows', and must wrap to the top of the grid.
        int r_wrapped = (r + rows) % rows;

        // If the source word 'w' is outside the grid, it wraps around.
        int w_wrapped = (w + wpr) % wpr;

        // Load if the original 'r' was inside the valid board dimensions for this block,
        // or if it's a valid halo row for an active block.
        // A simpler approach is to check if the source coordinates are valid before loading.
        if (r >= 0 && r < rows && w >= 0 && w < wpr) {
             tile[tile_y][tile_x] = g_in[r * wpr + w];
        } else {
             // For any coordinate that falls outside the original board, load the toroidally wrapped data.
             tile[tile_y][tile_x] = g_in[r_wrapped * wpr + w_wrapped];
        }
    }


    __syncthreads();

    const int r = r_base + ty;
    const int w = w_base + tx;

    if (r >= rows || w >= wpr) return;

    // --- Computation using shared memory tile ---

    uint64_t n_word = tile[ty][tx + 1];
    uint64_t c_word = tile[ty + 1][tx + 1];
    uint64_t s_word = tile[ty + 2][tx + 1];

    uint64_t n0, n2, n3, n5, n6, n8;

    // Default bit-shifting for non-boundary words
    n0 = (n_word << 1) | (tile[ty][tx] >> 63);
    n2 = (n_word >> 1) | (tile[ty][tx + 2] << 63);
    n3 = (c_word << 1) | (tile[ty + 1][tx] >> 63);
    n5 = (c_word >> 1) | (tile[ty + 1][tx + 2] << 63);
    n6 = (s_word << 1) | (tile[ty + 2][tx] >> 63);
    n8 = (s_word >> 1) | (tile[ty + 2][tx + 2] << 63);

    int tail_bits = cols % 64;
    if (tail_bits == 0) tail_bits = 64;

    // Special handling for bit-shifting at the global board edges
    if (w == 0) {
        uint64_t msb_n = (tile[ty][tx] >> (tail_bits - 1)) & 1ULL;
        uint64_t msb_c = (tile[ty + 1][tx] >> (tail_bits - 1)) & 1ULL;
        uint64_t msb_s = (tile[ty + 2][tx] >> (tail_bits - 1)) & 1ULL;
        n0 = (n_word << 1) | msb_n;
        n3 = (c_word << 1) | msb_c;
        n6 = (s_word << 1) | msb_s;
    }
    if (w == wpr - 1) {
        uint64_t lsb_n = tile[ty][tx + 2] & 1ULL;
        uint64_t lsb_c = tile[ty + 1][tx + 2] & 1ULL;
        uint64_t lsb_s = tile[ty + 2][tx + 2] & 1ULL;
        n2 = (n_word >> 1) | (lsb_n << (tail_bits - 1));
        n5 = (c_word >> 1) | (lsb_c << (tail_bits - 1));
        n8 = (s_word >> 1) | (lsb_s << (tail_bits - 1));
    }
    
    uint64_t neighbors[] = {n0, n_word, n2, n3, n5, n6, s_word, n8};

    uint64_t c0 = 0, c1 = 0, c2 = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t x = neighbors[i];
        uint64_t t0 = c0 ^ x;
        uint64_t c_0 = c0 & x;
        uint64_t t1 = c1 ^ c_0;
        uint64_t c_1 = c1 & c_0;
        uint64_t t2 = c2 ^ c_1;
        c0 = t0; c1 = t1; c2 = t2;
    }

    uint64_t result = (~c2 & c1) & (c_word | c0);

    if (w == wpr - 1 && cols % 64 != 0) {
        uint64_t tail_mask = (1ULL << (cols % 64)) - 1;
        result &= tail_mask;
    }

    g_out[r * wpr + w] = result;
}

/**
 * @brief Launches the CUDA kernel to compute the next generation.
 * 
 * Configures the grid and block dimensions and launches the kernel.
 * 
 * @param board The Board object containing grid dimensions.
 * @param d_current Device pointer to the current grid state.
 * @param d_next Device pointer to the next grid state.
 */
void evolve(Board& board, uint64_t* d_current, uint64_t* d_next) {
    int rows = board.rows;
    int cols = board.cols;
    int wpr = board.words_per_row;

    dim3 threadsPerBlock(BLOCK_W, BLOCK_H);
    dim3 numBlocks((wpr + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_current, d_next, rows, cols, wpr);
}

/**
 * @brief Main entry point of the program.
 * 
 * Parses command-line arguments, initializes the board, manages GPU memory,
 * runs the simulation loop, and saves the results.
 * 
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, 1 on error.
 */
int main(int argc, char* argv[]) {
    Config config;
    Timing* timing = Timing::getInstance();
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--load" && i + 1 < argc) {
            config.loadFile = argv[++i];
        } else if (arg == "--save" && i + 1 < argc) {
            config.saveFile = argv[++i];
        } else if (arg == "--generations" && i + 1 < argc) {
            config.generations = std::stoi(argv[++i]);
        } else if (arg == "--measure") {
            config.measure = true;
        }
    }

    if (config.loadFile.empty() || config.saveFile.empty()) {
        std::cerr << "Usage: " << argv[0] << " --load <file> --save <file> --generations <n> [--measure]" << std::endl;
        return 1;
    }

    timing->startSetup();

    Board board;
    {
        std::ifstream infile(config.loadFile);
        if (!infile.is_open()) {
            std::cerr << "Error opening file: " << config.loadFile << std::endl;
            return 1;
        }
        
        std::string line;
        if (std::getline(infile, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            size_t commaPos = line.find(',');
            if (commaPos != std::string::npos) {
                try {
                    int cols = std::stoi(line.substr(0, commaPos));
                    int rows = std::stoi(line.substr(commaPos + 1));
                    board.resize(rows, cols);
                } catch (...) {
                     std::cerr << "Invalid header format." << std::endl;
                     return 1;
                }
            }
        }

        for (int r = 0; r < board.rows; ++r) {
            if (std::getline(infile, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                for (int c = 0; c < board.cols && c < (int)line.length(); ++c) {
                    if (line[c] == 'x') {
                        board.set(r, c, true);
                    }
                }
            }
        }
    }

    timing->stopSetup();


    uint64_t *d_board_grid, *d_next_board_grid;
    size_t board_size = board.grid.size() * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc(&d_board_grid, board_size));
    CUDA_CHECK(cudaMalloc(&d_next_board_grid, board_size));
    CUDA_CHECK(cudaMemcpy(d_board_grid, board.grid.data(), board_size, cudaMemcpyHostToDevice));

    timing->startComputation();

    for (int gen = 0; gen < config.generations; ++gen) {
        evolve(board, d_board_grid, d_next_board_grid);
        std::swap(d_board_grid, d_next_board_grid); 
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    timing->stopComputation();

    CUDA_CHECK(cudaMemcpy(board.grid.data(), d_board_grid, board_size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_board_grid));
    CUDA_CHECK(cudaFree(d_next_board_grid));

    timing->startFinalization();

    {
        std::ofstream outfile(config.saveFile);
        if (!outfile.is_open()) {
            std::cerr << "Error opening file for writing: " << config.saveFile << std::endl;
            return 1;
        }

        outfile << board.cols << "," << board.rows << "\n";
        for (int r = 0; r < board.rows; ++r) {
            for (int c = 0; c < board.cols; ++c) {
                outfile << (board.get(r, c) ? 'x' : '.');
            }
            outfile << "\n";
        }
    }

    timing->stopFinalization();
    std::string output = timing->getResults();

    if (config.measure) {
        std::cout << output << std::endl;
    }

    return 0;
}
