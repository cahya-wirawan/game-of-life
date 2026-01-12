/**
 * @file gol.cpp
 * @brief Conway's Game of Life implementation using Bitboards and OpenMP.
 *
 * This program implements Conway's Game of Life using a bitboard representation
 * for memory efficiency and performance. It supports parallel execution using
 * OpenMP and sequential execution.
 *
 * Usage:
 *   ./gol --load <file> --save <file> --generations <n> [--measure] [--threads <n>] [--mode <omp|seq>]
 *
 * Arguments:
 *   --load <file>       Input file containing the initial state.
 *   --save <file>       Output file to save the final state.
 *   --generations <n>   Number of generations to simulate.
 *   --measure           (Optional) Measure and print execution time.
 *   --threads <n>       (Optional) Number of threads to use (default: 1).
 *   --mode <omp|seq>    (Optional) Execution mode: "omp" for OpenMP (default) or "seq" for sequential.
 */
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <omp.h>
#include "Timing.h"

/**
 * @brief Configuration structure for the Game of Life simulation.
 *
 * Holds command-line arguments and settings.
 */
struct Config
{
    std::string loadFile;     ///< Path to input file.
    std::string saveFile;     ///< Path to output file.
    int generations = 0;      ///< Number of generations to simulate.
    bool measure = false;     ///< Flag to enable performance measurement.
    int threads = 1;          ///< Number of OpenMP threads to use.
    std::string mode = "omp"; ///< Execution mode: "omp" or "seq".
};

/**
 * @brief Represents the game board using a bitboard.
 *
 * Each cell state is stored as a bit in a 64-bit integer (uint64_t).
 * Rows are padded to align with 64-bit word boundaries.
 */
struct Board
{
    int rows = 0;               ///< Number of rows in the grid.
    int cols = 0;               ///< Number of columns in the grid.
    int words_per_row = 0;      ///< Number of 64-bit words required per row.
    std::vector<uint64_t> grid; ///< Flat vector storing the bitboard.

    /**
     * @brief Resizes the board to the specified dimensions.
     *
     * @param r Number of rows.
     * @param c Number of columns.
     */
    void resize(int r, int c)
    {
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
    void set(int r, int c, bool val)
    {
        if (!val)
            return;
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
    bool get(int r, int c) const
    {
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
void print_duration(std::chrono::duration<double> d)
{
    auto total_seconds = d.count();
    int minutes = static_cast<int>(total_seconds / 60);
    double seconds = total_seconds - (minutes * 60);

    std::cout << std::setfill('0') << std::setw(2) << minutes << ":"
              << std::setfill('0') << std::setw(6) << std::fixed << std::setprecision(3) << seconds;
}

/**
 * @brief Computes the next generation of the game board.
 *
 * This function calculates the next state of the grid based on Conway's Game of Life rules.
 * It uses bitwise operations to process 64 cells at a time.
 * Parallelization is applied using OpenMP if enabled in config.
 *
 * @param current The current state of the board.
 * @param next The board to store the next state.
 * @param config Configuration settings (threads, mode).
 */
void evolve(Board &current, Board &next, const Config &config)
{
    if (next.grid.size() != current.grid.size())
    {
        next.resize(current.rows, current.cols);
    }
    else
    {
        next.rows = current.rows;
        next.cols = current.cols;
        next.words_per_row = current.words_per_row;
    }

    int rows = current.rows;
    int cols = current.cols;
    int wpr = current.words_per_row;
    int tail_bits = cols % 64;
    uint64_t tail_mask = (tail_bits == 0) ? ~0ULL : ((1ULL << tail_bits) - 1);

    bool run_parallel = (config.mode == "omp");
#pragma omp parallel for num_threads(config.threads) if (run_parallel)
    for (int r = 0; r < rows; ++r)
    {
        int r_prev = (r == 0) ? rows - 1 : r - 1;
        int r_next = (r == rows - 1) ? 0 : r + 1;

        const uint64_t *row_curr = &current.grid[r * wpr];
        const uint64_t *row_prev = &current.grid[r_prev * wpr];
        const uint64_t *row_next = &current.grid[r_next * wpr];
        uint64_t *row_out = &next.grid[r * wpr];

        for (int w = 0; w < wpr; ++w)
        {
            int w_prev = (w == 0) ? wpr - 1 : w - 1;
            int w_next = (w == wpr - 1) ? 0 : w + 1;

            auto get_left_shifted = [&](uint64_t val, uint64_t val_prev, int prev_idx)
            {
                uint64_t msb;
                if (prev_idx == wpr - 1 && tail_bits != 0)
                {
                    msb = (val_prev >> (tail_bits - 1)) & 1ULL;
                }
                else
                {
                    msb = val_prev >> 63;
                }
                return (val << 1) | msb;
            };

            auto get_right_shifted = [&](uint64_t val, uint64_t val_next, int curr_idx)
            {
                uint64_t lsb = val_next & 1ULL;
                if (curr_idx == wpr - 1 && tail_bits != 0)
                {
                    return (val >> 1) | (lsb << (tail_bits - 1));
                }
                else
                {
                    return (val >> 1) | (lsb << 63);
                }
            };

            uint64_t nw = row_prev[w];
            uint64_t cw = row_curr[w];
            uint64_t sw = row_next[w];

            uint64_t n0 = get_left_shifted(nw, row_prev[w_prev], w_prev);
            uint64_t n1 = nw;
            uint64_t n2 = get_right_shifted(nw, row_prev[w_next], w);

            uint64_t n3 = get_left_shifted(cw, row_curr[w_prev], w_prev);
            uint64_t n5 = get_right_shifted(cw, row_curr[w_next], w);

            uint64_t n6 = get_left_shifted(sw, row_next[w_prev], w_prev);
            uint64_t n7 = sw;
            uint64_t n8 = get_right_shifted(sw, row_next[w_next], w);

            uint64_t neighbors[] = {n0, n1, n2, n3, n5, n6, n7, n8};

            uint64_t c0 = 0, c1 = 0, c2 = 0;
            for (uint64_t x : neighbors)
            {
                uint64_t t0 = c0 ^ x;
                uint64_t c_0 = c0 & x;
                uint64_t t1 = c1 ^ c_0;
                uint64_t c_1 = c1 & c_0;
                uint64_t t2 = c2 ^ c_1;
                c0 = t0;
                c1 = t1;
                c2 = t2;
            }

            uint64_t result = (~c2 & c1) & (cw | c0);

            if (w == wpr - 1)
            {
                result &= tail_mask;
            }

            row_out[w] = result;
        }
    }
}

/**
 * @brief Main entry point of the program.
 *
 * Parses command-line arguments, sets up the board, runs the simulation,
 * and saves the result. Handles setup, computation, and finalization timing.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, 1 on error (invalid arguments or file I/O).
 */
int main(int argc, char *argv[])
{
    Config config;
    Timing *timing = Timing::getInstance();

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--load" && i + 1 < argc)
        {
            config.loadFile = argv[++i];
        }
        else if (arg == "--save" && i + 1 < argc)
        {
            config.saveFile = argv[++i];
        }
        else if (arg == "--generations" && i + 1 < argc)
        {
            config.generations = std::stoi(argv[++i]);
        }
        else if (arg == "--measure")
        {
            config.measure = true;
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            config.threads = std::stoi(argv[++i]);
        }
        else if (arg == "--mode" && i + 1 < argc)
        {
            config.mode = argv[++i];
        }
    }

    if (config.loadFile.empty() || config.saveFile.empty())
    {
        std::cerr << "Usage: " << argv[0] << " --load <file> --save <file> --generations <n> [--measure] [--threads <n>] [--mode <omp|seq>]\n";
        return 1;
    }

    timing->startSetup();

    Board board;
    {
        std::ifstream infile(config.loadFile);
        if (!infile.is_open())
        {
            std::cerr << "Error opening file: " << config.loadFile << "\n";
            return 1;
        }

        std::string line;
        if (std::getline(infile, line))
        {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            size_t commaPos = line.find(',');
            if (commaPos != std::string::npos)
            {
                try
                {
                    int cols = std::stoi(line.substr(0, commaPos));
                    int rows = std::stoi(line.substr(commaPos + 1));
                    board.resize(rows, cols);
                }
                catch (...)
                {
                    std::cerr << "Invalid header format.\n";
                    return 1;
                }
            }
        }

        for (int r = 0; r < board.rows; ++r)
        {
            if (std::getline(infile, line))
            {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                for (int c = 0; c < board.cols && c < (int)line.length(); ++c)
                {
                    if (line[c] == 'x')
                    {
                        board.set(r, c, true);
                    }
                }
            }
        }
    }

    timing->stopSetup();

    timing->startComputation();

    Board nextBoard;
    nextBoard.resize(board.rows, board.cols);

    for (int gen = 0; gen < config.generations; ++gen)
    {
        evolve(board, nextBoard, config);
        std::swap(board.grid, nextBoard.grid);
    }

    timing->stopComputation();

    timing->startFinalization();

    {
        std::ofstream outfile(config.saveFile);
        if (!outfile.is_open())
        {
            std::cerr << "Error opening file for writing: " << config.saveFile << "\n";
            return 1;
        }

        outfile << board.cols << "," << board.rows << "\n";
        for (int r = 0; r < board.rows; ++r)
        {
            for (int c = 0; c < board.cols; ++c)
            {
                outfile << (board.get(r, c) ? 'x' : '.');
            }
            outfile << "\n";
        }
    }

    timing->stopFinalization();
    std::string output = timing->getResults();

    if (config.measure)
    {
        std::cout << output << std::endl;
    }

    return 0;
}