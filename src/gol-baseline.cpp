#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "Timing.h"

struct Config {
    std::string loadFile;
    std::string saveFile;
    int generations = 0;
    bool measure = false;
};

struct Board {
    int rows = 0;
    int cols = 0;
    std::vector<bool> grid;

    bool get(int r, int c) const {
        // Wrap-around logic
        int wrapped_r = (r + rows) % rows;
        int wrapped_c = (c + cols) % cols;
        return grid[wrapped_r * cols + wrapped_c];
    }

    void set(int r, int c, bool val) {
        grid[r * cols + c] = val;
    }
    
    void resize(int r, int c) {
        rows = r;
        cols = c;
        grid.resize(r * c, false);
    }
};

void print_duration(std::chrono::duration<double> d) {
    // Format: 00:00.000 (MM:SS.ms) assuming the example 00:00.002 matches this.
    // However, 00:00.002 is very small. It might be HH:MM:SS or MM:SS.mmm
    // Given the context of a CLI program, MM:SS.mmm is logical for fast operations.
    
    auto total_seconds = d.count();
    int minutes = static_cast<int>(total_seconds / 60);
    double seconds = total_seconds - (minutes * 60);

    std::cout << std::setfill('0') << std::setw(2) << minutes << ":"
              << std::setfill('0') << std::setw(6) << std::fixed << std::setprecision(3) << seconds;
}

int count_neighbors(const Board& board, int r, int c) {
    int count = 0;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;
            if (board.get(r + dr, c + dc)) {
                count++;
            }
        }
    }
    return count;
}

void evolve(Board& current, Board& next) {
    next.rows = current.rows;
    next.cols = current.cols;
    // We assume next is already sized correctly or we resize it.
    // Ideally we avoid reallocating every step.
    if (next.grid.size() != current.grid.size()) {
        next.resize(current.rows, current.cols);
    }

    for (int r = 0; r < current.rows; ++r) {
        for (int c = 0; c < current.cols; ++c) {
            int neighbors = count_neighbors(current, r, c);
            bool alive = current.get(r, c);
            bool next_alive = false;

            if (alive) {
                if (neighbors == 2 || neighbors == 3) {
                    next_alive = true;
                }
            } else {
                if (neighbors == 3) {
                    next_alive = true;
                }
            }
            next.set(r, c, next_alive);
        }
    }
}

int main(int argc, char* argv[]) {
    Config config;
    Timing* timing = Timing::getInstance();

    // Simple argument parsing
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

    // Setup.
    timing->startSetup();

    // Load Board
    Board board;
    {
        std::ifstream infile(config.loadFile);
        if (!infile.is_open()) {
            std::cerr << "Error opening file: " << config.loadFile << std::endl;
            return 1;
        }
        
        std::string line;
        if (std::getline(infile, line)) {
            // Remove any trailing carriage returns (e.g. from Windows formatted files)
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            size_t commaPos = line.find(',');
            if (commaPos != std::string::npos) {
                try {
                    int cols = std::stoi(line.substr(0, commaPos));
                    int rows = std::stoi(line.substr(commaPos + 1));
                    // std::cout << "Rows: " << rows << " Cols: " << cols << std::endl;
                    board.resize(rows, cols);
                } catch (...) {
                     std::cerr << "Invalid header format." << std::endl;
                     return 1;
                }
            } else {
                 std::cerr << "Invalid header format." << std::endl;
                 return 1;
            }
        }

        for (int r = 0; r < board.rows; ++r) {
            if (std::getline(infile, line)) {
                // Handle potential windows line endings
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                for (int c = 0; c < board.cols && c < (int)line.length(); ++c) {
                    if (line[c] == 'x') {
                        board.set(r, c, true);
                    }
                }
            }
        }
    }

    timing->stopSetup();

    // Calculation.
    timing->startComputation();

    // Simulation
    Board nextBoard;
    nextBoard.resize(board.rows, board.cols);

    // Double buffering pointers/references could be optimized, but swapping logic is safer
    for (int gen = 0; gen < config.generations; ++gen) {
        evolve(board, nextBoard);
        // Swap buffers
        std::swap(board.grid, nextBoard.grid); 
        // rows/cols remain same
    }

    timing->stopComputation();

    // Finalization.
    timing->startFinalization();

    // Save Board
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
    // Save this string.
    std::string output = timing->getResults();

    if (config.measure) {
        std::cout << output << std::endl;
    }

    return 0;
}
