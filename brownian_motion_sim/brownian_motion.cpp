// Brownian motion (Wiener process) simulator.
//
// Simulates N independent particles undergoing d-dimensional Brownian
// motion via the discretized SDE:  x(t+dt) = x(t) + sqrt(2*D*dt) * N(0,1)
//
// Build:   g++ -O2 -std=c++17 -o brownian_motion brownian_motion.cpp
// Run:     ./brownian_motion --particles 200 --steps 1000 --dt 0.01 --D 1.0 --dim 2 --out trajectories.csv

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

struct Config {
    int particles = 100;
    int steps = 1000;
    double dt = 0.01;
    double D = 1.0;      // diffusion coefficient
    int dim = 2;         // spatial dimension (1, 2, or 3)
    unsigned seed = std::random_device{}();
    std::string out = "trajectories.csv";
};

static void printUsage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "  --particles N   number of particles (default 100)\n"
        "  --steps N       number of time steps (default 1000)\n"
        "  --dt X          time step size (default 0.01)\n"
        "  --D X           diffusion coefficient (default 1.0)\n"
        "  --dim N         spatial dimension: 1, 2, or 3 (default 2)\n"
        "  --seed N        RNG seed (default random)\n"
        "  --out FILE      output CSV path (default trajectories.csv)\n"
        "  --help          show this message\n";
}

static bool parseArgs(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--particles") cfg.particles = std::stoi(next("--particles"));
        else if (arg == "--steps") cfg.steps = std::stoi(next("--steps"));
        else if (arg == "--dt") cfg.dt = std::stod(next("--dt"));
        else if (arg == "--D") cfg.D = std::stod(next("--D"));
        else if (arg == "--dim") cfg.dim = std::stoi(next("--dim"));
        else if (arg == "--seed") cfg.seed = static_cast<unsigned>(std::stoul(next("--seed")));
        else if (arg == "--out") cfg.out = next("--out");
        else if (arg == "--help" || arg == "-h") { printUsage(argv[0]); std::exit(0); }
        else { std::cerr << "Unknown argument: " << arg << "\n"; printUsage(argv[0]); return false; }
    }
    if (cfg.dim < 1 || cfg.dim > 3) {
        std::cerr << "--dim must be 1, 2, or 3\n";
        return false;
    }
    if (cfg.particles <= 0 || cfg.steps <= 0 || cfg.dt <= 0.0 || cfg.D < 0.0) {
        std::cerr << "particles/steps/dt must be positive, D must be non-negative\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    Config cfg;
    if (!parseArgs(argc, argv, cfg)) return 1;

    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double stepStd = std::sqrt(2.0 * cfg.D * cfg.dt);

    // positions[p][k] = current coordinate k of particle p
    std::vector<std::vector<double>> positions(cfg.particles, std::vector<double>(cfg.dim, 0.0));

    std::ofstream csv(cfg.out);
    if (!csv) {
        std::cerr << "Failed to open output file: " << cfg.out << "\n";
        return 1;
    }
    csv << "step,time,particle";
    static const char* axisNames[3] = {"x", "y", "z"};
    for (int k = 0; k < cfg.dim; ++k) csv << "," << axisNames[k];
    csv << "\n";

    auto writeRow = [&](int step, double t, int p) {
        csv << step << "," << t << "," << p;
        for (int k = 0; k < cfg.dim; ++k) csv << "," << positions[p][k];
        csv << "\n";
    };

    for (int p = 0; p < cfg.particles; ++p) writeRow(0, 0.0, p);

    for (int step = 1; step <= cfg.steps; ++step) {
        double t = step * cfg.dt;
        for (int p = 0; p < cfg.particles; ++p) {
            for (int k = 0; k < cfg.dim; ++k) {
                positions[p][k] += stepStd * normal(rng);
            }
            writeRow(step, t, p);
        }
    }
    csv.close();

    // Sanity check: estimate D from the mean squared displacement at the
    // final time and compare against the configured value.
    double msd = 0.0;
    for (int p = 0; p < cfg.particles; ++p) {
        double sq = 0.0;
        for (int k = 0; k < cfg.dim; ++k) sq += positions[p][k] * positions[p][k];
        msd += sq;
    }
    msd /= cfg.particles;
    double finalT = cfg.steps * cfg.dt;
    double D_est = msd / (2.0 * cfg.dim * finalT);

    std::cout << "Simulated " << cfg.particles << " particles, " << cfg.steps
              << " steps, dim=" << cfg.dim << ", dt=" << cfg.dt
              << ", D=" << cfg.D << ", seed=" << cfg.seed << "\n";
    std::cout << "Final MSD = " << msd << "  ->  estimated D = " << D_est
              << " (configured D = " << cfg.D << ")\n";
    std::cout << "Wrote trajectories to " << cfg.out << "\n";

    return 0;
}
