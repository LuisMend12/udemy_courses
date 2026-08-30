#!/usr/bin/env julia
# Brownian motion (Wiener process) simulator.
#
# Simulates N independent particles undergoing d-dimensional Brownian
# motion via the discretized SDE:  x(t+dt) = x(t) + sqrt(2*D*dt) * N(0,1)
#
# Run:   julia brownian_motion.jl --particles 200 --steps 1000 --dt 0.01 --D 1.0 --dim 2 --out trajectories.csv
#
# Port of brownian_motion.cpp. The RNG stream will not match the C++ binary
# bit-for-bit -- Julia's default RNG (Xoshiro256++) and normal-variate
# algorithm differ from libstdc++'s mt19937_64 + std::normal_distribution --
# but both draw from the same N(0,1) distribution, so the simulated process
# and its statistics (e.g. the recovered diffusion coefficient) match.
#
# Particle indices in the output CSV are 0-indexed to match the original
# tool's format, even though Julia arrays are 1-indexed internally.

using Random
using Printf

mutable struct Config
    particles::Int
    steps::Int
    dt::Float64
    D::Float64          # diffusion coefficient
    dim::Int            # spatial dimension (1, 2, or 3)
    seed::UInt64
    out::String
end

Config() = Config(100, 1000, 0.01, 1.0, 2, rand(RandomDevice(), UInt64), "trajectories.csv")

function print_usage(prog::AbstractString)
    println("""
    Usage: $prog [options]
      --particles N   number of particles (default 100)
      --steps N       number of time steps (default 1000)
      --dt X          time step size (default 0.01)
      --D X           diffusion coefficient (default 1.0)
      --dim N         spatial dimension: 1, 2, or 3 (default 2)
      --seed N        RNG seed (default random)
      --out FILE      output CSV path (default trajectories.csv)
      --help          show this message""")
end

function parse_config(args::Vector{String})
    cfg = Config()
    prog = PROGRAM_FILE == "" ? "brownian_motion.jl" : basename(PROGRAM_FILE)

    i = 1
    next_val = (name) -> begin
        if i + 1 > length(args)
            println(stderr, "Missing value for $name")
            exit(1)
        end
        i += 1
        args[i]
    end

    while i <= length(args)
        arg = args[i]
        if arg == "--particles"
            cfg.particles = parse(Int, next_val("--particles"))
        elseif arg == "--steps"
            cfg.steps = parse(Int, next_val("--steps"))
        elseif arg == "--dt"
            cfg.dt = parse(Float64, next_val("--dt"))
        elseif arg == "--D"
            cfg.D = parse(Float64, next_val("--D"))
        elseif arg == "--dim"
            cfg.dim = parse(Int, next_val("--dim"))
        elseif arg == "--seed"
            cfg.seed = parse(UInt64, next_val("--seed"))
        elseif arg == "--out"
            cfg.out = next_val("--out")
        elseif arg == "--help" || arg == "-h"
            print_usage(prog)
            exit(0)
        else
            println(stderr, "Unknown argument: $arg")
            print_usage(prog)
            exit(1)
        end
        i += 1
    end

    if cfg.dim < 1 || cfg.dim > 3
        println(stderr, "--dim must be 1, 2, or 3")
        exit(1)
    end
    if cfg.particles <= 0 || cfg.steps <= 0 || cfg.dt <= 0.0 || cfg.D < 0.0
        println(stderr, "particles/steps/dt must be positive, D must be non-negative")
        exit(1)
    end
    return cfg
end

const AXIS_NAMES = ("x", "y", "z")

function simulate(cfg::Config)
    rng = Xoshiro(cfg.seed)
    step_std = sqrt(2.0 * cfg.D * cfg.dt)

    # positions[k, p] = current coordinate k of particle p
    positions = zeros(Float64, cfg.dim, cfg.particles)

    io = open(cfg.out, "w")
    print(io, "step,time,particle")
    for k in 1:cfg.dim
        print(io, ",", AXIS_NAMES[k])
    end
    print(io, "\n")

    function write_row(step, t, p)
        print(io, step, ",", t, ",", p - 1)
        for k in 1:cfg.dim
            print(io, ",", positions[k, p])
        end
        print(io, "\n")
    end

    for p in 1:cfg.particles
        write_row(0, 0.0, p)
    end

    for step in 1:cfg.steps
        t = step * cfg.dt
        for p in 1:cfg.particles
            for k in 1:cfg.dim
                positions[k, p] += step_std * randn(rng)
            end
            write_row(step, t, p)
        end
    end
    close(io)

    # Sanity check: estimate D from the mean squared displacement at the
    # final time and compare against the configured value.
    msd = 0.0
    for p in 1:cfg.particles
        sq = 0.0
        for k in 1:cfg.dim
            sq += positions[k, p]^2
        end
        msd += sq
    end
    msd /= cfg.particles
    final_t = cfg.steps * cfg.dt
    D_est = msd / (2.0 * cfg.dim * final_t)

    @printf("Simulated %d particles, %d steps, dim=%d, dt=%g, D=%g, seed=%d\n",
            cfg.particles, cfg.steps, cfg.dim, cfg.dt, cfg.D, cfg.seed)
    @printf("Final MSD = %g  ->  estimated D = %g (configured D = %g)\n",
            msd, D_est, cfg.D)
    println("Wrote trajectories to ", cfg.out)
end

function main()
    cfg = parse_config(ARGS)
    simulate(cfg)
end

main()
