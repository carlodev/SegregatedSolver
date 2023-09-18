using SegregatedSolver
using Gridap, GridapGmsh

mesh_file_path = joinpath(@__DIR__, "../models", "DU89_2D_A1_v2.msh")
model = GmshDiscreteModel(mesh_file_path)

writevtk(model, "Airfoil_v2")

using Plots
dt = 0.1
time_step = collect(0.0:dt:1)
actual_time = [0.0]
actual_time_step = [0.0]
val = rand(10000)

function run_time(time_step)
    i = 1

    for tn in time_step[1:end-1]
        nref = 1
        time_compute = tn
        t0 = tn
        while time_compute <  tn + dt
            time_try = t0 + dt/nref
            i = i +1
            if  val[i] < 0.75
                t0 = time_try
                time_compute = time_try
                push!(actual_time,time_compute)
                push!(actual_time_step, dt/nref)

            else
                nref = nref*2
            end

        end

    end
end

run_time(time_step)
actual_time
actual_time_step

plotly()
# scatter(actual_time,actual_time_step,legend=:outertopright)
sum(actual_time_step)
