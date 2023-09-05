using Pkg
Pkg.activate(".")


using Revise
using Gridap
using GridapDistributed
using IterativeSolvers
using LinearAlgebra
using PartitionedArrays
using MPI
using Parameters
using SparseArrays
using GridapPETSc
using GridapDistributed: Algebra
using Gridap.FESpaces
using Gridap.Arrays
using Gridap.CellData
#add GridapSolvers#block-preconditioners
using GridapSolvers.LinearSolvers

include("AnalyticalSolution.jl")
include("SpaceConditions.jl")
include("parallel_matrix.jl")
include("AddNewTags.jl")
include("HParam.jl")


params = Dict(
      :N => 10,
      :D => 2, #Dimension
      :order => 1, 
      :t0 => 0.0,
      :dt => 0.01,
      :tF => 0.2,
      :case => "TaylorGreen",
:θ => 0.5)

    diameter = 0.5 #0.5 [m] vortex dimension
  
    Vs = 1 #1[m/s]swirling speed
    Ua = 0.3 #0.3 [m/s]convective velocity in x
    Va = 0.2 #0.2 [m/s]convective velocity in y
    params[:ν] = 0.001 #0.001 m2/s 
  
    #Domain and mesh definition
    domain = (-diameter, diameter, -diameter, diameter)
    partition = (params[:N], params[:N])
    model = CartesianDiscreteModel(domain, partition; isperiodic=(true, true))
  
    # hf_gen!(params)
    velocity, pa, ωa = analytical_solution(diameter, Vs, Ua, Va, params[:ν])
    merge!(params, Dict(:u0 => velocity, :model => model))
    V, Q, U, P, Y, X, model = CreateTGSpaces(model, params, pa) #We update model with the new label of the center point
  
  
     
    degree = 4 * params[:order]
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    new_dict = Dict(:U => U,
      :P => P,
      :X => X,
      :Y => Y,
      :Ω => Ω,
      :dΩ => dΩ,
      :degree => degree,
      :force_params => nothing,
      :p0 => pa)
  
    merge!(params, new_dict)
  
    @unpack D, N, t0, dt, tF, ν, θ = params
    h = h_param(Ω, D)
  
    U0 = U(0.0)
    P0 = P(0.0)
    X0 = X(0.0)
  
  
    time_step = dt:dt:tF
  
  
  
    #SEGREGATED 
    uh0 = interpolate_everywhere(velocity(0.0), U(0.0))
    ph0 = interpolate_everywhere(pa(0.0), P(0.0))
  
    u_adv = interpolate_everywhere(velocity(0.0), U(0))
    
    trials = [U,P]
    tests = [V,Q]
  
    merge!(params, Dict(:h=>h,:dΩ=>dΩ))
  
    Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, 
    Mat_ML, Mat_inv_ML, Mat_S, vec_Auu, vec_Aup,vec_Apu, vec_App = initialize_matrices_and_vectors(trials,tests, 0.0, u_adv, params)
  
    vec_Ap = vec_App+ vec_Apu
    vec_Au = vec_Aup+ vec_Auu


# include("LinearUtilities.jl")

coeff = [2.1875, -2.1875, 1.3125, -0.3125]

function vel_kspsetup(ksp)
  pc = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[], GridapPETSc.PETSC.KSPGMRES)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[], pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[], GridapPETSc.PETSC.PCGAMG)
end

function pres_kspsetup(ksp)
  pc = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[], GridapPETSc.PETSC.KSPCG)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[], pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[], GridapPETSc.PETSC.PCGAMG)
end


    vec_pm = get_free_dof_values(ph0)
    vec_um = get_free_dof_values(uh0)
  
    lpdofs = length(vec_pm)
    ludofs = length(vec_um)
  
    vec_am = zeros(ludofs)
    vec_sum_pm = zeros(lpdofs)
    Δa_star = zeros(ludofs)
    Δpm1 = zeros(lpdofs)
    Δa = zeros(ludofs)
  
    b1 = zeros(ludofs)
    b2 = zeros(lpdofs)
  
  # vec_vec_um = create_ũ_vector(vec_um)
  

  M = 6
  

  solve_press = :gmres

  p_time = Float64[]
  u_time = Float64[]
  assembly_time = Float64[]

    for (it,tn) in enumerate(time_step)
      err = 1
      m = 0

      while m<M
  

  
          println("solving velocity")
          tu = @elapsed begin  
            @time b1 = -Mat_Auu * vec_um - Mat_Aup * vec_pm - Mat_ML * vec_am +
            Mat_Auu * dt * vec_am + (1 - θ) * Mat_Aup * vec_sum_pm + vec_Au
  
            Δa_star = LinearSolvers.allocate_col_vector(Mat_ML)
            GridapPETSc.with() do
              solver_v = PETScLinearSolver(vel_kspsetup)
              ss1 = symbolic_setup(solver_v, Mat_ML)
              ns1 = numerical_setup(ss1, Mat_ML)
              @time Gridap.solve!(Δa_star, ns1, b1)
             end #end Gridap
          end # end begin, solving velocity    
  
          println("solving pressure")
  
  
          tp = @elapsed begin
              # -Vec_A because changing sign in the continuity equations
            
             b2 .= Mat_Tpu * Δa_star + Mat_Apu * (vec_um + dt * Δa_star) + Mat_App * vec_pm + Mat_Tpu * vec_am - vec_Ap
             Δpm1 .= zeros(lpdofs)
             GridapPETSc.with() do
              solver_p = PETScLinearSolver(pres_kspsetup)
              ss2 = symbolic_setup(solver_p, Mat_S)
              ns2 = numerical_setup(ss2, Mat_S)
              Gridap.solve!(Δpm1,ns2,b2)
            end #end GridapPETSc
            
               
  
          end
          
          GridapPETSc.with() do
            GridapPETSc.GridapPETSc.gridap_petsc_gc()
          end
          
          push!(p_time,tp)
          push!(u_time,tu)

  
  
  
        println("update end")
  
          Δa .= Δa_star - θ .* Mat_inv_ML .* (Mat_Aup * Δpm1)
  
          vec_um .+=  dt * Δa
          vec_pm .+= Δpm1
  
  
          println("inner iter = $m")
          if m == 0
            vec_sum_pm .= Δpm1
            vec_am .= Δa
          else
            vec_sum_pm .+= Δpm1
            vec_am .+= Δa
  
          end
  
          m = m + 1
          ua_n = get_free_dof_values( interpolate_everywhere(velocity(tn), U(tn)))
          err = norm(ua_n - vec_um)
  
        println("error = $err")
  
      end #end while


      # update_ũ_vector!(vec_vec_um, vec_um)
      # vec_um .= update_ũ(vec_vec_um, coeff)
      # update_free_values!(u_adv, vec_um)

      println("update_matrices")

    a_time = @elapsed begin

    uh_tn = FEFunction(U(tn), vec_um)
    ph_tn = FEFunction(P(tn), vec_pm)

    Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, 
    Mat_ML, Mat_inv_ML, Mat_S, vec_Auu, vec_Aup,vec_Apu, vec_App = _matrices_and_vectors!(trials, tests, tn, uh_tn, params)
    vec_Ap = vec_App+ vec_Apu
    vec_Au = vec_Aup+ vec_Auu

      


  end #end begin

  push!(assembly_time,a_time)


      writevtk(Ω, "TG_segregated_$tn.vtu", cellfields = ["uh" => uh_tn, "uh_analytic"=> velocity(tn), "ph" => ph_tn, "ph_analytic"=> pa(tn)])
end #end for
  
  
using Statistics, Plots
    p_time = p_time[2:end]
    u_time = u_time[2:end]
    assembly_time =  assembly_time[2:end]
x_time = 1:1:(length(p_time))
# plotly()
plot(x_time,u_time, label = "u")
plot!(x_time,p_time, label = "p")

Statistics.mean(u_time) * M
Statistics.mean(p_time) * M
Statistics.mean(assembly_time)

# x_time