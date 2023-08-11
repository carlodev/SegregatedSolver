using Pkg
Pkg.activate(".")
using Revise
using GridapDistributed: Algebra
include("../../ExoFlow.jl")
include("../SpaceConditions.jl")
include("../AnalyticalSolution.jl")

include("parallel_matrix.jl")
using Statistics, LinearAlgebra

#Domain and mesh definition

val(x) = x
val(x::Gridap.Fields.ForwardDiff.Dual) = x.value
function τsu(u, h)
  r = 1
  τ₂ = h^2 / (4 * ν)
  τ₃ = dt / 2

  u = val(norm(u))
  if iszero(u)
    println("is zero")
    return (1 / τ₂^r + 1 / τ₃^r)^(-1 / r)
  end
  τ₁ = h / (2 * u)
  return (1 / τ₁^r + 1 / τ₂^r + 1 / τ₃^r)^(-1 / r)

end

function τb(u, h)
  return (u ⋅ u) * τsu(u, h)
end








partition = (2, 2)
backend = SequentialBackend() #MPIBackend()



function petsc_options()
  return ""
end




with_backend(backend, partition) do parts
  if get_backend(parts) == MPIBackend()
    comm = MPI.COMM_WORLD
    #To avoid multiple printing of the same line in parallel
    if MPI.Comm_rank(comm) != 0
      redirect_stderr(devnull)
      redirect_stdout(devnull)
    end
  end

  params = Dict(
    :N => 10,
    :D => 2, #Dimension
    :order => 1, :t0 => 0.0,
    :dt => 0.01,
    :tF => 0.1,
    :t_endramp => 1.0, :case => "TaylorGreen",
    :solver => :petsc,
    :method => :VMS,
    :ode_method => :AlphaMethod,
    :θ => 1.0,
    :ρ∞ => 0.8,
    :Re => 100_000,
    :c => 1, #chord lenght [m], used for naca (=1), cylinder (=0.1), liddriven = (1), 0.5
    :u_in => 1.0,  # =1.0 for lid driven 
    :periodic => false, :printmodel => false, :mesh_gen => false, :linear => false,
    :steady => false, :debug_mode => false, :mesh_file => "NACA0012_2D_improved.msh",
    :Cᵢ => [4, 36],
    :options => petsc_options(),
    :nls_trace => true,
    :nls_iter => 20, :ν => 1.0e-5,  #channel = 0.0001472, 
    :ρ => 1.0, #kg/m3 density
    :body_force => 0.0, #channel = 0.00337204
    :np_x => 2, #number of processors in X
    :np_y => 2, #number of processors in Y
    :np_z => 1, #number of processors in Z
    :restart => false,
    :restart_file => "Du89_2p1.csv",
    :TI => 0.01,)
  params = initialize_parameters(params)
  diameter = 0.5 #0.5 [m] vortex dimension

  Vs = 1 #1[m/s]swirling speed
  Ua = 0.3 #0.3 [m/s]convective velocity in x
  Va = 0.2 #0.2 [m/s]convective velocity in y
  params[:ν] = 0.001 #0.001 m2/s 

  #Domain and mesh definition
  domain = (-diameter, diameter, -diameter, diameter)
  partition = (params[:N], params[:N])
  model = CartesianDiscreteModel(parts, domain, partition; isperiodic=(true, true))

  hf_gen!(params)
  velocity, pa, ωa = analytical_solution(diameter, Vs, Ua, Va, params[:ν])
  merge!(params, Dict(:u0 => velocity, :model => model))
  V, Q, U, P, Y, X, model = CreateTGSpaces(model, params, pa) #We update model with the new label of the center point


  println("spaces created")



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
  global dt = 0.01
  global ν
  h = h_param(Ω, D)

  U0 = U(0.0)
  P0 = P(0.0)
  X0 = X(0.0)


  #Assemble Matrices

  u_dofs = size(get_free_dof_ids(U0))[1]
  p_dofs = size(get_free_dof_ids(P0))[1]


  as = assembly_spaces(Q, V, P0, U0)

  time_step = dt:dt:100*dt


  cconv(u_adv, ∇u) = u_adv ⋅ (∇u)

  Tuu(u, v) = ∫((v + τsu ∘ (u_adv, h) * (cconv ∘ (u_adv, ∇(v)))) ⊙ u)dΩ
  Tpu(u, q) = ∫((τsu ∘ (u_adv, h)) * (∇(q)) ⊙ u)dΩ
  Auu1(u, v) = ∫(ν * ∇(v) ⊙ ∇(u) + (cconv ∘ (u_adv, ∇(u))) ⋅ v + ((τsu ∘ (u_adv, h)) * (cconv ∘ (u_adv, ∇(v)))) ⊙ (cconv ∘ (u_adv, ∇(u))))dΩ

  Auu2(u, v) = ∫(((τb ∘ (u_adv, h)) * (∇ ⋅ v)) ⊙ (∇ ⋅ u) + 0.5 .* u_adv ⋅ (v + (τsu ∘ (u_adv, h)) * (cconv ∘ (u_adv, ∇(v)))) ⋅ (∇ ⋅ u))dΩ

  Auu(u, v) = Auu1(u, v) + Auu2(u, v)

  Aup(p, v) = ∫(-(∇ ⋅ v) * p + ((τsu ∘ (u_adv, h)) * (cconv ∘ (u_adv, ∇(v)))) ⊙ ∇(p))dΩ

  Apu(u, q) = ∫(q * (∇ ⋅ u) + 0.5 .* (τsu ∘ (u_adv, h)) ⋅ (∇(q)) ⋅ u_adv ⋅ (∇ ⋅ u) + (τsu ∘ (u_adv, h)) * (∇(q)) ⊙ (cconv ∘ (u_adv, ∇(u))))dΩ

  App(p, q) = ∫(((τsu ∘ (u_adv, h)) * ∇(q)) ⊙ (∇(p)))dΩ

  ML(u, v) = Tuu(u, v) + (θ * dt) * Auu(u, v)

  S(p, q) = -θ * ∫((dt .+ τsu ∘ (u_adv, h)) ⋅ ((∇(q)') ⊙ (∇(p))))dΩ

  function define_vectors(t, V, Q, P, U)
    rhs(v) = 0.0
    vec_Auu = get_vector(AffineFEOperator(Auu, rhs, U(t), V))
    vec_Aup = get_vector(AffineFEOperator(Aup, rhs, P(t), V))
    vec_Apu = get_vector(AffineFEOperator(Apu, rhs, U(t), Q))
    vec_App = get_vector(AffineFEOperator(App, rhs, P(t), Q))

    return vec_App + vec_Apu, vec_Aup + vec_Auu
  end

  function define_matrices(t, V, Q, P, U)
    rhs(v) = 0.0
    Mat_Auu = get_matrix(AffineFEOperator(Auu, rhs, U(t), V))
    Mat_Aup = get_matrix(AffineFEOperator(Aup, rhs, P(t), V))
    Mat_Apu = get_matrix(AffineFEOperator(Apu, rhs, U(t), Q))
    Mat_App = get_matrix(AffineFEOperator(App, rhs, P(t), Q))

    Mat_Tuu = get_matrix(AffineFEOperator(Tuu, rhs, U(t), V))
    Mat_Tpu = get_matrix(AffineFEOperator(Tpu, rhs, U(t), Q))

    Mat_ML = get_matrix(AffineFEOperator(ML, rhs, U(t), V))
    Mat_S = get_matrix(AffineFEOperator(S, rhs, P(t), Q))

    return Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, Mat_ML, Mat_S
  end


  function define_vec_A(t, V, Q, P, U)
    rhs(v) = 0.0
    vec_Auu = get_vector(AffineFEOperator(Auu, rhs, U(t), V))
    vec_Aup = get_vector(AffineFEOperator(Aup, rhs, P(t), V))
    vec_Apu = get_vector(AffineFEOperator(Apu, rhs, U(t), Q))
    vec_App = get_vector(AffineFEOperator(App, rhs, P(t), Q))

    return vec_App + vec_Apu, vec_Aup + vec_Auu
  end

  #SEGREGATED 
  uh0 = interpolate_everywhere(velocity(0.0), U(0.0))
  ph0 = interpolate_everywhere(pa(0.0), P(0.0))
  uh1 = interpolate_everywhere(velocity(dt), U(0.0))
  ph1 = interpolate_everywhere(pa(dt), P(0.0))


  u_adv = interpolate_everywhere(velocity(0.0), U(0))
  Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, Mat_ML, Mat_S = define_matrices(0.0, V, Q, P, U)
  # Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, Mat_ML, Mat_inv_ML = initialize_Mat(as, Tuu, Auu, Aup, Apu, App, ML, S)

  # update_Mat!(Mat_Tuu, Mat_Tpu, Mat_Auu,  Mat_Aup,  Mat_Apu,  Mat_App, Mat_ML, as)

  Mat_inv_ML = inv_lump_vel_mass!(Mat_ML)


  #Vectors initialization

  vec_Ap, vec_Au = define_vectors(0.0, V, Q, P, U)

  vec_pm = GridapDistributed.change_ghost(get_free_dof_values(ph0), Mat_Aup.cols)
  vec_um = GridapDistributed.change_ghost(get_free_dof_values(uh0), Mat_Auu.cols)
  vec_am = PVector(0.0, Mat_ML.cols)
  vec_sum_pm = PVector(0.0, Mat_Aup.cols)
  Δa_star = PVector(0.0, Mat_Apu.cols)
  Δpm1 = PVector(0.0, Mat_Aup.cols)

  b1 = PVector(0.0, vec_Au.rows)
  b2 = PVector(0.0, vec_Ap.rows)

  M = 5
  tol = 1e-4
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

  for tn in time_step
    err = 1
    m = 0

    while err > tol

      GridapPETSc.with() do
        vec_pm = GridapDistributed.change_ghost(vec_pm, Mat_Aup.cols)
        vec_um = GridapDistributed.change_ghost(vec_um, Mat_Auu.cols)
        vec_am = GridapDistributed.change_ghost(vec_am, Mat_ML.cols)
        vec_sum_pm = GridapDistributed.change_ghost(vec_sum_pm, Mat_Aup.cols)

        solver_v = PETScLinearSolver(vel_kspsetup)
        println("solving velocity")
        @time begin
          ss1 = symbolic_setup(solver_v, Mat_ML)
          ns1 = numerical_setup(ss1, Mat_ML)
          
          @time b1 .= -Mat_Auu * vec_um - Mat_Aup * vec_pm - Mat_ML * vec_am +
                      Mat_Auu * dt * vec_am + (1 - θ) * Mat_Aup * vec_sum_pm + vec_Au
        end #end begin

        @time solve!(Δa_star, ns1, b1)

        vec_um = GridapDistributed.change_ghost(vec_um, Mat_Apu.cols)
        vec_pm = GridapDistributed.change_ghost(vec_pm, Mat_App.cols)
        vec_am = GridapDistributed.change_ghost(vec_am, Mat_Tpu.cols)


        println("solving pressure")



        #-Vec_A because changing sign in the continuity equations
        @time begin
          b2 .= Mat_Tpu * Δa_star + Mat_Apu * (vec_um + dt * Δa_star) + Mat_App * vec_pm + Mat_Tpu * vec_am - vec_Ap

          solver_p = PETScLinearSolver(pres_kspsetup)

          ss2 = symbolic_setup(solver_p, Mat_S)
          ns2 = numerical_setup(ss2, Mat_S)
         end #end begin
        @time solve!(Δpm1, ns2, b2)

      end #end do GridapPETSc


      println("update end")
      @time begin
        Δa = Δa_star - θ .* Mat_inv_ML .* (Mat_Aup * Δpm1)

        vec_um .= vec_um + dt * Δa
        vec_pm .= vec_pm + Δpm1


        println("inner iter = $m")
        if m == 0
          vec_sum_pm .= Δpm1
          vec_am .= Δa
        else
          vec_sum_pm .+= Δpm1
          vec_am .+= Δa

        end

        m = m + 1
        err = norm(Δa) / N
      end #end begin
      println("error = $err")

      println("garbge collection")

    end #end while
    @time GridapPETSc.gridap_petsc_gc() # Do garbage collection of PETSc objects

    u_adv = FEFunction(U(tn), vec_um)
    println("update_matrices")
    @time Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, Mat_ML, Mat_S = define_matrices(tn, V, Q, P, U)
    Mat_inv_ML = inv_lump_vel_mass!(Mat_ML)
    vec_Ap, vec_Au = define_vectors(tn, V, Q, P, U)


    uh_tn = FEFunction(U(tn), vec_um)
    ph_tn = FEFunction(P(tn), vec_pm)

    # writevtk(Ω, "TG_segregated_$tn.vtu", cellfields = ["uh" => uh_tn, "uh_analytic"=> velocity(tn), "ph" => ph_tn, "ph_analytic"=> pa(tn)])
  end #end for

end #end with_backend


#mpiexecjl --project=. -n 4 julia Taylor_Green_Segregated_parallel.jl
