"""
It creates the finite elements spaces accordingly to the previously generated dirichelet tags
"""
function creation_fe_spaces(params::Dict{Symbol,Any}, u_diri_tags, u_diri_values, p_diri_tags, p_diri_values)
    reffeᵤ = ReferenceFE(lagrangian, VectorValue{params[:D],Float64}, params[:order])
    reffeₚ = ReferenceFE(lagrangian, Float64, params[:order])


    V = TestFESpace(params[:model], reffeᵤ, conformity=:H1, dirichlet_tags=u_diri_tags)
    U = TransientTrialFESpace(V, u_diri_values)

    Q = TestFESpace(params[:model], reffeₚ, conformity=:H1, dirichlet_tags=p_diri_tags)
    P = TrialFESpace(Q, p_diri_values)

    Y = MultiFieldFESpace([V, Q])
    X = TransientMultiFieldFESpace([U, P])

    return V, U, P, Q, Y, X
end


function create_initial_conditions(params::Dict{Symbol,Any})
    @unpack U,P,u0, D, restart,dt,t0 = params


        uh0 = interpolate_everywhere(u0(t0), U(t0))
        uh_adv = interpolate_everywhere(u0(t0+dt), U(t0))

        if !restart
          if haskey(params,:p0)
            @unpack p0 = params
            ph0 = interpolate_everywhere(p0(t0), P(t0))
          else
            ph0 = interpolate_everywhere(t0, P(t0))

          end
        end

    return uh0,uh_adv,ph0
end


function create_PETSc_setup(M::AbstractMatrix,ksp_setup::Function)
      solver = PETScLinearSolver(ksp_setup)
      ss = symbolic_setup(solver, M)
      ns = numerical_setup(ss, M)

      return ns
end

function solve_case(params::Dict{Symbol,Any})

@unpack M, petsc_options, time_step, θ, dt,t0, t_endramp, case, benchmark, method, trials, tests, Ω = params
@unpack U,P,u0 = params


uh0,uh_adv, ph0 = create_initial_conditions(params)

matrices = initialize_matrices_and_vectors(trials,tests, t0+dt, uh_adv, params; method=method)

Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, 
Mat_App, Mat_ML, Mat_inv_ML, Mat_S, Vec_Au, Vec_Ap = matrices

vec_pm,vec_um,vec_am,vec_sum_pm,Δa_star,Δpm1,Δa,b1,b2,ũ_vector = initialize_vectors(matrices,uh0,ph0)


if case == "TaylorGreen"
  @unpack u0,p0 = params
end

for (ntime,tn) in enumerate(time_step)
    m = 0
    println("norm Au")
    println(norm(Vec_Au))

    println("norm Ap")
    println(norm(Vec_Ap))

    GridapPETSc.with(args=split(petsc_options)) do

        ns1 = create_PETSc_setup(Mat_ML,vel_kspsetup)
        ns2 = create_PETSc_setup(Mat_S,pres_kspsetup)


      time_solve = @elapsed begin 


        vec_am .= pzeros(Mat_ML)
        vec_sum_pm .= pzeros(Mat_Aup)
    while m<=M

        Δpm1 .=  pzeros(Mat_S)
        Δa_star .= pzeros(Mat_ML)

        vec_um = GridapDistributed.change_ghost(vec_um, Mat_Auu)
        vec_pm = GridapDistributed.change_ghost(vec_pm, Mat_Aup)
        vec_am = GridapDistributed.change_ghost(vec_am, Mat_ML)


        println("solving velocity")
          
          b1 .= -Mat_Auu * vec_um - Mat_Aup * vec_pm - Mat_ML * vec_am +
          Mat_Auu * dt * vec_am + (1 - θ) * Mat_Aup * vec_sum_pm + Vec_Au

          @time solve!(Δa_star,ns1,b1)
          


        vec_um = GridapDistributed.change_ghost(vec_um, Mat_Apu)
        vec_pm = GridapDistributed.change_ghost(vec_pm, Mat_App)
        vec_am = GridapDistributed.change_ghost(vec_am, Mat_Tpu)

        Δa_star = GridapDistributed.change_ghost(Δa_star, Mat_Tpu)

        println("solving pressure")

        #-Vec_A because changing sign in the continuity equations
        b2 .= Mat_Tpu * Δa_star + Mat_Apu * (vec_um + dt * Δa_star) + Mat_App * vec_pm + Mat_Tpu * vec_am - Vec_Ap

        @time solve!(Δpm1,ns2,b2)

      
      println("update end")
      Δpm1 = GridapDistributed.change_ghost(Δpm1, Mat_Aup)

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
      
    end  #end while
  end #end elapsed
  println("solution time")
  println(time_solve)
    GridapPETSc.GridapPETSc.gridap_petsc_gc()
  end #end GridapPETSc


  if !benchmark 
    uh_tn = FEFunction(U(tn), vec_um)
    ph_tn = FEFunction(P(tn), vec_pm)
    
    if case == "TaylorGreen"
        writevtk(Ω, "$(case)_$tn.vtu", cellfields = ["uh" => uh_tn, "uh_analytic"=> u0(tn), "ph" => ph_tn, "ph_analytic"=> p0(tn)])
    else
        writevtk(Ω, "$(case)_$tn.vtu", cellfields = ["uh" => uh_tn, "ph" => ph_tn])

    end
  end

  update_ũ_vector!(ũ_vector,vec_um)
  
  if tn>t_endramp
    uh_tn = FEFunction(U(tn+dt), update_ũ(ũ_vector))

  end
  
  println("update_matrices")
    @time begin
     Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, 
      Mat_ML, Mat_inv_ML, Mat_S, Vec_Au, Vec_Ap = matrices_and_vectors(trials, tests, tn+dt, uh_tn, params; method=method)
    end

  end #end for

end