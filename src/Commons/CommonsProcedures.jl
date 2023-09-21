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
    @unpack U,P,u0, D, restart,t0,Ω = params


        uh0 = interpolate_everywhere(u0(t0), U(t0))
        ph0 = interpolate_everywhere(0.0, P(t0))

        if !restart
          if haskey(params,:p0)
            @unpack p0 = params
            ph0 = interpolate_everywhere(p0(t0), P(t0))
          
          end
        else

          uh_0 = restart_uh_field(params)
          ph_0 = restart_ph_field(params)
          uh0 = interpolate_everywhere(uh_0, U(t0))
          ph0 = interpolate_everywhere(ph_0, P(t0))

        end

  writevtk(Ω,"Initial_Conditions", cellfields=["uh0"=>uh0,"ph0"=>ph0])
    
  return uh0,ph0
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


uh0, ph0 = create_initial_conditions(params)

matrices = initialize_matrices_and_vectors(trials,tests, t0+dt, uh0, params; method=method)

Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, 
Mat_App, Mat_ML, Mat_inv_ML, Mat_S, Vec_Au, Vec_Ap = matrices

vec_pm,vec_um,vec_am,vec_sum_pm,Δa_star,Δpm1,Δa,b1,b2,ũ_vector = initialize_vectors(matrices,uh0,ph0)

if case == "TaylorGreen"
  @unpack u0,p0 = params
end

local_unique_idx = get_nodes(params)


for (ntime,tn) in enumerate(time_step)

    m = 0
    GridapPETSc.with(args=split(petsc_options)) do

        ns1 = create_PETSc_setup(Mat_ML,vel_kspsetup)
        ns2 = create_PETSc_setup(Mat_S,pres_kspsetup)


      time_solve = @elapsed begin 


        vec_am .= pzeros(Mat_ML)
        vec_sum_pm .= pzeros(Mat_Aup)
        
        norm_Δa0 = 10
        norm_Δp0 = 10
        err_norm_Δa0 = 1
        err_norm_Δp0 = 1
        
        M
      
      while (m<= M) && (err_norm_Δa0<200)

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
          norm_Δa0 = norm(Δa)
          norm_Δp0 = norm(Δpm1)

        else
          vec_sum_pm .+= Δpm1
          vec_am .+= Δa

        end
        
        err_norm_Δa0 = norm_Δa0/norm(Δa)
        err_norm_Δp0 = norm_Δp0/norm(Δpm1)

        println("err a")
        println(err_norm_Δa0)

        println("err p")
        println(err_norm_Δp0)

        m = m + 1
      
    end  #end while
  end #end elapsed


  println("solution time")
  println(time_solve)
    GridapPETSc.GridapPETSc.gridap_petsc_gc()
  end #end GridapPETSc

  update_ũ_vector!(ũ_vector,vec_um)
  # uh_tn_updt = FEFunction(U(tn+dt), update_ũ(ũ_vector))
  uh_tn_updt = FEFunction(U(tn+dt), vec_um)
#  if tn<t_endramp
#     uh_tn_updt = FEFunction(U(tn+dt), vec_um)
#  end

uh_tn = FEFunction(U(tn), vec_um)
ph_tn = FEFunction(P(tn), vec_pm)
save_path = "$(case)_$(tn)_.vtu"
  if !benchmark && (mod(ntime,100)==0 || ntime<10) 


    if case == "TaylorGreen"
        writevtk(Ω, save_path, cellfields = ["uh" => uh_tn, "uh_analytic"=> u0(tn), "ph" => ph_tn, "ph_analytic"=> p0(tn)])
    else
       @time writevtk(Ω, save_path, cellfields = ["uh" => uh_tn, "uh_updt" => uh_tn_updt, "ph" => ph_tn])

    end
  end
  export_fields(params, local_unique_idx, tn, uh_tn, ph_tn)

  if mod(ntime,10)==0
    println("update_matrices")

    @time begin
     Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, 
      Mat_ML, Mat_inv_ML, Mat_S, Vec_Au, Vec_Ap = matrices_and_vectors(trials, tests, tn+dt, uh_tn_updt, params; method=method)
    end
  end

  end #end for


end