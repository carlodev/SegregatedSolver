function create_initial_conditions(params::Dict{Symbol,Any})
    @unpack U,P,u0,p0,D = params


        uh0 = interpolate_everywhere(u0(0.0), U(0.0))
        ph0 = interpolate_everywhere(p0(0.0), P(0.0))


    return uh0,ph0
end


function create_PETSc_setup(M::AbstractMatrix,ksp_setup::Function)
      solver = PETScLinearSolver(ksp_setup)
      ss = symbolic_setup(solver, M)
      ns = numerical_setup(ss, M)

      return ns
end

function solve_case(params::Dict{Symbol,Any})

@unpack M, petsc_options, time_step, θ, dt, case, benchmark, method, trials, tests, Ω = params
@unpack U,P = params


uh0, ph0 = create_initial_conditions(params)

matrices = initialize_matrices_and_vectors(trials,tests, 0.0, uh0, params; method=method)

Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, 
Mat_App, Mat_ML, Mat_inv_ML, Mat_S, Vec_Au, Vec_Ap = matrices

vec_pm,vec_um,vec_am,vec_sum_pm,Δa_star,Δpm1,Δa,b1,b2,ũ_vector = initialize_vectors(matrices,uh0,ph0)



@unpack u0,p0 = params



for tn in time_step
    m = 0
    GridapPETSc.with(args=split(petsc_options)) do

        ns1 = create_PETSc_setup(Mat_ML,vel_kspsetup)
        ns2 = create_PETSc_setup(Mat_S,pres_kspsetup)


      time_solve = @elapsed begin 
    while m<=M

        @time Δpm1 .=  pzeros(Mat_S)
        @time Δa_star .= pzeros(Mat_ML)

        vec_um = GridapDistributed.change_ghost(vec_um, Mat_Auu)
        vec_pm = GridapDistributed.change_ghost(vec_pm, Mat_Aup)
        vec_am = GridapDistributed.change_ghost(vec_am, Mat_ML)


        println("solving velocity")
          
          @time b1 .= -Mat_Auu * vec_um - Mat_Aup * vec_pm - Mat_ML * vec_am +
          Mat_Auu * dt * vec_am + (1 - θ) * Mat_Aup * vec_sum_pm + Vec_Au

          @time solve!(Δa_star,ns1,b1)
          


        vec_um = GridapDistributed.change_ghost(vec_um, Mat_Apu)
        vec_pm = GridapDistributed.change_ghost(vec_pm, Mat_App)
        vec_am = GridapDistributed.change_ghost(vec_am, Mat_Tpu)

        @time Δa_star = GridapDistributed.change_ghost(Δa_star, Mat_Tpu)

        println("solving pressure")

        #-Vec_A because changing sign in the continuity equations
        @time b2 .= Mat_Tpu * Δa_star + Mat_Apu * (vec_um + dt * Δa_star) + Mat_App * vec_pm + Mat_Tpu * vec_am - Vec_Ap

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
  uh_tn = FEFunction(U(tn), update_ũ(ũ_vector))

    println("update_matrices")
    @time begin

  Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, 
  Mat_ML, Mat_inv_ML, Mat_S, Vec_Au, Vec_Ap = matrices_and_vectors(trials, tests, tn, uh_tn, params; method=method)
  

  end

  end #end for

end