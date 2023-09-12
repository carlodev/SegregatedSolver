function main(params)
    init_params(params)
    @unpack case, backend, rank_partition = params
    


    if case == "TaylorGreen"
        run_function = run_taylorgreen
    elseif case == "Airfoil"
        run_function = run_airfoil
    elseif case == "LidDriven"
        run_function = run_liddriven
    else
        @error "Case $case not recognized as valid"
    end

    backend() do distribute
        if backend == with_mpi
            comm = MPI.COMM_WORLD
            #To avoid multiple printing of the same line in parallel
            if MPI.Comm_rank(comm) != 0
              redirect_stderr(devnull)
              redirect_stdout(devnull)
            end
           
        end

        run_function(params,distribute)
    end

end
