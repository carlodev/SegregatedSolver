function vel_kspsetup(ksp)
    pc = Ref{GridapPETSc.PETSC.PC}()
    @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[], GridapPETSc.PETSC.KSPGMRES)
    # @check_error_code GridapPETSc.PETSC.KSPSetReusePreconditioner(ksp[], GridapPETSc.PETSC.PETSC_TRUE)
    @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[], pc)
    @check_error_code GridapPETSc.PETSC.PCSetType(pc[], GridapPETSc.PETSC.PCGAMG)  
  end
  
  function pres_kspsetup(ksp)
    pc = Ref{GridapPETSc.PETSC.PC}()
    @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[], GridapPETSc.PETSC.KSPCG)
    # @check_error_code GridapPETSc.PETSC.KSPSetReusePreconditioner(ksp[], GridapPETSc.PETSC.PETSC_TRUE)
    @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[], pc)
    @check_error_code GridapPETSc.PETSC.PCSetType(pc[], GridapPETSc.PETSC.PCGAMG)
    
  end