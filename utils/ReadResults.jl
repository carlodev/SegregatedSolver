using CSV, DataFrames, Plots, XLSX
using UnicodePlots
using Trapz
include("ReadResults_Utils.jl")

Re = 250000
u0 = 1.0
c = 1.0
rho = 1.0
μ = u0*c*rho/Re
α = 1.0

global results_path = "Results/"

cellfields = Dict("ph"=>(:scalar_value,1), "friction"=>(:scalar_value,2))




file_idx, dir_idx, path = get_file_dir_idx(results_path)

field_file_idx, field_dir_idx = get_file_dir_idx_fields(path, file_idx, dir_idx, cellfields)

nodes_unique,unique_idx = get_nodes(path)
n_Γ = get_normals(path,unique_idx)

move_files(path,dir_idx)

read_field_directories(path, dir_idx, field_dir_idx, cellfields, unique_idx)

clear_directories(path,file_idx,dir_idx,field_file_idx, field_dir_idx,cellfields)


#ReadFiles

Ph = average_field(path, "ph", cellfields, file_idx, field_file_idx, unique_idx;offset = 1)
Friction = average_field(path, "friction", cellfields, file_idx, field_file_idx, unique_idx;offset = 1)


top_nodesx,bottom_nodesx,top_nodesy,bottom_nodesy,cp_top,cp_bottom,friction_top,friction_bottom,n_Γ_airfoil_top,n_Γ_airfoil_bottom= extract_airfoil_features(nodes_unique, n_Γ, Ph, Friction; u0=u0, A=c, rho=rho, α = α)






t_Γ_airfoil_top = map(x->get_tangent_x([n_Γ_airfoil_top[x,:]...]), 1:1:size(n_Γ_airfoil_top)[1] )
t_Γ_airfoil_bottom = map(x->get_tangent_x([n_Γ_airfoil_bottom[x,:]...]), 1:1:size(n_Γ_airfoil_bottom)[1] )


star_top_Cp = DataFrame(XLSX.readtable("Star.xlsx","Cp_top"))
star_bottom_Cp = DataFrame(XLSX.readtable("Star.xlsx","Cp_bottom"))

star_top_Cf = DataFrame(XLSX.readtable("Star.xlsx","Cf_top"))
star_bottom_Cf = DataFrame(XLSX.readtable("Star.xlsx","Cf_bottom"))


gr()
plt_Cf = plot(xlabel ="x/c",ylabel="Cf",ylims=([-0.02,0.02]))
plot!(top_nodesx, friction_top,linecolor =:red, label = "vms")
plot!(bottom_nodesx, friction_bottom,linecolor =:red,label = false)
plot!(star_top_Cf.xf./0.2,star_top_Cf.Cf, linecolor =:blue, label = "star")
plot!(star_bottom_Cf.xf./0.2,star_bottom_Cf.Cf, linecolor =:blue, label = false)

Plots.savefig(plt_Cf, "Cf_2D_du89.pdf")

plt_Cp = plot(xlabel ="x/c",ylabel="Cp")
plot!(top_nodesx, cp_top,linecolor=:red, label = "vms")
plot!(bottom_nodesx, cp_bottom,linecolor =:red, label = false)
plot!(star_top_Cp.xp ./0.2,star_top_Cp.Cp, linecolor =:blue, label = "star")
plot!(star_bottom_Cp.xp ./0.2,star_bottom_Cp.Cp , linecolor =:blue, label = false)
yflip!()
Plots.savefig(plt_Cp, "Cp_2D_du89.pdf")


CL_p = trapz(bottom_nodes,cp_bottom.* (abs.(n_Γ_airfoil_bottom.y) )) - trapz(top_nodes,cp_top.* abs.(n_Γ_airfoil_top.y))
CD_p = -trapz(top_nodes,cp_top.*map(x->x[2], t_Γ_airfoil_top))+trapz(bottom_nodes,cp_bottom.*map(x->x[2], t_Γ_airfoil_bottom))

CD_f = trapz(bottom_nodes,friction_bottom.*map(x->x[1], t_Γ_airfoil_bottom)) + trapz(bottom_nodes,friction_bottom.*map(x->x[1], t_Γ_airfoil_bottom))
CD = CD_p+CD_f
CL_p/CD


plt_Cf = lineplot(top_nodes, friction_top,color =:red, name = "vms", width = 150,height=50)
lineplot!(plt_Cf,bottom_nodes, friction_bottom,color =:red)
lineplot!(plt_Cf,Float64.(star_top_Cf.xf)./0.2, Float64.(star_top_Cf.Cf), color =:blue, name = "star")
lineplot!(plt_Cf,Float64.(star_bottom_Cf.xf)./0.2,Float64.(star_bottom_Cf.Cf), color =:blue )



plt_Cp = lineplot(top_nodes, cp_top,color =:red, name = "vms", width = 150,height=50)
lineplot!(plt_Cp,bottom_nodes, cp_bottom,color =:red)
lineplot!(plt_Cp,Float64.(star_top_Cp.xp)./0.2, Float64.(star_top_Cp.Cp), color =:blue, name = "star")
lineplot!(plt_Cp,Float64.(star_bottom_Cp.xp)./0.2,Float64.(star_bottom_Cp.Cp), color =:blue )




dt = 0.00025
DT = 0.1
Int64.(round(2.5))