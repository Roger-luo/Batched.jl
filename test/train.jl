using MPSFilter, Lattices, LatticeSites, Flux, Distributed, JLD2
using Lattices: Chain
using Flux.Optimise: ADAM, SGD


function train(plan::TrainingPlan, opt)
    history = Float64[]
    H_m = nothing

    println(plan)
    procs = addprocs(plan.nchains; exeflags="--project")
    Distributed.remotecall_eval(Main, procs, :(using MPSFilter))

    for i in 1:plan.nsteps
        try
            println("step ", i, " :")
            samples = psample(plan.model, plan.lattice; nsamples=plan.nsamples, nburn=plan.nburn, nchains=plan.nchains)
            loss, _, _ = syn_grad!(plan.parameters, plan.model, plan.lattice,
                plan.hamiltonian, samples; progressbar=plan.log_config.sample_progress)

            if plan.log_config.exact_energy
                Psi = [plan.model(each) for each in HilbertSpace(plan.lattice)]
                exact_loss = (Psi' * Matrix{Float64}(plan.hamiltonian) * Psi) / (Psi' * Psi)
                println("energy (exact) = ", exact_loss / length(plan.lattice))
            end
            println("energy (sampled) = ", loss / length(plan.lattice))
            push!(history, loss)
            opt()
        catch e
            if e isa InterruptException
                save(plan, history)
            end
            rethrow(e)
        end
    end
    save(plan, history)
    history
end

# function train(model, lattice, opt; epoch=2000)
#     history = Float64[]
#     parameters = params(model)
#
#     for i in 1:epoch
#         Psi = [model(each) for each in HilbertSpace(lattice)]
#         exact_loss = (Psi' * AFM_m(lattice) * Psi) / (Psi' * Psi) / length(lattice)
#
#         println("step ", i, " :")
#         println("energy (exact) = ", exact_loss)
#         push!(history, data(exact_loss))
#
#         Flux.Tracker.back!(exact_loss)
#         opt()
#     end
#     history
# end
#
function vmc_train(model, lattice, opt; nsamples=i->(500+50i), nburn=i->1000, nchains=4)
    history = Float64[]
    parameters = params(model)

    for i in 1:2000
        Psi = [model(each) for each in HilbertSpace(lattice)]
        exact_loss = (Psi' * AFM_m(lattice) * Psi) / (Psi' * Psi) / length(lattice)
        samples = psample(model, lattice; nsamples=nsamples(i), nburn=nburn(i), nchains=nchains)
        loss, _, _ = syn_grad!(parameters, model, lattice, AFM(lattice), samples)
        # loss = energy(model, AFM(lattice); nsamples=nsamples(i), nburn=nburn) / length(lattice)

        println("step ", i, " :")
        println("energy (exact) = ", exact_loss)
        println("energy (sampled) = ", loss / length(lattice))
        push!(history, loss)
        opt()
    end
    history
end


# plan = TrainingPlan(
#     model=BlackKingBar(4, nkernel=10, bonds=(2, 2), shape=(16, 8, 1)),
#     nsteps=1000,
#     lattice=Chain(4),
#     hamiltonian=AFM,
#     nsamples=1000,
#     nburn=1000,
#     nchains=1,
#     optimizer=ps->ADAM(ps, 0.001; β1 = 0.98, β2 = 0.989, ϵ = 1e-08, decay = 0),
#     log=LogConfig(
#         exact_energy=true,
#         sample_progress=false,
#     )
# )
#
# train(plan)

lattice = Chain(4)
model=BlackKingBar(4, nkernel=10, bonds=(2, 2), shape=(16, 8, 1))
H = AFM(lattice)
opt = ADAM(params(model), 0.001; β1 = 0.98, β2 = 0.989, ϵ = 1e-08, decay = 0)
history = vmc_train(model, lattice, opt; nsamples=i->(500+50 * (i ÷ 50)), nburn=i->1000, nchains=nprocs())
