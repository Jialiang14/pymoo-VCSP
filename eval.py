from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# problem = get_problem("vcsp1")
problem = get_problem("zdt1")

# NSGA2
# algorithm = NSGA2(pop_size=20)
# res = minimize(problem,
#                algorithm,
#                ('n_gen', 20),
#                seed=1,
#                verbose=True)
#
# plot = Scatter()
# # plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, color="red")
# plot.show()

# MOEA/D
# from pymoo.util.ref_dirs import get_reference_directions
# ref_dirs = get_reference_directions("uniform", 2, n_partitions=12)
#
# algorithm = MOEAD(
#     ref_dirs,
#     n_neighbors=10,
#     prob_neighbor_mating=0.7,
# )
#
# res = minimize(problem,
#                algorithm,
#                ('n_gen', 20),
#                seed=1,
#                verbose=False)
#
# Scatter().add(res.F).show()


algorithm = AGEMOEA2(pop_size=20)

res = minimize(problem,
               algorithm,
               ('n_gen', 20),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
